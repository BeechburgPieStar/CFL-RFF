import torch
import torch.nn as nn
import random

class XInteract(nn.Module):
    def __init__(self, xi_rate=0.5):
        super().__init__()
        self.xi_rate = xi_rate
    
    def forward(self, xa, xb):
        if self.training:
            if random.random() > self.xi_rate:
                return xb, xa
        return xa, xb

class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size // 2
        if seq_len < patch_size:
            raise ValueError(f"seq_len({seq_len}) must be >= patch_size({patch_size}).")
        if stride <= 0:
            raise ValueError(f"stride({stride}) must be > 0.")
        num_patches = (seq_len - patch_size) // stride + 1
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x)
        x_out = x_out.transpose(1, 2) 
        return x_out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dr):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dr),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dr)
        )
    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dr):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dr)

    def forward(self, x):
        x = x + self.ffn(self.norm(x))
        return x

class PatchNet(nn.Module):
    def __init__(self,
                 seq_len=256,
                 in_chans=2,
                 patch_size=64,
                 embed_dim=128,
                 num_classes=6,
                 mlp_ratio=4.0,
                 use_xi=1,
                 lamb_rank=1.0,
                 dr=0.5,
                 stride=None):
        super().__init__()
        self.patch_embedding_a = PatchEmbed(seq_len, patch_size, in_chans, embed_dim, stride=stride)
        self.bottleneck_a = MLP(embed_dim, mlp_ratio=mlp_ratio, dr=dr)
        self.cls_head_a = nn.Linear(embed_dim, num_classes)

        self.patch_embedding_b = PatchEmbed(seq_len, patch_size, in_chans, embed_dim, stride=stride)
        self.bottleneck_b = MLP(embed_dim, mlp_ratio=mlp_ratio, dr=dr)
        self.cls_head_b = nn.Linear(embed_dim, num_classes)

        self.interact = XInteract(xi_rate=0.5)
        self.use_xi = use_xi
        self.lamb_rank = lamb_rank

    def forward(self, xa, xb):
        x_patch_a = self.patch_embedding_a(xa)
        x_patch_b = self.patch_embedding_b(xb)

        x_neck_a = self.bottleneck_a(x_patch_a)
        x_neck_b = self.bottleneck_b(x_patch_b)

        if self.use_xi:
            x_neck_a, x_neck_b = self.interact(x_neck_a, x_neck_b)

        x_mean_a = x_neck_a.mean(dim=1)
        x_mean_b = x_neck_b.mean(dim=1)

        logit_a = self.cls_head_a(x_mean_a)
        logit_b = self.cls_head_b(x_mean_b)

        energy_a = torch.log(torch.sum(torch.exp(logit_a), dim=1))
        energy_b = torch.log(torch.sum(torch.exp(logit_b), dim=1))

        conf_a0 = energy_a / 10
        conf_b0 = energy_b / 10

        conf_a = conf_a0.unsqueeze(1)
        conf_b = conf_b0.unsqueeze(1)

        if self.lamb_rank == 0:
            logit_ab = logit_a * conf_a.detach() + logit_b * conf_b.detach()
        else:
            logit_ab = (logit_a + logit_b) / 2
        return logit_ab, logit_a, logit_b, conf_a0, conf_b0

from thop import profile, clever_format
import logging

# 立即设置日志级别
logging.getLogger('thop').setLevel(logging.ERROR)

# 计算模型的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 自动格式化数字并转换单位
def format_size(size):
    # 参数量单位转换（B -> K -> M -> G -> T）
    for unit in ['B', 'K', 'M', 'G', 'T']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} T"

# 计算前向计算的 FLOPS (Floating Point Operations)
def compute_flops(model, input_tensor_a, input_tensor_b):
    # 使用thop计算 FLOPS 和参数
    # Combine inputs to fit the profiling requirements
    input_tensor = (input_tensor_a, input_tensor_b)  # Packing both tensors in a tuple
    flops, params = profile(model, inputs=input_tensor)
    return flops, params

# 计算模型复杂度并打印
def calculate_model_complexity(model, input_tensor_a, input_tensor_b):
    # 计算模型参数量
    num_params = count_parameters(model)
    print(f"Model Parameters: {format_size(num_params)}")
    
    # 计算计算量 (FLOPS)
    flops, params = compute_flops(model, input_tensor_a, input_tensor_b)
    print(f"Model FLOPS: {clever_format(flops, '%.3f')}")
    print(f"Model Params: {clever_format(params, '%.3f')}")

if __name__ == "__main__":
    # 输入参数设置
    seq_len = 256
    in_chans = 2
    patch_size = 64
    embed_dim = 128
    num_classes = 6
    batch_size = 8
    use_xi = 1

    # 创建模型实例
    model = PatchNet(
        seq_len=seq_len,
        in_chans=in_chans,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        mlp_ratio=2.0,
        use_xi=use_xi,
        dr=0.5
    )

    # 假设输入是batch_size x in_chans x seq_len大小的张量
    xa = torch.randn(batch_size, in_chans, seq_len)
    xb = torch.randn(batch_size, in_chans, seq_len)

    # 前向传播，计算logits和confidences
    logit_ab, logit_a, logit_b, conf_a0, conf_b0 = model(xa, xb)

    # 打印输出张量的形状
    print("logit_ab shape:", logit_ab.shape)
    print("logit_a shape:", logit_a.shape)
    print("logit_b shape:", logit_b.shape)
    print("conf_a0 shape:", conf_a0.shape)
    print("conf_b0 shape:", conf_b0.shape)
    print("\nlogit_ab sample:\n", logit_ab[0])
    print("\nConfidence a:", conf_a0)
    print("Confidence b:", conf_b0)

    # 计算模型的复杂度
    print("\nCalculating Model Complexity:")
    calculate_model_complexity(model, xa, xb)