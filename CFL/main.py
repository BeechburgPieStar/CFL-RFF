import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from utils.load_data import load_single_dataset_pair
from backbones.PatchNet import PatchNet

def parse_args():
    parser = argparse.ArgumentParser("PatchNet")
    # 训练通用
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=None, help="梯度裁剪阈值；None 表示不裁剪")
    parser.add_argument('--patience_early_stop', type=int, default=10)
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument('--dataset_name', type=str, default="ManySig")
    parser.add_argument('--exp', type=str, default="CRD")
    parser.add_argument('--train_date', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--all_test_round', type=int, default=4)
    parser.add_argument('--test_round', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--mlp_ratio', type=float, default=2.0)
    parser.add_argument('--lamb_rank', type=float, default=1.0, help="weight for rank loss")
    parser.add_argument('--use_xi', type=int, default=1, help="use cross-branch interaction")
    parser.add_argument('--w_eq', type=int, default=1, help="use eq branch")
    parser.add_argument('--wo_eq', type=int, default=1, help="use non_eq branch")
    parser.add_argument('--code_state', type=str, default="only_test", choices=["only_train", "only_test", "train_test"])
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_receivers(all_num=12, all_test_round=4, test_round=0):
    if not (0 <= test_round < all_test_round):
        raise ValueError(f"test_round {test_round} not in [0, {all_test_round-1}]")
    if all_num % all_test_round != 0:
        raise ValueError(f"total rx num {all_num} not divisible by rounds {all_test_round}")

    receivers = list(range(all_num))
    per_round = all_num // all_test_round
    start = test_round * per_round
    end = all_num if test_round == all_test_round - 1 else start + per_round
    test = receivers[start:end]
    train = [r for r in receivers if r not in test]
    return train, test

def prepare_dataset(dataset_name, rx_indexes, date_indexes, tx_num, is_train, seed):
    x_all_a, x_all_b = [], []
    y_all = []
    for rx_index in rx_indexes:
        for date_index in date_indexes:
            xa, xb, y = load_single_dataset_pair(dataset_name, rx_index, date_index, tx_num)
            x_all_a.append(xa)
            x_all_b.append(xb)
            y_all.append(y)

    x_all_a = np.concatenate(x_all_a, axis=0)
    x_all_b = np.concatenate(x_all_b, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    if is_train:
        indices = np.arange(len(y_all))
        train_idx, val_idx = train_test_split(indices, test_size=0.3, random_state=seed)

        x_a_train, x_a_val = x_all_a[train_idx, :, :], x_all_a[val_idx, :, :]
        x_b_train, x_b_val = x_all_b[train_idx, :, :], x_all_b[val_idx, :, :]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        return (x_a_train, x_b_train, y_train), (x_a_val, x_b_val, y_val)

    return x_all_a, x_all_b, y_all


def _to_device(data_a, data_b, target, device):
    data_a = data_a.to(device).float()
    data_b = data_b.to(device).float()
    target = target.to(device).long()
    return data_a, data_b, target

class IndexedTensorDataset(TensorDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return (*data, torch.tensor(idx, dtype=torch.long))

class History:
    def __init__(self, n_samples, momentum: float = 0.9):
        self.momentum = momentum
        self.n = n_samples
        self.avg_loss = torch.zeros(n_samples, dtype=torch.float32)
        self.avg_conf = torch.zeros(n_samples, dtype=torch.float32)

    @torch.no_grad()
    def correctness_update(self, idx: torch.Tensor, loss_vec: torch.Tensor, conf_vec: torch.Tensor):
        idx_cpu = idx.detach().cpu()
        loss_cpu = loss_vec.detach().cpu().to(torch.float32)
        conf_cpu = conf_vec.detach().cpu().to(torch.float32)

        m = self.momentum
        self.avg_loss[idx_cpu] = m * self.avg_loss[idx_cpu] + (1 - m) * loss_cpu
        self.avg_conf[idx_cpu] = m * self.avg_conf[idx_cpu] + (1 - m) * conf_cpu

    def get_target_margin(self, idx1: torch.Tensor, idx2: torch.Tensor):
        l1 = self.avg_loss[idx1.detach().cpu()]
        l2 = self.avg_loss[idx2.detach().cpu()]
        diff = l1 - l2
        rank_target = torch.where(diff > 0, torch.tensor(-1.0), torch.tensor(1.0))
        rank_margin = diff.abs().to(torch.float32)

        device = idx1.device
        return rank_target.to(device), rank_margin.to(device)


def rank_loss(confidence: torch.Tensor, idx: torch.Tensor, history: History):
    if confidence.dim() == 2 and confidence.size(1) == 1:
        confidence = confidence.squeeze(1)

    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, shifts=-1, dims=0)
    idx2 = torch.roll(idx, shifts=-1, dims=0)

    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1.0
    rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

    loss_fn = nn.MarginRankingLoss(margin=0.0)
    ranking_loss = loss_fn(rank_input1, rank_input2, -rank_target)
    return ranking_loss


def train_epoch(model, criterion, train_loader, optimizer, epoch, device,
                lamb_rank: float = 1.0, 
                w_eq: int = 1, 
                wo_eq: int = 1,
                grad_clip=None, ce_item=None, hist_a=None, hist_b=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in train_loader:
        if len(batch) == 4:
            data_a, data_b, target, idx = batch
        else:
            data_a, data_b, target = batch
            idx = torch.arange(data_a.size(0), dtype=torch.long)

        data_a, data_b, target = _to_device(data_a, data_b, target, device)
        idx = idx.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, logit_a, logit_b, conf_a, conf_b = model(data_a, data_b)
        cls_loss_ab = criterion(logits, target) 

        cls_loss_a = criterion(logit_a, target)

        cls_loss_b = criterion(logit_b, target)

        with torch.no_grad():
            loss_a_items = ce_item(logit_a, target)  # (B,)
            loss_b_items = ce_item(logit_b, target)  # (B,)

        rloss_a = rank_loss(conf_a.squeeze(), idx, hist_a) if (hist_a is not None) else 0.0
        rloss_b = rank_loss(conf_b.squeeze(), idx, hist_b) if (hist_b is not None) else 0.0
        rloss = rloss_a + rloss_b

        if w_eq and wo_eq:
            loss = cls_loss_ab + cls_loss_a + cls_loss_b + lamb_rank * rloss
        elif not w_eq and wo_eq:
            loss = cls_loss_a
        elif w_eq and not wo_eq:
            loss = cls_loss_b
        else:
            raise ValueError(f"Invalid combination: w_eq={w_eq}, wo_eq={wo_eq}")

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if (hist_a is not None) and (hist_b is not None):
            hist_a.correctness_update(idx, loss_a_items, conf_a.squeeze())
            hist_b.correctness_update(idx, loss_b_items, conf_b.squeeze())

        total_loss += loss.item() * data_a.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data_a.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print(f"Train Epoch: {epoch}\tLoss: {avg_loss:.6f}, Acc: {correct}/{total} ({acc:.2f}%)")
    return avg_loss, acc


@torch.no_grad()
def evaluate_epoch(model, criterion, val_loader, epoch, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for data_a, data_b, target in val_loader:
        data_a, data_b, target = _to_device(data_a, data_b, target, device)
        logits, _, _, _, _ = model(data_a, data_b)
        loss = criterion(logits, target)
        total_loss += loss.item() * data_a.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data_a.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    print(f"\nValidation set: Loss: {avg_loss:.4f}, Acc: {correct}/{total} ({acc:.2f}%)\n")
    return avg_loss, acc


@torch.no_grad()
def test_epoch(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    for data_a, data_b, target in test_loader:
        data_a, data_b, target = _to_device(data_a, data_b, target, device)
        logits, _, _, _, _ = model(data_a, data_b)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data_a.size(0)
    acc = correct / max(total, 1)
    print(f"Test Accuracy: {acc:.4f}")
    return acc


def train_and_evaluate(model, train_loader, val_loader, epochs, save_path,
                       lr=1e-3, weight_decay=0.0, patience_early_stop=10,
                       scheduler_patience=5, min_lr=1e-6, grad_clip=None, device="cuda",
                       lamb_rank=1.0, w_eq=1, wo_eq=1):
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    ce_item = nn.CrossEntropyLoss(reduction='none').to(device)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                  patience=scheduler_patience, verbose=True, min_lr=min_lr)

    hist_a = History(len(train_loader.dataset))
    hist_b = History(len(train_loader.dataset))

    best_val_loss = float('inf')
    no_improve = 0

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_epoch(model, criterion, train_loader, optimizer, epoch, device,
            lamb_rank=lamb_rank, w_eq=w_eq, wo_eq=wo_eq, grad_clip=grad_clip, ce_item=ce_item,
            hist_a=hist_a, hist_b=hist_b)

        val_loss, _ = evaluate_epoch(model, criterion, val_loader, epoch, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            print(f"Validation loss improved {best_val_loss:.6f} -> {val_loss:.6f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s).")

        if no_improve >= patience_early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break

        print("------------------------------------------------")
    return best_val_loss


def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu
    setup_seed(conf.seed)

    if conf.dataset_name == "ManySig":
        tx_num, rx_num = 6, 12
    elif conf.dataset_name == "ManyRx":
        tx_num, rx_num = 10, 32
    model = PatchNet(patch_size=conf.patch_size, embed_dim=conf.embed_dim,
                     num_classes=tx_num, mlp_ratio=conf.mlp_ratio, use_xi=conf.use_xi, lamb_rank = conf.lamb_rank)

    rx_train, rx_test = split_receivers(rx_num, conf.all_test_round, conf.test_round)
    print(f"Train receivers: {rx_train}")
    print(f"Test receivers:  {rx_test}")

    (x_a_train, x_b_train, y_train), (x_a_val, x_b_val, y_val) = prepare_dataset(
        conf.dataset_name, rx_train, conf.train_date, tx_num, True, conf.seed
    )
    x_a_train = torch.tensor(x_a_train, dtype=torch.float32)
    x_b_train = torch.tensor(x_b_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_a_val = torch.tensor(x_a_val, dtype=torch.float32)
    x_b_val = torch.tensor(x_b_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_set = IndexedTensorDataset(x_a_train, x_b_train, y_train)
    train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True, drop_last=False)

    val_loader = DataLoader(TensorDataset(x_a_val, x_b_val, y_val),
                            batch_size=conf.batch_size, shuffle=False, drop_last=False)

    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    train_dates_str = "_".join(map(str, conf.train_date))
    save_name = (
        f"{conf.dataset_name}_"
        f"{conf.exp}_"
        f"date{train_dates_str}_"
        f"round{conf.test_round}_"
        f"seed{conf.seed}_"
        f"p{conf.patch_size}_"
        f"d{conf.embed_dim}_"
        f"mlp{conf.mlp_ratio}_"
        f"xi{conf.use_xi}_"
        f"lamb{conf.lamb_rank}_"
        f"eq{conf.w_eq}_woeq{conf.wo_eq}.pth"
    )
    save_path = os.path.join(save_dir, save_name)

    if conf.code_state in ["only_train", "train_test"]:
        train_and_evaluate(
            model, train_loader, val_loader,
            epochs=conf.epochs, save_path=save_path,
            lr=conf.lr, weight_decay=conf.wd,
            patience_early_stop=conf.patience_early_stop,
            scheduler_patience=conf.scheduler_patience,
            min_lr=conf.min_lr, grad_clip=conf.grad_clip, device="cuda",
            lamb_rank=conf.lamb_rank,
            w_eq=conf.w_eq, 
            wo_eq=conf.wo_eq
        )

    if conf.code_state in ["only_test", "train_test"]:
        model_test = PatchNet(patch_size=conf.patch_size, embed_dim=conf.embed_dim,
                              num_classes=tx_num, mlp_ratio=conf.mlp_ratio)
        state = torch.load(save_path, map_location="cpu")
        model_test.load_state_dict(state)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_test = model_test.to(device)

        if conf.exp == "CR":
            x_a_test, x_b_test, y_test = prepare_dataset(conf.dataset_name, rx_test, conf.train_date, tx_num, False, conf.seed)
        else:
            other_days = [d for d in [1, 2, 3, 4] if d not in conf.train_date]
            x_a_test, x_b_test, y_test = prepare_dataset(conf.dataset_name, rx_test, other_days, tx_num, False, conf.seed)

        x_a_test = torch.tensor(x_a_test, dtype=torch.float32)
        x_b_test = torch.tensor(x_b_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        test_loader = DataLoader(TensorDataset(x_a_test, x_b_test, y_test), batch_size=32, shuffle=False, drop_last=False)
        test_epoch(model_test, test_loader, device)


if __name__ == '__main__':
    main()