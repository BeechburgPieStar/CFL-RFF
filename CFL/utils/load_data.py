import os
import pickle
import numpy as np

# 当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 各数据集对应的接收机索引
rx_indexes_of_manysig = [
    '1-1', '1-19', '2-1', '2-19', '3-19', '7-7', '7-14',
    '8-8', '14-7', '18-2', '19-2', '20-1'
]

rx_indexes_of_manyrx = [
    '1-1', '1-19', '1-20', '2-1', '2-19', '3-19',
    '7-7', '7-14', '8-7', '8-8', '8-14',
    '13-7', '13-14', '14-7',
    '18-2', '18-19',
    '19-1', '19-2', '19-19', '19-20',
    '20-1', '20-19', '20-20',
    '23-1', '23-3', '23-5', '23-6', '23-7',
    '24-5', '24-6', '24-13', '24-16'
]

# all_normal_rx_indexes_of_manyrx = [
#     '1-1', '1-19', '1-20', '2-1', '2-19', '3-19', '7-7',
#     '8-7', '8-8', '8-14', '13-14', '14-7', '18-2', '18-19', '19-1',
#     '20-1', '20-19', '20-20'
# ]


def preprocessing(x):
    """
    对输入数据进行归一化，使每个样本的功率为1
    """
    for i in range(x.shape[0]):
        power = np.sum(x[i, 0, :]**2 + x[i, 1, :]**2) / x.shape[2]
        x[i] = x[i] / np.sqrt(power)
    return x

def load_single_dataset_pair(dataset, rx_index, date_index, tx_num, max_per_class=100, verbose=True):
    """
    同时读取 non_equalized 与 equalized 两支数据
    以类别为单位做对齐校验
    仅保留两支样本量一致的类别
    不一致的类别直接跳过，并输出异常类别信息

    返回
    xa: non_equalized  [N, 2, L]
    xb: equalized      [N, 2, L]
    y : 标签            [N]
    bad_classes: list  异常类别列表，每个元素是 (tx_index, n_non_eq, n_eq)
    """
    if dataset == "ManySig":
        rx_indexes = rx_indexes_of_manysig
    elif dataset == "ManyRx":
        rx_indexes = rx_indexes_of_manyrx
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    def _load(eq_flag):
        folder_path = os.path.join(current_dir, "..", "dataset", f"{dataset}/{eq_flag}")
        file_path = os.path.join(folder_path, f"date{date_index}/rx_{rx_indexes[rx_index]}_data.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    data_a = _load("non_equalized")
    data_b = _load("equalized")

    xa_list, xb_list, y_list = [], [], []
    bad_classes = []

    for tx_index in range(tx_num):
        tx_a = np.transpose(data_a["data"][tx_index], (0, 2, 1))
        tx_b = np.transpose(data_b["data"][tx_index], (0, 2, 1))

        na = int(min(max_per_class, tx_a.shape[0]))
        nb = int(min(max_per_class, tx_b.shape[0]))

        if na != nb:
            bad_classes.append((tx_index, na, nb))
            continue

        if na == 0:
            bad_classes.append((tx_index, na, nb))
            continue

        xa_list.append(tx_a[:na])
        xb_list.append(tx_b[:nb])
        y_list.extend([tx_index] * na)

    if verbose and bad_classes:
        bad_str = ", ".join([f"{c}(non_eq={na}, eq={nb})" for c, na, nb in bad_classes])
        print(f"[rx={rx_index}, date={date_index}] skipped classes: {bad_str}")

    if len(xa_list) == 0:
        if verbose:
            print(f"[rx={rx_index}, date={date_index}] all classes skipped, return empty")
        xa = np.empty((0, 2, 0), dtype=np.float32)
        xb = np.empty((0, 2, 0), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        return xa, xb, y, bad_classes

    xa = np.concatenate(xa_list, axis=0)
    xb = np.concatenate(xb_list, axis=0)

    xa = preprocessing(xa)
    xb = preprocessing(xb)

    y = np.array(y_list, dtype=np.int64)
    return xa, xb, y