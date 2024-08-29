import torch
import os
import glob
import time
import pandas as pd
from typing import List, Dict, Tuple


def sequence_mask(sequence: torch.tensor, valid_len: torch.tensor):
    """为序列生成相应的mask， 标记pad

    Args:
        sequence: (B, L) or (B, L, h_dim)
        valid_len: (B,)
    """
    max_len = sequence.size(1)
    mask = torch.arange((max_len), dtype=torch.float32,
                        device=sequence.device)[None, :] < valid_len[:, None]
    return ~mask


node_type_to_prefix = {
    'labitem':"ITEMS",
    'drug': "DRUGS"
}


def get_latest_threshold(bth_path):
    files = glob.glob(os.path.join(bth_path, '*.pickle'))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return os.path.basename(latest_file)


def get_latest_model_ckpt(folder_path):
    # 获取指定文件夹下的所有.pt文件路径
    files = glob.glob(os.path.join(folder_path, '*.pt'))

    # 如果文件夹为空，返回 None
    if not files:
        return None

        # 找到最新的文件
    latest_file = max(files, key=os.path.getmtime)

    # 返回最新文件的文件名
    return os.path.basename(latest_file)


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


class EarlyStopper:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.is_stop = False
        self.best_model_wts = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score:  # Assuming lower score is better (e.g., loss)
            self.best_score = score
            self.counter = 0  # 重置耐心计数器
            self.best_model_wts = model.state_dict()
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.is_stop = True
                print("Early stop due to increasing valid loss!")

    def _log(self, path_to_save, model_name, notes):
        # 保存模型checkpoint时，往log.csv记录这次的实验信息，供后续查阅
        assert notes is not None  # 实验备注必须填写！
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        new_row = {
            "timestamp": timestamp,
            "valid_loss": self.best_score,
            "model_name": model_name,
            "notes": notes,  # 应记录实验描述，以及运行实验脚本的参数
        }
        log_file = os.path.join(path_to_save, "log.csv")
        if os.path.exists(log_file):
            df = pd.read_csv(log_file, index_col=0)
            df = df.append(new_row, ignore_index=True)
        else:
            df = pd.DataFrame(new_row, index=[0])
        df.to_csv(log_file, index=False)

    def save_checkpoint(self, path_to_save, model_name, notes):
        self._log(path_to_save, model_name, notes)
        torch.save(self.best_model_wts, os.path.join(path_to_save, model_name))
