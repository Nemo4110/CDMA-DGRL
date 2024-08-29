import sys
import pandas as pd
import numpy as np
import os
import torch
import time

from sklearn.metrics import \
    auc, \
    roc_auc_score, \
    accuracy_score, \
    jaccard_score, \
    f1_score, \
    precision_score, \
    recall_score, \
    average_precision_score
from utils.ddi import DDICalculator
from typing import List


sys.path.append('..')
# ddi_calculator = DDICalculator()


def flat_indices_to_voc_size(indices: List[int], voc_size, exclude_indices=None) -> np.ndarray:
    if exclude_indices is not None:
        indices = [elm for elm in indices if elm not in exclude_indices]

    tmp = np.zeros(voc_size)
    tmp[indices] = 1

    return tmp


def flat_probs(probs: List[torch.tensor], preds: List[int]):
    tmp = torch.stack([prob[:-3] for prob in probs], dim=0)  # 去掉SOS, EOS, PAD
    tmp = torch.max(tmp, dim=0).values
    # tmp = np.max(tmp, axis=0)  # 对于没预测的药物，取每个位置上最大的概率，否则直接取对应的概率
    for i, pred in enumerate(preds):
        if pred < tmp.size(0):  # 去掉SOS, EOS, PAD
            tmp[pred] = probs[i][pred]

    return tmp


def cal_jaccard(y_true: torch.tensor, y_pred: torch.tensor):
    if y_true.numel() == 0 or y_pred.numel() == 0:
        return 0
    set1 = set(y_true.tolist())
    set2 = set(y_pred.tolist())
    a, b = len(set1 & set2), len(set1 | set2)
    return a / b


def rocauc(cur_day_preds, cur_day_probs, cur_day_labels, **kwargs):
    try:
        return roc_auc_score(cur_day_labels, cur_day_probs, average='macro')
    except ValueError:
        return 0


def accuracy(cur_day_preds, cur_day_probs, cur_day_labels, **kwargs):
    return accuracy_score(cur_day_labels, cur_day_preds)


def jaccard(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
    if is_01:
        return jaccard_score(cur_day_labels, cur_day_preds)

    target = np.where(cur_day_labels == 1)[0]
    preds  = np.where(cur_day_preds  == 1)[0]

    inter = set(preds) & set(target)
    union = set(preds) | set(target)
    score = 0 if len(union) == 0 else len(inter) / len(union)

    return score


def prauc(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
    return average_precision_score(cur_day_labels, cur_day_probs, average='macro')


def precision(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
    if is_01:
        return precision_score(cur_day_labels, cur_day_preds, zero_division=0)
    else:
        target = np.where(cur_day_labels == 1)[0]
        preds  = np.where(cur_day_preds  == 1)[0]
        inter = set(preds) & set(target)
        score = 0 if len(preds) == 0 else len(inter) / len(preds)

        return score


def recall(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
    if is_01:
        return recall_score(cur_day_labels, cur_day_preds, zero_division=0)
    else:
        target = np.where(cur_day_labels == 1)[0]
        preds  = np.where(cur_day_preds  == 1)[0]
        inter = set(preds) & set(target)
        score = 0 if len(target) == 0 else len(inter) / len(target)

        return score


def calculate_f1(row):
    prc = row['precision']
    rec = row['recall']
    if prc + rec == 0:
        return 0.0
    return 2 * (prc * rec) / (prc + rec)


# def ddi_trues(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
#     if is_01:
#         labels = kwargs['cur_day_ground_true_d_seq']
#     else:
#         labels = np.nonzero(cur_day_labels)[0]
#     return ddi_calculator.calc_ddi_rate(labels)
#
#
# def ddi_preds(cur_day_preds, cur_day_probs, cur_day_labels, is_01: bool = False, **kwargs):
#     if is_01:
#         preds = kwargs['cur_day_predicted_d_seq']
#     else:
#         preds = np.nonzero(cur_day_preds)[0]
#     return ddi_calculator.calc_ddi_rate(preds)


# def calc_metrics_for_curr_adm_v2(
#     idx, all_day_logits, all_day_labels, batch_d_seq_to_be_judged, best_thresholds_by_days,
#     metric_functions=(rocauc, prauc, accuracy, jaccard, precision, recall, ddi_preds, ddi_trues)
# ):
#
#     # 适配当前取值范围[0,1]
#     result = pd.DataFrame(columns=['id', 'day'] + [mf.__name__ for mf in metric_functions])
#
#     # 测试集上batch_size = 1
#     all_day_logits = all_day_logits[0]
#     all_day_labels = all_day_labels[0]
#     all_day_d_seq_to_be_judged = batch_d_seq_to_be_judged[0]
#
#     for i, (cur_day_logits, cur_day_labels, cur_day_d_seq_to_be_judged) in enumerate(
#         zip(all_day_logits, all_day_labels, all_day_d_seq_to_be_judged)):
#         cur_day_logits = cur_day_logits.cpu().sigmoid()  # 需要先sigmoid！
#         cur_day_labels = cur_day_labels.cpu().bool()
#
#         cur_day = i + 1
#
#         # 若这天实际上没有正样本，则不进行指标计算
#         # 否则会出触发sklearn警告，说没有正样本计算某些指标是无意义的
#         if cur_day_labels.sum().item() == 0:
#             continue
#
#         cur_day_bth = best_thresholds_by_days[cur_day]  # 获取当天的最佳阈值
#         cur_day_preds = cur_day_logits > cur_day_bth  # logits转换为[0, 1]
#
#         # 从cur_day_d_seq_to_be_judged中获取对应预测的药物
#         cur_day_predicted_d_ind = torch.nonzero(cur_day_preds).squeeze(1)
#         cur_day_predicted_d_seq = torch.index_select(cur_day_d_seq_to_be_judged, 0, cur_day_predicted_d_ind)
#
#         cur_day_ground_true_d_ind = torch.nonzero(cur_day_labels).squeeze(1)
#         cur_day_ground_true_d_seq = torch.index_select(cur_day_d_seq_to_be_judged, 0, cur_day_ground_true_d_ind)
#
#         result.loc[len(result)] = ([idx, cur_day] +
#                                    [mf(cur_day_preds, cur_day_logits, cur_day_labels, is_01=True,
#                                        cur_day_predicted_d_seq=cur_day_predicted_d_seq,
#                                        cur_day_ground_true_d_seq=cur_day_ground_true_d_seq)
#                                     for mf in metric_functions])
#
#     result['f1'] = result.apply(calculate_f1, axis=1)
#     return result


def convert2df(logits: torch.tensor, labels: torch.tensor) -> pd.DataFrame:
    each_day_collector = []
    for d, (y_hat, y) in enumerate(zip(logits, labels)):
        y_hat = y_hat.sigmoid().cpu().tolist()
        y = y.cpu().tolist()

        day = d + 1  # 因为是从住院的第二天开始预测的，以此偏移量为1
        curr_day_df = pd.DataFrame({
            'score': y_hat,
            'label': y,
            'day': [day] * len(y)
        })
        each_day_collector.append(curr_day_df)
    return pd.concat(each_day_collector, axis=0)


def calc_metrics(results):
    return {
        "rocauc":       roc_auc_score(results['label'].values, results['score'].values, average='macro'),
        "ap": average_precision_score(results['label'].values, results['score'].values, average='macro'),
        "acc":         accuracy_score(results['label'].values, (results['score'] > 0.5).values),
        "precision":  precision_score(results['label'].values, (results['score'] > 0.5).values),
        "recall":        recall_score(results['label'].values, (results['score'] > 0.5).values, zero_division=0),
        "f1_score":          f1_score(results['label'].values, (results['score'] > 0.5).values, zero_division=0)
    }


def save_results(path_dir_results, results, ckpt_filename, notes):
    assert notes is not None  # 实验备注必须填写！
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    new_row = {
        "timestamp": timestamp,
        "ckpt_filename": ckpt_filename,
        "notes": notes,
        **calc_metrics(results)
    }
    result_file = os.path.join(path_dir_results, "results.csv")
    if os.path.exists(result_file):
        df_results = pd.read_csv(result_file, index_col=0)
        df_results = df_results.append(new_row, ignore_index=True)
    else:
        df_results = pd.DataFrame(new_row, index=[0])
    df_results.to_csv(result_file, index=False)
