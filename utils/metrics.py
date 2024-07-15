import sys
import pandas as pd
import numpy as np
import os
import torch
import pickle

from tqdm import tqdm
from sklearn.metrics import \
    auc, \
    roc_auc_score, \
    accuracy_score, \
    jaccard_score, \
    f1_score, \
    precision_score, \
    recall_score, \
    precision_recall_curve, \
    average_precision_score
from utils.ddi import DDICalculator
from typing import List

sys.path.append('..')
ddi_calculator = DDICalculator()


def flat_indices_to_voc_size(indices: List[int], voc_size, exclude_indices=None) -> np.ndarray:
    if exclude_indices is not None:
        indices = [elm for elm in indices if elm not in exclude_indices]

    tmp = np.zeros(voc_size)
    tmp[indices] = 1

    return tmp


def flat_probs(probs: List[torch.tensor], preds: List[int]):
    tmp = [prob[:-3] for prob in probs]  # 去掉SOS, EOS, PAD
    # TODO: 检查这里np.max后，是否得到正确形状
    tmp = np.max(tmp, axis=0)  # 对于没预测的药物，取每个位置上最大的概率，否则直接取对应的概率
    for i, pred in enumerate(preds):
        tmp[pred] = probs[i][pred]

    return tmp


def cal_jaccard(y_true: torch.tensor, y_pred: torch.tensor):
    if y_true.numel() == 0 or y_pred.numel() == 0:
        return 0
    set1 = set(y_true.tolist())
    set2 = set(y_pred.tolist())
    a, b = len(set1 & set2), len(set1 | set2)
    return a / b


def rocauc(cur_day_preds, cur_day_probs, cur_day_labels):
    try:
        return roc_auc_score(cur_day_labels, cur_day_probs, average='macro')
    except ValueError:
        return 0


def accuracy(cur_day_preds, cur_day_probs, cur_day_labels):
    return accuracy_score(cur_day_labels, cur_day_preds)


def jaccard(cur_day_preds, cur_day_probs, cur_day_labels):
    target = np.where(cur_day_labels == 1)
    inter = set(cur_day_preds) & set(target)
    union = set(cur_day_preds) | set(target)
    score = 0 if union == 0 else len(inter) / len(union)

    return score


def prauc(cur_day_preds, cur_day_probs, cur_day_labels):
    return average_precision_score(cur_day_labels, cur_day_probs, average='macro')


def precision(cur_day_preds, cur_day_probs, cur_day_labels):
    target = np.where(cur_day_labels == 1)
    inter = set(cur_day_preds) & set(target)
    score = 0 if len(cur_day_preds) == 0 else len(inter) / len(cur_day_preds)

    return score


def recall(cur_day_preds, cur_day_probs, cur_day_labels):
    target = np.where(cur_day_labels == 1)
    inter = set(cur_day_preds) & set(target)
    score = 0 if len(target) == 0 else len(inter) / len(target)

    return score


def ddi_trues(cur_day_preds, cur_day_probs, cur_day_labels):
    return ddi_calculator.calc_ddi_rate(np.nonzero(cur_day_labels))


def ddi_preds(cur_day_preds, cur_day_probs, cur_day_labels):
    return ddi_calculator.calc_ddi_rate(np.nonzero(cur_day_preds))


def calc_metrics_for_curr_adm(
        idx, all_day_preds, all_day_probs, all_day_labels,
        metric_functions=(rocauc, prauc, accuracy, jaccard, precision, recall, ddi_preds, ddi_trues)
    ):
    """
    以df的形式，记录患者在这次住院的每天的各项指标;
    方便后续进行细致的结果数据分析
    """
    d_voc_size = all_day_probs[0][0].size(-1) - 3  # 去掉SOS, EOS, PAD

    SOS = d_voc_size
    EOS = d_voc_size + 1
    PAD = d_voc_size + 2

    all_day_probs  = [flat_probs(cur_day_probs, cur_day_preds) for cur_day_probs, cur_day_preds in zip(all_day_probs, all_day_preds)]
    all_day_preds  = [flat_indices_to_voc_size(cur_day_preds,  d_voc_size, [SOS, EOS, PAD]) for cur_day_preds  in all_day_preds]
    all_day_labels = [flat_indices_to_voc_size(cur_day_labels, d_voc_size, [SOS, EOS, PAD]) for cur_day_labels in all_day_labels]

    # TODO: 对每个mf跑些单元测试
    result = pd.DataFrame(columns=['day'] + [mf.__name__ for mf in metric_functions])

    for cur_day, (cur_day_preds, cur_day_probs, cur_day_labels) in enumerate(
              zip(all_day_preds, all_day_probs, all_day_labels)):
        # 新增一行（天的）指标结果
        result.loc[len(result)] = [idx, cur_day] + [mf(cur_day_preds, cur_day_probs, cur_day_labels) for mf in metric_functions]

    # TODO: 用precision, recall这两列计算每天的f1
    # result['f1'] =   # use `apply()` ?

    return result


class Logger:
    r"""For logging testing metrics.

    Note: <https://math.stackexchange.com/questions/4205313/total-average-of-averages-not-same-as-the-average-of-total-values>
    """
    def __init__(self, max_timestep, save_dir_path: str, best_thresholdspath: str, is_calc_ddi: bool = False):
        self.save_dir_path = save_dir_path
        self.is_calc_ddi = is_calc_ddi

        self.list_preds_each_timestep = [[] for _ in range(max_timestep)]
        self.list_trues_each_timestep = [[] for _ in range(max_timestep)]

        with open(best_thresholdspath, 'rb') as f:
            self.dict_best_threholds = pickle.load(f)

        if self.is_calc_ddi:
            self.ddi_calculator = DDICalculator()
            self.list_edge_indices_each_timestep = [[] for _ in range(max_timestep)]
            self.list_ddi_pred = [[] for _ in range(max_timestep)]
            self.list_ddi_true = [[] for _ in range(max_timestep)]

    def log(self, list_y_pred, list_y_true, list_edge_indices=None):
        if self.is_calc_ddi and list_edge_indices is not None:
            for timestep, edge_indices in enumerate(list_edge_indices):
                self.list_edge_indices_each_timestep[timestep].append(edge_indices.cpu())

        for timestep, (y_pred, y_true) in enumerate(zip(list_y_pred, list_y_true)):
            self.list_preds_each_timestep[timestep].append(y_pred.cpu())
            self.list_trues_each_timestep[timestep].append(y_true.cpu())

    def get_results(self):
        list_preds_each_timestep_concated = [torch.cat(list_preds, dim=0) for list_preds in self.list_preds_each_timestep]
        list_tures_each_timestep_concated = [torch.cat(list_trues, dim=0) for list_trues in self.list_trues_each_timestep]

        # total performance & best threshold
        preds_total_concated = torch.cat(list_preds_each_timestep_concated, dim=0)
        trues_total_concated = torch.cat(list_tures_each_timestep_concated, dim=0)

        dict_total_performance = {}
        dict_total_performance['rocauc'] = roc_auc_score(trues_total_concated, preds_total_concated)
        precision, recall, ____ = precision_recall_curve(trues_total_concated, preds_total_concated)
        dict_total_performance['prauc'] = auc(recall, precision)
        # fpr, tpr, thresholds = roc_curve(trues_total_concated, preds_total_concated)
        # best_threshold = thresholds[np.argmax(tpr - fpr)]  # youden index
        preds_total_concated_bool = preds_total_concated > self.dict_best_threholds['total']
        dict_total_performance['accuracy']   = accuracy_score(trues_total_concated, preds_total_concated_bool)
        dict_total_performance['jaccard']     = jaccard_score(trues_total_concated, preds_total_concated_bool)
        dict_total_performance['precision'] = precision_score(trues_total_concated, preds_total_concated_bool)
        dict_total_performance['recall']       = recall_score(trues_total_concated, preds_total_concated_bool)
        dict_total_performance['f1']               = f1_score(trues_total_concated, preds_total_concated_bool)

        # performance of each timestep
        df_performance_each_timestep = pd.DataFrame()

        list_rocauc    = []
        list_prauc     = []
        list_accuracy  = []
        list_jaccard   = []
        list_precision = []
        list_recall    = []
        list_f1        = []

        for current_timestep, (preds_current_timestep, trues_current_timestep) in enumerate(
                zip(list_preds_each_timestep_concated, list_tures_each_timestep_concated)):
            list_rocauc.append(roc_auc_score(trues_current_timestep, preds_current_timestep))
            precision, recall, _ = precision_recall_curve(trues_current_timestep, preds_current_timestep)
            list_prauc.append(auc(recall, precision))

            # fpr, tpr, thresholds = roc_curve(trues_current_timestep, preds_current_timestep)
            # best_threshold_curr_timestep = thresholds[np.argmax(tpr - fpr)]  # youden index
            preds_current_timestep_bool = preds_current_timestep > self.dict_best_threholds['each_timestep'][current_timestep]

            list_accuracy.append(  accuracy_score(trues_current_timestep, preds_current_timestep_bool))
            list_jaccard.append(    jaccard_score(trues_current_timestep, preds_current_timestep_bool))
            list_precision.append(precision_score(trues_current_timestep, preds_current_timestep_bool))
            list_recall.append(      recall_score(trues_current_timestep, preds_current_timestep_bool))
            list_f1.append(              f1_score(trues_current_timestep, preds_current_timestep_bool))

        df_performance_each_timestep['rocauc']    = list_rocauc
        df_performance_each_timestep['prauc']     = list_prauc
        df_performance_each_timestep['accuracy']  = list_accuracy
        df_performance_each_timestep['jaccard']   = list_jaccard
        df_performance_each_timestep['precision'] = list_precision
        df_performance_each_timestep['recall']    = list_recall
        df_performance_each_timestep['f1']        = list_f1

        # DDI
        if self.is_calc_ddi:
            # iterate every timestep
            for timestep, (preds_current_timestep, trues_current_timestep, edge_indices_current_timestep) in tqdm(
                    enumerate(zip(self.list_preds_each_timestep, self.list_trues_each_timestep, self.list_edge_indices_each_timestep)),
                    leave=False
            ):
                # iterate every hadm batches
                for y_pred, y_true, edge_indices in tqdm(zip(preds_current_timestep, trues_current_timestep,
                                                             edge_indices_current_timestep), leave=False):
                    y_pred_bool = y_pred > self.dict_best_threholds['each_timestep'][timestep]
                    self.list_ddi_pred[timestep].extend(self.ddi_calculator.calc_ddis_for_batch_admi(y_pred_bool, edge_indices))
                    self.list_ddi_true[timestep].extend(self.ddi_calculator.calc_ddis_for_batch_admi(y_true,      edge_indices))

            total_ddis_pred = [np.array(ddi_pred_current_timestep) for ddi_pred_current_timestep in self.list_ddi_pred]
            total_ddis_true = [np.array(ddi_true_current_timestep) for ddi_true_current_timestep in self.list_ddi_true]
            dict_total_performance['ddi_pred'] = np.concatenate(total_ddis_pred, axis=None).mean()
            dict_total_performance['ddi_true'] = np.concatenate(total_ddis_true, axis=None).mean()

            df_performance_each_timestep['ddi_pred'] = [np.mean(ddi_pred_current_timestep) for ddi_pred_current_timestep in self.list_ddi_pred]
            df_performance_each_timestep['ddi_true'] = [np.mean(ddi_true_current_timestep) for ddi_true_current_timestep in self.list_ddi_true]

        return dict_total_performance, df_performance_each_timestep

    def save(self, description: str):
        self.dict_total, self.df = self.get_results()

        path4dict = os.path.join(self.save_dir_path, "total_dicts")
        path4df   = os.path.join(self.save_dir_path, "dfs")

        os.mkdir(path4dict) if not os.path.exists(path4dict) else None
        os.mkdir(path4df)   if not os.path.exists(path4df)   else None

        with open(os.path.join(path4dict, f"{description}.pickle"), 'wb') as f:
            pickle.dump(self.dict_total, f)

        self.df.to_csv(os.path.join(path4df, f"{description}.csv"))
