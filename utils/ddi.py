r"""
Class used to calculate the DDI.
"""

import pandas as pd
import os
import pickle
import torch

from queue import Queue


class DDICalculator:
    def __init__(self, 
                 path_ddi_dataset=r"/data/data2/041/datasets/DDI") -> None:
        self.df_map_of_idx4ndc_rxcui_atc4_cids = pd.read_csv(os.path.join(path_ddi_dataset, "MAP_IDX4NDC_RXCUI_ATC4_CIDS.csv"))
        self.df_map_of_idx4ndc_rxcui_atc4_cids.sort_values(by='idx', inplace=True)

        with open(os.path.join(path_ddi_dataset, "ddi_adj_matrix.pickle"), 'rb') as f:
            self.ddi_adj = pickle.load(f)

    def calc_ddi_rate(self, pred_drug_ndc_scores_per_admi: torch.BoolTensor):
        df_drugs_curr_admi = self.df_map_of_idx4ndc_rxcui_atc4_cids.loc[
            pred_drug_ndc_scores_per_admi.cpu().detach().numpy()
        ]
        df_drugs_can_calc_ddi = df_drugs_curr_admi[df_drugs_curr_admi.list_cid_idx.notnull()]

        q = Queue()
        for list_cid_idx in list(df_drugs_can_calc_ddi.list_cid_idx.values):
            q.put(list_cid_idx)

        cnt_all_pair = 0
        cnt_ddi_pair = 0
        while q.qsize() > 1:
            list_curr_cid_idx = q.get()
            for curr_cid_idx in eval(list_curr_cid_idx):
                for list_other_cid_idx in q.queue:
                    for other_cid_idx in eval(list_other_cid_idx):
                        if self.ddi_adj[curr_cid_idx][other_cid_idx] > 0:
                            cnt_ddi_pair += 1
                        cnt_all_pair += 1
                    
        return 0 if cnt_all_pair == 0 else cnt_ddi_pair / cnt_all_pair
