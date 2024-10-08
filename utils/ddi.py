r"""
Class used to calculate the DDI.
"""

import os
import sys; sys.path.append("..")
import pickle
import torch
import dill
import pandas as pd
import utils.constant as constant

from tqdm import tqdm

# https://stackoverflow.com/questions/20625582/
pd.options.mode.chained_assignment = None  # default='warn'


class DDICalculator:
    def __init__(self, path_ddi_dataset=constant.PATH_DDI_DATA) -> None:
        self._check_mapping_file_else_generate(file_path=os.path.join(path_ddi_dataset, "MAP_IDX4NDC_RXCUI_ATC4_CIDS.csv"))

        # use `index_col=0` argument to avoid `unnamed :0`
        # <https://stackoverflow.com/questions/53988226/pd-read-csv-add-column-named-unnamed-0>
        self.df_map_of_idx4ndc_rxcui_atc4_cids = pd.read_csv(os.path.join(path_ddi_dataset, "MAP_IDX4NDC_RXCUI_ATC4_CIDS.csv"), index_col=0)
        self.df_map_of_idx4ndc_rxcui_atc4_cids = self.df_map_of_idx4ndc_rxcui_atc4_cids.drop(columns=['list_cid'])

        self.df_map_of_idx4ndc_rxcui_atc4_cids['ATC3'] = self.df_map_of_idx4ndc_rxcui_atc4_cids['ATC4'].map(lambda x: x[:4], na_action='ignore')

        self.df_map_of_idx4ndc_rxcui_atc4_cids.sort_values(by='idx', inplace=True)

        # `voc_final.pkl`, `ddi_A_final.pkl` obtain by running the `python processing.py` cmd
        # under data dir of WWW'22 COGNet repo: https://github.com/BarryRun/COGNet
        with open(os.path.join(path_ddi_dataset, "voc_final.pkl"), 'rb') as f:
            self.med_voc = dill.load(f)['med_voc']
        self.med_unique_word = list(self.med_voc.word2idx.keys())
        with open(os.path.join(path_ddi_dataset, "ddi_A_final.pkl"), 'rb') as f:
            self.ddi_adj = dill.load(f)

    def _check_mapping_file_else_generate(self, file_path: str):
        if os.path.exists(file_path):
            return
        else:
            DDICalculator.generate_mapping_file(path_ddi=constant.PATH_DDI_DATA, path_iii=constant.PATH_MIMIC_III_ETL_OUTPUT)

    @staticmethod
    def generate_mapping_file(path_ddi, path_iii):
        # RXCUI to ATC4
        df_ndc2atc = pd.read_csv(os.path.join(path_ddi, "ndc2atc_level4.csv"))
        df_ndc2atc.drop(columns=['YEAR', 'MONTH', 'NDC'], inplace=True)
        df_ndc2atc.drop_duplicates(subset=['RXCUI'], inplace=True)

        df_drug_ndc_feat = pd.read_csv(os.path.join(path_iii, "DRUGS_NDC_FEAT.csv.gz"))  # generated by `preprocess_drugs.py`
        df_drug_ndc_feat.drop(columns=['Unnamed: 0'], inplace=True)
        df_drug_ndc_feat.drop(columns=[
            'DRUG_TYPE_MAIN_Proportion',
            'DRUG_TYPE_BASE_Proportion',
            'DRUG_TYPE_ADDITIVE_Proportion',
            'FORM_UNIT_DISP_Freq_1',
            'FORM_UNIT_DISP_Freq_2',
            'FORM_UNIT_DISP_Freq_3',
            'FORM_UNIT_DISP_Freq_4',
            'FORM_UNIT_DISP_Freq_5'], inplace=True)
        df_drug_ndc_feat.sort_values(by="NDC", inplace=True)
        df_drug_ndc_feat['idx'] = range(len(df_drug_ndc_feat))

        # ndc not real (< 1000)
        df_drug_ndc_feat_ndc_not_real = df_drug_ndc_feat[df_drug_ndc_feat.NDC < 1000].copy()
        df_drug_ndc_feat_ndc_not_real.rename(columns={'rxnorm_id': 'RXCUI'}, inplace=True)

        # ndc is real, but rxcui does not exist
        temp_df = df_drug_ndc_feat[df_drug_ndc_feat.NDC > 1000]
        df_drug_ndc_feat_ndc_real_rxcui_not_exist = temp_df[temp_df.rxnorm_id.isnull()].copy()
        df_drug_ndc_feat_ndc_real_rxcui_not_exist.rename(columns={'rxnorm_id': 'RXCUI'}, inplace=True)

        # ndc is real and rxcui(s) exist
        df_drug_ndc_feat_ndc_real = df_drug_ndc_feat[df_drug_ndc_feat.NDC > 1000].copy()
        df_drug_ndc_feat_ndc_real.dropna(inplace=True)
        df_drug_ndc_feat_ndc_real.rename(columns={'rxnorm_id': 'RXCUI'}, inplace=True)

        df_drug_ndc_feat_ndc_real = df_drug_ndc_feat_ndc_real.merge(df_ndc2atc, how='left', on=["RXCUI"])

        df_drug_ndc_feat_ndc_real_atc4_exist = df_drug_ndc_feat_ndc_real[df_drug_ndc_feat_ndc_real.ATC4.notnull()]

        df_drug_ndc_feat_ndc_real_ATC4_unique = df_drug_ndc_feat_ndc_real_atc4_exist.drop_duplicates(subset=['ATC4'])
        list_atc4_exist = list(df_drug_ndc_feat_ndc_real_ATC4_unique.ATC4.unique())

        # ATC2CID
        cid_atc = os.path.join(path_ddi, "drug-atc.csv")
        cid2atc_dic = {}
        with open(cid_atc, 'r') as f:
            for line in f:
                line_ls = line[:-1].split(',')
                cid = line_ls[0]
                atcs = [act[:5] for act in line_ls[1:]]
                cid2atc_dic[cid] = atcs

        act2cid = {}
        for curr_atc in tqdm(list_atc4_exist):

            unique_cid = []
            for cid, list_atc in cid2atc_dic.items():
                if curr_atc in list_atc:
                    unique_cid.append(cid)
            if len(unique_cid) != 0:
                act2cid[curr_atc] = unique_cid  # each exist act4 code may have multiple cid codes

        df_drug_ndc_feat_ndc_real['list_cid'] = df_drug_ndc_feat_ndc_real['ATC4'].map(act2cid)

        # final dataframe
        df_map_of_idx4ndc_rxcui_atc4_cids = pd.concat([
            df_drug_ndc_feat_ndc_real_rxcui_not_exist,
            df_drug_ndc_feat_ndc_real,
            df_drug_ndc_feat_ndc_not_real
        ])
        df_map_of_idx4ndc_rxcui_atc4_cids.to_csv(os.path.join(path_ddi, "MAP_IDX4NDC_RXCUI_ATC4_CIDS.csv"))

    def calc_ddis_for_batch_admi(self, edge_labels, edge_indices):
        existing_edge_indices = torch.index_select(edge_indices, dim=1, index=torch.nonzero(edge_labels).flatten())
        set_admi = existing_edge_indices[0].unique()

        ddis = []
        for curr_admi in set_admi:
            indices_curr_hadm = torch.nonzero(existing_edge_indices[0] == curr_admi).flatten()
            durg_idxes_curr_admi = torch.index_select(existing_edge_indices, dim=1, index=indices_curr_hadm)[1]
            ddis.append(self.calc_ddi_rate(durg_idxes_curr_admi))

        return ddis

    def calc_ddi_rate(self, durg_idxes_curr_admi: torch.tensor):
        # `durg_idxes_curr_admi`: the indeices of drugs of current patient, waiting to calculate the DDI score
        mask = self.df_map_of_idx4ndc_rxcui_atc4_cids.idx.isin(durg_idxes_curr_admi.tolist())  # MUST tolist !!!
        df_drugs_curr_admi = self.df_map_of_idx4ndc_rxcui_atc4_cids.loc[mask]

        df_drugs_can_calc_ddi = df_drugs_curr_admi[df_drugs_curr_admi.ATC3.notnull()]
        df_drugs_can_calc_ddi = df_drugs_can_calc_ddi[df_drugs_can_calc_ddi.ATC3.isin(self.med_unique_word)]
        atc3s = df_drugs_can_calc_ddi.ATC3.unique()

        cnt_all = 0
        cnt_ddi = 0
        for i, atc3_i in enumerate(atc3s):
            idx_drug_i = self.med_voc.word2idx[atc3_i]

            for j, atc3_j in enumerate(atc3s):
                if j <= i: continue
                cnt_all += 1

                idx_drug_j = self.med_voc.word2idx[atc3_j]
                if self.ddi_adj[idx_drug_i, idx_drug_j] == 1 or \
                   self.ddi_adj[idx_drug_j, idx_drug_i] == 1:
                    cnt_ddi += 1

        if cnt_all == 0:
            return 0
        return cnt_ddi / cnt_all
