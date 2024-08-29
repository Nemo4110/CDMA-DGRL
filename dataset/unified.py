"""加载原始DF数据，后续转换成各种类型的数据都从这里走"""
import sys; sys.path.append("..")
import pandas as pd
import os
import torch
import torch.utils.data as torchdata
import random
import utils.constant as constant
import torch_geometric.transforms as T

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List, Dict, Union
from tqdm import tqdm

from utils.enum_type import FeatureType, FeatureSource
from utils.config import max_adm_length


# 各个表的特征列
list_selected_admission_columns = [
    'ADMISSION_TYPE',
    'ADMISSION_LOCATION',
    'DISCHARGE_LOCATION',
    'INSURANCE',
    'LANGUAGE',
    'RELIGION',
    'MARITAL_STATUS',
    'ETHNICITY'
]
list_selected_labitems_columns = ['FLUID', 'CATEGORY']
list_selected_drug_ndc_columns = [
    "DRUG_TYPE_MAIN_Proportion",
    "DRUG_TYPE_BASE_Proportion",
    "DRUG_TYPE_ADDITIVE_Proportion",
    "FORM_UNIT_DISP_Freq_1",
    "FORM_UNIT_DISP_Freq_2",
    "FORM_UNIT_DISP_Freq_3",
    "FORM_UNIT_DISP_Freq_4",
    "FORM_UNIT_DISP_Freq_5"
]
list_selected_prescriptions_columns = [
    "DRUG_TYPE",
    "PROD_STRENGTH",
    "DOSE_VAL_RX",
    "DOSE_UNIT_RX",
    "FORM_VAL_DISP",
    "FORM_UNIT_DISP",
    "ROUTE"
]
list_selected_labevents_columns = ['VALUENUM_Z-SCORED', 'CATAGORY']  # 先float，再token


# 上面每个特征列的名称都是唯一的，因此可以全局一个字段，映射特征列名称到相应特征类型
field2type = {
    'item_id': FeatureType.TOKEN,
    'user_id': FeatureType.TOKEN,

    # user id
    'HADM_ID': FeatureType.TOKEN,

    # item id
    'NDC': FeatureType.TOKEN,
    'ITEMID': FeatureType.TOKEN,
    
    # 患者(user)住院信息
    'ADMISSION_TYPE':     FeatureType.TOKEN,
    'ADMISSION_LOCATION': FeatureType.TOKEN,
    'DISCHARGE_LOCATION': FeatureType.TOKEN,
    'INSURANCE':      FeatureType.TOKEN,
    'LANGUAGE':       FeatureType.TOKEN,
    'RELIGION':       FeatureType.TOKEN,
    'MARITAL_STATUS': FeatureType.TOKEN,
    'ETHNICITY':      FeatureType.TOKEN,

    # 检验item
    'FLUID': FeatureType.TOKEN,
    'CATEGORY': FeatureType.TOKEN,

    # 药品item
    'DRUG_TYPE_MAIN_Proportion': FeatureType.FLOAT,
    'DRUG_TYPE_BASE_Proportion': FeatureType.FLOAT,
    'DRUG_TYPE_ADDITIVE_Proportion': FeatureType.FLOAT,
    'FORM_UNIT_DISP_Freq_1': FeatureType.TOKEN,
    'FORM_UNIT_DISP_Freq_2': FeatureType.TOKEN,
    'FORM_UNIT_DISP_Freq_3': FeatureType.TOKEN,
    'FORM_UNIT_DISP_Freq_4': FeatureType.TOKEN,
    'FORM_UNIT_DISP_Freq_5': FeatureType.TOKEN,

    # 患者-检验项目 interaction
    'CATAGORY':          FeatureType.TOKEN,
    'VALUENUM_Z-SCORED': FeatureType.FLOAT,

    # 患者-药品 interaction
    'DRUG_TYPE':      FeatureType.TOKEN,
    'PROD_STRENGTH':  FeatureType.TOKEN,
    'DOSE_VAL_RX':    FeatureType.TOKEN,
    'DOSE_UNIT_RX':   FeatureType.TOKEN,
    'FORM_VAL_DISP':  FeatureType.TOKEN,
    'FORM_UNIT_DISP': FeatureType.TOKEN,
    'ROUTE':          FeatureType.TOKEN
}

field2source = {
    'user_id': FeatureSource.USER_ID,
    'item_id': FeatureSource.ITEM_ID,

    # user id
    'HADM_ID': FeatureSource.USER_ID,

    # item id
    'NDC': FeatureSource.ITEM_ID,
    'ITEMID': FeatureSource.ITEM_ID,


    # 患者(user)住院信息
    'ADMISSION_TYPE':     FeatureSource.USER,
    'ADMISSION_LOCATION': FeatureSource.USER,
    'DISCHARGE_LOCATION': FeatureSource.USER,
    'INSURANCE':      FeatureSource.USER,
    'LANGUAGE':       FeatureSource.USER,
    'RELIGION':       FeatureSource.USER,
    'MARITAL_STATUS': FeatureSource.USER,
    'ETHNICITY':      FeatureSource.USER,

    # 检验item
    'FLUID': FeatureSource.ITEM,
    'CATEGORY': FeatureSource.ITEM,

    # 药品item
    'DRUG_TYPE_MAIN_Proportion': FeatureSource.ITEM,
    'DRUG_TYPE_BASE_Proportion': FeatureSource.ITEM,
    'DRUG_TYPE_ADDITIVE_Proportion': FeatureSource.ITEM,
    'FORM_UNIT_DISP_Freq_1': FeatureSource.ITEM,
    'FORM_UNIT_DISP_Freq_2': FeatureSource.ITEM,
    'FORM_UNIT_DISP_Freq_3': FeatureSource.ITEM,
    'FORM_UNIT_DISP_Freq_4': FeatureSource.ITEM,
    'FORM_UNIT_DISP_Freq_5': FeatureSource.ITEM,

    # 患者-检验项目 interaction
    'CATAGORY':          FeatureSource.INTERACTION,
    'VALUENUM_Z-SCORED': FeatureSource.INTERACTION,

    # 患者-药品 interaction
    'DRUG_TYPE':      FeatureSource.INTERACTION,
    'PROD_STRENGTH':  FeatureSource.INTERACTION,
    'DOSE_VAL_RX':    FeatureSource.INTERACTION,
    'DOSE_UNIT_RX':   FeatureSource.INTERACTION,
    'FORM_VAL_DISP':  FeatureSource.INTERACTION,
    'FORM_UNIT_DISP': FeatureSource.INTERACTION,
    'ROUTE':          FeatureSource.INTERACTION
}

field2dtype = {
    "HADM_ID": 'int64',
    "ITEMID":  'int64',

    # df_admissions
    'ADMISSION_TYPE':     'int64',
    'ADMISSION_LOCATION': 'int64',
    'DISCHARGE_LOCATION': 'int64',
    'INSURANCE':          'int64',
    'LANGUAGE':           'int64',
    'RELIGION':           'int64',
    'MARITAL_STATUS':     'int64',
    'ETHNICITY':          'int64',

    # df_labitems
    'LABEL':    'string',
    'FLUID':    'int64',
    'CATEGORY': 'int64',

    # df_labevents
    'CATAGORY':          'int64',
    'VALUENUM_Z-SCORED': 'float64',
    'TIMESTEP':          'int64',

    # df_prescriptions
    'DRUG':              'string',
    'DRUG_NAME_POE':     'string',
    'DRUG_NAME_GENERIC': 'string',
    'FORMULARY_DRUG_CD': 'string',
    'GSN':               'string',
    'NDC':               'int64',
    'DRUG_TYPE':         'int64',
    'PROD_STRENGTH':     'int64',
    'DOSE_VAL_RX':       'int64',
    'DOSE_UNIT_RX':      'int64',
    'FORM_VAL_DISP':     'int64',
    'FORM_UNIT_DISP':    'int64',
    'ROUTE':             'int64'

}


class SourceDataFrames:
    def __init__(self, path_etl_output: str = constant.PATH_MIMIC_III_ETL_OUTPUT):
        self.path_etl_output = path_etl_output

        # 读取etl处理后的数据
        print("> loading .csv files...")
        self.df_admissions    = pd.read_csv(os.path.join(self.path_etl_output, "ADMISSIONS_NEW.csv.gz"),             index_col=0, dtype=field2dtype)
        self.df_labitems      = pd.read_csv(os.path.join(self.path_etl_output, "D_LABITEMS_NEW.csv.gz"),             index_col=0, dtype=field2dtype)
        self.df_labevents     = pd.read_csv(os.path.join(self.path_etl_output, "LABEVENTS_PREPROCESSED.csv.gz"),     index_col=0, dtype=field2dtype)
        self.df_prescriptions = pd.read_csv(os.path.join(self.path_etl_output, "PRESCRIPTIONS_PREPROCESSED.csv.gz"), index_col=0, dtype=field2dtype)
        self.df_drug_ndc_feat = pd.read_csv(os.path.join(self.path_etl_output, "DRUGS_NDC_FEAT.csv.gz"),             index_col=0, dtype=field2dtype)
        print("> finish loading!")

        # 截断一下最大值最小值
        self.df_labevents['VALUENUM_Z-SCORED'] = self.df_labevents['VALUENUM_Z-SCORED'].clip(lower=-100., upper=100.)

        self.df_admissions.sort_values(by='HADM_ID', inplace=True)
        self.df_labitems.sort_values(by='ITEMID', inplace=True)
        self.df_drug_ndc_feat.sort_values(by='NDC', inplace=True)

        self.adm_both = self._filter_out_adm_len_lt_2()
        self.adm_train, self.adm_val, self.adm_test = self._train_val_test_split()

        self.field2type = field2type
        self.field2source = field2source
        self.tokenfields2mappedid = self._prepare_mapping_for_token_type_fields()

        # user&item的特征 (注：因为上面排序过，因此每行直接对应新map之后的id)
        # 在预处理时，统一用0填充了nan，因此实际值从1开始
        # 对于token类型的特征列，需要用tokenfields2mappedid转换为最终映射mappedID
        for field in list_selected_admission_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                self.df_admissions[field] = self._map_token_field_to_mapped_id(field, self.df_admissions)
        for field in list_selected_labitems_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                self.df_labitems[field] = self._map_token_field_to_mapped_id(field, self.df_labitems)
        for field in list_selected_drug_ndc_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                self.df_drug_ndc_feat[field] = self._map_token_field_to_mapped_id(field, self.df_drug_ndc_feat)

        # 同理，行为表的token特征列也需要映射
        for field in list_selected_labevents_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                self.df_labevents[field] = self._map_token_field_to_mapped_id(field, self.df_labevents)
        for field in list_selected_prescriptions_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                self.df_prescriptions[field] = self._map_token_field_to_mapped_id(field, self.df_prescriptions)

        self.feat_admis = torch.from_numpy(self.df_admissions[list_selected_admission_columns].values)
        self.feat_items = torch.from_numpy(self.df_labitems[list_selected_labitems_columns].values)
        self.feat_drugs = torch.from_numpy(self.df_drug_ndc_feat[list_selected_drug_ndc_columns].values)

        # groupby hadm_id
        self.g_admi = self.df_admissions.groupby('HADM_ID')
        self.g_labe = self.df_labevents.groupby('HADM_ID')
        self.g_pres = self.df_prescriptions.groupby('HADM_ID')

    def _filter_out_adm_len_lt_2(self):
        """过滤掉住院长度小于2的HADM_ID,也就是说至少要有2天的记录"""
        length_per_hadm_l = self.df_labevents.groupby('HADM_ID')[['TIMESTEP']].nunique()
        length_per_hadm_p = self.df_prescriptions.groupby('HADM_ID')[['TIMESTEP']].nunique()
        length_per_hadm_l_multidays = length_per_hadm_l[length_per_hadm_l.TIMESTEP > 1]
        length_per_hadm_p_multidays = length_per_hadm_p[length_per_hadm_p.TIMESTEP > 1]

        adm_l = set(list(length_per_hadm_l_multidays.index))
        adm_p = set(list(length_per_hadm_p_multidays.index))

        both = list(set.intersection(adm_l, adm_p))
        both = list(map(int, both))
        print(f"> total adm whose length > 1: {len(both)}")

        return both

    def _train_val_test_split(self):
        adm_train_val, adm_test = train_test_split(self.adm_both, test_size=0.1, random_state=10043)
        adm_train, adm_val = train_test_split(adm_train_val, test_size=1. / 36, random_state=10043)
        print(f"> total adm for training: {len(adm_train)}, "
              f"validating: {len(adm_val)}, "
              f"testing: {len(adm_test)}")
        return adm_train, adm_val, adm_test
    
    def _prepare_mapping_for_token_type_fields(self):
        """为所有 token类型 的 特征列 生成从0开始的映射"""
        tokenfields2mappedid = {}
        
        # admission
        for field in list_selected_admission_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                tokenfields2mappedid[field] = self._get_id_map_for_token_field(field, self.df_admissions)

        # lab items
        for field in list_selected_labitems_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                tokenfields2mappedid[field] = self._get_id_map_for_token_field(field, self.df_labitems)

        # drug items
        for field in list_selected_drug_ndc_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                tokenfields2mappedid[field] = self._get_id_map_for_token_field(field, self.df_drug_ndc_feat)

        # lab events
        for field in list_selected_labevents_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                tokenfields2mappedid[field] = self._get_id_map_for_token_field(field, self.df_labevents)

        # prescriptions
        for field in list_selected_prescriptions_columns:
            if self.field2type[field] == FeatureType.TOKEN:
                tokenfields2mappedid[field] = self._get_id_map_for_token_field(field, self.df_prescriptions)

        # 将物品（实验室检验项目、药物）的原始id映射到从0开始
        unique_hadm_id = self.df_admissions.HADM_ID.sort_values().unique()
        unique_item_id = self.df_labitems.ITEMID.sort_values().unique()
        unique_ndc_id = self.df_drug_ndc_feat.NDC.sort_values().unique()
        self.hadmid2mappedip = pd.DataFrame(data={'HADM_ID': unique_hadm_id, 'mappedID': pd.RangeIndex(len(unique_hadm_id))})
        self.itemid2mappedid = pd.DataFrame(data={ 'ITEMID': unique_item_id, 'mappedID': pd.RangeIndex(len(unique_item_id))})
        self.drugid2mappedid = pd.DataFrame(data={    'NDC': unique_ndc_id,  'mappedID': pd.RangeIndex(len(unique_ndc_id))})
        tokenfields2mappedid['HADM_ID'] = self.hadmid2mappedip
        tokenfields2mappedid['ITEMID'] = self.itemid2mappedid
        tokenfields2mappedid['NDC'] = self.drugid2mappedid

        return tokenfields2mappedid

    def _get_id_map_for_token_field(self, token_field, source_df):
        unique_tokens = source_df[token_field].sort_values().unique()
        return pd.DataFrame(data={f'{token_field}': unique_tokens,
                                  'mappedID': pd.RangeIndex(len(unique_tokens))})

    def _map_token_field_to_mapped_id(self, token_field: str, ori_df: pd.DataFrame):
        assert self.field2type[token_field] == FeatureType.TOKEN
        map_df = self.tokenfields2mappedid[token_field]
        map_sr = pd.Series(map_df['mappedID'].values, index=map_df[token_field].values)
        return ori_df[token_field].map(map_sr)

    def get_mapped_id(self, id_filed, src_id):
        assert self.field2source[id_filed] in [FeatureSource.USER_ID, FeatureSource.ITEM_ID]
        map_df = self.tokenfields2mappedid[id_filed]
        mapped_id = map_df[map_df[id_filed] == src_id].mappedID.values[0]
        return mapped_id


class OneAdm(Dataset):
    """
    得到表示一次住院过程的DataFrames，
    后续各种模型需要的数据集，继承这个类，然后写自己的转换器adaptor；
    P.S. 其实只要重写__getitem__方法就好
    """
    def __init__(self, source_dfs: SourceDataFrames, split):
        super().__init__()

        assert split in ("train", "test", "val")
        self.source_dfs = source_dfs
        self.split = split

        if split == "train":
            self.admissions = self.source_dfs.adm_train
        elif split == "val":
            self.admissions = self.source_dfs.adm_val
        else:
            self.admissions = self.source_dfs.adm_test

    def __getitem__(self, idx):
        # id = self.admissions[idx]
        raise NotImplementedError

    def __len__(self):
        return len(self.admissions)


class OneAdmOneHG(OneAdm):
    """将单次住院过程表示为一张异质图"""
    def __getitem__(self, idx):
        id = self.admissions[idx]
        return self._convert_to_hetero_graph(id)

    def _convert_to_hetero_graph(self, id):
        curr_id_df_admi = self.source_dfs.g_admi.get_group(id)
        curr_id_df_labe = self.source_dfs.g_labe.get_group(id)
        curr_id_df_pres = self.source_dfs.g_pres.get_group(id)

        curr_id_df_labe = curr_id_df_labe.sort_values(by=["TIMESTEP", "CHARTTIME", "ROW_ID"])
        curr_id_df_pres = curr_id_df_pres.sort_values(by=["TIMESTEP", "STARTDATE", "ENDDATE", "ROW_ID"])

        # --- get corr tensor shards ---
        # nodes
        mapped_id = self.source_dfs.get_mapped_id('HADM_ID', id)
        nf_curr_adm = self.source_dfs.feat_admis[mapped_id].unsqueeze(0)  # 这里要增加一个维度

        nf_items = self.source_dfs.feat_items
        nf_drugs = self.source_dfs.feat_drugs

        # edges
        ## Edge indexes
        unique_hadm_id = curr_id_df_admi.HADM_ID.sort_values().unique()
        unique_hadm_id = pd.DataFrame(data={
            'HADM_ID': unique_hadm_id,
            'mappedID': pd.RangeIndex(len(unique_hadm_id)),
        })
        ### Perform merge to obtain the edges from HADM_ID and ITEMID:
        unique_item_id = self.source_dfs.itemid2mappedid
        unique_ndc_id = self.source_dfs.drugid2mappedid

        ratings_hadm_id = pd.merge(
            curr_id_df_labe['HADM_ID'], unique_hadm_id, left_on='HADM_ID', right_on='HADM_ID', how='left')
        ratings_item_id = pd.merge(
            curr_id_df_labe['ITEMID'], unique_item_id, left_on='ITEMID', right_on='ITEMID', how='left')

        ratings_hadm_id_drug = pd.merge(
            curr_id_df_pres['HADM_ID'], unique_hadm_id, left_on='HADM_ID', right_on='HADM_ID', how='left')
        ratings_ndc_id = pd.merge(
            curr_id_df_pres['NDC'], unique_ndc_id, left_on='NDC', right_on='NDC', how='left')

        ratings_hadm_id_items = torch.from_numpy(ratings_hadm_id['mappedID'].values)
        ratings_hadm_id_drugs = torch.from_numpy(ratings_hadm_id_drug['mappedID'].values)
        ratings_item_id = torch.from_numpy(ratings_item_id['mappedID'].values)
        ratings_drug_id = torch.from_numpy(ratings_ndc_id['mappedID'].values)

        edge_index_hadm_to_item = torch.stack([ratings_hadm_id_items, ratings_item_id], dim=0)
        edge_index_hadm_to_drug = torch.stack([ratings_hadm_id_drugs, ratings_drug_id], dim=0)

        ## Edge features
        ef_items = torch.from_numpy(
            curr_id_df_labe[list_selected_labevents_columns].values)
        ef_drugs = torch.from_numpy(
            curr_id_df_pres[list_selected_prescriptions_columns].values)
        edges_timestep_items = torch.from_numpy(curr_id_df_labe['TIMESTEP'].values)
        edges_timestep_drugs = torch.from_numpy(curr_id_df_pres['TIMESTEP'].values)

        ### assemble ####

        hetero_graph = HeteroData()
        ## Node
        ### node indices
        hetero_graph["admission"].node_id = torch.arange(len(unique_hadm_id))
        hetero_graph["labitem"  ].node_id = torch.arange(len(unique_item_id))
        hetero_graph["drug"     ].node_id = torch.arange(len(unique_ndc_id))
        ### node features:
        hetero_graph["admission"].x = nf_curr_adm
        hetero_graph["labitem"  ].x = nf_items
        hetero_graph["drug"     ].x = nf_drugs

        ## edge:
        hetero_graph["admission", "did", "labitem"].edge_index = edge_index_hadm_to_item
        hetero_graph["admission", "did", "labitem"].x          = ef_items
        hetero_graph["admission", "did", "labitem"].timestep   = edges_timestep_items

        hetero_graph["admission", "took", "drug"].edge_index = edge_index_hadm_to_drug
        hetero_graph["admission", "took", "drug"].x          = ef_drugs
        hetero_graph["admission", "took", "drug"].timestep   = edges_timestep_drugs

        return hetero_graph

    @staticmethod
    def split_by_day(hg: HeteroData):
        hgs = []

        last_lbe_day = hg["admission", "did", "labitem"].timestep.max().int().item()
        last_pre_day = hg["admission", "took", "drug"].timestep.max().int().item()
        adm_len = max(last_lbe_day, last_pre_day)

        device = hg["admission", "did", "labitem"].timestep.device

        for cur_day in range(adm_len + 1):
            sub_hg = HeteroData()

            # NODE (copied directly)
            for node_type in hg.node_types:
                sub_hg[node_type].node_id = hg[node_type].node_id.clone()
                sub_hg[node_type].x = hg[node_type].x.clone()

            # Edges
            for edge_type in hg.edge_types:
                mask = (hg[edge_type].timestep == cur_day).to(device)

                edge_index = hg[edge_type].edge_index[:, mask]
                ex = hg[edge_type].x[mask, :]

                sub_hg[edge_type].edge_index = edge_index.clone()
                sub_hg[edge_type].x = ex.clone()

            sub_hg = T.ToUndirected()(sub_hg)

            hgs.append(sub_hg)

        return hgs

    @staticmethod
    def pack_batch(hgs: List[HeteroData], batch_size: int):
        r"""
        Args:
            hgs:
            batch_size: how many sub graphs (days) to pack into a batch
        """
        loader = DataLoader(hgs, batch_size=batch_size)
        return next(iter(loader))

    @staticmethod
    def neg_sample_for_cur_day(pos_indices, num_itm_nodes: int, strategy: int = 2):
        """
        Args:
            strategy: int
                - 2: 2:1 (neg:pos)
                - >=10 and < num_itm_nodes: sample `strategy` negative samples
                - -1: full item set
        """
        # 一张图里的病人结点固定为1个
        num_pos_edges = pos_indices.size(1)

        if num_pos_edges == 0:  # 若当前天没有正样本
            num_neg_samples = 10  # 保证最少有10个负样本
        else:
            if strategy == 2:
                num_neg_samples = num_pos_edges * 2
            elif 10 <= strategy < num_itm_nodes:
                num_neg_samples = strategy
            elif strategy == -1:
                num_neg_samples = num_itm_nodes
            else:
                raise f"invalid negative sample `strategy` args: {strategy}!"

        neg_indices = negative_sampling(pos_indices, (1, num_itm_nodes), num_neg_samples=num_neg_samples)

        return neg_indices


class SingleItemType(OneAdm):
    """只考虑 单种 用户-物品关系，如患者-检验项目 or 患者-药物"""
    def __init__(self, source_dfs: SourceDataFrames, split, item_type: str):
        super(SingleItemType, self).__init__(source_dfs, split)
        self.item_type = item_type
        if self.item_type == "labitem":
            self.interaction = self.source_dfs.df_labevents.copy()
            self.interaction.sort_values(by=["HADM_ID", "TIMESTEP", "CHARTTIME", "ROW_ID"], inplace=True)
            self._prep_interaction(
                self.interaction,
                cols_to_drop=['ROW_ID', 'SUBJECT_ID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'FLAG', 'CATAGORY', 'VALUENUM_Z-SCORED'],
                cols_to_remap={
                    'HADM_ID': self.source_dfs.tokenfields2mappedid['HADM_ID'],
                    'ITEMID':  self.source_dfs.tokenfields2mappedid['ITEMID'],
                },
                cols_to_rename={
                    'HADM_ID':  'user_id',
                    'ITEMID':   'item_id',
                    'TIMESTEP': 'day',
                })
            self.original_item_id_field = 'ITEMID'
            self.item_feat_fields = list_selected_labitems_columns
            self.item_feat_values = self.source_dfs.feat_items
        elif self.item_type == "drug":
            self.interaction = self.source_dfs.df_prescriptions.copy()
            self.interaction.sort_values(by=["HADM_ID", "TIMESTEP", "STARTDATE", "ENDDATE", "ROW_ID"], inplace=True)
            self.interaction = self._prep_interaction(
                self.interaction,
                cols_to_drop=['ROW_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STARTDATE', 'ENDDATE', 'DRUG', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN'] + list_selected_prescriptions_columns,
                cols_to_remap={
                    'HADM_ID': self.source_dfs.tokenfields2mappedid['HADM_ID'],
                    'NDC':     self.source_dfs.tokenfields2mappedid['NDC'],
                },
                cols_to_rename={
                    'HADM_ID':  'user_id',
                    'NDC':      'item_id',
                    'TIMESTEP': 'day',
                })
            self.original_item_id_field = 'NDC'
            self.item_feat_fields = list_selected_drug_ndc_columns
            self.item_feat_values = self.source_dfs.feat_drugs
        else:
            raise NotImplementedError

        self.gb_uid = self.interaction.groupby('user_id')
        self.num_items = self.num(self.original_item_id_field)
        self.user_feat_fields = list_selected_admission_columns
        self.user_feat_values = self.source_dfs.feat_admis
        self.original_user_id_field = 'HADM_ID'

        # 这里用映射后的'user_id'和'item_id'，方便直接从对应用户/物品特征tensor中取相应特征行
        # 由于负采样的行无实际interaction features，因此不使用FeatureSource.INTERACTION
        # 顺序上注意遵循先用户、后物品
        self.available_fields = ['user_id',] + self.user_feat_fields + ['item_id',] + self.item_feat_fields

    def _prep_interaction(self,
                          interaction,
                          cols_to_drop: List,
                          cols_to_remap: Dict,
                          cols_to_rename: Dict):
        interaction.drop(columns=cols_to_drop, inplace=True)
        interaction['label'] = 1
        for col, map_df in cols_to_remap.items():
            map_s = pd.Series(map_df['mappedID'].values, index=map_df[col].values)
            interaction[col] = interaction[col].map(map_s)
        interaction.rename(columns=cols_to_rename, inplace=True)
        return interaction

    def _cur_day_neg_sample(self, pos_items: List, num_neg_samples: int):
        all_items = self.source_dfs.tokenfields2mappedid[
            self.original_item_id_field].mappedID.values.tolist()
        available = list(set(all_items) - set(pos_items))
        num_neg_samples = num_neg_samples if num_neg_samples <= len(available) else len(available)
        return random.sample(available, k=num_neg_samples)

    def _all_day_neg_samples(self, pos_shard, mappedid):
        gb_day = pos_shard.groupby('day')

        # 按天进行负采样
        mix_shards = []
        days = pos_shard.day.unique().tolist()
        days = [day for day in days if day < max_adm_length]  # 设置最长住院长度限制
        for d in days:
            cur_day_pos_shard = gb_day.get_group(d)

            pos_items = cur_day_pos_shard['item_id'].values.tolist()
            neg_items = self._cur_day_neg_sample(pos_items, num_neg_samples=2*len(pos_items))  # 暂用2:1负采样率

            cur_day_neg_shard = pd.DataFrame()
            cur_day_neg_shard['item_id'] = neg_items
            cur_day_neg_shard['user_id'] = mappedid
            cur_day_neg_shard['label'] = 0
            cur_day_neg_shard['day'] = d

            cur_day_mix_shard = pd.concat([cur_day_pos_shard, cur_day_neg_shard], axis=0)
            cur_day_mix_shard = cur_day_mix_shard.sample(frac=1)

            mix_shards.append(cur_day_mix_shard)

        # 统一变成了以下形式的DF数据，示例：
        # user_id, item_id, label,  day
        # -----------------------------
        #       0,       5,     1,    0
        #       0,      11,     1,    1
        #       0,     225,     0,    2
        return pd.concat(mix_shards, axis=0) if len(mix_shards) > 0 else pd.DataFrame()

    def __getitem__(self, idx):
        uid = self.admissions[idx]
        mappedid = self.source_dfs.get_mapped_id('HADM_ID', uid)
        pos_shard = self.gb_uid.get_group(mappedid)
        if len(pos_shard) > 0:
            interaction = self._all_day_neg_samples(pos_shard, mappedid)
            if len(interaction) == 0:
                return interaction
            else:
                pass
        else:
            interaction = pd.DataFrame()  # Empty

        return interaction

    def get_user_feature(self, uids=None):
        if uids is None:
            return self.source_dfs.feat_admis
        else:
            return self.source_dfs.feat_admis[uids]

    def get_item_feature(self, iids=None):
        if self.item_type == "labitem":
            if iids is None:
                return self.source_dfs.feat_items
            else:
                return self.source_dfs.feat_items[iids]
        elif self.item_type == "drug":
            if iids is None:
                return self.source_dfs.feat_drugs
            else:
                return self.source_dfs.feat_drugs[iids]
        else:
            raise NotImplementedError

    def num(self, token_field):
        """return the num of unique values of **token type field**, for embedding"""
        assert self.source_dfs.field2type[token_field] == FeatureType.TOKEN

        if token_field == 'user_id':
            token_field = self.original_user_id_field
        elif token_field == 'item_id':
            token_field = self.original_item_id_field
        else:
            pass

        return len(self.source_dfs.tokenfields2mappedid[token_field])

    def fields(self, ftype=None, source=None):
        """Given type and source of features, return all the field name of this type and source.
        If ``ftype == None``, the type of returned fields is not restricted.
        If ``source == None``, the source of returned fields is not restricted.

        Args:
            ftype (FeatureType, optional): Type of features. Defaults to ``None``.
            source (FeatureSource, optional): Source of features. Defaults to ``None``.

        Returns:
            list: List of field names.
        """
        ftype = set(ftype) if ftype is not None else set(FeatureType)
        source = set(source) if source is not None else set(FeatureSource)
        ret = []
        for field in self.available_fields:
            tp = self.source_dfs.field2type[field]
            src = self.source_dfs.field2source[field]
            if tp in ftype and src in source:
                ret.append(field)
        return ret


def get_pos_or_neg_shard(interaction: pd.DataFrame, is_pos: bool):
    """得到interaction正或负样本的部分"""
    if is_pos:
        return interaction[interaction.label == 1]
    else:
        return interaction[interaction.label == 0]


class SingleItemTypeForContextAwareRec(SingleItemType):
    def __getitem__(self, idx):
        uid = self.admissions[idx]
        mappedid = self.source_dfs.get_mapped_id('HADM_ID', uid)
        pos_shard = self.gb_uid.get_group(mappedid)
        if len(pos_shard) > 0:
            interaction = self._all_day_neg_samples(pos_shard, mappedid)
            if len(interaction) == 0:  # 没东西
                return interaction
            else:
                interaction = self._concat_corr_user_item_feat(interaction)
        else:
            interaction = pd.DataFrame()  # Empty
        return interaction

    def _concat_corr_user_item_feat(self, interaction):
        # 按user_id, item_id，取出相应的用户、物品特征列，一起放入最终的DataFrame(interaction)中
        user_feat_shard = pd.DataFrame(
            self.get_user_feature(interaction['user_id'].values),
            columns=self.user_feat_fields
        )
        item_feat_shard = pd.DataFrame(
            self.get_item_feature(interaction['item_id'].values),
            columns=self.item_feat_fields
        )

        # https://stackoverflow.com/questions/35084071
        interaction.reset_index(inplace=True, drop=True)
        user_feat_shard.reset_index(inplace=True, drop=True)
        item_feat_shard.reset_index(inplace=True, drop=True)

        return pd.concat([interaction, user_feat_shard, item_feat_shard], axis=1)


class SingleItemTypeForSequentialRec(SingleItemType):
    def __getitem__(self, idx):
        uid = self.admissions[idx]
        mappedid = self.source_dfs.get_mapped_id('HADM_ID', uid)
        pos_shard = self.gb_uid.get_group(mappedid)
        if len(pos_shard) > 0:
            interaction = self._all_day_neg_samples(pos_shard, mappedid)
            if len(interaction) == 0:
                return interaction
            else:
                interaction = self._add_history_seq(pos_shard, interaction)
        else:
            interaction = pd.DataFrame()  # Empty
        return interaction

    def _add_history_seq(self, pos_shard, interaction):
        """从第二天开始，为每条记录添加历史序列，"""
        pos_gb_day = pos_shard.groupby('day')
        int_gb_day = interaction.groupby('day')

        collector = []
        days = pos_shard.day.unique().tolist()
        days = [day for day in days if day < max_adm_length]
        for i, d in enumerate(days):
            if i == 0:
                continue
            pos_pre_day = pos_gb_day.get_group(days[i-1])  # 前一个有记录的天
            int_cur_day = int_gb_day.get_group(d)

            history = pos_pre_day['item_id'].values.tolist()
            history_len = len(history)

            history = [history for _ in range(len(int_cur_day))]
            history_len = [history_len for _ in range(len(int_cur_day))]

            df_hist = pd.DataFrame({'history': history,
                                    'history_len': history_len})

            int_cur_day.reset_index(inplace=True, drop=True)
            df_hist.reset_index(inplace=True, drop=True)

            collector.append(pd.concat([int_cur_day, df_hist], axis=1))

        return pd.concat(collector, axis=0)


class DFDataset(Dataset):
    r"""供基线模型使用的DataFrame Dataset"""
    def __init__(self, pre_dataset):
        name = pre_dataset.__class__.__name__
        split = pre_dataset.split
        item_type = pre_dataset.item_type
        self.dataframe = self._get_preprocessed(name, split, item_type)  # 如果处理过，就直接加载

        if self.dataframe is None:
            self._collect_all_shard(pre_dataset)
            self._save(name, split, item_type)

    def _collect_all_shard(self, pre_dataset: Union[SingleItemType,
                                                    SingleItemTypeForContextAwareRec,
                                                    SingleItemTypeForSequentialRec]):
        print("> in DFDataset, concat all single admission instances...")
        # 遍历，收集，拼成一个大的
        all_adm_interaction = []
        for sgl_adm_interaction in tqdm(pre_dataset, leave=False, ncols=80):
            all_adm_interaction.append(sgl_adm_interaction)
        self.dataframe = pd.concat(all_adm_interaction, axis=0)
        print("> done!")

    def _get_preprocessed(self, name, split, item_type):
        data_folder = self._get_data_folder(name)
        filename = os.path.join(data_folder, f"{split}_{item_type}.csv.gz")
        if os.path.isfile(filename):
            return pd.read_csv(filename, index_col=0, dtype={"history": "string"})
        else:
            return None

    def _save(self, name, split, item_type):
        data_folder = self._get_data_folder(name)
        os.makedirs(data_folder, exist_ok=True)
        filename = os.path.join(data_folder, f"{split}_{item_type}.csv.gz")
        self.dataframe.to_csv(filename, compression='gzip')

    def _get_data_folder(self, name):
        return os.path.join("data", name)  # 正确前提：运行于项目一级目录下的主脚本

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 获取数据行
        return self.dataframe.iloc[idx]

    @staticmethod
    def collect_fn(rows):
        df = pd.DataFrame(rows)
        if 'history' in df.columns:
            df['history'] = df['history'].astype("string")
            df['history'] = df['history'].apply(string2list)
        return df


# https://stackoverflow.com/questions/69959719
def string2list(row_value):
    list_str = row_value.strip('][').replace('"', '').split(',')
    if len(list_str) > 0 and list_str[0] != '':
        return list(map(int, list_str))
    else:
        return []


if __name__ == '__main__':
    sources_dfs = SourceDataFrames(r"..\data\mimic-iii-clinical-database-1.4")
    pre_dataset = SingleItemTypeForContextAwareRec(sources_dfs, "val", "labitem")
    itr_dataset = DFDataset(pre_dataset)
    itr_dataloader = torchdata.DataLoader(
        itr_dataset, batch_size=256, shuffle=False, pin_memory=True, collate_fn=DFDataset.collect_fn)
    for interaction in itr_dataloader:
        print(interaction)
        break

