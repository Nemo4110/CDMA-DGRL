import os.path as path
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime, timedelta


def preprocess_admission(src_csv_path, dst_csv_path, value_na=0):
    df_admissions = pd.read_csv(src_csv_path)

    df_admissions["ADMITTIME"] = pd.to_datetime(df_admissions["ADMITTIME"], format="%Y-%m-%d %H:%M:%S")
    df_admissions["DISCHTIME"] = pd.to_datetime(df_admissions["DISCHTIME"], format="%Y-%m-%d %H:%M:%S")

    # Node_feature
    list_str_type_columns = ['ADMISSION_TYPE',
                             'ADMISSION_LOCATION',
                             'DISCHARGE_LOCATION',
                             'INSURANCE',
                             'LANGUAGE',
                             'RELIGION',
                             'MARITAL_STATUS',
                             'ETHNICITY']

    for c in list_str_type_columns:
        m = df_admissions[c].value_counts()
        m = pd.Series(index=m.index, data=range(1, len(m) + 1))
        df_admissions[c] = df_admissions[c].map(m)

    values_fillna = {}
    for c in list_str_type_columns:
        values_fillna[c] = value_na
    df_admissions.fillna(value=values_fillna, inplace=True)
    df_admissions.to_csv(dst_csv_path)


def preprocess_labitems(src_csv_path, dst_csv_path, value_na=0):
    df_d_labitems = pd.read_csv(src_csv_path)

    list_str_type_columns = ['FLUID', 'CATEGORY']
    for c in list_str_type_columns:
        m = df_d_labitems[c].value_counts()
        m = pd.Series(index=m.index, data=range(1, len(m) + 1))
        df_d_labitems[c] = df_d_labitems[c].map(m)

    values_fillna = {}
    for c in list_str_type_columns:
        values_fillna[c] = value_na
    df_d_labitems.fillna(value=values_fillna, inplace=True)

    df_d_labitems.to_csv(dst_csv_path)


def preprocess_labevents(src_csv_path, dst_csv_path, value_na=0):
    df_labevents = pd.read_csv(src_csv_path)
    df_labevents.dropna(subset=['HADM_ID', 'ITEMID'], inplace=True)
    df_labevents.sort_values(by=["HADM_ID", "ITEMID"], inplace=True)
    df_labevents["CHARTTIME"] = pd.to_datetime(df_labevents["CHARTTIME"], format="%Y-%m-%d %H:%M:%S")

    grouped_by_itemid_value_type_only = df_labevents[df_labevents.VALUENUM.notnull()].groupby("ITEMID")
    grouped_by_itemid_not_value_type = df_labevents[df_labevents.VALUENUM.isnull()].groupby("ITEMID")

    # **************************************************************************************************************** #
    print("*** Solving multi value type ***")
    set_itemid_value_type = list(grouped_by_itemid_value_type_only.groups.keys())
    set_itemid_not_value_type = list(grouped_by_itemid_not_value_type.groups.keys())

    set_itemid_pure_value_type = np.setdiff1d(set_itemid_value_type, set_itemid_not_value_type)
    set_itemid_mixed_value_type = np.setdiff1d(set_itemid_value_type, set_itemid_pure_value_type)
    set_itemid_pure_non_value_type = np.setdiff1d(set_itemid_not_value_type, set_itemid_mixed_value_type)

    # >>> Pure non-value type itemid <<< Need to re-map by catagrory
    for itemid in tqdm(set_itemid_pure_non_value_type):
        s = df_labevents[df_labevents.ITEMID == itemid].VALUE.value_counts()
        m = pd.Series(index=s.index, data=range(1, len(s) + 1))
        df_labevents.loc[df_labevents.ITEMID == itemid, 'CATAGORY'] = df_labevents[df_labevents.ITEMID == itemid].VALUE.map(m)

    df_labevents.CATAGORY.fillna(value_na, inplace=True)

    # >>> Mix value type itemid <<<
    for itemid in tqdm(set_itemid_mixed_value_type):
        mask = df_labevents[df_labevents.ITEMID == itemid].VALUENUM.isnull()
        list_index = df_labevents[df_labevents.ITEMID == itemid].loc[mask, :].index
        df_labevents.drop(list_index, inplace=True)

    # **************************************************************************************************************** #
    print("*** Z-SCORE ***")
    def box_analysis(data: pd.Series):
        # via. [利用 Pandas 进行数据处理](https://juejin.cn/post/6859254388021133326#heading-0)
        qu = data.quantile(q=0.75)
        ql = data.quantile(q=0.25)

        iqr = qu - ql
        up = qu + 1.5 * iqr
        low = ql - 1.5 * iqr

        mask = (data < up) & (data > low)
        return mask

    def z_score_4_value_type_labitem(df_grp):
        dfx = df_grp.copy()
        dfx_normal = dfx[dfx.FLAG != 'abnormal']

        mask = box_analysis(dfx_normal.VALUENUM)  # using box analysis filter out abnormal value whose FLAG != 'abnormal'
        dfx_normal_boxed = dfx_normal.loc[mask, :]

        mean = dfx_normal_boxed.VALUENUM.mean()
        std = dfx_normal_boxed.VALUENUM.std()

        dfx['VALUENUM_Z-SCORED'] = dfx['VALUENUM'].apply(lambda x: (x - mean) / (std + 1e-7))  # avoid division by zero

        return dfx

    df_itemid_value_type_only_zscore = grouped_by_itemid_value_type_only.apply(z_score_4_value_type_labitem)
    df_labevents = df_labevents.merge(df_itemid_value_type_only_zscore[['ROW_ID', 'VALUENUM_Z-SCORED']], how='left', on='ROW_ID')
    df_labevents['VALUENUM_Z-SCORED'].fillna(value_na, inplace=True)

    # **************************************************************************************************************** #
    print("*** Adding TIMESTEP ***")
    grouped_by_hadmid = df_labevents.groupby("HADM_ID")
    def add_timestep_per_hadmid(df_grouped_by_hadmid: pd.DataFrame):
        interval_hour = 24  # chosen interval
        df_grouped_by_hadmid = df_grouped_by_hadmid.sort_values(by="CHARTTIME")

        st = df_grouped_by_hadmid.CHARTTIME.iloc[0]  # st <- start time
        et = df_grouped_by_hadmid.CHARTTIME.iloc[-1]  # et <- end time
        st = datetime.strptime(f"{st.year}-{st.month}-{st.day} {st.hour // interval_hour * interval_hour:2}:00:00", "%Y-%m-%d %H:%M:%S")
        et = datetime.strptime(f"{et.year}-{et.month}-{et.day} {(((et.hour // interval_hour) + 1) * interval_hour) - 1:2}:59:59", "%Y-%m-%d %H:%M:%S")
        interval = timedelta(hours=interval_hour)

        dfx = df_grouped_by_hadmid.copy()
        dfx.insert(len(dfx.columns), "TIMESTEP", np.NaN)

        timestep = 0
        while st < et:
            mask = (st <= dfx.CHARTTIME) & (dfx.CHARTTIME <= st + interval)
            if len(dfx.loc[mask]) > 0:
                dfx.loc[mask, 'TIMESTEP'] = timestep
                timestep += 1
            st += interval

        return dfx

    df_grouped_by_hadmid_timestep_added = grouped_by_hadmid.apply(add_timestep_per_hadmid)
    df_labevents = df_labevents.merge(df_grouped_by_hadmid_timestep_added[['ROW_ID', 'TIMESTEP']], how='left', on='ROW_ID', copy=False)

    # **************************************************************************************************************** #
    print("*** Merging repeat edges ***")
    gb_hadmid = df_labevents.groupby("HADM_ID")
    drop_indexs = []
    for hadm_id in tqdm(df_labevents.HADM_ID.unique()):
        df_curr_hadmid = df_labevents[df_labevents.HADM_ID == hadm_id]

        for timestep in range(int(df_curr_hadmid.TIMESTEP.max()) + 1):
            df_curr_hadmid_curr_timestep = df_curr_hadmid[df_curr_hadmid.TIMESTEP == timestep]

            sr_itemid_value_counts = df_curr_hadmid_curr_timestep.ITEMID.value_counts()
            sr_itemid_repeat = sr_itemid_value_counts[sr_itemid_value_counts > 1]

            for itemid_repeat in list(sr_itemid_repeat.index):
                deprecate_entry_rowid = df_curr_hadmid_curr_timestep[df_curr_hadmid_curr_timestep.ITEMID == itemid_repeat].sort_values(by="CHARTTIME").ROW_ID.iloc[0:-1]
                deprecate_entry_index = list(deprecate_entry_rowid.index)
                drop_indexs.extend(deprecate_entry_index)

    df_labevents.drop(drop_indexs, inplace=True)
    df_labevents.to_csv(dst_csv_path)


if __name__ == "__main__":
    path_dataset = r"/data/data2/041/datasets/mimic-iii-clinical-database-1.4"
    preprocess_admission(src_csv_path=path.join(path_dataset, "ADMISSIONS.csv.gz"),
                         dst_csv_path=path.join(path_dataset, "ADMISSIONS_NEW.csv.gz"))
    preprocess_labitems(src_csv_path=path.join(path_dataset, "D_LABITEMS.csv.gz"),
                        dst_csv_path=path.join(path_dataset, "D_LABITEMS_NEW.csv.gz"))
    preprocess_labevents(src_csv_path=path.join(path_dataset, "LABEVENTS.csv.gz"),
                         dst_csv_path=path.join(path_dataset, "LABEVENTS_NEW_remove_duplicate_edges.csv.gz"))
