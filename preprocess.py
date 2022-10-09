import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime

from conf import MIMIC_DIR, PROCESSED_DIR, MODEL_DIR, FILE_LABEL, MIMIC_II_FLAG
import utils


def load_list_from_txt(filepath):
    with open(filepath, 'r') as f:
        return f.read().split()


def make_icds_histogram(df):
    return df.ICD9_CODE.explode().value_counts()

def MIMIC_Preprocessor():
    if MIMIC_II_FLAG:
        return MIMIC_II_Preprocessor()
    return MIMIC_III_Preprocessor()

class MIMIC_II_Preprocessor:
    def __init__(self) -> None:
        self.name = 'II'

    def load_preprocessed(self, path=PROCESSED_DIR, data='final'):
        if data == 'df':
            with open(f'{path}mimic_data.pkl', 'rb') as file:
                self.df = pickle.load(file)
        elif data == 'multi':
            with open(f'{path}mimic_data_multi.pkl', 'rb') as file:
                self.df_multi = pickle.load(file)
        elif data == 'tree':
            with open(f'{path}mimic_data_tree.pkl', 'rb') as file:
                self.df_tree = pickle.load(file)
        elif data == 'leaf':
            self.df_leaves = utils.load_pickle(f'{path+FILE_LABEL}sample_leaves.pkl')
        elif data == 'final':
            self.df_final = utils.load_pickle(
                f'{path}{FILE_LABEL}_mimic_data_final.pkl')

    def save_preprocessed(self, path=PROCESSED_DIR):
        if hasattr(self, 'df_tree'):
            pd.to_pickle(self.df_tree, f'{path}mimic_data_tree.pkl')
        if hasattr(self, 'df'):
            pd.to_pickle(self.df, f'{path}mimic_data.pkl')
        if hasattr(self, 'df_multi'):
            pd.to_pickle(self.df_multi, f'{path}mimic_data_multi.pkl')
        if hasattr(self, 'df_final'):
            pd.to_pickle(self.df_final,
                         f'{path}{FILE_LABEL}_mimic_data_final.pkl')

    def preprocess_text(self, verbose=1):
        df_text = (pd.read_csv(f'{MIMIC_DIR}noteevents.csv').query(
            "category == 'DISCHARGE_SUMMARY'").drop_duplicates(
                'text').drop_duplicates('hadm_id')[[
                    'subject_id', 'hadm_id', 'text'
                ]])
        df_icds = pd.read_csv(f'{MIMIC_DIR}icd9.csv').dropna().groupby(
            'hadm_id')['code'].unique().reset_index()

        self.df = pd.merge(df_icds, df_text, on='hadm_id', how='inner')
        self.df.rename(columns={
            "text": "TEXT",
            "hadm_id": "HADM_ID",
            "code": "ICD9_CODE"
        }, inplace=True)

        if verbose:
            print(f'''
            -------------
            Total unique ICD codes: {self.df.ICD9_CODE.explode().nunique()}
            Total samples: {self.df.shape[0]}
            Data preprocessed!
            ''')

    def gen_hadm_map(self):
        df_not = (pd.read_csv(f'{MIMIC_DIR}noteevents.csv'))
        df_not = df_not[["hadm_id", "icustay_id"]].dropna(subset=["hadm_id", "icustay_id"])
        df_not = df_not.groupby("icustay_id").agg(HADM_ID=("hadm_id", set)).reset_index()
        icu2hadm = {}
        for _, k in df_not.iterrows():
            icu2hadm[k["icustay_id"]] = k["HADM_ID"]
        df_adm = (pd.read_csv(f'{MIMIC_DIR}admissions.csv'))
        df_admt = df_adm.groupby("subject_id").agg(HADM_ID=("hadm_id", list)).reset_index()
        sub2hadmt = {}
        for _, k in df_admt.iterrows():
            tmp = set()
            for hadm_id in k["HADM_ID"]:
                tmp.add(hadm_id)
            sub2hadmt[k["subject_id"]] = tmp
        self.icu2hadm = icu2hadm
        self.sub2hadm = sub2hadmt

    def df_insert_hadm(self, df):
        def helper(x):
            hs, hs2 = set(), set()
            if x.subject_id in self.sub2hadm:
                hs = self.sub2hadm[x.subject_id]
                if len(hs) == 1:
                    return list(hs)[0]
            if x.icustay_id in self.icu2hadm:
                hs2 = self.icu2hadm[x.icustay_id]
                if len(hs2) == 1:
                    return list(hs2)[0]
            hs2 = hs2.intersection(hs)
            if len(hs2) > 0:
                return list(hs2)[0]
            if len(hs) > 0:
                return list(hs)[0]
            return None
        df["hadm_id"] = df.apply(helper, axis=1)
        return df

    def preprocess_multi_modal(self):
        self.gen_hadm_map()
        df_pat = (pd.read_csv(f'{MIMIC_DIR}d_patients.csv')) # dob
        df_adm = (pd.read_csv(f'{MIMIC_DIR}admissions.csv')) # admit_dt disch_dt
        df_mr = pd.merge(df_adm, df_pat, how='left', on='subject_id')
        df_de_detail = (pd.read_csv(f'{MIMIC_DIR}demographic_detail.csv')) # admission_type_descr
        df_mr = pd.merge(df_mr, df_de_detail, how='left', on='hadm_id')
        df_info = df_mr[["hadm_id", "admission_type_descr"]]

        def gap(x, begin, end, dur):
            adm_time = datetime.strptime(x[begin], "%d/%m/%Y %H:%M:%S")
            born_time = datetime.strptime(x[end], "%d/%m/%Y %H:%M:%S")
            return ((adm_time-born_time).days+dur//2) // dur

        df_info["HOSPITAL_EXPIRE_FLAG"] = df_mr.apply(
            lambda x: 1 if x.hospital_expire_flg == 'Y' else 0, axis=1)
        df_info["STAY"] = df_mr.apply(
            lambda x: gap(x, "admit_dt", "disch_dt", 1), axis=1)
        df_info["AGE"] = df_mr.apply(lambda x: gap(x, "admit_dt", "dob", 365),
                                     axis=1)
        df_info["GENDER"] = df_mr.apply(lambda x: 1 if x.sex=='M' else 0, axis=1)

        df_lab = (pd.read_csv(f'{MIMIC_DIR}labevents.csv').dropna(
            subset=['hadm_id', 'flag']).query("flag =='abnormal'").groupby(
                "hadm_id").agg(LAB_ITEMID=('itemid', set)).reset_index())
        df_micro = (pd.read_csv(f'{MIMIC_DIR}microbiologyevents.csv')
                    .fillna({'org_itemid': 0, 'ab_itemid': 0})
                    .astype({'org_itemid': int, 'ab_itemid': int})
                    .dropna(subset=['hadm_id'])
                    .groupby('hadm_id').agg(
                        SPEC_ITEMID=('spec_itemid', list), # 测试细菌的样本 92
                        #   SPEC_TYPE_DESC=('SPEC_TYPE_DESC', list), # 对SPEC_ITEMID的描述
                        ORG_ITEMID=('org_itemid', list), # 测试时生长的生物体 309
                        #   ISOLATE_NUM=('ISOLATE_NUM', list), # 菌落数量
                        AB_ITEMID=('ab_itemid', list), # 抗生素 30
                        INTERPRETATION=('interpretation', list) # 敏感性 S敏感，R抗性，I中间，P待定
                    ).reset_index())
        df_drug = (pd.read_csv(f'{MIMIC_DIR}medevents.csv'))
        df_drug = self.df_insert_hadm(df_drug)
        df_drug = (df_drug.dropna(subset=["itemid", "hadm_id"]).groupby("hadm_id").agg(
            GSN=("itemid", set)).reset_index())

        self.df_multi = self.concat_dfs(df_drug, df_micro, df_lab, df_info, on="hadm_id")

    @staticmethod
    def concat_dfs(*dfs, on='HADM_ID', how='outer'):
        res = dfs[0]
        for df in dfs[1:]:
            res = pd.merge(res, df, on=on, how=how)
        return res

    def split(self, full=False, verbose=1):

        # Load ordered list of ICD classes (sorted list of all available ICD codes)
        self.all_icds = load_list_from_txt(f'{PROCESSED_DIR}top50_icds.txt')

        self.mlb = MultiLabelBinarizer(classes=self.all_icds)

        train_ids = load_list_from_txt(f'{PROCESSED_DIR}train_50_2_hadm_ids.csv')
        val_ids = load_list_from_txt(f'{PROCESSED_DIR}dev_50_2_hadm_ids.csv')
        test_ids = load_list_from_txt(f'{PROCESSED_DIR}test_50_2_hadm_ids.csv')
        self.mlb.fit(self.df_final[self.df_final['HADM_ID'].isin(train_ids)]['ICD9_CODE'])
        hadm_ids = [train_ids, val_ids, test_ids]
        hadm_ids = [[int(i) for i in ids] for ids in hadm_ids]

        assert not np.in1d(hadm_ids[0], hadm_ids[1]).any(), 'Data leakage!'
        assert not np.in1d(hadm_ids[0], hadm_ids[2]).any(), 'Data leakage!'
        assert not np.in1d(hadm_ids[2], hadm_ids[1]).any(), 'Data leakage!'

        ((self.x_train, self.y_train), (self.x_val, self.y_val),
         (self.x_test, self.y_test)) = [
             (self.df_final[self.df_final['HADM_ID'].isin(ids)].drop('ICD9_CODE', axis=1).reset_index(drop=True),
              pd.DataFrame(self.mlb.transform(
                  self.df_final[self.df_final['HADM_ID'].isin(ids)]['ICD9_CODE'])))
             for ids in hadm_ids
         ]

        if verbose:
            utils.print_time(f'''
            Data Split Y: {self.y_train.shape[0]}, {self.y_val.shape[0]}, {self.y_test.shape[0]}
            Data Split X: {self.x_train.shape[0]}, {self.x_val.shape[0]}, {self.x_test.shape[0]}
            ''')

    def preprocess_final(self):
        self.df.rename(columns={"": ""})
        self.df_final = pd.merge(self.df, self.df_leaves, how='left', on='HADM_ID')
        print(f'''
        -------------
        Total unique ICD codes: {self.df.ICD9_CODE.explode().nunique()}
        Total samples: {self.df.shape[0]}
        Data preprocessed!
        ''')

class MIMIC_III_Preprocessor:
    def __init__(self):
        self.name = 'III'

    def load_preprocessed(self, path=PROCESSED_DIR, data='final'):
        if data == 'df':
            with open(f'{path}mimic3_data.pkl', 'rb') as file:
                self.df = pickle.load(file)
        elif data == 'multi':
            with open(f'{path}mimic3_data_multi.pkl', 'rb') as file:
                self.df_multi = pickle.load(file)
        elif data == 'tree':
            with open(f'{path}mimic3_data_tree.pkl', 'rb') as file:
                self.df_tree = pickle.load(file)
        elif data == 'leaf':
            self.df_leaves = utils.load_pickle(f'{path+FILE_LABEL}sample_leaves.pkl')
        elif data == 'final':
            self.df_final = utils.load_pickle(
                f'{path}{FILE_LABEL}_mimic3_data_final.pkl')

    def save_preprocessed(self, path=PROCESSED_DIR):
        if hasattr(self, 'df_tree'):
            pd.to_pickle(self.df_tree, f'{path}mimic3_data_tree.pkl')
        if hasattr(self, 'df'):
            pd.to_pickle(self.df, f'{path}mimic3_data.pkl')
        if hasattr(self, 'df_multi'):
            pd.to_pickle(self.df_multi, f'{path}mimic3_data_multi.pkl')
        if hasattr(self, 'df_final'):
            pd.to_pickle(self.df_final,
                         f'{path}{FILE_LABEL}_mimic3_data_final.pkl')

    def preprocess_text(self, verbose=1):
        df_text = (pd.read_csv(f'{MIMIC_DIR}NOTEEVENTS.csv.gz').query(
            "CATEGORY == 'Discharge summary'").drop_duplicates(
                'TEXT').drop_duplicates('HADM_ID')[[
                    'SUBJECT_ID', 'HADM_ID', 'TEXT'
                ]])
        df_diag = pd.read_csv(f'{MIMIC_DIR}DIAGNOSES_ICD.csv.gz')
        df_proc = pd.read_csv(f'{MIMIC_DIR}PROCEDURES_ICD.csv.gz')
        df_diag.ICD9_CODE = df_diag.ICD9_CODE.apply(lambda x: utils.reformat(str(x), True))
        df_proc.ICD9_CODE = df_proc.ICD9_CODE.apply(lambda x: utils.reformat(str(x), False))
        df_icds = (pd.concat([df_diag, df_proc]).dropna().
                   groupby('HADM_ID')['ICD9_CODE'].unique().reset_index())

        self.df = pd.merge(df_icds, df_text, on='HADM_ID', how='inner')

        if verbose:
            print(f'''
            -------------
            Total unique ICD codes: {self.df.ICD9_CODE.explode().nunique()}
            Total samples: {self.df.shape[0]}
            Data preprocessed!
            ''')

    def preprocess_multi_modal(self):
        df_pat = (pd.read_csv(f'{MIMIC_DIR}PATIENTS.csv.gz'))
        df_adm = (pd.read_csv(f'{MIMIC_DIR}ADMISSIONS.csv.gz'))
        df_mr = pd.merge(df_adm, df_pat, how='left', on='SUBJECT_ID')
        df_info = df_mr[["HADM_ID", "ADMISSION_TYPE", "HOSPITAL_EXPIRE_FLAG"]]
        def get_age(x):
            adm_time = datetime.strptime(x.ADMITTIME, "%Y-%m-%d %H:%M:%S")
            born_time = datetime.strptime(x.DOB, "%Y-%m-%d %H:%M:%S")
            return ((adm_time-born_time).days+183) // 365

        df_info["AGE"] = df_mr.apply(lambda x:get_age(x), axis=1)
        df_info["GENDER"] = df_mr.apply(lambda x: 1 if x.GENDER=='M' else 0, axis=1)

        df_lab = (pd.read_csv(f'{MIMIC_DIR}LABEVENTS.csv.gz').dropna(
            subset=['HADM_ID']).query("FLAG =='abnormal'").groupby(
                "HADM_ID").agg(LAB_ITEMID=('ITEMID', set)).reset_index())
        df_callout = (pd.read_csv(f'{MIMIC_DIR}CALLOUT.csv.gz').dropna(
            subset=['DISCHARGE_WARDID', 'HADM_ID']
        ).query("CALLOUT_OUTCOME == 'Discharged'").query(
            "ACKNOWLEDGE_STATUS == 'Acknowledged'").groupby("HADM_ID").agg(
                REQUEST_TELE=('REQUEST_TELE', 'max'),
                REQUEST_RESP=('REQUEST_RESP', 'max'),
                REQUEST_CDIFF=('REQUEST_CDIFF', 'max'),
                REQUEST_MRSA=('REQUEST_MRSA', 'max'),
                REQUEST_VRE=('REQUEST_VRE', 'max')).reset_index())
        df_drug = (pd.read_csv(f'{MIMIC_DIR}PRESCRIPTIONS.csv.gz')
                     .dropna(subset=['HADM_ID', 'GSN'])
                    #  .apply(func=lambda x: x.strip().split(',')[0])
                     .groupby('HADM_ID')
                     .agg(GSN=('GSN', set)).reset_index()) # 需要清理
        df_micro = (pd.read_csv(f'{MIMIC_DIR}MICROBIOLOGYEVENTS.csv.gz')
              .fillna({'ORG_ITEMID': 0, 'AB_ITEMID': 0})
              .astype({'ORG_ITEMID': int, 'AB_ITEMID': int})
              .dropna(subset=['HADM_ID'])
              .groupby('HADM_ID').agg(
                  SPEC_ITEMID=('SPEC_ITEMID', list), # 测试细菌的样本 92
                #   SPEC_TYPE_DESC=('SPEC_TYPE_DESC', list), # 对SPEC_ITEMID的描述
                  ORG_ITEMID=('ORG_ITEMID', list), # 测试时生长的生物体 309
                #   ISOLATE_NUM=('ISOLATE_NUM', list), # 菌落数量
                  AB_ITEMID=('AB_ITEMID', list), # 抗生素 30
                  INTERPRETATION=('INTERPRETATION', list) # 敏感性 S敏感，R抗性，I中间，P待定
              ).reset_index())
        df_chart = (pd.read_csv(f'{MIMIC_DIR}CHARTEVENTS.csv.gz')
              .dropna(subset=["VALUENUM", "ITEMID"])
              .astype({'ITEMID': int}) # ITEM_ID: 6463 NUM_ITEM_ID: 2884
              .query("ERROR != 1.0")
              .groupby(['HADM_ID', 'ITEMID']).agg(
                MAX_VALUE_NUM=('VALUENUM', 'max'),
                MEAN_VALUE_NUM=('VALUENUM', 'mean'),
                MIN_VALUE_NUM=('VALUENUM', 'min')
              ).reset_index()
              .groupby('HADM_ID').agg(
                CHART_ITEMID=('ITEMID', list),
                MAX_VALUE_NUM=('MAX_VALUE_NUM', list),
                MEAN_VALUE_NUM=('MEAN_VALUE_NUM', list),
                MIN_VALUE_NUM=('MIN_VALUE_NUM', list)
              ).reset_index())
        self.df_multi = self.concat_dfs(df_callout, df_drug, df_micro,
                                        df_chart, df_lab, df_info)
        # self.df_multi = pd.merge(df_icds, self.df_multi, on='HADM_ID', how='left')

    @staticmethod
    def concat_dfs(*dfs, on='HADM_ID', how='outer'):
        res = dfs[0]
        for df in dfs[1:]:
            res = pd.merge(res, df, on=on, how=how)
        return res


    def split(self, full=False, verbose=1):

        # Load ordered list of ICD classes (sorted list of all available ICD codes)
        if full:
            self.all_icds = load_list_from_txt(
                f'{PROCESSED_DIR}common_ordered_icd.txt')
        else:
            self.all_icds = load_list_from_txt(
                f'{PROCESSED_DIR}top50_icds.txt')

        self.mlb = MultiLabelBinarizer(classes=self.all_icds) #.fit(
        # self.df_final['ICD9_CODE'])

        if full:
            train_ids = load_list_from_txt(
                f'{PROCESSED_DIR}train_full_hadm_ids.csv')
            val_ids = load_list_from_txt(
                f'{PROCESSED_DIR}dev_full_hadm_ids.csv')
            test_ids = load_list_from_txt(
                f'{PROCESSED_DIR}test_full_hadm_ids.csv')
        else:
            train_ids = load_list_from_txt(
                f'{PROCESSED_DIR}train_50_hadm_ids.csv')
            val_ids = load_list_from_txt(
                f'{PROCESSED_DIR}dev_50_hadm_ids.csv')
            test_ids = load_list_from_txt(
                f'{PROCESSED_DIR}test_50_hadm_ids.csv')
        self.mlb.fit(self.df_final[self.df_final['HADM_ID'].isin(train_ids)]['ICD9_CODE'])
        hadm_ids = [train_ids, val_ids, test_ids]
        hadm_ids = [[int(i) for i in ids] for ids in hadm_ids]

        assert not np.in1d(hadm_ids[0], hadm_ids[1]).any(), 'Data leakage!'
        assert not np.in1d(hadm_ids[0], hadm_ids[2]).any(), 'Data leakage!'
        assert not np.in1d(hadm_ids[2], hadm_ids[1]).any(), 'Data leakage!'

        ((self.x_train, self.y_train), (self.x_val, self.y_val),
         (self.x_test, self.y_test)) = [
             (self.df_final[self.df_final['HADM_ID'].isin(ids)].drop('ICD9_CODE', axis=1).reset_index(drop=True),
              pd.DataFrame(self.mlb.transform(
                  self.df_final[self.df_final['HADM_ID'].isin(ids)]['ICD9_CODE'])))
             for ids in hadm_ids
         ]

        if verbose:
            utils.print_time(f'''
            Data Split Y: {self.y_train.shape[0]}, {self.y_val.shape[0]}, {self.y_test.shape[0]}
            Data Split X: {self.x_train.shape[0]}, {self.x_val.shape[0]}, {self.x_test.shape[0]}
            ''')

    def preprocess_final(self):
        self.df_final = pd.merge(self.df, self.df_leaves, how='left', on='HADM_ID')
        print(f'''
        -------------
        Total unique ICD codes: {self.df.ICD9_CODE.explode().nunique()}
        Total samples: {self.df.shape[0]}
        Data preprocessed!
        ''')

if __name__ == "__main__":
    p = MIMIC_Preprocessor()
    # for multi modal
    p.preprocess_text()
    p.preprocess_multi_modal()

    # for final
    # p.load_preprocessed(data='df')
    # p.load_preprocessed(data='leaf')
    # p.preprocess_final()

    p.save_preprocessed()
    utils.print_time("OVER")
