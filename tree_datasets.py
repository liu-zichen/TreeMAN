import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from preprocess import MIMIC_II_Preprocessor, MIMIC_III_Preprocessor, load_list_from_txt
from conf import MIMIC_DIR, PROCESSED_DIR, MIMIC_II_FLAG
import utils

def TreeDataset(args='full'):
    if MIMIC_II_FLAG:
        return TreeDatasetII(args)
    return TreeDatasetIII(args)

class TreeDatasetII(MIMIC_II_Preprocessor):
    binary_fields = [
        'GSN', 'SPEC_ITEMID', 'ORG_ITEMID', 'AB_ITEMID', 'LAB_ITEMID'
    ]

    def __init__(self, train_data_from='full'):
        super().__init__()
        self.all_icds = load_list_from_txt( # get top frequently code
            f'{PROCESSED_DIR}top50_icds.txt')
        self.label_encoder = LabelEncoder().fit(self.all_icds)
        self.train_data_from = train_data_from

    def save_preprocessed(self, path=PROCESSED_DIR):
        if hasattr(self, 'df_tree'):
            pd.to_pickle(self.df_tree.copy(), f'{path}mimic_data_tree.pkl')
            utils.save_json(self.re_feature_map, f'{path}re_feature_map.json')
            utils.save_json(self.feature_map, f'{path}feature_map.json')

    def load_tree_data(self):
        self.load_preprocessed(data='tree')
        utils.print_time("df_tree shape:%s" % str(self.df_tree.shape))

    def load_split_data(self, path=PROCESSED_DIR):
        self.x_train = utils.load_pickle(f'{path}split_x_train.pkl')
        self.y_train = utils.load_pickle(f'{path}split_y_train.pkl')
        self.x_val = utils.load_pickle(f'{path}split_x_val.pkl')
        self.y_val = utils.load_pickle(f'{path}split_y_val.pkl')
        self.x_test = utils.load_pickle(f'{path}split_x_test.pkl')
        self.y_test = utils.load_pickle(f'{path}split_y_test.pkl')
        utils.print_time("Split Data Loaded!")

    def save_split_data(self, path=PROCESSED_DIR):
        utils.save_pickle(self.x_train, f'{path}split_x_train.pkl')
        utils.save_pickle(self.y_train, f'{path}split_y_train.pkl')
        utils.save_pickle(self.x_val, f'{path}split_x_val.pkl')
        utils.save_pickle(self.y_val, f'{path}split_y_val.pkl')
        utils.save_pickle(self.x_test, f'{path}split_x_test.pkl')
        utils.save_pickle(self.y_test, f'{path}split_y_test.pkl')
        utils.print_time("Split Data Saved!")

    def get_feature_map(self):
        feature_id = 1
        feature_map = defaultdict(dict) # field2value2feature_id
        re_feature_map = dict() # field2tuple(field, value)

        def get_counter(field):
            return Counter([e for lis in self.df_multi[field].dropna() for e in lis])

        for field in self.binary_fields:
            f_set = [v[0] for v in get_counter(field).items() if v[1] > 2]
            for v in f_set:
                feature_map[field][v] = feature_id
                re_feature_map[feature_id] = (field, v)
                feature_id += 1
            print("Process: " +  field)
        utils.print_time(f"FeatureMap Generated num_feature_id: {feature_id-1}")
        return feature_map, re_feature_map

    def gen_tree_data(self):
        """
        将df_multi转为xgboost可用的DataFrame
        """
        self.feature_map, self.re_feature_map = self.get_feature_map()
        simple_fields = [
            'HOSPITAL_EXPIRE_FLAG', 'GENDER'
        ]
        df_tree = self.df_multi[['hadm_id', 'AGE', 'STAY']].copy()
        ADMISSION_TYPEs = ['ELECTIVE', 'EMERGENCY', 'NEWBORN', 'URGENT']
        for adm_type in ADMISSION_TYPEs:
            df_tree["ADMISSION_TYPE" + adm_type] = self.df_multi.apply(
                lambda x: x.admission_type_descr == adm_type,
                axis=1).astype(pd.SparseDtype(bool))
        for s_field in simple_fields:
            df_tree[s_field] = (self.df_multi[s_field]
                        .apply(lambda x: pd.notna(x) and x >= 1)
                        .astype(pd.SparseDtype(bool)))

        for b_field in self.binary_fields:  # 处理binary数据
            for value, f_id in self.feature_map[b_field].items():
                df_tree[f_id] = (
                    self.df_multi[b_field].apply(lambda x: isinstance(
                        x, (list, set)) and value in x).astype(
                            pd.SparseDtype(bool)))
            utils.print_time(
                f"Processed field: {b_field} Len: {len(self.feature_map[b_field])}"
            )

        def by_index(x, f, v, index_field):
            return x[f][x[index_field].index(v)] if isinstance(
                x[index_field], list) and v in x[index_field] else pd.NA

        def fab(x, v):
            t = by_index(x, 'INTERPRETATION', v, 'AB_ITEMID')
            return pd.notna(t) and t == 'S'

        for v, f_id in self.feature_map['AB_ITEMID'].items():
            df_tree[f_id] = self.df_multi.apply(lambda x:fab(x, v),axis=1).astype(pd.SparseDtype(bool))
        utils.print_time("Processed field: AB_ITEMID")

        # 合并 icd code
        df_icds = pd.read_csv(f'{MIMIC_DIR}icd9.csv').dropna().groupby(
            'hadm_id')['code'].unique().reset_index()
        self.df_tree = pd.merge(df_icds, df_tree, how='left', on='hadm_id')
        self.df_tree.rename(columns={"hadm_id": "HADM_ID", "code": "ICD9_CODE"}, inplace=True)

    def split_train(self):
        train_ids = load_list_from_txt(f'{PROCESSED_DIR}train_50_2_hadm_ids.csv')
        val_ids = load_list_from_txt(f'{PROCESSED_DIR}dev_50_2_hadm_ids.csv')
        test_ids = load_list_from_txt(f'{PROCESSED_DIR}test_50_2_hadm_ids.csv')
        def get_by_id(ids):
            ids = [int(v) for v in ids]
            y = self.df_tree[self.df_tree['HADM_ID'].isin(ids)]['ICD9_CODE']
            x = self.df_tree[self.df_tree['HADM_ID'].isin(ids)].drop(
                labels=['HADM_ID', 'ICD9_CODE'], axis=1)
            return x.copy(), pd.Series(y)
        print(self.df_tree['HADM_ID'].dtype)
        self.x_train, self.y_train = get_by_id(train_ids)
        self.x_test, self.y_test = get_by_id(test_ids)
        self.x_val, self.y_val = get_by_id(val_ids)

    def preprocess_tree(self, mod='final'):
        if mod == 'final':
            self.load_split_data()
            return
        if mod == 'ground':
            self.load_preprocessed(data='multi')
            self.gen_tree_data()
            self.save_preprocessed()
        elif mod == 'normal':
            self.load_tree_data()
        utils.print_time("Data Processed/Loaded! Begin split!")
        self.split_train()
        self.save_split_data()

    def dtrain(self, code_index=0):
        code = self.all_icds[code_index]
        y = self.y_train.apply(lambda x: code in x)
        return xgb.DMatrix(self.x_train,
                           label=y,
                           enable_categorical=True)

    def dtest(self, code_index=0):
        code = self.all_icds[code_index]
        y = self.y_test.apply(lambda x: code in x)
        return xgb.DMatrix(self.x_test,
                           label=y,
                           enable_categorical=True)

    def dval(self, code_index=0):
        code = self.all_icds[code_index]
        y = self.y_val.apply(lambda x: code in x)
        return xgb.DMatrix(self.x_val,
                           label=y,
                           enable_categorical=True)

class TreeDatasetIII(MIMIC_III_Preprocessor):
    binary_fields = [
        'GSN', 'SPEC_ITEMID', 'ORG_ITEMID', 'AB_ITEMID', 'LAB_ITEMID'
    ]

    def __init__(self, train_data_from='full'):
        super().__init__()
        self.all_icds = load_list_from_txt( # get top frequently code
            f'{PROCESSED_DIR}top50_icds.txt')
        self.label_encoder = LabelEncoder().fit(self.all_icds)
        self.train_data_from = train_data_from

    def save_preprocessed(self, path=PROCESSED_DIR):
        if hasattr(self, 'df_tree'):
            pd.to_pickle(self.df_tree.copy(), f'{path}mimic3_data_tree.pkl')
            utils.save_json(self.re_feature_map, f'{path}re_feature_map.json')
            utils.save_json(self.feature_map, f'{path}feature_map.json')

    def load_tree_data(self):
        self.load_preprocessed(data='tree')
        utils.print_time("df_tree shape:%s" % str(self.df_tree.shape))

    def load_split_data(self, path=PROCESSED_DIR):
        self.x_train = utils.load_pickle(f'{path}split_x_train.pkl')
        self.y_train = utils.load_pickle(f'{path}split_y_train.pkl')
        self.x_val = utils.load_pickle(f'{path}split_x_val.pkl')
        self.y_val = utils.load_pickle(f'{path}split_y_val.pkl')
        self.x_test = utils.load_pickle(f'{path}split_x_test.pkl')
        self.y_test = utils.load_pickle(f'{path}split_y_test.pkl')
        utils.print_time("Split Data Loaded!")

    def save_split_data(self, path=PROCESSED_DIR):
        utils.save_pickle(self.x_train, f'{path}split_x_train.pkl')
        utils.save_pickle(self.y_train, f'{path}split_y_train.pkl')
        utils.save_pickle(self.x_val, f'{path}split_x_val.pkl')
        utils.save_pickle(self.y_val, f'{path}split_y_val.pkl')
        utils.save_pickle(self.x_test, f'{path}split_x_test.pkl')
        utils.save_pickle(self.y_test, f'{path}split_y_test.pkl')
        utils.print_time("Split Data Saved!")

    def get_feature_map(self):
        feature_id = 1
        feature_map = defaultdict(dict) # field2value2feature_id
        re_feature_map = dict() # field2tuple(field, value)

        def get_counter(field):
            return Counter([e for lis in self.df_multi[field].dropna() for e in lis])

        for field in self.binary_fields:
            f_set = [v[0] for v in get_counter(field).items() if v[1] > 10]
            for v in f_set:
                feature_map[field][v] = feature_id
                re_feature_map[feature_id] = (field, v)
                feature_id += 1
            print("Process: " +  field)
        chart_items = [
            v[0] for v in get_counter('CHART_ITEMID').most_common(200)
            if v[1] > 10
        ]
        for item in chart_items: # chart data include max,min,avg
            feature_map['CHART_ITEMID'][item] = feature_id
            for f_id in range(feature_id, feature_id+3):
                re_feature_map[f_id] = ('CHART_ITEMID', item)
            feature_id += 3 # 3 for max,mim,avg
        utils.print_time(f"FeatureMap Generated num_feature_id: {feature_id-1}")
        return feature_map, re_feature_map

    def gen_tree_data(self):
        """
        将df_multi转为xgboost可用的DataFrame
        """
        self.feature_map, self.re_feature_map = self.get_feature_map()
        simple_fields = [
            'REQUEST_TELE', 'REQUEST_RESP', 'REQUEST_CDIFF', 'REQUEST_MRSA',
            'REQUEST_VRE', 'HOSPITAL_EXPIRE_FLAG', 'GENDER'
        ]
        df_tree = self.df_multi[['HADM_ID', 'AGE']].copy()
        ADMISSION_TYPEs = ['ELECTIVE', 'EMERGENCY', 'NEWBORN', 'URGENT']
        for adm_type in ADMISSION_TYPEs:
            df_tree["ADMISSION_TYPE" + adm_type] = self.df_multi.apply(
                lambda x: x.ADMISSION_TYPE == adm_type,
                axis=1).astype(pd.SparseDtype(bool))
        for s_field in simple_fields:
            df_tree[s_field] = (self.df_multi[s_field]
                        .apply(lambda x: pd.notna(x) and x >= 1)
                        .astype(pd.SparseDtype(bool)))

        for b_field in self.binary_fields:  # 处理binary数据
            for value, f_id in self.feature_map[b_field].items():
                df_tree[f_id] = (
                    self.df_multi[b_field].apply(lambda x: isinstance(
                        x, (list, set)) and value in x).astype(
                            pd.SparseDtype(bool)))
            utils.print_time(
                f"Processed field: {b_field} Len: {len(self.feature_map[b_field])}"
            )

        def by_index(x, f, v, index_field):
            return x[f][x[index_field].index(v)] if isinstance(
                x[index_field], list) and v in x[index_field] else pd.NA

        def fab(x, v):
            t = by_index(x, 'INTERPRETATION', v, 'AB_ITEMID')
            return pd.notna(t) and t == 'S'

        for v, f_id in self.feature_map['AB_ITEMID'].items():
            df_tree[f_id] = self.df_multi.apply(lambda x:fab(x, v),axis=1).astype(pd.SparseDtype(bool))
        utils.print_time("Processed field: AB_ITEMID")

        f = lambda x,f,v:by_index(x, f, v, 'CHART_ITEMID')

        for v, f_id in self.feature_map['CHART_ITEMID'].items():
            df_tree[f_id] = self.df_multi.apply(lambda x:f(x, 'MAX_VALUE_NUM', v), axis=1).astype(pd.SparseDtype(float))
            df_tree[f_id + 1] = self.df_multi.apply(lambda x:f(x, 'MEAN_VALUE_NUM', v), axis=1).astype(pd.SparseDtype(float))
            df_tree[f_id + 2] = self.df_multi.apply(lambda x:f(x, 'MIN_VALUE_NUM', v), axis=1).astype(pd.SparseDtype(float))
        utils.print_time("Processed field: CHART_ITEMID")
        # 合并 icd code
        df_diag = pd.read_csv(f'{MIMIC_DIR}DIAGNOSES_ICD.csv.gz')
        df_proc = pd.read_csv(f'{MIMIC_DIR}PROCEDURES_ICD.csv.gz')
        df_diag.ICD9_CODE = df_diag.ICD9_CODE.apply(lambda x: utils.reformat(str(x), True))
        df_proc.ICD9_CODE = df_proc.ICD9_CODE.apply(lambda x: utils.reformat(str(x), False))
        df_icds = (pd.concat([df_diag, df_proc]).dropna().
                   groupby('HADM_ID')['ICD9_CODE'].unique().reset_index())
        self.df_tree = pd.merge(df_icds, df_tree, how='left', on='HADM_ID')
        # return self.df_tree

    def split_train(self):
        if self.train_data_from == 'full':
            """
            train dataset shouldn't show ids from val and test datasets.
            """
            train_ids = load_list_from_txt(
                f'{PROCESSED_DIR}train_full_hadm_ids.csv')
            train_ids.extend(load_list_from_txt(
                f'{PROCESSED_DIR}dev_full_hadm_ids.csv'))
            train_ids.extend(load_list_from_txt(
                f'{PROCESSED_DIR}test_full_hadm_ids.csv'))
            val_ids = set(load_list_from_txt(
                f'{PROCESSED_DIR}dev_50_hadm_ids.csv'))
            test_ids = set(load_list_from_txt(
                f'{PROCESSED_DIR}test_50_hadm_ids.csv'))
            train_ids = set(train_ids)
            train_ids = train_ids - val_ids
            train_ids = train_ids - test_ids
        else:
            train_ids = load_list_from_txt(
                f'{PROCESSED_DIR}train_50_hadm_ids.csv')
            val_ids = load_list_from_txt(
                f'{PROCESSED_DIR}dev_50_hadm_ids.csv')
            test_ids = load_list_from_txt(
                f'{PROCESSED_DIR}test_50_hadm_ids.csv')
        def get_by_id(ids):
            ids = [int(v) for v in ids]
            y = self.df_tree[self.df_tree['HADM_ID'].isin(ids)]['ICD9_CODE']
            x = self.df_tree[self.df_tree['HADM_ID'].isin(ids)].drop(
                labels=['HADM_ID', 'ICD9_CODE'], axis=1)
            return x.copy(), pd.Series(y)
        print(self.df_tree['HADM_ID'].dtype)
        self.x_train, self.y_train = get_by_id(train_ids)
        self.x_test, self.y_test = get_by_id(test_ids)
        self.x_val, self.y_val = get_by_id(val_ids)

    def preprocess_tree(self, mod='final'):
        if mod == 'final':
            self.load_split_data()
            return
        if mod == 'ground':
            self.load_preprocessed(data='multi')
            self.gen_tree_data()
            self.save_preprocessed()
        elif mod == 'normal':
            self.load_tree_data()
        utils.print_time("Data Processed/Loaded! Begin split!")
        self.split_train()
        self.save_split_data()

    def dtrain(self, code_index=0):
        code = self.all_icds[code_index]
        y = self.y_train.apply(lambda x: code in x)
        return xgb.DMatrix(self.x_train,
                           label=y,
                           enable_categorical=True)

    def dtest(self, code_index=0):
        code = self.all_icds[code_index]
        y = self.y_test.apply(lambda x: code in x)
        return xgb.DMatrix(self.x_test,
                           label=y,
                           enable_categorical=True)

    def dval(self, code_index=0):
        code = self.all_icds[code_index]
        y = self.y_val.apply(lambda x: code in x)
        return xgb.DMatrix(self.x_val,
                           label=y,
                           enable_categorical=True)


if __name__ == "__main__":
    p = TreeDataset()
    p.load_preprocessed(data='multi')
    p.gen_tree_data()
    p.save_preprocessed()
    utils.print_time("OVER")