import xgboost as xgb
import pandas as pd
import re
import numpy as np
import os

from conf import MODEL_DIR, tree_config, FILE_LABEL, PROCESSED_DIR
import utils
from tree_datasets import TreeDataset
from preprocess import MIMIC_Preprocessor

class TreeMethod:
    def __init__(self, config) -> None:
        self.config = config
        print(config)
        self.dataset = TreeDataset(config.train_data_from)
        self.bsts = []

    def train(self):
        self.dataset.preprocess_tree('normal')
        params = self.config.params
        utils.print_time("Begin Training! xtrain shape:%s" %
                         str(self.dataset.x_train.shape))
        for i in range(self.config.codes_num):
            dtrain, dval = self.dataset.dtrain(i), self.dataset.dval(i)
            bst = xgb.train(params,
                            dtrain=dtrain,
                            evals=[(dtrain, 'train'),
                                (dval, 'val')],
                            verbose_eval=True,
                            num_boost_round=self.config.num_boost_round)
            self.bsts.append(bst)
            utils.print_time(f"Code {str(i)} Training Completed")

    def save_model(self):
        utils.print_time("Begin To Save Model!")
        model_dir = f"{MODEL_DIR}{FILE_LABEL}/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        for i, bst in enumerate(self.bsts):
            bst.save_model(f"{model_dir}tree_model{str(i)}.json") # xgb doc: must use json
            bst.dump_model(f"{model_dir}xgbst{str(i)}.raw.txt")

    def load_model(self):
        model_dir = f"{MODEL_DIR}{FILE_LABEL}/"
        for i in range(self.config.codes_num):
            bst = xgb.Booster()
            bst.load_model(f"{model_dir}tree_model{str(i)}.json")
            self.bsts.append(bst)
        utils.print_time("Tree Model Loaded")

    def predict_save_idx(self):
        """
        get leaf idx for all samples
        """
        self.dataset.load_tree_data()
        self.dataset.df_tree.drop('ICD9_CODE', axis=1, inplace=True)
        self.dataset.df_tree.drop_duplicates(subset='HADM_ID', inplace=True)
        utils.print_time("Begin Predicting! Data shape:%s" %
                         str(self.dataset.df_tree.shape))
        df_chunks = np.array_split(self.dataset.df_tree, 10)

        def predict_chunk(df_chunk):
            labels = df_chunk[['HADM_ID']]
            labels.reset_index(inplace=True, drop=True)
            df_chunk.drop('HADM_ID', axis=1, inplace=True)
            cons = [labels]
            for bst in self.bsts:
                leaf_idxs = pd.DataFrame(
                    bst.predict(xgb.DMatrix(df_chunk, enable_categorical=True),
                                pred_leaf=True))
                cons.append(leaf_idxs)
                utils.print_time("Predicted! leaf_idxs shape:%s" %
                                str(leaf_idxs.shape))
            con = pd.concat(cons, axis=1)
            utils.print_time("Concat! con shape:%s" % str(con.shape))
            return con

        self.sample_leaves = pd.concat([predict_chunk(df) for df in df_chunks])
        utils.print_time("Predicted! predicted shape:%s" %
                         str(self.sample_leaves.shape))
        utils.save_pickle(self.sample_leaves, f"{PROCESSED_DIR+FILE_LABEL}sample_leaves.pkl")

def train():
    t = TreeMethod(utils.DotDict(tree_config))
    t.train()
    t.save_model()

def predict():
    t = TreeMethod(utils.DotDict(tree_config))
    t.load_model()
    t.predict_save_idx()

if __name__ == "__main__":
    train()
    predict()

    # build final dataset
    p = MIMIC_Preprocessor()
    p.load_preprocessed(data='df')
    p.load_preprocessed(data='leaf')
    p.preprocess_final()
    utils.print_time("OVER!!!")
