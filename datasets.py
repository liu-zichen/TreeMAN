from shutil import ExecError
import numpy as np
from torch.utils.data import Dataset
from preprocess import MIMIC_Preprocessor
import torch
import utils


class MIMIC_Dataset(Dataset):
    def __init__(self,
                 tokenizer,
                 full=True,
                 codes_num=50, # one vs rest 中训练的code数量
                 tree_num=2): # 每个code采用树的数量
        super().__init__()
        self.mod = "train"
        self.full = full
        self.codes_num, self.tree_num = codes_num, tree_num
        self.tokenizer = tokenizer
        self.processor_init()
        self.MAXLEN = 0
        self.leaves_num = max(self.processor.df_final[[
            i for i in range(tree_num * codes_num)
        ]].max()) + 1
        self.label_num = len(self.processor.mlb.classes_)
        utils.print_time("leaves_num: " + str(self.leaves_num))
        utils.print_time("labels_num: " + str(self.label_num))

    def processor_init(self):
        self.processor = MIMIC_Preprocessor()
        self.processor.load_preprocessed()
        self.processor.df_final = self.processor.df_final[
            [i for i in range(self.tree_num)] +
            ["HADM_ID", "TEXT", "ICD9_CODE"]].astype(
                {i: int for i in range(self.tree_num)})
        d = {i: [ii + i*self.codes_num for ii in range(self.codes_num)] for i in range(self.tree_num)}
        self.processor.df_final.rename(columns=lambda c: d[c].pop(0)
                                        if c in d.keys() else c,
                                        inplace=True)
        # convert tree node id => embed idx
        base_num = 1 # 0 for padding idx
        for i in range(self.tree_num * self.codes_num):
            leaf2idx = dict()
            for value in self.processor.df_final[i]:
                if value not in leaf2idx:
                    leaf2idx[value] = base_num
                    base_num+=1
            self.processor.df_final[i].replace(leaf2idx, inplace=True)
        # 预处理 text->embed_ids
        self.processor.df_final['INPUTS'] = self.processor.df_final[
            'TEXT'].apply(lambda x: self.tokenizer.get_one(x))
        self.processor.split(self.full)

    def set_mod(self, mod):
        self.mod = mod

    def __len__(self):
        if self.mod == "train":
            return len(self.processor.x_train)
        elif self.mod == "test":
            return len(self.processor.x_test)
        elif self.mod == "val":
            return len(self.processor.x_val)
        raise ExecError(f"Wrong mod {self.mod}")

    def __getitem__(self, index):
        x, y = self.get_row(index)
        leaves = torch.LongTensor(
            x.drop(['HADM_ID', 'TEXT', 'INPUTS']).to_numpy(dtype=np.int64))
        input_ids = x['INPUTS']
        return torch.LongTensor(input_ids), leaves, y.to_numpy()

    def get_row(self, index):
        if self.mod == "train":
            x, y = self.processor.x_train, self.processor.y_train
        elif self.mod == "test":
            x, y = self.processor.x_test, self.processor.y_test
        elif self.mod == "val":
            x, y = self.processor.x_val, self.processor.y_val
        else:
            raise ExecError(f"Wrong mod {self.mod}")
        return x.loc[index], y.loc[index]
