import torch
from torch import Tensor
from torch import nn
from .modules import LWAttention, CrossAttention
from text_models import ConvTextModel, LSTMTextModel
from typing import Union
from torch.optim import Adam, SparseAdam
import torch.nn.functional as F

TEXT_MODEL = Union[ConvTextModel, LSTMTextModel]
XLMLossFct = nn.BCELoss()

class LwanTextModel(nn.Module):
    """
    LWAN + TEXT
    """
    def __init__(self, text_model: TEXT_MODEL, labels_num, label_size=None):
        super().__init__()
        self.text_model = text_model
        self.lwan = LWAttention(labels_num,
                                text_model.hidden_size,
                                label_size=label_size)

    def forward(self, input_ids):
        text_outputs, masks = self.text_model(input_ids) # N, S, hidden_size
        return self.lwan(text_outputs, masks) # N, labels_num


class LwanCrossModel(nn.Module):
    """
    LWAN + TEXT + GBDT + CrossAttention
    """
    def __init__(self,
                 text_model: TEXT_MODEL,
                 labels_num,
                 leaves_emb: nn.Embedding,
                 att_type="cross",
                 tree_num=50,
                 leaf_k=None,
                 label_size=None,
                 residual=True,
                 normalize=False,
                 dropout=0.5):

        super().__init__()
        self.text_model = text_model
        self.lwan = LWAttention(labels_num,
                                text_model.hidden_size,
                                label_size=label_size)
        self.leaves_emb = leaves_emb
        self.cross_attention = CrossAttention(
            text_model.hidden_size,
            leaves_emb.embedding_dim,
            text_model.hidden_size,
            leaf_k=leaf_k,
            att_type=att_type,
            tree_num=tree_num,
            normalize=normalize,
            residual=residual,
            dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, leaf_ids):
        text_outputs, masks = self.text_model(input_ids) # N, T, H
        leaf_embs = self.leaves_emb(leaf_ids) # N, Tree_Num, E
        leaf_embs = self.dropout(leaf_embs)
        outputs = self.cross_attention(text_outputs, leaf_embs) # N, T, H
        outputs = self.dropout(outputs)
        x = self.lwan(outputs, masks)
        return x # N, Label_num
