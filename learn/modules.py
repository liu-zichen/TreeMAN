from unittest import TestCase
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class Embedding(nn.Module):
    def __init__(self, emb_init=None, emb_num=None, emb_dim=None) -> None:
        super().__init__()
        self.emb = nn.Embedding(emb_num,
                                emb_dim,
                                padding_idx=0,
                                _weight=emb_init,
                                sparse=True)

    def forward(self, inputs):
        return self.emb(inputs)

class LWAttention(nn.Module):
    def __init__(self, labels_num, hidden_size, label_size=None) -> None:
        super().__init__()
        self.label_size = label_size
        if label_size:
            self.W_k = nn.Linear(hidden_size, label_size)
        self.U = nn.Linear(label_size if label_size else hidden_size, labels_num, False)
        self.final = nn.Linear(hidden_size, labels_num)

    def forward(self, x: Tensor, masks=None):
        """
        x: B, S, H
        masks: B, S
        U: L, H
        F: L, H
        """
        if self.label_size:
            k = self.W_k(x)
            att = self.U(k)
        else:
            att = self.U(x)
        if masks is not None:
            masks = torch.unsqueeze(masks, 2) # B, S, 1
            att = att.masked_fill(masks < 1, -torch.inf)
        att = F.softmax(att, dim=1) # B, S, L
        x = att.transpose(1, 2).matmul(x) # B, L, H
        x = self.final.weight.mul(x).sum(dim=2).add(self.final.bias) # B, L
        return x.sigmoid()


def lw_attention(inputs, masks, labels, c_weight, c_bias):
    """
    label_wise attention + Linear Regression
    INPUT:
        text-inputs: N, Token_num, hidden
        # tree-inputs: N, Tree_num,  hidden2
        masks:       N, Token_num (0 for mask)
        labels:      candidates, label_dim=hidden
        c_weight:    candidates, hidden
        c_bias:      candidates
    OUTPUT: 
        scores: N, candidates, 1
    """
    masks = torch.unsqueeze(masks, 2) # N, Token_num, 1
    att = (F.linear(inputs, weight=labels) # N, candidates, Token_num
            .masked_fill(masks < 1, -torch.inf)
            .transpose(1, 2).softmax(2))
    x = att @ inputs # N, candidates, hidden
    x = c_weight.mul(x).sum(dim=2).add(c_bias) # N, candidates
    return x.sigmoid() # N, candiates

class CrossAttention(nn.Module):
    def __init__(self,
                 text_size,
                 leaf_size,
                 out_size,
                 leaf_k=None,
                 att_type="cross",
                 tree_num=50,
                 residual=True,
                 dropout=0.5,
                 normalize=False):
        """
        att_type: "cross", "maxpool", "average"
        """
        super().__init__()
        self.W_q = nn.Linear(text_size, leaf_size if leaf_k is None else leaf_k)
        have_bias = True
        self.k = nn.Linear(leaf_k, tree_num,
                           bias=have_bias) if leaf_k is not None else None
        if self.k is not None and have_bias:
            self.k.bias = nn.Parameter(
                torch.zeros(self.k.bias.size(), dtype=self.k.bias.dtype))
        self.convertor = nn.Linear(text_size + leaf_size, out_size)
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.nor_layer = nn.LayerNorm(out_size) if normalize else None
        self.att_type = att_type
        self.max_pool = nn.MaxPool1d(tree_num)

        assert (not residual) or out_size == text_size

    def forward(self, text_inputs: Tensor, leaves: Tensor):
        """
        text_inputs: B, L, W
        leaves: B, T, E
        """
        if self.att_type == "cross":
            q = self.W_q(text_inputs) # B, L, E
            if self.k is None:
                att = torch.matmul(leaves, q.transpose(1, 2)).transpose(1, 2) # B, L, T
            else:
                att = self.k(q)
            x = att.softmax(2) @ leaves # B, L, E
        elif self.att_type == "average":
            x = torch.mean(leaves, dim=1) # B, E
            x = x.repeat(text_inputs.size()[1], 1, 1).transpose(0, 1)
        else:
            x = self.max_pool(leaves.transpose(1, 2))
            x = x.repeat(1, 1, text_inputs.size()[1]).transpose(1, 2)
        x = self.dropout(x)
        x = torch.cat((text_inputs, x), dim=2) # B, L, W+E
        x = self.convertor(x) # B, L, label_size
        if self.nor_layer is not None:
            x = self.nor_layer(x)
        # x = x.tanh()
        if self.residual:
            x = text_inputs + x
        return x.tanh()
