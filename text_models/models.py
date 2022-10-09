import utils
from torch import Tensor, nn
import numpy as np
import torch

class BaseTextModel(nn.Module):
    def __init__(self, word_vectors: np.ndarray, hidden_size=100, dropout=0.1, require_grad=False) -> None:
        super().__init__()
        self.embed_size = word_vectors.shape[1]
        self.hidden_size = hidden_size
        word_vectors = Tensor(np.r_[np.array([[0] * self.embed_size]),
                                    word_vectors])
        self.word_embeds = nn.Embedding(word_vectors.shape[0],
                                        word_vectors.shape[1],
                                        padding_idx=0,
                                        sparse=False,
                                        _weight=word_vectors)
        self.word_embeds.requires_grad_(require_grad)
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = 0

    def forward_embed(self, inputs: Tensor):
        """
        return:
            masks B, L
        """
        lengths, masks = (inputs != self.padding_idx).sum(dim=-1), inputs != self.padding_idx
        inputs = inputs[:, :lengths.max()]
        embeds = self.dropout(self.word_embeds(inputs))
        masks = masks[:, :lengths.max()]
        return embeds, lengths.cpu(), masks

    def forward(self, x: Tensor):
        x, _, _ = self.forward_embed(x)
        return x

class ConvTextModel(BaseTextModel):
    def __init__(self, word_vectors: np.ndarray, hidden_size=100, kernel_sizes=[4], dropout=0.1, require_grad=False) -> None:
        super().__init__(word_vectors, hidden_size, dropout, require_grad)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.embed_size,
                      hidden_size // len(kernel_sizes),
                      kernel_size=kernel_size,
                      padding=kernel_size // 2) for kernel_size in kernel_sizes
        ])

    def forward(self, x: Tensor):
        """
        x: B, L
        """
        x, _, masks = self.forward_embed(x) # B, L, E
        x = x.transpose(1, 2)
        x = torch.cat([conv(x) for conv in self.convs], dim=1) # B, L, hidden_size
        return x.transpose(1, 2).tanh(), masks

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers_num, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=layers_num,
                            dropout=dropout if layers_num > 1 else 0,
                            batch_first=True,
                            bidirectional=True)
        self.init_state = nn.Parameter(
            torch.zeros(2 * 2 * layers_num, 1, hidden_size))

    def forward(self, inputs, lengths, **kwargs):
        self.lstm.flatten_parameters()
        init_state = self.init_state.repeat([1, inputs.size(0), 1])
        cell_init, hidden_init = init_state[:init_state.size(0)//2], init_state[init_state.size(0)//2:]
        idx = torch.argsort(lengths, descending=True)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs[idx], lengths[idx], batch_first=True)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            self.lstm(packed_inputs, (hidden_init, cell_init))[0], batch_first=True)
        return outputs[torch.argsort(idx)]

class LSTMTextModel(BaseTextModel):
    def __init__(self,
                 word_vectors: np.ndarray,
                 hidden_size=100,
                 dropout=0.1,
                 layers_num=3,
                 require_grad=False,
                 normalize=True):
        super().__init__(word_vectors, hidden_size, dropout, require_grad)
        self.lstm = LSTMEncoder(self.embed_size, self.hidden_size // 2,
                                layers_num, dropout)
        self.norm_layer = nn.LayerNorm(hidden_size) if normalize else None

    def forward(self, x: Tensor):
        x, lengths, masks = self.forward_embed(x)
        x = self.lstm(x, lengths) # N, L, (hidden_size/2) * 2
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return self.dropout(x), masks
