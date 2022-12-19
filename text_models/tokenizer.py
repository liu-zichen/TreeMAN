import re
from nltk.tokenize import RegexpTokenizer
from shutil import ExecError
from typing import List, Dict
import torch
import numpy as np

class Tokenizer(object):
    def __init__(self, w2idx) -> None:
        super().__init__()
        self.w2idx = {w:(idx+1) for w, idx in w2idx.items()}
        self.w2idx["[pad]"] = 0

    def __call__(self, texts: List[str], max_length=4000):
        """
        return: 
            input_ids: [[idx...]...]
            masks: [[1..0..]...]
        """
        idxs_list = []
        masks = []
        max_words = 0
        for text in texts:
            tokens = NoteExtractor.to_doc(text)
            idxs = [
                self.w2idx[tok] for i, tok in enumerate(tokens)
                if tok in self.w2idx and i < max_length
            ]
            max_words = max(len(idxs), max_words)
            idxs_list.append(idxs)
        for idxs in idxs_list:
            masks.append([1] * len(idxs) + [0] * (max_words - len(idxs)))
            idxs.extend([0] * (max_words - len(idxs)))
        return torch.LongTensor(idxs_list), torch.LongTensor(masks)

    def get_one(self, text, max_length=4000):
        tokens = NoteExtractor.to_doc(text)
        idxs = [
            self.w2idx[tok] for i, tok in enumerate(tokens)
            if tok in self.w2idx and i < max_length
        ]
        idxs = idxs + [0] * (max_length - len(idxs))
        # masks = [1] * len(idxs) + [0] * (max_length - len(idxs))
        return np.array(idxs)


class NoteExtractor(object):
    """
    deal clinical note
    """
    tokenizer = RegexpTokenizer(r'\w+')

    @classmethod
    def to_doc(cls, doc: str) -> List[str]:
        doc = doc.lower()
        return [t for t in cls.tokenizer.tokenize(doc) if not t.isnumeric()]
