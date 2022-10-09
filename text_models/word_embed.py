import csv

import gensim.models.word2vec as w2v
import pandas as pd
import utils
from conf import EMBED_DIR, MIMIC_DIR

from tokenizer import NoteExtractor


class ProcessedIter(object):
    def __init__(self):
        self.df = pd.read_csv(MIMIC_DIR + "NOTEEVENTS.csv.gz")

    def __iter__(self):
        for _, text in self.df['TEXT'].items():
            yield(NoteExtractor.to_doc(text))

def word_embeddings(out_file, embedding_size, min_count, n_iter):
    sentences = ProcessedIter()
    model = w2v.Word2Vec(vector_size=embedding_size,
                         min_count=min_count,
                         workers=64,
                         sg=0,
                         negative=5,
                         window=5)
    model.build_vocab(sentences)
    utils.print_time("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=n_iter)
    utils.print_time("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file

if __name__ == "__main__":
    word_embeddings(EMBED_DIR + "128_0_10_cb_5n_5w.embeds", 128, 0, 10)

# nohup python3.9 text_models/word_embed.py >e1.out 2>&1 &