# TreeMAN: Tree-enhanced Multimodal Attention Network for ICD Coding

Code for COLING'22 paper "TreeMAN: Tree-enhanced Multimodal Attention Network for ICD Coding"


## Data Preparation (MIMIC-III 50)

- Place [MIMIC-III dataset](https://mimic.mit.edu/) files under `data/mimic-data/`
- The id files under `data/preprocessed/` are from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)
- `data/preprocessed/top50_icds.txt` contains the top 50 icd codes (same as [caml-mimic/dataproc_mimic_III.ipynb](https://github.com/jamesmullenbach/caml-mimic/blob/master/notebooks/dataproc_mimic_III.ipynb))

## Run

1. Preprocess dataset: `preprocess.py`
2. Preprocess dataset for decision tree: `tree_datasets.py`
3. Train decision trees and generate the leaf information: `tree_method.py`
4. Train word embedding: `text_models/word_embed.py` (modify this file to change it's config)
5. Train *TreeMAN*, predict with *TreeMAN* and evaluate the result: `run.py`

## Configuration

To change the configuration, see `conf.py` which contains all configuration for this project (except the configuration for training word embedding).
