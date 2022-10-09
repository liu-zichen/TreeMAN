from pathlib import Path
from datetime import datetime

now = datetime.now()

REPO_PATH = str(Path(__file__).resolve().parents[0])
MIMIC_II_FLAG = False # False means mimic_iii, True means mimic_ii
FILE_LABEL = "10-99-full-5" # num_boost_round-lr-train_data_from-max_depth
DATA_DIR = REPO_PATH + "/data" + ("ii" if MIMIC_II_FLAG else "") + "/"
MIMIC_DIR = DATA_DIR + "mimic-data/"
PROCESSED_DIR = DATA_DIR + "preprocessed/"
MODEL_DIR = DATA_DIR + "models/"
EMBED_DIR = MODEL_DIR + "embeds/"

common_config = dict(
    log_path = DATA_DIR + "logs/",
    log_label = "test",
    n_gpu = "0",
    seed = 42,
    full = False
)

leaves_config = dict(
    codes_num = 50,
    tree_num = 1, # how many tree used for each icd code
)

lwan_text_config = dict(
    model_name = "lwan_text",
    model_save_path = MODEL_DIR + now.strftime("%d-%H-%M") + "lwantext.model",
    model_load_path = MODEL_DIR + "22-07-35lwantext.model",
    # text model
    text_model_name = "lstm", # lstm cnn
    word_embed_path = EMBED_DIR + "128_0_10_cb_5n_5w.embeds",
    # word_embed_path = EMBED_DIR + "word2vec_sg0_100.model",
    require_grad = False,
    hidden_size = 512,
    kernel_sizes = [5, 7, 9, 11],
    layers_num = 1,
    # lwan
    label_size = 256, # None 表示不启用 w_k
    # for training
    optim = "AdamW", # SGD, AdamW, SparseAdam
    batch_size = 8,
    batch_steps = 1,
    lr = 1e-3,
    dropout = 0.3,
    eps = 1e-8,
    n_epochs = 128,
    schedule = "plateau", # linear / plateau
    num_warmup_steps = 10,
    patience = 5,
    factor = 0.9,
    break_epoch = 30
)

lwan_cross_config = dict(
    model_name = "lwan_cross",
    model_save_path = MODEL_DIR + now.strftime("%d-%H-%M") + common_config["n_gpu"] + "lwancross.model",
    # model_load_path = MODEL_DIR + "22-07-34lwancross.model",
    # text model
    text_model_name = "lstm", # lstm
    word_embed_path = EMBED_DIR + "128_0_10_cb_5n_5w.embeds",
    # word_embed_path = EMBED_DIR + "word2vec_sg0_100.model",
    require_grad = False,
    hidden_size = 512,
    kernel_sizes = [5, 7, 9, 11],
    layers_num = 1,
    # lwan
    label_size = 256, # None 表示不启用 w_k
    # for training
    optim = "AdamW", # SGD, AdamW, SparseAdam
    batch_size = 8,
    batch_steps = 1,
    lr = 1e-3,
    dropout = 0.3,
    eps = 1e-8,
    n_epochs = 100,
    schedule = "plateau", # linear / plateau
    num_warmup_steps = 10,
    patience = 5,
    factor = 0.9,
    break_epoch = 50,
    # special
    leaf_size = 20,
    residual = False,
    normalize = True,
    text_norm = False,
    leaf_k_size = 128, # dim for tree embedding in paper
    att_type = "cross", # "cross", "maxpool", "average"
)

tree_config = dict(
    params={
        'booster': 'gbtree',
        'max_depth': 5,
        'num_parallel_tree': 1,
        'eta': 0.99, # learning_rate
        'tree_method': 'hist',
        'objective': "reg:logistic",
        # 'enable_categorical': True,  # 类别类型数据
        # 'gpu_id': 1,
        # 'nthread': 1
        # 'subsample': 0.1,
        # 'sampling_method': 'gradient_based',
        'eval_metric': 'rmse'
    },
    num_boost_round = 10,
    codes_num = 50, # all=8907
    train_data_from = "full", # '50'/'full'
)

train_config = {**common_config, **lwan_cross_config, **leaves_config}
train_config["log_label"] = "_".join([
    train_config["model_name"], # train_config["text_model_name"],
    train_config["log_label"]
])
train_config['FILE_LABEL'] = FILE_LABEL