import datasets
import gensim.models.word2vec as w2v
import torch
import transformers
import utils
from evaluation import all_metrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import BaseTrainer
from torch import optim
from conf import MIMIC_II_FLAG

from .models import *
from text_models import Tokenizer, ConvTextModel


class Trainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.wv = w2v.Word2Vec.load(args.word_embed_path).wv
        self.tokenizer = Tokenizer(self.wv.key_to_index)
        self.dataset = datasets.MIMIC_Dataset(self.tokenizer,
                                              full=args.full,
                                              codes_num=args.codes_num,
                                              tree_num=args.tree_num)
        self.args["leaves_num"] = self.dataset.leaves_num
        utils.print_time("Trainer Init!!!")

    def train(self):
        self.log_init()
        self.model_init()
        self.model, self.device = utils.model_device(self.args.n_gpu, self.model)

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=True, drop_last=True)
        params = self.model.parameters()
        if self.args.optim == 'SGD':
            self.optimizer = optim.SGD(params, lr=self.args.lr)
        elif self.args.optim == 'AdamW':
            self.optimizer = transformers.AdamW(params, self.args.lr, eps=self.args.eps)
        elif self.args.optim == 'SparseAdam':
            self.optimizer = utils.DenseSparseAdam(params, self.args.lr, eps=self.args.eps)

        if self.args.schedule == "linear":
            self.scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.n_epochs)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.args.factor,
                patience=self.args.patience,
                min_lr=1e-4)

        best_eval, best_epoch, best_save = 0, 0, 0
        for epoch in range(self.args.n_epochs):
            self.train_epoch(epoch=epoch)
            results = self.eval('val')
            results['lr'] = self.optimizer.param_groups[0]['lr']
            all_score = self.overall_score(results)
            results['all_score'] = all_score
            self.log_tensor_json(results, epoch, "MIMIC_III")
            results['epoch'] = epoch

            if all_score > best_eval:
                best_eval, best_epoch = all_score, epoch
                self.save_model(flag="best")
            self.log_json(results)

            if best_epoch < epoch-self.args.break_epoch:
                break
            if self.args.schedule == "linear":
                self.scheduler.step()
            else:
                self.scheduler.step(all_score)
        del self.optimizer
        del self.scheduler

    def text_model_init(self):
        if self.args.text_model_name == "cnn":
            return ConvTextModel(self.wv.vectors,
                                 hidden_size=self.args.hidden_size,
                                 kernel_sizes=self.args.kernel_sizes,
                                 dropout=self.args.dropout,
                                 require_grad=self.args.require_grad)
        else:
            return LSTMTextModel(self.wv.vectors,
                                 hidden_size=self.args.hidden_size,
                                 layers_num=self.args.layers_num,
                                 dropout=self.args.dropout,
                                 require_grad=self.args.require_grad,
                                 normalize=self.args.text_norm)

    def model_init(self):
        text_model = self.text_model_init()
        self.model = LwanTextModel(text_model,
                                   self.dataset.label_num,
                                   label_size=self.args.label_size)

    def train_epoch(self, epoch):
        self.dataset.set_mod('train')
        self.model.train()
        base_iter = epoch * len(self.dataloader)
        for i, batch in enumerate(self.dataloader):
            logits, labels = self.forward_batch(batch)
            loss = XLMLossFct(logits, labels.to(self.device).float())
            loss.backward()
            if (i+1) % self.args.batch_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            if epoch < 1:
                self.log_tensorboard(loss.item(), i + base_iter, "MIMIC_III")
        self.optimizer.zero_grad()

    def predict(self, mod='test'):
        self.dataset.set_mod(mod)
        p_dataloader = DataLoader(self.dataset,
                                  batch_size=self.args.batch_size*4,
                                  shuffle=False,
                                  drop_last=False)
        self.model.eval()
        results, ground_trues = [], [],
        for batch in p_dataloader:
            with torch.no_grad():
                logits, labels = self.forward_batch(batch) # batch_size, label_num
            results.append(logits.cpu())
            ground_trues.append(labels)
        return torch.cat(results), torch.cat(ground_trues) # samples, label_num

    def eval(self, mod='test'):
        results, ground_trues = self.predict(mod=mod)
        ground_trues = ground_trues > 0

        def result_to_binary(x: Tensor):
            # x: samples, label_num
            res = x >= 0.5
            max_r = x.argmax(1)
            for i, idx in enumerate(max_r):
                res[i][idx] = True
            return res

        res = all_metrics((result_to_binary(results)).numpy(),
                          ground_trues.numpy(),
                          yhat_raw=results.numpy(),
                          k=3 if MIMIC_II_FLAG else 5)
        print(res)
        # metric2(ground_trues, results)
        return res

    @staticmethod
    def overall_score(results):
        return results["auc_macro"] + results["auc_micro"] + results[
            "f1_macro"] / 7 + results["f1_micro"] / 7 + results[
                "prec_at_" + "3" if MIMIC_II_FLAG else "5"] / 5

    def forward_batch(self, batch):
        input_ids, leaves, labels = batch
        logits = self.model(input_ids=input_ids.to(self.device))
        return logits, labels

    def save_model(self, flag=None):
        save_path = self.args.model_save_path + flag if flag else self.args.model_save_path
        utils.save_model(self.model, save_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.args.model_load_path
        self.model_init()
        self.model.load_state_dict(torch.load(model_path))
        self.model, self.device = utils.model_device(self.args.n_gpu, self.model)

    def load_embeddings(self):
        raise NotImplementedError("load_embeddings")

class LwanCrossTrainer(Trainer):
    """
    Trainer of LwanCrossModel
    """
    def model_init(self):
        text_model = self.text_model_init()
        leaves_emb = nn.Embedding(self.dataset.leaves_num,
                                  self.args.leaf_size,
                                  padding_idx=0,
                                  sparse=False)
        self.model = LwanCrossModel(text_model,
                                    self.dataset.label_num,
                                    leaves_emb,
                                    att_type=self.args.att_type,
                                    tree_num=self.args.codes_num * self.args.tree_num,
                                    residual=self.args.residual,
                                    label_size=self.args.label_size,
                                    leaf_k=self.args.leaf_k_size,
                                    normalize=self.args.normalize,
                                    dropout=self.args.dropout)

    def load_embeddings(self):
        if hasattr(self.args, "leaves_weight_path"):
            self.model.leaves_emb.weight = nn.Parameter(torch.load(
                self.args.leaves_weight_path)["leaves_emb.weight"])
            utils.print_time("INIT leaves_emb.weight!!!")
        if hasattr(self.args, "lwan_weight_path"):
            state_dict = torch.load(self.args.lwan_weight_path)
            self.model.lwan.final.weight = nn.Parameter(state_dict["lwan.final.weight"])
            self.model.lwan.final.bias = nn.Parameter(state_dict["lwan.final.bias"])
            self.model.lwan.U.weight = nn.Parameter(state_dict["lwan.U.weight"])
            utils.print_time("INIT LWAN weight!!!")

    def forward_batch(self, batch):
        input_ids, leaves, labels = batch
        logits = self.model(input_ids=input_ids.to(self.device),
                            leaf_ids=leaves.to(self.device))
        return logits, labels


Model2Trainer = {
    "lwan_text": Trainer,
    "lwan_cross": LwanCrossTrainer
}
