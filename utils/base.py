from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from .utils import seed_everything
import json

class BaseTrainer(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self._log_path = os.path.join(args.log_path, args.log_label, str(datetime.datetime.now()))

        # Global Seed Init
        seed_everything(0 if self.args.seed is None else self.args.seed)

    def log_init(self):
        if self.args.log_path and not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)
        self._summary_writer = SummaryWriter(self._log_path)
        self.log_json(self.args)
    
    def log_json(self, info):
        with open(self._log_path + "config.json", 'a') as fp:
            json.dump(info, fp)
            fp.write("\n")

    def log_tensorboard(self, data, iteration: int, dataset_label: str, data_label: str="loss"):
        self._summary_writer.add_scalar('data/%s/%s' % (dataset_label, data_label), data, iteration)

    def log_tensor_json(self, datas, iteration: int, dataset_label: str):
        for k, v in datas.items():
            self.log_tensorboard(v, iteration, dataset_label, k)

