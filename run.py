from learn import Model2Trainer, Trainer
import utils
import conf
import torch

def train_and_eval():
    args = utils.DotDict(conf.train_config)
    trainer: Trainer = Model2Trainer[args.model_name](args)
    trainer.train()
    torch.cuda.empty_cache()
    trainer.load_model(trainer.args.model_save_path + "best")
    results = trainer.eval('test')
    results["final"] = True
    trainer.log_json(results)
    return results

def main():
    results = []
    for i in range(5):
        conf.train_config['seed'] += 1
        r = train_and_eval()
        results.append(r)
    t = results[0]
    print({k: sum([r[k] for r in results]) / len(results) for k in t.keys()})


def eval():
    args = utils.DotDict(conf.train_config)
    trainer = Model2Trainer[args.model_name](args)
    trainer.load_model("./data/models/09-05lwantext.model")
    trainer.eval('test')

if __name__ == "__main__":
    train_and_eval()
    # main() train_and_eval()
    

# nohup python3.9 run.py >run1.out 2>&1 &
