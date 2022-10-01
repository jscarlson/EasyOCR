import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
import argparse
import wandb

cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if os.path.isfile(opt.lang_char):
        with open(opt.lang_char) as f:
            opt.character = "".join(chr(int(i)) for i in f.read().split())
            if "en" in opt.lang_char:
                opt.character += " "
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    opt = get_config(args.config_file)
    wandb.init(project="EasyOCR_v2", name=opt.experiment_name)
    train(opt, amp=False)
