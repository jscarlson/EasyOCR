import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd

cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    with open(opt.lang_char) as f:
        opt.character = "".join(f.read().split())
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

if __name__ == '__main__':

    opt = get_config("config_files/pr_config.yaml")
    train(opt, amp=False)
