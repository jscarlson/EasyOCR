from numpy import save
import pandas as pd
import os
from glob import glob
import numpy as np
from shutil import copy
import json


if __name__ == "__main__":

    save_dir = "./easyocr_data"
    os.makedirs(save_dir, exist_ok=True)
    train_dir = os.path.join(save_dir, "tk1957_train")
    val_dir = os.path.join(save_dir, "tk1957_val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    root_dir = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ocr_datasets/teikoku/1957"
    chars_dir = os.path.join(root_dir, "char_crops")
    seg_dir_unlab = os.path.join(root_dir, "seg_unlabeled")
    seg_dir_lab = os.path.join(root_dir, "seg_labeled")
    coco_train_path = os.path.join(root_dir, "tk1957_ann_file_train70.json")
    coco_val_path = os.path.join(root_dir, "tk1957_ann_file_test30.json")
    seg_paths = glob(os.path.join(seg_dir_unlab, "*.png"))
    seg_basenames = [os.path.basename(x) for x in seg_paths]

    with open(coco_train_path) as f: coco_train = json.load(f)
    with open(coco_val_path) as f: coco_val = json.load(f)
    
    train_seg_basenames = [x['file_name'] for x in coco_train["images"]]
    val_seg_basenames = [x['file_name'] for x in coco_val["images"]]
    print(f"Len val segs {len(val_seg_basenames)}; len train segs {len(train_seg_basenames)}")

    train_labels = []
    for sbname in train_seg_basenames:
        sbname_w_labels = [x for x in os.listdir(seg_dir_lab) if "_".join(x.split("_")[:-1]) == os.path.splitext(sbname)[0]][0]
        seq_str = os.path.splitext(sbname_w_labels)[0].split("_")[-1]
        if "ゲ" in seq_str:
                seq_str = seq_str.replace("ゲ", "ゲ")   
        train_labels.append((sbname, seq_str))
        copy(os.path.join(seg_dir_unlab, sbname), train_dir)
    train_labels_df = pd.DataFrame(train_labels, columns=["filename", "words"])
    train_labels_df.to_csv(os.path.join(train_dir, "labels.csv"))

    val_labels = []
    for sbname in val_seg_basenames:
        sbname_w_labels = [x for x in os.listdir(seg_dir_lab) if "_".join(x.split("_")[:-1]) == os.path.splitext(sbname)[0]][0]
        seq_str = os.path.splitext(sbname_w_labels)[0].split("_")[-1]
        if "ゲ" in seq_str:
            seq_str = seq_str.replace("ゲ", "ゲ") 
        val_labels.append((sbname, seq_str))
        copy(os.path.join(seg_dir_unlab, sbname), val_dir)
    val_labels_df = pd.DataFrame(val_labels, columns=["filename", "words"])
    val_labels_df.to_csv(os.path.join(val_dir, "labels.csv"))