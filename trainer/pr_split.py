from numpy import save
import pandas as pd
import os
from glob import glob
import numpy as np
from shutil import copy


def extract_seg_id(p):
    return p.split("_")[p.split("_").index("SEG")+1].split("-")[-1]


if __name__ == "__main__":

    save_dir = "/home/jscarlson/Downloads/easyocr_data"
    os.makedirs(save_dir)
    train_dir = os.path.join(save_dir, "pr_train")
    val_dir = os.path.join(save_dir, "pr_val")
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    root_dir = "/media/jscarlson/BackupPlus/Japan/personnelrecords/deep-learning-pipeline/segment-annotations/v2-mapped/corrected_crops"
    chars_dir = os.path.join(root_dir, "chars")
    seg_dir = os.path.join(root_dir, "seg")
    seg_paths = glob(os.path.join(seg_dir, "*.png"))
    seg_basenames = [os.path.basename(x) for x in seg_paths]

    seg_ids = [extract_seg_id(p) for p in seg_paths]
    uni_seg_ids = sorted(list(set(seg_ids)))
    np.random.seed(99)
    np.random.shuffle(uni_seg_ids)
    EVAL_PCT = 0.3
    val_seg_ids = uni_seg_ids[:int(EVAL_PCT*len(uni_seg_ids))]
    train_seg_ids = uni_seg_ids[int(EVAL_PCT*len(uni_seg_ids)):]
    print(f"Len val {len(val_seg_ids)}; len train {len(train_seg_ids)}")

    train_seg_basenames = [x for x in seg_basenames if extract_seg_id(x) in train_seg_ids]
    val_seg_basenames = [x for x in seg_basenames if extract_seg_id(x) in val_seg_ids]
    print(f"Len val segs {len(val_seg_basenames)}; len train segs {len(train_seg_basenames)}")

    train_labels = []
    for sbname in train_seg_basenames:
        seq_str = os.path.splitext(sbname)[0].split("_")[-1]
        train_labels.append((sbname, seq_str))
        copy(os.path.join(seg_dir, sbname), train_dir)
    train_labels_df = pd.DataFrame(train_labels, columns=["filename", "words"])
    train_labels_df.to_csv(os.path.join(train_dir, "labels.csv"))

    val_labels = []
    for sbname in val_seg_basenames:
        seq_str = os.path.splitext(sbname)[0].split("_")[-1]
        val_labels.append((sbname, seq_str))
        copy(os.path.join(seg_dir, sbname), val_dir)
    val_labels_df = pd.DataFrame(val_labels, columns=["filename", "words"])
    val_labels_df.to_csv(os.path.join(val_dir, "labels.csv"))