import pandas as pd
import os
from shutil import copy
import json


def create_datasets(seg_basenames, seg_texts, seg_dir, save_dir):
    labeled_pairs = []
    for fn, txt in zip(seg_basenames, seg_texts):
        labeled_pairs.append((fn, txt))
        copy(os.path.join(seg_dir, fn), save_dir)
    labels_df = pd.DataFrame(labeled_pairs, columns=["filename", "words"])
    labels_df.to_csv(os.path.join(save_dir, "labels.csv"))


if __name__ == "__main__":

    # set up initial dirs
    save_dir = "./easyocr_data"
    train_dir = os.path.join(save_dir, "accnd_train")
    val_dir = os.path.join(save_dir, "accnd_val")

    # make dirs
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # set up paths
    root_dir = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ocr_datasets/newspaper"
    chars_dir = os.path.join(root_dir, "noisy_chars_commas_t2b")
    seg_dir = os.path.join(root_dir, "noisy_lines2")
    coco_train_path = os.path.join(root_dir, "noisy_train70sofar_highres_expanded_comma_corrected.json")
    coco_val_path = os.path.join(root_dir, "noisy_test30sofar_highres_expanded_comma_corrected.json")

    # open train and test coco jsons
    with open(coco_train_path) as f: coco_train = json.load(f)
    with open(coco_val_path) as f: coco_val = json.load(f)
    
    # gather texts and file names
    train_seg_basenames = [x['file_name'] for x in coco_train["images"]]
    val_seg_basenames = [x['file_name'] for x in coco_val["images"]]
    train_seg_texts = [x['text'] for x in coco_train["images"]]
    val_seg_texts = [x['text'] for x in coco_val["images"]]
    print(f"Len val segs {len(val_seg_basenames)}; len train segs {len(train_seg_basenames)}")

    # create datasets for train and test
    create_datasets(train_seg_basenames, train_seg_texts, seg_dir, train_dir)
    create_datasets(val_seg_basenames, val_seg_texts, seg_dir, val_dir)
