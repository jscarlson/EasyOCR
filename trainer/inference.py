import pandas as pd
import PIL
from PIL import Image
from PIL import ImageDraw
import torch
import easyocr
import argparse
from tqdm import tqdm
from nltk.metrics.distance import edit_distance
import os
import json


def gt_collect(results, gts):
    gt_pred_pairs = []
    for fn, gt in gts:
        pred = results.get(fn, None)
        if pred is None:
            gt_pred_pairs.append((gt, ""))
        else:
            gt_pred_pairs.append((gt, pred))
    return gt_pred_pairs


def string_cleaner(s):
    return (s
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("''", "\"")
        .replace("‘‘", "\"")
        .replace("’’", "\"")
        .replace("\n", "")
    )


def inference(img_name, reader):
    bounds = reader.recognize(img_name)
    try:
        return bounds[0][-2]
    except IndexError:
        return ""


def textline_evaluation(
        pairs,
        print_incorrect=False, 
        no_spaces_in_eval=False, 
        norm_edit_distance=False, 
        uncased=False
    ):

    n_correct = 0
    edit_count = 0
    length_of_data = len(pairs)
    n_chars = sum(len(gt) for gt, _ in pairs)

    for gt, pred in pairs:

        # eval w/o spaces
        pred, gt = string_cleaner(pred), string_cleaner(gt)
        gt = gt.strip() if not no_spaces_in_eval else gt.strip().replace(" ", "")
        pred = pred.strip() if not no_spaces_in_eval else pred.strip().replace(" ", "")
        if uncased:
            pred, gt = pred.lower(), gt.lower()
        
        # textline accuracy
        if pred == gt:
            n_correct += 1
        else:
            if print_incorrect:
                print(f"GT: {gt}\nPR: {pred}\n")

        # ICDAR2019 Normalized Edit Distance
        if norm_edit_distance:
            if len(gt) > len(pred):
                edit_count += edit_distance(pred, gt) / len(gt)
            else:
                edit_count += edit_distance(pred, gt) / len(pred)
        else:
            edit_count += edit_distance(pred, gt)

    accuracy = n_correct / float(length_of_data) * 100
    
    if norm_edit_distance:
        cer = edit_count / float(length_of_data)
    else:
        cer = edit_count / n_chars

    return accuracy, cer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", type=str, required=True,
        help="Path to COCO JSON file with training data")
    parser.add_argument("--image_dir", type=str, required=True,
        help="Path to relevant image directory")
    parser.add_argument("--lang", type=str, required=True,
        help="Path to relevant image directory")
    parser.add_argument("--dataset_name", type=str, required=True,
        help="Path to relevant image directory")
    args = parser.parse_args()

    with open(args.coco_json) as f:
            coco = json.load(f)
    coco_images = [os.path.join(args.image_dir, x["file_name"]) for x in coco["images"]]

    reader = easyocr.Reader([args.lang], gpu=True,
        recog_network=args.dataset_name,
        model_storage_directory="/srv/ocr/github_repos/EasyOCR/trainer/custom_models",
        user_network_directory="/srv/ocr/github_repos/EasyOCR/trainer/custom_networks",
        detector=False,
    )

    inference_results = {}
    with torch.no_grad():
        for path in tqdm(coco_images):
            output = inference(path, reader=reader)
            inference_results[os.path.basename(path)] = output

    gts = []
    for x in coco["images"]:
        filename = x["file_name"]
        gt_chars = x["text"]
        gts.append((filename, gt_chars))
    gt_pred_pairs = gt_collect(inference_results, gts)

    acc, norm_ED = textline_evaluation(gt_pred_pairs, print_incorrect=False, 
        no_spaces_in_eval=False, norm_edit_distance=False, uncased=True)

    print(f"EasyOCR | Textline accuracy = {acc} | CER = {norm_ED}")