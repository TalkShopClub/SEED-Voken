#!/usr/bin/env python3
"""
Evaluate IBQ reconstruction quality using OmniParser.
Compares GT vs Reconstructed images by:
1. Running OCR (EasyOCR) on both -> text bounding boxes
2. Running YOLO icon detection on both -> icon bounding boxes
3. Hungarian matching of boxes, computing IoU metrics.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'OmniParser'))
from util.utils import check_ocr_box, get_yolo_model, predict_yolo


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def hungarian_match_iou(gt_boxes, pred_boxes):
    """
    Match GT boxes to predicted boxes via Hungarian algorithm (maximize IoU).
    Returns: matched IoUs, num_unmatched_gt, num_unmatched_pred
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return [], 0, 0
    if len(gt_boxes) == 0:
        return [], 0, len(pred_boxes)
    if len(pred_boxes) == 0:
        return [], len(gt_boxes), 0

    # Build cost matrix (negative IoU for minimization)
    cost = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            cost[i, j] = -compute_iou(gb, pb)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_ious = [-cost[r, c] for r, c in zip(row_ind, col_ind)]

    n_unmatched_gt = len(gt_boxes) - len(row_ind)
    n_unmatched_pred = len(pred_boxes) - len(col_ind)
    return matched_ious, n_unmatched_gt, n_unmatched_pred


def detect_elements(image_pil, yolo_model):
    """
    Run OCR + YOLO detection on a PIL image.
    Returns: text_boxes (list of xyxy), icon_boxes (list of xyxy)
    """
    w, h = image_pil.size

    # OCR detection
    (texts, ocr_bboxes), _ = check_ocr_box(
        image_pil, display_img=False, output_bb_format='xyxy',
        easyocr_args={'text_threshold': 0.8}, use_paddleocr=False
    )
    text_boxes = [list(bb) for bb in ocr_bboxes]  # already xyxy in pixel coords

    # YOLO icon detection
    xyxy, conf, phrases = predict_yolo(
        model=yolo_model, image=image_pil,
        box_threshold=0.05, imgsz=None, scale_img=False, iou_threshold=0.7
    )
    icon_boxes = xyxy.cpu().tolist()  # xyxy in pixel coords

    return text_boxes, icon_boxes


def split_gt_recon(combined_path, size):
    """Split a side-by-side image (GT left, Recon right) into two PIL images."""
    img = Image.open(combined_path).convert("RGB")
    w, h = img.size
    gt = img.crop((0, 0, w // 2, h))
    recon = img.crop((w // 2, 0, w, h))
    return gt, recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon_dir", default="results/reconstruct_pretrained_262144_1024",
                        help="Directory with side-by-side GT|Recon images")
    parser.add_argument("--yolo_model", default="OmniParser/weights/icon_detect/model.pt")
    parser.add_argument("--n_images", type=int, default=None, help="Limit number of images")
    parser.add_argument("--output", default="results/omniparser_eval_iou.json")
    args = parser.parse_args()

    print("Loading YOLO icon detection model...")
    yolo_model = get_yolo_model(args.yolo_model)

    # Collect images
    exts = {".png", ".jpg", ".jpeg"}
    all_files = sorted([
        f for f in os.listdir(args.recon_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if args.n_images:
        all_files = all_files[:args.n_images]

    text_ious_all = []
    icon_ious_all = []
    per_image_results = []

    # Metrics accumulators
    gt_text_counts = []
    recon_text_counts = []
    gt_icon_counts = []
    recon_icon_counts = []

    for fname in tqdm(all_files, desc="Evaluating"):
        path = os.path.join(args.recon_dir, fname)
        try:
            gt_img, recon_img = split_gt_recon(path, None)
        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue

        # Detect elements
        try:
            gt_text, gt_icons = detect_elements(gt_img, yolo_model)
            recon_text, recon_icons = detect_elements(recon_img, yolo_model)
        except Exception as e:
            print(f"Detection failed for {fname}: {e}")
            continue

        # Text box IoU
        text_matched, text_unmatched_gt, text_unmatched_pred = hungarian_match_iou(gt_text, recon_text)
        text_mean_iou = np.mean(text_matched) if text_matched else 0.0

        # Icon box IoU
        icon_matched, icon_unmatched_gt, icon_unmatched_pred = hungarian_match_iou(gt_icons, recon_icons)
        icon_mean_iou = np.mean(icon_matched) if icon_matched else 0.0

        text_ious_all.extend(text_matched)
        icon_ious_all.extend(icon_matched)
        gt_text_counts.append(len(gt_text))
        recon_text_counts.append(len(recon_text))
        gt_icon_counts.append(len(gt_icons))
        recon_icon_counts.append(len(recon_icons))

        per_image_results.append({
            "file": fname,
            "text_iou": round(text_mean_iou, 4),
            "icon_iou": round(icon_mean_iou, 4),
            "gt_text_count": len(gt_text),
            "recon_text_count": len(recon_text),
            "gt_icon_count": len(gt_icons),
            "recon_icon_count": len(recon_icons),
            "text_matched": len(text_matched),
            "icon_matched": len(icon_matched),
        })

    # Summary
    summary = {
        "n_images": len(per_image_results),
        "text_box_iou_mean": round(np.mean(text_ious_all), 4) if text_ious_all else 0.0,
        "text_box_iou_median": round(np.median(text_ious_all), 4) if text_ious_all else 0.0,
        "icon_box_iou_mean": round(np.mean(icon_ious_all), 4) if icon_ious_all else 0.0,
        "icon_box_iou_median": round(np.median(icon_ious_all), 4) if icon_ious_all else 0.0,
        "avg_gt_text_count": round(np.mean(gt_text_counts), 2),
        "avg_recon_text_count": round(np.mean(recon_text_counts), 2),
        "avg_gt_icon_count": round(np.mean(gt_icon_counts), 2),
        "avg_recon_icon_count": round(np.mean(recon_icon_counts), 2),
        "text_detection_rate": round(np.mean(recon_text_counts) / (np.mean(gt_text_counts) + 1e-6), 4),
        "icon_detection_rate": round(np.mean(recon_icon_counts) / (np.mean(gt_icon_counts) + 1e-6), 4),
    }

    print("\n" + "=" * 60)
    print("OmniParser Evaluation Results")
    print("=" * 60)
    print(f"Images evaluated:        {summary['n_images']}")
    print(f"Text Box IoU (mean):     {summary['text_box_iou_mean']}")
    print(f"Text Box IoU (median):   {summary['text_box_iou_median']}")
    print(f"Icon Box IoU (mean):     {summary['icon_box_iou_mean']}")
    print(f"Icon Box IoU (median):   {summary['icon_box_iou_median']}")
    print(f"Avg GT text boxes:       {summary['avg_gt_text_count']}")
    print(f"Avg Recon text boxes:    {summary['avg_recon_text_count']}")
    print(f"Text detection rate:     {summary['text_detection_rate']}")
    print(f"Avg GT icon boxes:       {summary['avg_gt_icon_count']}")
    print(f"Avg Recon icon boxes:    {summary['avg_recon_icon_count']}")
    print(f"Icon detection rate:     {summary['icon_detection_rate']}")
    print("=" * 60)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_image": per_image_results}, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
