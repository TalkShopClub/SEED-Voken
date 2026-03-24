#!/usr/bin/env python3
"""
Pre-compute OmniParser annotations (text + icon bounding boxes) for a dataset of images.

Runs YOLO (icon detection) + EasyOCR (text detection) only — skips Florence2 captioning
since we only need bounding boxes for mask generation.

Usage:
    python scripts/precompute_omniparser_masks.py \
        --data_dir /workspace/users/mike/data/ibq_finetune_stages/stage_d \
        --omniparser_dir /workspace/users/mike/repos/OmniParser \
        --output_path /workspace/users/mike/data/omniparser_annotations.json \
        --max_images 100
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute OmniParser bounding box annotations")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--omniparser_dir", type=str, required=True, help="Path to OmniParser repo")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path")
    parser.add_argument("--max_images", type=int, default=0, help="Max images to process (0 = all)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for parallelism")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index for parallelism (-1 = end)")
    parser.add_argument("--yolo_model_path", type=str, default=None,
                        help="Path to YOLO model (default: <omniparser_dir>/weights/icon_detect/model.pt)")
    parser.add_argument("--box_threshold", type=float, default=0.05, help="YOLO detection threshold")
    parser.add_argument("--text_threshold", type=float, default=0.9, help="EasyOCR text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def collect_image_paths(data_dir, max_images=0, start_idx=0, end_idx=-1):
    """Collect image file paths from directory."""
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    paths = []
    for fname in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            paths.append(os.path.join(data_dir, fname))

    # Apply index range
    if end_idx == -1:
        end_idx = len(paths)
    paths = paths[start_idx:end_idx]

    # Apply max limit
    if max_images > 0:
        paths = paths[:max_images]

    return paths


def run_ocr(image_path, easyocr_reader, text_threshold=0.9):
    """Run EasyOCR on an image, return bounding boxes in ratio coordinates."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    image_np = np.array(image)

    results = easyocr_reader.readtext(image_np, paragraph=False, text_threshold=text_threshold)

    text_bboxes = []
    for (bbox_pts, text, conf) in results:
        # bbox_pts is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        x1, x2 = min(xs) / w, max(xs) / w
        y1, y2 = min(ys) / h, max(ys) / h
        # Clamp to [0, 1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        text_bboxes.append([x1, y1, x2, y2])

    return text_bboxes


def run_yolo(image_path, yolo_model, box_threshold=0.05):
    """Run YOLO icon detection, return bounding boxes in ratio coordinates."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    results = yolo_model(image_path, verbose=False)
    icon_bboxes = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            conf = boxes.conf[i].item()
            if conf < box_threshold:
                continue
            # xyxy format in pixel coordinates
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            # Convert to ratio
            x1 /= w
            y1 /= h
            x2 /= w
            y2 /= h
            # Clamp
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            icon_bboxes.append([x1, y1, x2, y2])

    return icon_bboxes


def main():
    args = parse_args()

    # Add OmniParser to path for model loading utilities
    sys.path.insert(0, args.omniparser_dir)

    # Collect image paths
    image_paths = collect_image_paths(args.data_dir, args.max_images, args.start_idx, args.end_idx)
    print(f"Processing {len(image_paths)} images from {args.data_dir}")

    # Load YOLO model
    yolo_model_path = args.yolo_model_path or os.path.join(
        args.omniparser_dir, "weights", "icon_detect", "model.pt"
    )
    print(f"Loading YOLO model from {yolo_model_path}...")
    from ultralytics import YOLO
    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(args.device)
    print("YOLO model loaded")

    # Load EasyOCR
    print("Loading EasyOCR...")
    import easyocr
    easyocr_reader = easyocr.Reader(["en"], gpu=(args.device == "cuda"))
    print("EasyOCR loaded")

    # Process images
    annotations = {}
    t_start = time.time()

    for idx, image_path in enumerate(image_paths):
        fname = os.path.basename(image_path)
        try:
            text_bboxes = run_ocr(image_path, easyocr_reader, args.text_threshold)
            icon_bboxes = run_yolo(image_path, yolo_model, args.box_threshold)

            annotations[fname] = {
                "text_bboxes": text_bboxes,
                "icon_bboxes": icon_bboxes,
            }
        except Exception as e:
            print(f"  ERROR processing {fname}: {e}")
            annotations[fname] = {"text_bboxes": [], "icon_bboxes": []}

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(image_paths) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(image_paths)}] {fname}: "
                  f"text={len(annotations[fname]['text_bboxes'])}, "
                  f"icons={len(annotations[fname]['icon_bboxes'])} "
                  f"({rate:.1f} img/s, ETA {eta:.0f}s)")

    # Save annotations
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(annotations, f)

    elapsed = time.time() - t_start
    print(f"\nDone! Processed {len(annotations)} images in {elapsed:.1f}s")
    print(f"Annotations saved to {args.output_path}")

    # Summary stats
    n_text = sum(len(v["text_bboxes"]) for v in annotations.values())
    n_icon = sum(len(v["icon_bboxes"]) for v in annotations.values())
    n_empty = sum(1 for v in annotations.values() if not v["text_bboxes"] and not v["icon_bboxes"])
    print(f"Total text boxes: {n_text}, icon boxes: {n_icon}")
    print(f"Images with no detections: {n_empty}/{len(annotations)}")


if __name__ == "__main__":
    main()
