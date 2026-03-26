"""
Pre-extract OCR text for all images using PaddleOCR-VL-1.5 and save results to a
JSON cache file. Run this once before training so the OCR model does not need to
be loaded per-batch during the training loop.

Usage:
    python scripts/preextract_ocr.py \
        --data_root data \
        --cache_path ocr_cache.json \
        --batch_size 8 \
        --max_new_tokens 256

The OCR cache (JSON) maps  file_path → ocr_text  and is read automatically by
OCREnhancedIBQ during training (see src/IBQ/models/ocr_enhanced_ibq.py).
"""

import os
import json
import glob
import argparse
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default="data",          help="Root directory of images")
    p.add_argument("--cache_path",     default="ocr_cache.json",help="Output JSON cache file")
    p.add_argument("--model_name",     default="PaddlePaddle/PaddleOCR-VL-1.5")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size",     type=int, default=4,     help="Images per VLM call")
    p.add_argument("--extensions",     nargs="+",
                   default=[".png", ".jpg", ".jpeg", ".webp", ".bmp"])
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(model_name, device):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"Loading {model_name} …")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return processor, model


@torch.no_grad()
def extract_one(image_pil, processor, model, device, max_new_tokens):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text",  "text": "OCR:"},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:-1])


def main():
    args = parse_args()

    # Gather image paths
    exts = set(args.extensions)
    paths = sorted([
        p for p in glob.glob(os.path.join(args.data_root, "**", "*"), recursive=True)
        if os.path.splitext(p)[1].lower() in exts
    ])
    print(f"Found {len(paths)} images in {args.data_root}")

    # Load existing cache
    cache = {}
    if os.path.exists(args.cache_path):
        with open(args.cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results from {args.cache_path}")

    # Only process images not yet cached
    todo = [p for p in paths if p not in cache]
    print(f"{len(todo)} images need OCR extraction")

    if not todo:
        print("Nothing to do.")
        return

    processor, model = load_model(args.model_name, args.device)

    save_every = 100   # flush to disk every N images
    for i, path in enumerate(tqdm(todo, desc="OCR")):
        try:
            img = Image.open(path).convert("RGB")
            text = extract_one(img, processor, model, args.device, args.max_new_tokens)
            cache[path] = text
        except Exception as e:
            print(f"  WARNING: failed on {path}: {e}")
            cache[path] = ""

        if (i + 1) % save_every == 0:
            with open(args.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

    # Final save
    with open(args.cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cache)} results to {args.cache_path}")


if __name__ == "__main__":
    main()
