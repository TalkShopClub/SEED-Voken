#!/usr/bin/env python3
"""Load image from path and print dimensions (including multi-frame)."""
import sys
sys.path.insert(0, "/workspace/scripts/SEED-Voken")

path = "/workspace/models/EWM-DataCollection/images/TongUI/180914/0056d77b-b7c3-4570-acb3-e0720f73acde.jpg"

print("Path:", path)
print()

# 1) torchvision read_image (same as base.load_image)
try:
    from torchvision.io import read_image
    t = read_image(path)
    print("torchvision.io.read_image:")
    print("  shape:", t.shape)
    print("  dtype:", t.dtype)
    if t.dim() == 4:
        print("  dim 0 (frames/batch):", t.shape[0])
        print("  dim 1 (channels):", t.shape[1])
        print("  dim 2 (height):", t.shape[2])
        print("  dim 3 (width):", t.shape[3])
    elif t.dim() == 3:
        print("  dim 0 (channels):", t.shape[0])
        print("  dim 1 (height):", t.shape[1])
        print("  dim 2 (width):", t.shape[2])
except Exception as e:
    print("torchvision read_image failed:", e)

print()

# 2) PIL (multi-page / n_frames)
try:
    from PIL import Image
    img = Image.open(path)
    print("PIL Image.open:")
    print("  size (W, H):", img.size)
    print("  mode:", img.mode)
    n_frames = getattr(img, "n_frames", 1)
    print("  n_frames:", n_frames)
    if n_frames > 1:
        # Sample first few frame shapes (palette mode can vary)
        for i in [0, 1, n_frames - 1]:
            img.seek(i)
            f = img.copy()
            a = __import__("numpy").array(f)
            print(f"  frame {i} shape: {a.shape}")
except Exception as e:
    print("PIL failed:", e)

print()

# 3) base.load_image (project's loader)
try:
    from src.IBQ.data.base import load_image
    t = load_image(path)
    print("base.load_image:")
    print("  shape:", t.shape)
    if t.dim() == 4:
        print("  -> layout (T, C, H, W); dim 0 = frames:", t.shape[0])
except Exception as e:
    print("base.load_image failed:", e)

# 4) After fix: preprocess_image output (first frame, resized if size set)
print()
try:
    from src.IBQ.data.base import IterableImagePaths
    ds = IterableImagePaths([path], size=384)
    sample = ds._get_sample(0)
    img = sample["image"]
    print("IterableImagePaths.preprocess_image (size=384) output:")
    print("  shape:", img.shape)
    print("  -> (H, W, C) = (384, 384, 3) expected")
except Exception as e:
    import traceback
    print("preprocess_image failed:", e)
    traceback.print_exc()
