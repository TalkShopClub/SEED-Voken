#!/usr/bin/env python3
"""
Profile all images from the datapath in configs/IBQ/gpu/finetune_256_ocr_smart_resize.yaml
to find values that may cause NaN in the discriminator and VQ-VAE (src/IBQ).

NaN risk sources (from src/IBQ):
- Discriminator: BatchNorm/ActNorm with zero or near-zero variance; extreme inputs.
- VQ-VAE quantize: large encoder outputs (overflow in d = z² + e² - 2ez); softmax(logits) with extreme logits; log(0) in entropy.
- LPIPS/perceptual: extreme or constant inputs.

Checks: NaN/Inf in tensor, value range, per-channel and global std (constant images),
out-of-range normalized values, L2 norm (overflow risk), load/preprocess errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# Repo root for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import albumentations
from PIL import Image

# Default config path (datapath loaded from this)
DEFAULT_CONFIG = REPO_ROOT / "configs/IBQ/gpu/finetune_256_ocr_smart_resize.yaml"

# Thresholds chosen to match numerical risks in discriminator and VQ-VAE
MIN_STD = 1e-3  # below this: constant/near-constant image -> BatchNorm/div issues
OUT_OF_RANGE_LOW = -1.1
OUT_OF_RANGE_HIGH = 1.1
MAX_L2_PER_PIXEL = 1e3  # very large L2 could contribute to overflow in matmul (VQ distance)


@dataclass
class ImageStats:
    path: str
    split: str
    error: str | None = None
    has_nan: bool = False
    has_inf: bool = False
    min_val: float | None = None
    max_val: float | None = None
    mean: float | None = None
    std: float | None = None
    std_per_channel: list[float] | None = None
    mean_per_channel: list[float] | None = None
    shape_after: tuple[int, ...] | None = None
    shape_before: tuple[int, ...] | None = None
    out_of_range_count: int = 0
    l2_norm: float | None = None
    num_pixels: int | None = None
    suspicious: bool = False
    reasons: list[str] = field(default_factory=list)


def load_image_pil(path: str) -> np.ndarray:
    """Load image using PIL only (no torchvision). Returns (C, H, W) uint8.
    Avoids segfaults from torchvision/libpng on problematic PNGs (e.g. iCCP)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)  # (H, W, C)
    return np.transpose(arr, (2, 0, 1))  # (C, H, W)


def get_preprocessor(size: int, random_crop: bool = False):
    """Same logic as IterableImagePaths in src/IBQ/data/base.py (SmallestMaxSize + crop)."""
    if size is None or size <= 0:
        return lambda **kw: {"image": kw["image"]}
    if not random_crop:
        cropper = albumentations.CenterCrop(height=size, width=size)
    else:
        cropper = albumentations.RandomCrop(height=size, width=size)
    preprocessor = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=size),
        cropper,
    ])
    return preprocessor


def load_and_preprocess(path: str, size: int, random_crop: bool = False):
    """Load image and preprocess like IterableImagePaths.preprocess_image (deterministic with random_crop=False).
    Uses PIL-only loading to avoid segfaults from torchvision/libpng on bad PNGs."""
    image = load_image_pil(path)  # (C, H, W) uint8 numpy
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    elif image.shape[0] == 4:
        image = image[:3]
    if len(image.shape) == 4:
        image = image[0]
    shape_before = tuple(image.shape)
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8)  # (H, W, C) for albumentations
    preprocessor = get_preprocessor(size, random_crop=random_crop)
    image = preprocessor(image=image)["image"]
    image = (image / 127.5 - 1.0).astype(np.float32)
    return image, shape_before


def profile_image(path: str, split: str, size: int, random_crop: bool = False) -> ImageStats:
    """Compute stats for one image; same preprocessing as dataloader (center crop for deterministic profile)."""
    stats = ImageStats(path=path, split=split)
    try:
        arr, shape_before = load_and_preprocess(path, size, random_crop=False)
    except Exception as e:
        stats.error = str(e)
        stats.suspicious = True
        stats.reasons.append("load_or_preprocess_error")
        return stats

    stats.shape_before = shape_before
    stats.shape_after = arr.shape
    stats.num_pixels = int(np.prod(arr.shape))

    has_nan = np.any(np.isnan(arr))
    has_inf = np.any(np.isinf(arr))
    stats.has_nan = bool(has_nan)
    stats.has_inf = bool(has_inf)
    if has_nan or has_inf:
        stats.suspicious = True
        if has_nan:
            stats.reasons.append("has_nan")
        if has_inf:
            stats.reasons.append("has_inf")

    stats.min_val = float(np.min(arr))
    stats.max_val = float(np.max(arr))
    stats.mean = float(np.mean(arr))
    stats.std = float(np.std(arr))

    if arr.ndim == 3:
        stats.mean_per_channel = [float(np.mean(arr[:, :, c])) for c in range(arr.shape[2])]
        stats.std_per_channel = [float(np.std(arr[:, :, c])) for c in range(arr.shape[2])]
        for i, sc in enumerate(stats.std_per_channel):
            if sc < MIN_STD:
                stats.suspicious = True
                stats.reasons.append(f"low_std_channel_{i}")
                break
    if stats.std < MIN_STD:
        stats.suspicious = True
        stats.reasons.append("low_std_global")

    out_of_range = np.sum((arr < OUT_OF_RANGE_LOW) | (arr > OUT_OF_RANGE_HIGH))
    stats.out_of_range_count = int(out_of_range)
    if stats.out_of_range_count > 0:
        stats.suspicious = True
        stats.reasons.append("out_of_range_normalized")

    flat = arr.ravel()
    l2_sq = np.sum(flat.astype(np.float64) ** 2)
    l2 = np.sqrt(l2_sq)
    stats.l2_norm = float(l2)
    if stats.num_pixels and l2 / stats.num_pixels > MAX_L2_PER_PIXEL:
        stats.suspicious = True
        stats.reasons.append("high_l2_per_pixel")

    return stats


def load_data_config(config_path: Path) -> dict:
    """Load data section from YAML; return dict of split -> config (manifest_path, root, size, etc.)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data = cfg.get("data", {})
    init = data.get("init_args", data)
    out = {}
    for split in ("train", "validation", "test"):
        split_cfg = init.get(split)
        if not split_cfg:
            continue
        params = split_cfg.get("params", split_cfg)
        if isinstance(params, dict) and "config" in params:
            out[split] = params["config"]
        else:
            out[split] = params or {}
    return out


def get_paths_from_manifest(manifest_path: str, root_from_config: str | None) -> list[str]:
    """Return list of absolute image paths from manifest JSON."""
    manifest_path = os.path.expanduser(manifest_path)
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(REPO_ROOT, manifest_path)
    with open(manifest_path) as f:
        data = json.load(f)
    root = os.path.abspath(os.path.expanduser(data["root"]))
    paths = data.get("paths", [])
    if paths and not os.path.isabs(paths[0]):
        paths = [os.path.join(root, p) for p in paths]
    return paths


def run_profiling(
    config_path: Path,
    output_json: Path | None = None,
    output_report: Path | None = None,
    output_suspicious_json: Path | None = None,
    limit: int | None = None,
    splits: list[str] | None = None,
) -> tuple[list[ImageStats], dict]:
    splits = splits or ["train", "validation", "test"]
    data_configs = load_data_config(config_path)
    all_stats: list[ImageStats] = []
    global_aggregates: dict = {
        "by_split": {},
        "suspicious_paths": [],
        "error_paths": [],
        "count_by_reason": {},
    }

    for split in splits:
        if split not in data_configs:
            continue
        cfg = data_configs[split]
        manifest_path = cfg.get("manifest_path")
        if not manifest_path:
            continue
        root = cfg.get("root")
        size = cfg.get("size", 256)
        random_crop = cfg.get("random_crop", False)

        try:
            paths = get_paths_from_manifest(manifest_path, root)
        except FileNotFoundError as e:
            global_aggregates["error_paths"].append({"split": split, "error": str(e)})
            continue

        if limit:
            paths = paths[:limit]
        n_suspicious = 0
        n_error = 0
        reasons_this_split: dict[str, int] = {}
        pbar = tqdm(enumerate(paths), total=len(paths), desc=f"Processing {split}")
        for i, path in pbar:
            if not 'tongui' in path.lower():
                continue
            st = profile_image(path, split, size, random_crop)
            all_stats.append(st)
            pbar.set_description(f"Errors: {n_error}, Sus: {n_suspicious}")
            if st.error:
                n_error += 1
                global_aggregates["error_paths"].append({"path": path, "split": split, "error": st.error})
            if st.suspicious:
                n_suspicious += 1
                global_aggregates["suspicious_paths"].append(path)
                for r in st.reasons:
                    reasons_this_split[r] = reasons_this_split.get(r, 0) + 1
                    global_aggregates["count_by_reason"][r] = global_aggregates["count_by_reason"].get(r, 0) + 1

        global_aggregates["by_split"][split] = {
            "total": len(paths),
            "suspicious": n_suspicious,
            "errors": n_error,
            "reasons": reasons_this_split,
        }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(
                [asdict(s) for s in all_stats],
                f,
                indent=2,
            )
        print(f"Wrote full stats to {output_json}")

    if output_suspicious_json and global_aggregates["suspicious_paths"]:
        output_suspicious_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_suspicious_json, "w") as f:
            json.dump({"skip_paths": global_aggregates["suspicious_paths"]}, f, indent=2)
        print(f"Wrote {len(global_aggregates['suspicious_paths'])} suspicious paths to {output_suspicious_json} (load_failed_paths format)")

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w") as f:
            f.write("# Image NaN-risk profile report\n\n")
            f.write(f"Config: {config_path}\n\n")
            f.write("## By split\n\n")
            for split, agg in global_aggregates["by_split"].items():
                f.write(f"- **{split}**: total={agg['total']}, suspicious={agg['suspicious']}, errors={agg['errors']}\n")
                if agg.get("reasons"):
                    f.write(f"  Reasons: {agg['reasons']}\n")
            f.write("\n## Count by reason\n\n")
            for r, c in sorted(global_aggregates["count_by_reason"].items(), key=lambda x: -x[1]):
                f.write(f"- {r}: {c}\n")
            f.write("\n## Suspicious paths (first 500)\n\n")
            for p in global_aggregates["suspicious_paths"][:500]:
                f.write(f"- {p}\n")
            if len(global_aggregates["suspicious_paths"]) > 500:
                f.write(f"\n... and {len(global_aggregates['suspicious_paths']) - 500} more.\n")
            f.write("\n## Error paths (first 200)\n\n")
            for item in global_aggregates["error_paths"][:200]:
                if isinstance(item, dict) and "path" in item:
                    f.write(f"- {item['path']}: {item.get('error', '')}\n")
                else:
                    f.write(f"- {item}\n")
        print(f"Wrote report to {output_report}")

    return all_stats, global_aggregates


def main():
    parser = argparse.ArgumentParser(description="Profile images for NaN risk (discriminator and VQ-VAE).")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to finetune config YAML (default: configs/IBQ/gpu/finetune_256_ocr_smart_resize.yaml)",
    )
    parser.add_argument(
        "-o", "--output-json",
        type=Path,
        default=None,
        help="Write full per-image stats JSON here",
    )
    parser.add_argument(
        "-r", "--output-report",
        type=Path,
        default=None,
        help="Write markdown report here",
    )
    parser.add_argument(
        "--output-suspicious-json",
        type=Path,
        default=None,
        help="Write suspicious paths as skip_paths JSON (for failed_samples_path / load_failed_paths)",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit number of images per split (for quick runs)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to profile",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    print(f"Using config: {args.config}")
    all_stats, agg = run_profiling(
        args.config,
        output_json=args.output_json,
        output_report=args.output_report,
        output_suspicious_json=args.output_suspicious_json,
        limit=args.limit,
        splits=args.splits,
    )
    print("\nSummary:")
    for split, s in agg["by_split"].items():
        print(f"  {split}: {s['suspicious']} suspicious, {s['errors']} errors out of {s['total']}")
    print(f"  Total suspicious paths: {len(agg['suspicious_paths'])}")
    print(f"  Total error paths: {len(agg['error_paths'])}")


if __name__ == "__main__":
    main()
