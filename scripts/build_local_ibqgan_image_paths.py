#!/usr/bin/env python3
"""
Build a JSON manifest of all image paths used by the local IBQ-GAN dataloader
(configs/IBQ/gpu/local_ibqgan_256.yaml). Uses the same root and image extensions
as LocalImages in src/IBQ/data/s3_images.py.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

# Extensions used by LocalImages (S3ImagesBase.IMAGE_EXTENSIONS)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
DEFAULT_ROOT = "/workspace/models/EWM-DataCollection/images"


def get_root_from_config(config_path):
    """Read data train config root from YAML without omegaconf/yaml."""
    with open(config_path) as f:
        content = f.read()
    # Match "root: /path" in the config (indented under data/init_args/train/params/config)
    m = re.search(r"^\s+root:\s*(.+)$", content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def list_local_images_find(root):
    """Use find to list image paths (faster on large trees), sorted."""
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Image root directory not found: {root}")
    # find root -type f \( -iname "*.jpg" -o -iname "*.jpeg" ... \)
    find_args = ["find", root, "-type", "f", "("]
    for i, e in enumerate(IMAGE_EXTENSIONS):
        find_args.extend(["-iname", f"*{e}"])
        if i < len(IMAGE_EXTENSIONS) - 1:
            find_args.append("-o")
    find_args.append(")")
    result = subprocess.run(
        find_args,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"find failed: {result.stderr}")
    paths = [p for p in result.stdout.strip().split("\n") if p]
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Build JSON manifest of image paths for local_ibqgan_256")
    parser.add_argument(
        "--config",
        default="configs/IBQ/gpu/local_ibqgan_256.yaml",
        help="Path to the YAML config (used to read data.train.params.config.root)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Override root directory (default: read from config)",
    )
    parser.add_argument(
        "-o", "--output",
        default="configs/IBQ/gpu/local_ibqgan_256_image_paths.json",
        help="Output JSON path",
    )
    parser.add_argument("--relative", action="store_true", help="Store paths relative to root in 'paths'")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream find output to JSON (low memory; use for very large dirs)",
    )
    parser.add_argument(
        "--max_paths",
        type=int,
        default=None,
        help="Use only the first N image paths (after sorting). Useful for eval subsets (e.g. 20000).",
    )
    args = parser.parse_args()

    root = args.root
    if root is None:
        if os.path.isfile(args.config):
            root = get_root_from_config(args.config)
        root = root or DEFAULT_ROOT
        if root is None:
            root = DEFAULT_ROOT
        print(f"Using root: {root}", file=sys.stderr)

    root = os.path.abspath(os.path.expanduser(root))
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.stream:
        find_args = ["find", root, "-type", "f", "("]
        for i, e in enumerate(IMAGE_EXTENSIONS):
            find_args.extend(["-iname", f"*{e}"])
            if i < len(IMAGE_EXTENSIONS) - 1:
                find_args.append("-o")
        find_args.append(")")
        print("Streaming find | sort to JSON (low memory)...", file=sys.stderr)
        proc = subprocess.Popen(find_args, stdout=subprocess.PIPE, text=True)
        sort_proc = subprocess.Popen(["sort"], stdin=proc.stdout, stdout=subprocess.PIPE, text=True)
        proc.stdout.close()
        count = 0
        # Use a temp file so we can skip overwriting existing good manifest when count==0
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=out_dir or ".")
        try:
            out = os.fdopen(tmp_fd, "w")
        except Exception:
            os.close(tmp_fd)
            raise
        with out:
            out.write('{\n  "root": ')
            json.dump(root, out)
            out.write(',\n  "paths": [\n')
            first = True
            for line in sort_proc.stdout:
                if args.max_paths is not None and count >= args.max_paths:
                    break
                path = line.rstrip("\n")
                if not path:
                    continue
                if not first:
                    out.write(",\n")
                val = os.path.relpath(path, root) if args.relative else path
                out.write("    ")
                json.dump(val, out)
                first = False
                count += 1
            out.write("\n  ],\n  \"count\": ")
            out.write(str(count))
            out.write("\n}\n")
        sort_proc.wait(timeout=3600)
        proc.wait(timeout=5)
        if count == 0 and os.path.isfile(args.output):
            try:
                with open(args.output) as f:
                    existing = json.load(f)
                if existing.get("paths") and len(existing["paths"]) > 0:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    print(
                        "WARNING: Found 0 images but output already has paths; leaving existing file unchanged.",
                        file=sys.stderr,
                    )
                    return
            except (json.JSONDecodeError, KeyError):
                pass
        os.replace(tmp_path, args.output)
        print(f"Wrote {count} image paths to {args.output}", file=sys.stderr)
        return

    print("Scanning for images (using find)...", file=sys.stderr)
    paths = list_local_images_find(root)
    if args.max_paths is not None:
        paths = paths[: args.max_paths]
        print(f"Limited to first {len(paths)} paths (--max_paths={args.max_paths})", file=sys.stderr)
    if args.relative:
        path_list = [os.path.relpath(p, root) for p in paths]
    else:
        path_list = paths

    # Do not overwrite an existing manifest that has paths with an empty list
    # (e.g. when build runs from a process where the image root is not visible).
    if len(path_list) == 0 and os.path.isfile(args.output):
        try:
            with open(args.output) as f:
                existing = json.load(f)
            if existing.get("paths") and len(existing["paths"]) > 0:
                print(
                    "WARNING: Found 0 images under root but output already has paths; leaving existing file unchanged.",
                    file=sys.stderr,
                )
                return
        except (json.JSONDecodeError, KeyError):
            pass

    out = {
        "root": root,
        "paths": path_list,
        "count": len(path_list),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(path_list)} image paths to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
