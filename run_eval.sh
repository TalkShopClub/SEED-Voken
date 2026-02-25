#!/usr/bin/env bash
# Evaluate IBQ finetune checkpoints at native resolution (no resizing) on first 20000
# images under /workspace/models/EWM-DataCollection/test.
# Comparison output: workspace/results/evals/native/{config_id}.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Number of images to evaluate. Override with MAX_PATHS env var (default: 20000).
# Manifest path includes this value so e.g. MAX_PATHS=100 uses a 100-image manifest.
MAX_PATHS="${MAX_PATHS:-20000}"
export MAX_PATHS
EVAL_MANIFEST="configs/IBQ/gpu/eval_test_${MAX_PATHS}_image_paths.json"
if [[ ! -f "$EVAL_MANIFEST" ]]; then
  echo "Building eval manifest (first $MAX_PATHS images from test dir)..."
  python scripts/build_local_ibqgan_image_paths.py \
    --root /workspace/models/EWM-DataCollection/test \
    -o "$EVAL_MANIFEST" \
    --max_paths "$MAX_PATHS"
fi

RESULTS_BASE="${RESULTS_BASE:-workspace/results/evals/native}"
mkdir -p "$RESULTS_BASE"

# finetune_256
CONFIG_ID="finetune_256"
CONFIG="configs/IBQ/gpu/eval_finetune_256.yaml"
CKPT="${CKPT_FINETUNE_256:-checkpoints/vqgan/ibq_finetune/last.ckpt}"
OUT_DIR="$RESULTS_BASE/$CONFIG_ID"
mkdir -p "$OUT_DIR"
echo "=== Evaluating $CONFIG_ID at native resolution (config: $CONFIG, ckpt: $CKPT) ==="
python evaluation_image.py \
  --config_file "$CONFIG" \
  --ckpt_path "$CKPT" \
  --model IBQ \
  --batch_size 1 \
  --save_comparison_dir "$OUT_DIR" \
  --save_native_resolution

# finetune_256_ocr
CONFIG_ID="finetune_256_ocr"
CONFIG="configs/IBQ/gpu/eval_finetune_256_ocr.yaml"
CKPT="${CKPT_FINETUNE_256_OCR:-checkpoints/vqgan/ibq_finetune_ocr/last.ckpt}"
OUT_DIR="$RESULTS_BASE/$CONFIG_ID"
mkdir -p "$OUT_DIR"
echo "=== Evaluating $CONFIG_ID at native resolution (config: $CONFIG, ckpt: $CKPT) ==="
python evaluation_image.py \
  --config_file "$CONFIG" \
  --ckpt_path "$CKPT" \
  --model IBQ \
  --batch_size 1 \
  --save_comparison_dir "$OUT_DIR" \
  --save_native_resolution

echo "Done. Results: $RESULTS_BASE/finetune_256 and $RESULTS_BASE/finetune_256_ocr"
