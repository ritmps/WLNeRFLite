#!/usr/bin/env bash
# scripts/grab_and_run_lego.sh
# End-to-end automated pipeline to grab the classic NeRF Lego tractor dataset,
# train a lightweight coordinate MLP in Wolfram Language, and render novel views.
#
# Usage:
#   ./scripts/grab_and_run_lego.sh
#   ./scripts/grab_and_run_lego.sh --rounds 3000 --res 200 --turntable
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ROOT_DIR/data/lego"
OUTPUT_DIR="$ROOT_DIR/output"

ROUNDS=3000
BATCH_SIZE=1024
RES=200
DEVICE="Automatic"
RENDER_TURNTABLE=false
SKIP_DOWNLOAD=false
COMPARE_VAL=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --rounds)
      ROUNDS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --res)
      RES="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --turntable)
      RENDER_TURNTABLE=true
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    -h|--help)
      echo "Usage: ./scripts/grab_and_run_lego.sh [options]"
      echo ""
      echo "Options:"
      echo "  --rounds N           Number of training rounds (default: 2000)"
      echo "  --batch-size N       Rays per batch (default: 1024)"
      echo "  --res N              Resolution for training views (default: 200)"
      echo "  --device CPU|GPU     Compute device (default: Automatic)"
      echo "  --turntable          Also render a 360-degree turntable GIF"
      echo "  --skip-download      Skip dataset download step"
      echo "  -h, --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

echo "=================================================================="
echo "WLNeRFLite Lego Tractor End-to-End Pipeline"
echo "=================================================================="
echo "Workspace : $ROOT_DIR"
echo "Data dir  : $DATA_DIR"
echo "Rounds    : $ROUNDS"
echo "Batch size: $BATCH_SIZE"
echo "Resolution: ${RES}x${RES}"
echo "Turntable : $RENDER_TURNTABLE"
echo "------------------------------------------------------------------"

# 1. Grab Lego dataset if needed
if [ "$SKIP_DOWNLOAD" = false ]; then
  if [ ! -f "$DATA_DIR/transforms_train.json" ] || [ ! -d "$DATA_DIR/train" ]; then
    echo "[Step 1/2] Lego dataset not found locally. Fetching via download script..."
    python3 "$SCRIPT_DIR/download_lego_data.py" --out-dir "$DATA_DIR" --splits train val
  else
    echo "[Step 1/2] Lego dataset found at $DATA_DIR. Verifying cache..."
    python3 "$SCRIPT_DIR/download_lego_data.py" --out-dir "$DATA_DIR" --splits train val
  fi
else
  echo "[Step 1/2] Skipping download (--skip-download specified)."
fi

# 2. Locate wolframscript / wolfram
WOLFRAM_BIN=""
if command -v wolframscript >/dev/null 2>&1; then
  WOLFRAM_BIN="wolframscript"
elif [ -x "/Applications/Wolfram 15.1.app/Contents/MacOS/wolfram" ]; then
  WOLFRAM_BIN="/Applications/Wolfram 15.1.app/Contents/MacOS/wolfram"
elif [ -x "/Applications/Wolfram.app/Contents/MacOS/wolfram" ]; then
  WOLFRAM_BIN="/Applications/Wolfram.app/Contents/MacOS/wolfram"
else
  echo "Error: Neither wolframscript nor wolfram binary was found on PATH or /Applications."
  exit 1
fi

echo ""
echo "[Step 2/2] Training WLNeRFLite on Lego Tractor and rendering views..."

CMD_ARGS=(
  -script "$SCRIPT_DIR/train_nerflite.wl"
  --dataset "$DATA_DIR"
  --rounds "$ROUNDS"
  --batch-size "$BATCH_SIZE"
  --res "$RES"
  --device "$DEVICE"
  --ray-bounds "2.0,6.0"
  --sample-fg-ratio "0.7"
  --output "$OUTPUT_DIR/lego_nerflite.wlnet"
  --render-test "$OUTPUT_DIR/lego_test.png"
  --compare-val "$OUTPUT_DIR/lego_val_compare.png"
)

if [ "$RENDER_TURNTABLE" = true ]; then
  CMD_ARGS+=(--render-turntable "$OUTPUT_DIR/lego_turntable.gif")
fi

"$WOLFRAM_BIN" "${CMD_ARGS[@]}"

echo ""
echo "=================================================================="
echo "Pipeline finished successfully!"
echo "Model Checkpoint : $OUTPUT_DIR/lego_nerflite.wlnet"
echo "Rendered Test    : $OUTPUT_DIR/lego_test.png"
echo "Val Comparison   : $OUTPUT_DIR/lego_val_compare.png"
if [ "$RENDER_TURNTABLE" = true ]; then
  echo "Turntable GIF    : $OUTPUT_DIR/lego_turntable.gif"
fi
echo "=================================================================="

