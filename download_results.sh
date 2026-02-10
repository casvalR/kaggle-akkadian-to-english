#!/bin/bash
# download_results.sh — asist-server3から結果・fine-tuned重みをダウンロード
#
# Usage:
#   bash download_results.sh                   # 結果 + experiment_log のみ
#   bash download_results.sh final_model       # fine-tuned重みもダウンロード
#   bash download_results.sh --all             # 全results/ディレクトリをダウンロード

set -e

REMOTE="asist-server3"
REMOTE_DIR="~/kaggle-akkadian"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_NAME="${1:-}"

echo "=== Downloading results from ${REMOTE} ==="

# --- ハイパラ探索結果 ---
echo "--- Hyperparameter search results ---"
rsync -avz "${REMOTE}:${REMOTE_DIR}/results/hyperparam/" "${SCRIPT_DIR}/results/hyperparam/" 2>/dev/null || echo "  (no hyperparam results found)"

# --- experiment_log.csv ---
echo "--- Experiment log ---"
rsync -avz "${REMOTE}:${REMOTE_DIR}/experiment_log.csv" "${SCRIPT_DIR}/" 2>/dev/null || echo "  (no experiment_log.csv found)"

# --- fine-tuned model weights ---
if [ "$MODEL_NAME" = "--all" ]; then
    echo "--- All results ---"
    rsync -avz "${REMOTE}:${REMOTE_DIR}/results/" "${SCRIPT_DIR}/results/"
elif [ -n "$MODEL_NAME" ]; then
    echo "--- Fine-tuned model: ${MODEL_NAME} ---"
    mkdir -p "${SCRIPT_DIR}/kaggle_models/${MODEL_NAME}"
    rsync -avz "${REMOTE}:${REMOTE_DIR}/results/${MODEL_NAME}/" \
        "${SCRIPT_DIR}/kaggle_models/${MODEL_NAME}/"
    echo "  Model saved to kaggle_models/${MODEL_NAME}/"
fi

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="

if [ -n "$MODEL_NAME" ] && [ "$MODEL_NAME" != "--all" ]; then
    echo ""
    echo "Next steps:"
    echo "  1. Generate inference-only notebook:"
    echo "     python generate_submission.py --inference-only --model-dir kaggle_models/${MODEL_NAME}"
    echo ""
    echo "  2. Upload model to Kaggle Dataset:"
    echo "     cd kaggle_models && kaggle datasets version -m 'Add ${MODEL_NAME}'"
    echo ""
    echo "  3. Push notebook:"
    echo "     kaggle kernels push"
fi
