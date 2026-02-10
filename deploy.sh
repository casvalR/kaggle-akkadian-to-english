#!/bin/bash
# deploy.sh — ローカル→リモート(asist-server3)のコード同期 + 環境セットアップ
#
# Usage:
#   bash deploy.sh                    # コード同期のみ
#   bash deploy.sh --install          # コード同期 + pip install
#   bash deploy.sh --sync-from-s4     # asist-server4からコードを取得してからデプロイ
#
# 前提:
#   - asist-server4: 現在のコードがある (同期元)
#   - asist-server3: 学習を実行する (デプロイ先)

set -e

REMOTE_TRAIN="asist-server3"
REMOTE_TRAIN_DIR="~/kaggle-akkadian"

REMOTE_SOURCE="asist-server4"
REMOTE_SOURCE_DIR="~/kaggle-akkadian"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- オプション解析 ---
DO_INSTALL=false
SYNC_FROM_S4=false

for arg in "$@"; do
    case "$arg" in
        --install)    DO_INSTALL=true ;;
        --sync-from-s4) SYNC_FROM_S4=true ;;
        *)            echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- Step 0: asist-server4からローカルに同期 (オプション) ---
if $SYNC_FROM_S4; then
    echo "=== Syncing from ${REMOTE_SOURCE} to local ==="
    rsync -avz --exclude='kaggle_models/' --exclude='data/' --exclude='.venv*' \
        --exclude='results/' --exclude='__pycache__/' --exclude='.git/' \
        "${REMOTE_SOURCE}:${REMOTE_SOURCE_DIR}/" "${SCRIPT_DIR}/"
    echo "  Done: local code updated from ${REMOTE_SOURCE}"
fi

# --- Step 1: ローカル→asist-server3にコード同期 ---
echo "=== Deploying to ${REMOTE_TRAIN} ==="

# リモートにディレクトリ作成
ssh "${REMOTE_TRAIN}" "mkdir -p ${REMOTE_TRAIN_DIR}"

# コード同期 (データ・モデル・仮想環境は除外)
rsync -avz \
    --exclude='kaggle_models/' \
    --exclude='data/' \
    --exclude='.venv*' \
    --exclude='results/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='.DS_Store' \
    --exclude='*.ipynb' \
    "${SCRIPT_DIR}/" "${REMOTE_TRAIN}:${REMOTE_TRAIN_DIR}/"

echo "  Done: code synced to ${REMOTE_TRAIN}:${REMOTE_TRAIN_DIR}"

# --- Step 2: pip install (オプション) ---
if $DO_INSTALL; then
    echo "=== Installing dependencies on ${REMOTE_TRAIN} ==="
    ssh "${REMOTE_TRAIN}" "cd ${REMOTE_TRAIN_DIR} && pip install -r requirements.txt"
    echo "  Done: dependencies installed"
fi

# --- 完了メッセージ ---
echo ""
echo "=========================================="
echo "Deploy complete!"
echo "=========================================="
echo ""
echo "To start training on ${REMOTE_TRAIN}:"
echo "  ssh ${REMOTE_TRAIN}"
echo "  cd ${REMOTE_TRAIN_DIR}"
echo "  screen -S training"
echo ""
echo "  # Hyperparameter search:"
echo "  python run_hyperparam_search.py --data-dir ./data/deep-past-initiative-machine-translation --model nllb_600m_dict --n-trials 20 --epochs 5"
echo ""
echo "  # Final training:"
echo "  python run_final_training.py --data-dir ./data/deep-past-initiative-machine-translation --model nllb_600m_dict --epochs 50"
