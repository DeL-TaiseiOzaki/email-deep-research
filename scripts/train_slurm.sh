#!/bin/bash
#SBATCH --job-name=art-e-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

set -euo pipefail

echo "=========================================="
echo "ART-E Training Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Date: $(date)"
echo "=========================================="

# Navigate to project root
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Ensure logs directory exists
mkdir -p logs

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# W&B configuration
export WANDB_PROJECT="art-e-email-agent"

# Pre-flight checks
echo "Pre-flight checks..."
nvidia-smi
uv run python -c "import art; print(f'openpipe-art OK')"
uv run python -c "from art_e.data.query_iterators import load_synthetic_queries; q = load_synthetic_queries(split='train', limit=1); print(f'Dataset OK: {len(q)} query loaded')"

# Step 1: Generate SQLite DB (idempotent, skips if exists)
echo "Ensuring email database exists..."
uv run python -c "from art_e.data.local_email_db import generate_database; generate_database()"

# Step 2: Run training
echo "Starting training..."
uv run python art_e/train.py

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
