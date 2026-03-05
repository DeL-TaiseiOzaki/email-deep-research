#!/bin/bash
#SBATCH --job-name=bench-art-e
#SBATCH --partition=a3megatpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=26
#SBATCH --output=logs/bench_%j.log
#SBATCH --error=logs/bench_%j.err
#SBATCH --exclude=megatpa-a3meganodeset-0

set -euo pipefail

echo "=========================================="
echo "ART-E Benchmark Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Date: $(date)"
echo "=========================================="

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

source .venv/bin/activate

mkdir -p logs

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Pre-flight
echo "Pre-flight checks..."
nvidia-smi
uv run python -c "import art; print('openpipe-art OK')"

# Ensure DB
echo "Ensuring email database exists..."
uv run python -c "from art_e.data.local_email_db import generate_database; generate_database()"

# ==========================================
# Phase 1: Start vLLM server for ART-E
# ==========================================
echo "Starting vLLM server for ART-E..."
CHECKPOINT_DIR=".art/email_agent/models/email-agent-qwen3-14b"
LATEST_STEP=$(ls -d ${CHECKPOINT_DIR}/[0-9][0-9][0-9][0-9] 2>/dev/null | sort -n | tail -1)

if [ -z "$LATEST_STEP" ]; then
    echo "ERROR: No checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi
echo "Using checkpoint: ${LATEST_STEP}"

uv run python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --served-model-name "email-agent-qwen3-14b" \
    --port 8000 \
    --api-key default \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --enable-lora \
    --lora-modules "email-agent-qwen3-14b=${LATEST_STEP}" \
    --max-lora-rank 16 \
    --return-tokens-as-token-ids \
    --disable-log-requests &

VLLM_PID=$!
echo "vLLM server PID: ${VLLM_PID}"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 600); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready! (took ${i}s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ==========================================
# Phase 2: Run benchmarks
# ==========================================
LIMIT=${BENCHMARK_LIMIT:-10}

echo ""
echo "=========================================="
echo "Running benchmark: all models, ${LIMIT} tasks"
echo "=========================================="

uv run python calc-cost/benchmark_latency.py \
    --models all \
    --limit "${LIMIT}" \
    --vllm-url http://localhost:8000/v1

# ==========================================
# Phase 3: Generate charts
# ==========================================
echo ""
echo "Generating comparison chart..."
uv run python calc-cost/plot_simple_comparison.py

# ==========================================
# Cleanup
# ==========================================
echo ""
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "Benchmark completed at $(date)"
echo "Results: calc-cost/benchmark_results.json"
echo "Chart:   calc-cost/simple_comparison.png"
echo "=========================================="
