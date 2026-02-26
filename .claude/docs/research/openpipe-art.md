# OpenPipe ART Library Research

## Overview
- **Full name**: Agent Reinforcement Trainer
- **GitHub**: https://github.com/OpenPipe/ART
- **PyPI**: openpipe-art v0.5.11 (2026-02-19)
- **Python**: >=3.11
- **License**: Apache-2.0
- **Docs**: https://art.openpipe.ai/

## Installation
```bash
pip install openpipe-art
# With backend (GPU training):
pip install openpipe-art[backend]
```

## Architecture
- **Client (Frontend)**: User-defined rollout code, reward functions
- **Server (Backend)**: vLLM inference + LoRA GRPO training
- Split design: frontend runs anywhere, backend needs GPU

## Core API Surface
- `art.TrainableModel` — Model wrapper with config, LoRA training
- `art.Model` — Non-trainable model for inference/eval
- `art.LocalAPI()` — Local backend API (runs vLLM server locally)
- `art.Trajectory` — Records messages/choices during rollout
- `art.TrajectoryGroup` — Groups trajectories for GRPO training
- `art.gather_trajectory_groups()` — Async collection of trajectory groups
- `art.gather_trajectories()` — Async collection of trajectories
- `art.TrainConfig` — Training hyperparameters (learning_rate etc.)
- `art.utils.iterate_dataset()` — Dataset batching with epoch tracking
- `art.utils.limit_concurrency()` — Concurrency limiter decorator

## Training Loop
1. Create TrainableModel with base_model + config
2. Register with LocalAPI (starts vLLM server)
3. Load dataset, iterate in batches
4. For each batch: generate trajectory groups (rollouts with rewards)
5. Call model.train(groups, config) — GRPO update with LoRA
6. Repeat for num_epochs

## GPU Requirements
- ART·E was trained on single H100 for ~$80
- Uses vLLM for inference → GPU memory for model + LoRA
- Qwen2.5-14B: ~28GB in fp16, fits on single H100 (80GB)
- Multi-GPU: possible via vLLM tensor parallelism

## Key Features
- GRPO (Group Relative Policy Optimization) — same as DeepSeek R1
- LoRA fine-tuning (via Unsloth)
- Multi-turn agent support
- OpenAI-compatible API interface
- W&B / Langfuse integration
- LangGraph support
- RULER (automatic reward generation)

## Serverless vs Local
- `art.LocalAPI()` — runs everything locally (needs GPU)
- `ServerlessBackend` — uses W&B Training managed infrastructure
- For our use case: LocalAPI on Slurm H100 cluster
