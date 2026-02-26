# ART-E Local Slurm Reproduction -- Implementation Design

> Architecture and implementation plan for adapting the ART-E email search RL agent
> to run locally on a Slurm cluster with H100 GPUs.
> Designed: 2026-02-26

## Executive Summary

Adapt OpenPipe's ART-E email search RL agent from SkyPilot/RunPod cloud to local Slurm H100 cluster.
Minimal changes: only modify what is necessary for local execution. Focus on training only (no evaluation).

**Target config**: agent_008 (use_tools=True, 4 traj/group, 12 groups/step, 3 epochs, Qwen2.5-14B)

---

## 1. Current Architecture

```
                   SkyPilot Cloud (RunPod)
                   ┌──────────────────────────────────┐
                   │  H100-SXM:1                      │
                   │                                   │
                   │  train.py                         │
                   │   ├── generate_database()         │
                   │   │   └── HF: corbt/enron-emails  │
                   │   ├── load_synthetic_queries()    │
                   │   │   └── HF: corbt/enron_emails_ │
                   │   │       sample_questions         │
                   │   ├── art.LocalAPI()              │
                   │   │   └── vLLM + LoRA backend     │
                   │   ├── S3 pull/push checkpoints    │
                   │   └── GRPO training loop          │
                   │       ├── rollout() per scenario  │
                   │       │   ├── Agent tool calls    │
                   │       │   ├── SQLite FTS5 search  │
                   │       │   └── Gemini judge (LLM)  │
                   │       └── model.train(groups)     │
                   └──────────────────────────────────┘
```

### Pain Points for Local Reproduction

| Issue | Location | Severity |
|-------|----------|----------|
| Python 3.10 in .python-version (openpipe-art needs >=3.11) | `.python-version` | Blocker |
| Editable path `../ART` for openpipe-art | `pyproject.toml` [tool.uv.sources] | Blocker |
| HF Hub loading for synthetic queries | `art_e/data/query_iterators.py` | Blocker |
| S3 checkpoint push/pull | `art_e/train.py` lines 91-96, 122-125, 149-152 | Blocker |
| SkyPilot launcher | `run_training_job.py` | Replace |
| `benchmark_model()` calls during training | `art_e/train.py` lines 119-120, 148 | Remove (training only) |
| Gemini API key for LLM judge | `art_e/rollout.py` line 173 | Needs .env |
| SkyPilot dependency in pyproject.toml | `pyproject.toml` line 26 | Remove |
| OpenPipe logging (optional) | `art_e/rollout.py` line 29 | Optional, disable |

---

## 2. Target Architecture

```
                   Slurm Cluster (Local)
                   ┌──────────────────────────────────┐
                   │  Slurm Job: H100 x 1 (or x 8)   │
                   │                                   │
                   │  scripts/train_slurm.sh           │
                   │   └── uv run python art_e/train.py│
                   │                                   │
                   │  train.py (modified)              │
                   │   ├── generate_database()         │
                   │   │   └── HF: corbt/enron-emails  │
                   │   ├── load_synthetic_queries()    │
                   │   │   └── LOCAL: data/art_e_      │
                   │   │       vince_kaminski/          │
                   │   ├── art.LocalAPI()              │
                   │   │   └── vLLM + LoRA (1x H100)  │
                   │   └── GRPO training loop          │
                   │       ├── rollout() per scenario  │
                   │       │   ├── Agent tool calls    │
                   │       │   ├── SQLite FTS5 search  │
                   │       │   └── Gemini judge (LLM)  │
                   │       └── model.train(groups)     │
                   │                                   │
                   │  Checkpoints: local filesystem    │
                   └──────────────────────────────────┘
```

### Key Differences

| Aspect | Before (Cloud) | After (Local Slurm) |
|--------|---------------|---------------------|
| Launch | SkyPilot (`run_training_job.py`) | Slurm (`sbatch scripts/train_slurm.sh`) |
| Python | 3.10 (via .python-version) | 3.11+ |
| openpipe-art | Editable from `../ART` | PyPI `openpipe-art[backend]` |
| Query data | HF Hub (`corbt/enron_emails_sample_questions`) | Local Arrow (`data/art_e_vince_kaminski/`) |
| Email DB | HF Hub (`corbt/enron-emails`) at runtime | Same (downloads to `data/enron_emails.db`) |
| Checkpoints | S3 bucket push/pull | Local filesystem (no S3) |
| Evaluation | `benchmark_model()` every N steps | Removed (training only) |
| GPU | H100-SXM:1 via RunPod | H100 x 1 via Slurm |

---

## 3. Implementation Plan

### Phase 0: Prerequisites (No code changes)

**Step 0.1**: Ensure GEMINI_API_KEY is available in `.env` for the LLM judge.
The LLM judge in `rollout.py` uses `gemini/gemini-2.0-flash` via litellm.
This requires `GEMINI_API_KEY` environment variable.

**Step 0.2**: Ensure HF_TOKEN is available if needed for `corbt/enron-emails` dataset download.
The SQLite DB generation (`local_email_db.py`) downloads from HF Hub at runtime.

### Phase 1: Environment Setup (2 files)

#### 1.1 `.python-version` -- Change Python version

**File**: `.python-version`
**Change**: `3.12` (already 3.12, confirm >=3.11 -- OK, no change needed)

> NOTE: .python-version already says `3.12` which satisfies >=3.11. No change needed.

#### 1.2 `pyproject.toml` -- Fix dependencies

**File**: `pyproject.toml`
**Changes**:
1. Update `requires-python` from `>=3.10` to `>=3.11`
2. Remove `skypilot[runpod]>=0.8.1` from dependencies
3. Remove `openpipe==4.49.0` (optional, conflicts may occur)
4. Remove `panza>=0.1.0` (unused, may cause install issues)
5. Remove `pip>=25.0.1` (unnecessary in uv-managed project)
6. Remove `[tool.uv.sources]` editable path for openpipe-art (use PyPI)
7. Add `openpipe-art[backend]` to ensure vLLM/training deps are included
8. Keep `art-e = { workspace = true }` in sources for the project itself

```toml
[project]
name = "art-e"
version = "0.1.0"
description = "Deep learning research for email search and processing"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "datasets>=3.4.1",
    "diskcache>=5.6.3",
    "huggingface-hub>=0.29.3",
    "langchain-core>=0.3.51",
    "litellm>=1.65.0.post1",
    "matplotlib>=3.10.1",
    "openpipe-art[backend]",
    "pandas>=1.3.0",
    "polars>=1.27.1",
    "python-dotenv>=1.1.0",
    "tabulate>=0.9.0",
    "tiktoken>=0.9.0",
    "tqdm>=4.62.0",
    "transformers>=4.50.3",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["art_e*"]

[tool.uv.sources]
art-e = { workspace = true }

[dependency-groups]
dev = [
    "art-e",
    "pytest>=8.3.5",
    "ruff>=0.8",
    "ipykernel>=6.29.5",
]
```

**Rationale**: Removing SkyPilot, panza, openpipe (separate from openpipe-art), pip, kaggle,
ipywidgets, mail-parser -- these are unused for training. Keep only what train.py and rollout.py need.

### Phase 2: Data Loading (1 file)

#### 2.1 `art_e/data/query_iterators.py` -- Load from local Arrow dataset

**File**: `art_e/data/query_iterators.py`
**Change**: Replace `load_dataset(HF_REPO_ID, split=split)` with `load_from_disk()` for local Arrow data

```python
from art_e.data.types_enron import SyntheticQuery
from typing import List, Optional
from datasets import load_from_disk, Dataset
import os
import random

# Path to local Arrow dataset (relative to project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATASET_PATH = os.path.join(BASE_DIR, "..", "..", "data", "art_e_vince_kaminski")

# A few spot-checked synthetic queries that we found to be ambiguous or
# contradicted by other source emails. We'll exclude them for a more accurate
# model.
bad_queries = [49, 101, 129, 171, 208, 266, 327]


def load_synthetic_queries(
    split: str = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    exclude_known_bad_queries: bool = True,
) -> List[SyntheticQuery]:
    dataset_dict = load_from_disk(LOCAL_DATASET_PATH)
    dataset: Dataset = dataset_dict[split]  # type: ignore

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if exclude_known_bad_queries:
        dataset = dataset.filter(lambda x: x["id"] not in bad_queries)

    if shuffle:
        dataset = dataset.shuffle()

    # Convert each row (dict) in the dataset to a SyntheticQuery object
    queries = [SyntheticQuery(**row) for row in dataset]  # type: ignore

    if max_messages is not None:
        queries = [query for query in queries if len(query.message_ids) <= max_messages]

    if shuffle:
        random.shuffle(queries)

    if limit is not None:
        return queries[:limit]
    else:
        return queries
```

**Key change**: `load_dataset(HF_REPO_ID, split=split)` -> `load_from_disk(LOCAL_DATASET_PATH)[split]`

**Risk**: The local Arrow dataset was saved by a newer version of `datasets` library (uses `List` type).
The current dataset_info.json has `"_type": "List"` which requires `datasets>=2.21.0`.
Our pyproject.toml requires `datasets>=3.4.1` so this should work in the uv environment.

### Phase 3: Training Script (1 file)

#### 3.1 `art_e/train.py` -- Remove cloud dependencies, simplify

**File**: `art_e/train.py`
**Changes**:
1. Remove S3 pull/push calls (`_experimental_pull_from_s3`, `_experimental_push_to_s3`)
2. Remove `benchmark_model()` calls (training only)
3. Remove `benchmark_model` import
4. Remove `BACKUP_BUCKET` env var references
5. Keep only agent_008 config (or keep all, but default to 008)
6. Add local checkpoint saving (optional)
7. Remove `delete_checkpoints()` call

```python
import art
import asyncio
from dotenv import load_dotenv
from typing import List
from rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.data.local_email_db import generate_database
from art.utils import iterate_dataset
from art_e.project_types import ProjectPolicyConfig, TrainingConfig

load_dotenv()

# Best model config from ART-E paper
agent_008 = art.TrainableModel(
    name="email-agent-008",
    project="email_agent",
    base_model="Qwen/Qwen2.5-14B-Instruct",
    config=ProjectPolicyConfig(
        max_turns=10,
        log_to_openpipe=False,  # Disable OpenPipe logging for local
        use_tools=True,
        training_config=TrainingConfig(
            trajectories_per_group=4,
            groups_per_step=12,
            learning_rate=1.2e-5,
            eval_steps=30,
            val_set_size=100,
            training_dataset_size=2510,  # Full Vince Kaminski train set
            num_epochs=3,
        ),
    ),
)


async def run_training(model: art.TrainableModel):
    # Step 1: Generate SQLite email database (downloads from HF if needed)
    generate_database()

    assert isinstance(model.config, ProjectPolicyConfig)
    if model.config.training_config is None:
        raise ValueError("Training config is not set")

    # Step 2: Initialize local API (starts vLLM server)
    api = art.LocalAPI()
    await model.register(api)

    # Step 3: Load training data from local Arrow dataset
    print("Loading training data...")
    train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
        split="train", limit=model.config.training_config.training_dataset_size
    )
    print(f"Training data size: {len(train_scenarios)}")

    # Step 4: Training loop (GRPO)
    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=model.config.training_config.groups_per_step,
        num_epochs=model.config.training_config.num_epochs,
        initial_step=await model.get_step(),
    )

    for batch, epoch, global_step, epoch_step in train_iterator:
        print(f"\n--- Step {global_step} (Epoch {epoch}, Step {epoch_step}) ---")

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(
                            model.config.training_config.trajectories_per_group
                        )
                    )
                )
                for scenario in batch
            )
        )

        await model.train(
            groups,
            config=art.TrainConfig(
                learning_rate=model.config.training_config.learning_rate
            ),
        )

    print("Training finished.")


if __name__ == "__main__":
    asyncio.run(run_training(agent_008))
```

**Key changes**:
- Removed all S3 references
- Removed `benchmark_model()` calls
- Removed `RUN_ID` env var dispatch (hardcode agent_008)
- Set `log_to_openpipe=False`
- Set `training_dataset_size=2510` (full Vince Kaminski train set)
- Simplified `__main__` block

### Phase 4: Slurm Job Script (1 new file)

#### 4.1 `scripts/train_slurm.sh` -- Slurm submission script

**File**: `scripts/train_slurm.sh` (NEW)

```bash
#!/bin/bash
#SBATCH --job-name=art-e-train
#SBATCH --partition=gpu           # Adjust to your cluster's GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                  # Single H100 (Qwen2.5-14B fits in 28GB fp16)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00           # 48 hours max (ART-E trained in ~hours)
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

# Load environment
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

# Step 1: Generate SQLite DB (idempotent, skips if exists)
echo "Ensuring email database exists..."
uv run python -c "from art_e.data.local_email_db import generate_database; generate_database()"

# Step 2: Run training
echo "Starting training..."
uv run python art_e/train.py

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
```

**Usage**: `sbatch scripts/train_slurm.sh`

### Phase 5: Environment File Template (1 new file)

#### 5.1 `.env.example` -- Required environment variables

**File**: `.env.example` (NEW)

```bash
# Required: Gemini API key for LLM judge (answer correctness evaluation)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Hugging Face token (for downloading corbt/enron-emails dataset)
# Only needed on first run when generating SQLite DB
HF_TOKEN=your_hf_token_here

# Optional: Weights & Biases (openpipe-art logs training metrics to W&B)
WANDB_API_KEY=your_wandb_api_key_here

# Optional: OpenPipe API key (for logging rollouts to OpenPipe dashboard)
# Set to empty or remove to disable
# OPENPIPE_API_KEY=
```

---

## 4. File Change Summary

### Files to MODIFY (4 files)

| File | Change | Lines Affected |
|------|--------|---------------|
| `pyproject.toml` | Remove SkyPilot/unused deps, fix openpipe-art source, bump Python | ~30 lines |
| `art_e/data/query_iterators.py` | HF Hub -> local Arrow dataset loading | ~10 lines |
| `art_e/train.py` | Remove S3/SkyPilot/eval, simplify to local-only | ~80 lines |
| `art_e/rollout.py` | No changes needed (Gemini judge stays, just needs API key) | 0 lines |

### Files to CREATE (2 files)

| File | Purpose |
|------|---------|
| `scripts/train_slurm.sh` | Slurm job submission script |
| `.env.example` | Environment variable template |

### Files to LEAVE UNCHANGED (5 files)

| File | Reason |
|------|--------|
| `art_e/rollout.py` | Works as-is; Gemini judge just needs GEMINI_API_KEY in .env |
| `art_e/email_search_tools.py` | SQLite tools, no cloud dependencies |
| `art_e/project_types.py` | Config classes, no changes needed |
| `art_e/data/local_email_db.py` | DB generation from HF, works as-is |
| `art_e/data/types_enron.py` | Data models, no changes needed |
| `.python-version` | Already 3.12, satisfies >=3.11 |

### Files to DELETE/IGNORE (1 file)

| File | Reason |
|------|--------|
| `run_training_job.py` | SkyPilot launcher, replaced by Slurm script |

---

## 5. Dependency Ordering

```
Phase 1: Environment Setup
  1.1 pyproject.toml (fix deps + Python version)
  1.2 uv sync (install dependencies)
       └── This may take 10-20 min for openpipe-art[backend] (vLLM, torch, etc.)

Phase 2: Data Loading
  2.1 query_iterators.py (local Arrow loading)
       └── Depends on: datasets library from Phase 1

Phase 3: Training Script
  3.1 train.py (remove cloud deps)
       └── Depends on: query_iterators.py changes from Phase 2

Phase 4: Slurm Script
  4.1 scripts/train_slurm.sh (NEW)
       └── Depends on: all Phase 1-3 changes

Phase 5: Environment
  5.1 .env.example (NEW)
  5.2 Create .env with actual API keys
       └── Independent, can be done anytime before running
```

**Critical path**: Phase 1 -> Phase 2 -> Phase 3 -> Phase 4

---

## 6. Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **openpipe-art[backend] install fails** | Cannot train | Pin version `openpipe-art[backend]==0.5.11`; check CUDA/torch compatibility |
| **vLLM + H100 CUDA version mismatch** | Cannot start inference | Check `nvidia-smi` CUDA version matches vLLM requirements (CUDA 12.1+) |
| **Gemini API rate limits** | Training stalls on reward computation | Add retry logic (already has `@retry(stop=stop_after_attempt(3))`); consider batch judge calls |
| **Local Arrow dataset incompatible format** | Cannot load training data | The `dataset_info.json` uses `List` type requiring datasets>=2.21; our deps require >=3.4.1 so should be OK |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Slurm OOM (memory)** | Job killed | Request 128GB RAM; Qwen2.5-14B needs ~28GB VRAM + system RAM for data |
| **SQLite concurrent access from vLLM workers** | DB locks | Already uses `check_same_thread=False` and read-only mode; should be OK |
| **Slurm time limit exceeded** | Training incomplete | Set 48h limit; ART-E originally trained in hours on single H100 |
| **openpipe-art API changes** | Code breaks | Pin version to 0.5.11 |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **HF dataset download for SQLite DB** | Network needed once | Can pre-download; DB is cached after first run |
| **W&B integration issues** | No metrics logging | W&B is optional; training proceeds without it |
| **litellm disk cache fills up** | Disk space | Clean `/tmp/litellm_cache` periodically |

---

## 7. GPU Resource Analysis

### Single H100 (Recommended)

| Component | VRAM Usage |
|-----------|------------|
| Qwen2.5-14B (fp16) | ~28 GB |
| vLLM KV cache | ~20-30 GB |
| LoRA adapters | ~1-2 GB |
| Training gradients | ~10-15 GB |
| **Total** | **~60-75 GB** |

Single H100 (80GB) should be sufficient. No multi-GPU needed.

**Decision (2026-02-26):** For `agent_008` (Qwen2.5-14B + GRPO), baseline training will run on **1x H100 80GB**. If OOM appears in practice, reduce concurrency first (`groups_per_step`, `trajectories_per_group`) before moving to multi-GPU.

### Multi-GPU Consideration

If 8x H100 are available and we want faster training:
- vLLM supports tensor parallelism (`--tensor-parallel-size N`)
- openpipe-art may support multi-GPU via vLLM config
- But single H100 is the validated configuration; start there

### Training Time Estimate

- Original: ~$80 on RunPod H100 (~$2.49/hr -> ~32 hours)
- With 2510 training samples, 3 epochs, 12 groups/step, 4 traj/group:
  - Total groups = ceil(2510/12) * 3 = 628 steps
  - Each step: 12 groups x 4 trajectories = 48 rollouts
  - Rollout: ~10-30 seconds (10 turns max, vLLM inference + Gemini judge)
  - Per step: ~5-15 minutes
  - Total: ~52-157 hours (wide range due to rollout variability)
- Realistic estimate: 24-72 hours on single H100

---

## 8. Verification Plan

### Pre-flight Checks

```bash
# 1. Python version
python --version  # Should be >= 3.11

# 2. CUDA available
nvidia-smi  # H100 visible, CUDA 12.x

# 3. Dependencies installed
uv run python -c "import art; print(art.__version__)"
uv run python -c "import vllm; print(vllm.__version__)"

# 4. Local dataset loads
uv run python -c "
from art_e.data.query_iterators import load_synthetic_queries
q = load_synthetic_queries(split='train', limit=5)
print(f'Loaded {len(q)} queries')
print(f'Sample: {q[0].question[:100]}')
"

# 5. SQLite DB generates
uv run python -c "
from art_e.data.local_email_db import generate_database
generate_database()
"

# 6. Gemini API works
uv run python -c "
from litellm import completion
r = completion(model='gemini/gemini-2.0-flash', messages=[{'role':'user','content':'Say hello'}], max_tokens=5)
print(r.choices[0].message.content)
"
```

### Smoke Test (Short training run)

Before full training, run a minimal test:
```python
# Modify train.py temporarily or add a flag:
# training_dataset_size=24 (2 steps worth)
# num_epochs=1
# This should complete in ~10-30 minutes
```

---

## 9. Rollout.py Analysis (No Changes Needed)

The rollout module works as-is because:

1. **LLM judge** (`determine_if_answer_is_correct`, line 162-181):
   - Uses `gemini/gemini-2.0-flash` via litellm
   - Just needs `GEMINI_API_KEY` in environment
   - Already has `@retry(stop=stop_after_attempt(3))`
   - Uses litellm disk cache for repeated queries

2. **Tool dispatch** (line 314-374):
   - search_emails, read_email: SQLite-based, fully local
   - return_final_answer: in-memory, no external calls

3. **OpenPipe logging** (line 384-403):
   - Guarded by `if model.config.log_to_openpipe and op_client is not None`
   - We set `log_to_openpipe=False` in train.py
   - Even if enabled, failure is caught and logged (non-blocking)

4. **Model inference** (line 245-256):
   - Uses `model.base_url` from art.LocalAPI
   - litellm routes to local vLLM server
   - No external API needed for the training model itself

---

## 10. Alternative: Replace Gemini Judge

If Gemini API key is unavailable or rate-limited, alternatives for the LLM judge:

| Option | Pros | Cons |
|--------|------|------|
| Keep Gemini (recommended) | Already implemented, cheap, fast | Needs API key, external dependency |
| Use the training model itself | No external API needed | Circular (model judges itself), quality degrades |
| Use local LLM (e.g., Qwen2.5-7B) | Fully offline | Need separate GPU/process, complex setup |
| String matching | Simple, fast | Misses semantic equivalence |
| Skip judge, use source-only reward | Simpler reward | Loses answer correctness signal |

**Recommendation**: Keep Gemini. Cost is negligible (~$0.01 per 1000 judge calls with flash).

---

## Changelog

| Date | Changes |
|------|---------|
| 2026-02-26 | Initial design for local Slurm reproduction |
| 2026-02-26 | Import/runtime safety review: `art_e/rollout.py` must not hard-import `openpipe`; make OpenPipe import optional/lazy. `art_e/train.py` should use `from art_e.rollout import rollout` (not `from rollout import rollout`) to avoid execution-mode-dependent imports. Keep benchmark validation removed for training-only runs, and treat `eval_steps`/`val_set_size` as optional metadata unless validation is reintroduced. |
| 2026-02-26 | Hardware decision confirmed: `agent_008` GRPO training baseline uses 1x H100 80GB; memory pressure should be handled first by reducing rollout concurrency (`groups_per_step`, `trajectories_per_group`) before adopting multi-GPU. |
