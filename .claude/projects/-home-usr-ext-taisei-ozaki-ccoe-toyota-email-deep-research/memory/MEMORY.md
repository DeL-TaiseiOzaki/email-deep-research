# Project Memory: email-deep-research (ART-E)

## Project Overview
ART-E is an email agent trained with reinforcement learning (GRPO) using the OpenPipe ART framework. The agent answers questions about the Enron email dataset by searching and reading emails via tool calls.

## Current State (2026-03-04)
- **Branch**: main (all changes unstaged, not committed)
- **Base model**: Qwen/Qwen3-14B (migrated from Qwen2.5-14B-Instruct)
- **Framework**: openpipe-art[backend]>=0.5.15 (vLLM 0.15.1, unsloth 2026.2.1)
- **Training status**: Active development; training configs ready but run not yet started with Qwen3

## Key Files
| File | Purpose |
|------|---------|
| `art_e/train.py` | Training entry point, model configs, GRPO loop |
| `art_e/rollout.py` | Rollout function, reward calculation, tool call handling |
| `art_e/project_types.py` | Pydantic configs (ProjectPolicyConfig, TrainingConfig) |
| `art_e/evaluate/benchmark.py` | Validation/benchmark runner |
| `art_e/data/` | Dataset loading, SQLite email DB, synthetic queries |
| `scripts/train_slurm.sh` | Base SLURM training script |
| `scripts/train_qwen3_thinking_slurm.sh` | Thinking-mode SLURM script |
| `pyproject.toml` | Dependencies and project config |

## Architecture
```
User query -> rollout() -> vLLM (Qwen3-14B) -> tool calls (search/read email) -> final answer
                |
                v
         reward_and_metrics() -> GRPO training step
```

## Model Configs
- `qwen3-14b`: Default, no thinking, max_tokens=2048
- `qwen3-14b-thinking`: Thinking enabled, max_tokens=4096, reasoning_parser="qwen3"

## Important Design Decisions
1. **Thinking mode is configurable** via `ProjectPolicyConfig.enable_thinking`
2. **strip_thinking_tags()** removes `<think>` blocks in non-tool-call path only
3. **Tool call parser**: hermes (Qwen3 Hermes-compatible)
4. **Checkpoint pruning**: keeps latest + top K by val/reward
5. **tool_choice**: "auto" for hosted_vllm, "required" otherwise
6. **content=None fix**: Set to "" for Qwen chat template compatibility

## Session Log
| Date | Summary |
|------|---------|
| 2026-03-04 | Qwen3 migration, thinking mode, training loop improvements, dependency upgrades |

## Known Issues / TODOs
- All changes are unstaged (need commit)
- Gemini CLI calls failing (ModelNotFoundError for gemini-3-pro-preview)
- Training run with Qwen3 not yet executed
- `calc-cost/` directory added (untracked) -- cost calculation utilities
