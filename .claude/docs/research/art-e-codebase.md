# ART·E Codebase Analysis

## Overview
OpenPipe ART·E (Art Email) — RL-trained email search agent using GRPO on Enron Email Dataset.

## Architecture
- **Base model**: Qwen/Qwen2.5-14B-Instruct
- **Training**: GRPO via openpipe-art library (LoRA fine-tuning on vLLM backend)
- **Environment**: 3 tools (search_emails, read_email, return_final_answer)
- **DB**: SQLite FTS5 full-text search on Enron emails (corbt/enron-emails on HF)
- **Dataset**: Synthetic Q&A pairs (corbt/enron_emails_sample_questions on HF)

## Key Modules
| File | Purpose |
|------|---------|
| `art_e/train.py` | Training script, model variant definitions, GRPO training loop |
| `art_e/rollout.py` | Agent-env interaction, reward function, tool dispatch, LLM judge |
| `art_e/email_search_tools.py` | SQLite FTS5 search + email retrieval |
| `art_e/project_types.py` | ProjectPolicyConfig + TrainingConfig |
| `art_e/data/local_email_db.py` | SQLite DB creation from HF dataset |
| `art_e/data/query_iterators.py` | HF dataset → SyntheticQuery loader |
| `art_e/data/types_enron.py` | Data models (SyntheticQuery, Email) |

## Reward Function (rollout.py:67-113)
- Format errors: -2 to -1
- Wrong answer: -1 to 0
- No answer / ran out of turns: 0 to 1
- Correct answer: 1 to 2 (bonuses for correct sources, fewer sources, fewer turns)
- Partial rewards (+0.1 each): finding right email, reading right email, not reading invalid, correct sources

## Data: Vince Kaminski (data/art_e_vince_kaminski/)
- Train: 2,510 synthetic Q&A pairs
- Test: 376 synthetic Q&A pairs
- All inbox_address: vince.kaminski@enron.com
- Features: id, question, answer, message_ids, how_realistic, inbox_address, query_date

## Dependencies
- `openpipe-art` (PyPI: v0.5.11, requires Python >=3.11)
- litellm, langchain-core, datasets, polars, pydantic
- SQLite (stdlib)
- vLLM (via openpipe-art backend)

## Original Training Config (agent_008 — best model)
- base_model: Qwen/Qwen2.5-14B-Instruct
- max_turns: 10
- use_tools: True
- trajectories_per_group: 4
- groups_per_step: 12
- learning_rate: 1.2e-5
- eval_steps: 30
- num_epochs: 3
- H100-SXM:1 via SkyPilot/RunPod
