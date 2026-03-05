# Codebase Analysis for Qwen3 Support

## 1. Directory Structure

```
email-deep-research/
├── art_e/                          # Main application package
│   ├── __init__.py
│   ├── project_types.py            # TrainingConfig + ProjectPolicyConfig
│   ├── train.py                    # Training loop (GRPO), model definition
│   ├── rollout.py                  # Agent rollout logic, tool calling, reward
│   ├── email_search_tools.py       # SQLite email search/read tools
│   ├── data/                       # Data loading, Enron email DB
│   │   ├── convert_enron_email_dataset.py
│   │   ├── generate_synthetic_question_data.py
│   │   ├── local_email_db.py
│   │   ├── query_iterators.py
│   │   └── types_enron.py
│   └── evaluate/                   # Benchmarking, charts, model push
│       ├── benchmark.py            # benchmark_model() function
│       ├── benchmark_prompted_models.py  # Benchmark GPT/Gemini/DeepSeek
│       ├── charts.py
│       ├── create_blog_charts.py
│       ├── display_run_html.py
│       ├── explore.py
│       ├── explore_o3_errors.py
│       ├── load_trajectories.py
│       └── push_agent_to_hf.py     # Push LoRA+tokenizer to HF
├── scripts/
│   ├── train_slurm.sh              # SLURM training job
│   └── benchmark_slurm.sh          # SLURM benchmark job
├── calc-cost/                      # Cost/latency comparison scripts
├── run_training_job.py             # SkyPilot launch script
├── pyproject.toml                  # Dependencies
└── .art/                           # ART output (checkpoints, trajectories)
```

## 2. Model Configuration and Loading

### Current Model Definition (art_e/train.py:56-76)

```python
agent_008 = art.TrainableModel(
    name="email-agent-008",
    project="email_agent",
    base_model="Qwen/Qwen2.5-14B-Instruct",  # <-- THE KEY LINE
    config=ProjectPolicyConfig(
        max_turns=10,
        training_config=TrainingConfig(
            trajectories_per_group=4,
            groups_per_step=12,
            learning_rate=1.2e-5,
            ...
        ),
    ),
)
```

### How base_model Is Used (ART Library Flow)

1. `art.TrainableModel.base_model` → passed to `dev.get_model_config(base_model, ...)`
2. `get_model_config()` → creates `InitArgs(model_name=base_model, ...)` with defaults:
   - `max_seq_length=32768`
   - `load_in_4bit=True`
   - `fast_inference=True` (enables vLLM)
   - `gpu_memory_utilization=0.79`
   - `max_lora_rank=8`
3. `InitArgs` → passed to `unsloth.FastLanguageModel.from_pretrained(**init_args)`
4. Unsloth dispatches based on `model_type` from the HF config:
   - `"qwen2"` → `FastQwen2Model` (used for both Qwen2 and Qwen2.5)
5. PEFT model created via `unsloth.FastLanguageModel.get_peft_model(model, **peft_args)`
6. LoRA target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

### vLLM OpenAI Server Configuration (art/dev/openai_server.py)

```python
server_args = ServerArgs(
    enable_auto_tool_choice=True,
    tool_call_parser="hermes",        # <-- Qwen2.5 uses Hermes format
    return_tokens_as_token_ids=True,
)
```

The `tool_call_parser="hermes"` is hardcoded in the ART library's server config.

### Benchmark SLURM Script (scripts/benchmark_slurm.sh)

Manually starts vLLM with:
```bash
--model Qwen/Qwen2.5-14B-Instruct
--enable-auto-tool-choice
--tool-call-parser hermes
--enable-lora
--lora-modules "email-agent-008=${LATEST_STEP}"
```

## 3. Unsloth Integration

### Version: 2025.3.19

### Model Dispatch (unsloth/models/loader.py)

```python
elif model_type == "qwen2":
    dispatch_model = FastQwen2Model
```

- Qwen2 and Qwen2.5 both use `model_type="qwen2"` in HF config
- **Qwen3 uses `model_type="qwen3"`** — NOT currently mapped in unsloth 2025.3.19

### Supported Model Types in Unsloth 2025.3.19

- `llama` → FastLlamaModel
- `mistral` → FastMistralModel
- `qwen2` → FastQwen2Model
- `granite` → FastGraniteModel
- `gemma` → (file exists but not in dispatch)
- `gemma2` → (file exists but not in dispatch)
- `cohere` → (file exists but disabled)

### Key Finding: **Unsloth 2025.3.19 does NOT support Qwen3**

The `FastQwen2Model` class (in `unsloth/models/qwen2.py`) imports from `transformers.models.qwen2.modeling_qwen2`, which is a different architecture from `transformers.models.qwen3.modeling_qwen3`.

### Chat Templates (unsloth/chat_templates.py)

Only has `qwen-2.5` / `qwen25` / `qwen2.5` templates registered. No Qwen3 template.

## 4. Training Pipeline

### train.py Flow
1. `generate_database()` — builds SQLite DB from Enron dataset
2. `art.LocalAPI()` — initializes local API (vLLM server internally)
3. `model.register(api)` — loads model via Unsloth, starts vLLM, creates LoRA
4. Load training data (synthetic email queries)
5. Training loop:
   - `art.gather_trajectory_groups()` → runs rollouts in parallel
   - `model.train(groups, config)` → GRPO training step
   - Periodic validation via `benchmark_model()`
   - Checkpoint pruning (keep top-K by reward)

### rollout.py Key Points
- Uses `model.openai_client()` for trainable models (gets logprobs via ART-patched OpenAI client)
- Uses `litellm.acompletion()` for non-trainable models (GPT-5, Gemini, etc.)
- Tool calling via OpenAI function calling format
- Answer evaluation uses `gpt-5-mini` for semantic comparison
- **Line 273**: Comment explicitly notes Qwen2.5 chat template issue:
  ```python
  # Ensure content is never None (tool call messages may omit content),
  # as Qwen2.5's chat template cannot concatenate str with NoneType.
  ```

### Tokenization (art/local/tokenize.py)
- Uses `tokenizer.apply_chat_template()` — relies on HF tokenizer's built-in template
- Creates sentinel tokens for logprob alignment
- Builds assistant_mask for GRPO loss calculation

## 5. Dependencies (pyproject.toml)

```toml
dependencies = [
    "datasets>=3.4.1",
    "huggingface-hub>=0.29.3",
    "langchain-core>=0.3.51",
    "litellm>=1.65.0.post1",
    "openpipe-art[backend]",      # Includes unsloth, vllm, trl, peft
    "transformers>=4.50.3",
]
```

Key transitive dependencies:
- **unsloth==2025.3.19** (via openpipe-art)
- **unsloth-zoo==2025.3.17** (via openpipe-art)
- **vllm==0.7.3** (via openpipe-art)
- **transformers==4.51.1** (has Qwen3 model support)
- **trl==0.15.2** (GRPO trainer)
- **peft** (LoRA adapters)

## 6. Model-Specific Logic

### In This Codebase (art_e/)
- Only ONE hardcoded base_model reference: `"Qwen/Qwen2.5-14B-Instruct"` in train.py
- The `None`-content fix in rollout.py (line 273-275) is Qwen2.5-specific
- No model-family-specific branching beyond what ART/Unsloth handle internally

### In ART Library
- `tool_call_parser="hermes"` hardcoded in openai_server.py
- Default LoRA target modules hardcoded to Llama-family projection names

### In Unsloth
- Model dispatch by `model_type` string from HF config
- Chat template dispatch by model family name

## 7. Qwen3 Architecture Differences

### Key Differences from Qwen2/Qwen2.5
Based on transformers 4.51.1 having `transformers.models.qwen3`:

1. **model_type**: `"qwen3"` (not `"qwen2"`)
2. **Thinking/reasoning mode**: Qwen3 supports `<think>...</think>` blocks
3. **Chat template**: Different from Qwen2.5 (includes thinking support)
4. **Architecture**: Similar transformer architecture but potentially different attention patterns
5. **Tool calling**: Uses different format than Hermes (may need different vLLM parser)

## 8. What Needs to Change for Qwen3 Support

### Layer 1: This Codebase (art_e/) — MINIMAL CHANGES
1. **train.py**: Change `base_model="Qwen/Qwen2.5-14B-Instruct"` to `"Qwen/Qwen3-14B"` (or appropriate variant)
2. **rollout.py**: The None-content fix (line 273-275) may still be needed; test with Qwen3's chat template
3. **scripts/*.sh**: Update model name references

### Layer 2: ART Library (openpipe-art) — POSSIBLE CHANGES
1. **openai_server.py**: `tool_call_parser="hermes"` may need to change for Qwen3
   - Qwen3 may use a different tool calling format
   - vLLM's `--tool-call-parser` needs to match the model's expected format
2. **dev/model.py**: LoRA target modules list is generic enough (q/k/v/o/gate/up/down proj) — likely compatible
3. **ServerArgs**: May need `enable_reasoning=True` and `reasoning_parser` for Qwen3 thinking mode

### Layer 3: Unsloth — BLOCKING DEPENDENCY
1. **unsloth 2025.3.19 does NOT have Qwen3 support**
   - No `qwen3` in model_type dispatch (loader.py)
   - No `FastQwen3Model` class
   - No Qwen3 chat template
2. **Options**:
   a. Upgrade unsloth to a version that adds Qwen3 support (if available)
   b. Check if Qwen3's `model_type` falls back to a compatible handler
   c. Bypass unsloth for Qwen3 (use HF transformers + PEFT directly)

### Layer 4: vLLM — POSSIBLE BLOCKING
1. **vLLM 0.7.3 does NOT have Qwen3 support**
   - No qwen3 references found
2. **Options**:
   a. Upgrade vLLM to a version with Qwen3 support
   b. Check if Qwen3 works via fallback mechanisms

### Layer 5: Thinking/Reasoning Mode
1. Qwen3 supports thinking mode with `<think>...</think>` blocks
2. If using thinking mode:
   - Need to handle think tokens in the chat template
   - May need `enable_thinking=True` parameter
   - Training should decide whether to train on thinking tokens or mask them
   - Token usage will increase significantly
3. If NOT using thinking mode:
   - May be able to disable via generation config
   - Simplifies integration

## Summary of Blocking Issues (Priority Order)

| Priority | Issue | Severity | Solution |
|----------|-------|----------|----------|
| 1 | Unsloth lacks Qwen3 model type | **BLOCKING** | Upgrade unsloth or find workaround |
| 2 | vLLM 0.7.3 lacks Qwen3 | **BLOCKING** | Upgrade vLLM |
| 3 | Tool call parser for Qwen3 | MEDIUM | Investigate Qwen3's tool format, update parser |
| 4 | Chat template handling | MEDIUM | Test with Qwen3 tokenizer's built-in template |
| 5 | Thinking mode decision | LOW | Design decision: enable or disable |
| 6 | Code changes in art_e/ | LOW | Simple string replacements |
