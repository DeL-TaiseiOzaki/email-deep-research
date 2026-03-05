# Qwen3 Compatibility Research (March 2026)

## 1. Unsloth + Qwen3

### Support Status: FULLY SUPPORTED

- **Initial support**: April 29, 2025 (same day as Qwen3 release)
- **Issue tracking**: [GitHub #2428 - Qwen3 Fine-tuning now in Unsloth!](https://github.com/unslothai/unsloth/issues/2428)
- **Installation**: `pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo`

### Model Classes

| Class | Location | Use Case |
|-------|----------|----------|
| `FastQwen3Model` | `unsloth/models/qwen3.py` | Dense Qwen3 models (internal) |
| `FastQwen3MoeModel` | `unsloth/models/qwen3_moe.py` | MoE Qwen3 models (internal) |
| `FastLanguageModel` | Public API | Dense models (0.6B-32B) |
| `FastModel` | Public API | **All models including MoE** (recommended) |

### Usage Pattern

```python
# For dense models (either works)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B",
    max_seq_length=4096,
    load_in_4bit=True,
)

# For MoE models (MUST use FastModel, not FastLanguageModel)
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-30B-A3B",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

### Version History

| Version | Date | Notes |
|---------|------|-------|
| 2025.4.29 | Apr 2025 | Initial Qwen3 support |
| 2025.5.13 | May 2025 | Fixed GRPO + vLLM compatibility, inference fix |
| 2025.9.11 | Sep 2025 | "Fast Qwen3 patching" confirmed |
| 2025.12.9 | Dec 2025 | Continued Qwen3 patching |

### Performance

- **2x faster** fine-tuning vs standard HF training
- **70% less VRAM** usage
- **8x longer context** lengths supported
- Qwen3-14B fits in **16GB VRAM** (Google Colab T4) with 4-bit quantization
- Qwen3-30B-A3B fits in **17.5GB VRAM**

---

## 2. vLLM + Qwen3

### Support Status: FULLY SUPPORTED (v0.8.4+)

| vLLM Version | Qwen3 Support |
|-------------|---------------|
| v0.8.4 | Initial Qwen3 + Qwen3MoE support |
| v0.8.5 | FP8 dense model fix; `enable_thinking=False` incompatibility noted |
| v0.9.0 | Dedicated `qwen3` reasoning parser; `enable_thinking=False` fix |
| v0.10.0+ | Stable support, used by OpenPipe ART |

### Basic Serving

```bash
# Dense model
vllm serve Qwen/Qwen3-8B \
  --enable-reasoning \
  --reasoning-parser qwen3

# MoE model
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 4 \
  --enable-reasoning \
  --reasoning-parser qwen3

# MoE with expert parallelism (for TP divisibility issues)
vllm serve Qwen/Qwen3-235B-A22B \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-reasoning \
  --reasoning-parser qwen3
```

### Known Issues

- **FP8 dense models**: Indexing error in v0.8.4, fixed in v0.8.5
- **FP8 MoE models**: Weight quantization block size mismatch with high TP — use `--tensor-parallel-size 4` or add `--enable-expert-parallel`
- **enable_thinking=False**: Incompatible with reasoning feature in v0.8.5, fixed in v0.9.0 with `qwen3` parser

---

## 3. Qwen3 Thinking Mode (`<think>...</think>`)

### How It Works

Qwen3 supports **hybrid thinking** — seamless switching between:
- **Thinking mode**: Deep reasoning for complex tasks (math, coding, logic)
- **Non-thinking mode**: Efficient, fast responses for simple dialogue

When thinking mode is enabled (default), the model outputs:
```
<think>
[internal reasoning steps]
</think>
[final answer]
```

### Controlling Thinking Mode

#### In vLLM (API level)

```python
# Enable thinking (default)
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[...],
)
# response.choices[0].message.reasoning_content → thinking steps
# response.choices[0].message.content → final answer

# Disable thinking
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[...],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    },
)
```

#### vLLM Server Flags

```bash
# With reasoning parser (extracts <think> into reasoning_content)
vllm serve Qwen/Qwen3-8B \
  --enable-reasoning \
  --reasoning-parser qwen3

# Parser options:
#   qwen3        — dedicated Qwen3 parser (v0.9.0+, recommended)
#   deepseek_r1  — also works (v0.8.4 initial approach)
```

#### Reasoning Parser Details

The `qwen3` reasoning parser (`vllm.reasoning.qwen3_reasoning_parser`):
- Extracts reasoning content via `extract_reasoning_content()`
- Supports streaming via `extract_reasoning_content_streaming()`
- Uses token IDs for faster processing in streaming mode
- Handles both `<think>` and `</think>` tag detection

### Fine-tuning Considerations

- **Multi-turn issue**: Default Qwen3 chat template **removes** `<think>` tokens from previous assistant turns
- **OpenPipe fix**: OpenPipe/Qwen3-14B-Instruct adds `<think></think>` tags to all assistant prompts to fix training/inference consistency
- **Recommendation**: For fine-tuning, ensure your chat template preserves thinking tags in conversation history

---

## 4. Qwen3 Tool Calling

### Format: Hermes-Compatible (Nous Format)

Qwen3's `tokenizer_config.json` has **built-in Hermes-style tool use** support. The default `fncall_prompt_type` is `'nous'` (Hermes format).

### vLLM Tool Calling Setup

```bash
# Option 1: Hermes parser (for standard Qwen3)
vllm serve Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-reasoning \
  --reasoning-parser qwen3

# Option 2: qwen3_coder parser (for Qwen3-Coder)
vllm serve Qwen/Qwen3-Coder-... \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-reasoning \
  --reasoning-parser qwen3
```

### Two Approaches

| Approach | Parser Location | Recommended For |
|----------|----------------|-----------------|
| **Qwen-Agent** | Client-side (Qwen-Agent parses) | Qwen3, QwQ — do NOT add `--tool-call-parser` to vLLM |
| **vLLM built-in** | Server-side (vLLM parses) | Qwen3-Coder — use `--tool-call-parser qwen3_coder` |

### Important Warning

> For reasoning models like Qwen3, **do NOT use ReAct-style (stopword-based) tool call templates**. The model may output stopwords inside `<think>` sections, causing premature tool call triggering.

---

## 5. Qwen3 Model Variants

### Dense Models

| Model | Parameters | Inference VRAM (4-bit) | LoRA QLoRA VRAM | LoRA BF16 VRAM |
|-------|-----------|----------------------|-----------------|----------------|
| Qwen3-0.6B | 0.6B | ~2GB | ~4-6GB | ~6-8GB |
| Qwen3-1.7B | 1.7B | ~4GB | ~8GB | ~12GB |
| Qwen3-4B | 4B | ~4GB | ~16GB | ~24GB |
| Qwen3-8B | 8B | ~6GB | ~12-16GB | ~24GB+ |
| Qwen3-14B | 14B | ~10GB | ~16GB (T4 OK) | ~40GB+ |
| Qwen3-32B | 32B | ~20GB | ~24-32GB | ~48GB+ |

### MoE Models

| Model | Total Params | Active Params | Inference VRAM (4-bit) | LoRA VRAM |
|-------|-------------|---------------|----------------------|-----------|
| Qwen3-30B-A3B | 30B | 3B | ~19GB | ~40GB+ (ZeRO-2) |
| Qwen3-235B-A22B | 235B | 22B | ~142GB | Multi-GPU enterprise |

### Performance Equivalences (per Qwen team)

| Qwen3 Model | Equivalent to |
|-------------|---------------|
| Qwen3-1.7B | Qwen2.5-3B |
| Qwen3-4B | Qwen2.5-7B / Qwen2.5-72B-Instruct (!) |
| Qwen3-8B | Qwen2.5-14B |
| Qwen3-14B | Qwen2.5-32B |
| Qwen3-32B | Qwen2.5-72B |
| Qwen3-30B-A3B | Outcompetes QwQ-32B |

### Recommended for Fine-tuning on Limited GPU

| GPU | Recommended Model | Quantization |
|-----|-------------------|-------------|
| **16GB** (T4, RTX 4060 Ti) | Qwen3-8B or Qwen3-14B | 4-bit QLoRA |
| **24GB** (RTX 3090/4090) | Qwen3-14B or Qwen3-32B | 4-bit QLoRA |
| **40GB** (A100 40GB) | Qwen3-32B | 4-bit or 8-bit QLoRA |
| **80GB** (A100 80GB) | Qwen3-32B | BF16 LoRA |
| **Multi-GPU** | Qwen3-30B-A3B (MoE) | ZeRO-2 + LoRA |

### Updated Variants (July 2025 - Qwen3-2507)

Qwen released updated "-2507" versions for 4B, 30B, and 235B models with improved thinking/non-thinking capabilities.

### Qwen3.5 (Late 2025)

Newer generation with sizes: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B. Unsloth supports these as well.

---

## 6. OpenPipe ART + Qwen3

### Support Status: SUPPORTED (with caveats)

- **GitHub**: [OpenPipe/ART](https://github.com/OpenPipe/ART)
- **License**: Apache-2.0
- **Algorithm**: GRPO + GSPO (Group Sequence Policy Optimization)
- **vLLM version**: Upgraded to v0.10.0

### Supported Qwen3 Models in ART

| Model | Tool Calling | Notes |
|-------|-------------|-------|
| OpenPipe/Qwen3-14B-Instruct | Yes | Fixed chat template (thinking tags preserved) |
| Qwen3-30B-A3B-Instruct | Yes | MoE, stronger reasoning |
| Other Qwen3 variants | Varies | Check chat template compatibility |

### Key Limitation: Multi-turn Training

- **Single-turn**: Fully supported
- **Multi-turn**: Qwen3 chat template **removes `<think>` tokens** from prior turns
- **Workaround**: Use `additional_histories` trajectory parameter to split turns into separate message histories

### OpenPipe/Qwen3-14B-Instruct

OpenPipe created a fixed variant of Qwen3-14B that:
- Adds `<think></think>` tags to ALL assistant prompts (not just the current turn)
- Ensures training/inference consistency
- 240,000+ downloads in first month
- Addresses gap: original Qwen3 lacked 14B Instruct (non-thinking) variant

### GSPO Algorithm

Used to train Qwen3-235B-A22B-Instruct-2507:
- Sequence-level optimization (vs token-level in GRPO)
- Improved stability for MoE models
- Infrastructure-friendly design

---

## 7. Known Gotchas & Best Practices

### Fine-tuning Gotchas

1. **MoE + LoRA + DeepSpeed**: Use ZeRO Stage 2, NOT Stage 3 (incompatible with LoRA on MoE)
2. **Multi-turn thinking tags**: Default template removes them — use OpenPipe's fixed template or custom template
3. **Unsloth MoE loading**: Must use `FastModel`, not `FastLanguageModel`
4. **vLLM enable_thinking=False**: Requires vLLM >= 0.9.0 with `qwen3` parser
5. **ReAct tool calling**: Avoid stopword-based templates — thinking section may contain stopwords

### Best Practices

1. **Start with Qwen3-8B or 14B** for initial experiments (fits on consumer GPU)
2. **Use 4-bit QLoRA** with Unsloth for maximum VRAM efficiency
3. **Use `qwen3` reasoning parser** in vLLM (not `deepseek_r1`)
4. **Use `hermes` tool call parser** for standard Qwen3 models
5. **Consider OpenPipe/Qwen3-14B-Instruct** if doing multi-turn training
6. **Qwen3-4B rivals Qwen2.5-72B-Instruct** — don't underestimate smaller models
7. **For ART/GRPO training**, ensure vLLM v0.10.0+ for best compatibility

---

## Sources

- [Unsloth Qwen3 Documentation](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [Unsloth Qwen3 Blog](https://unsloth.ai/blog/qwen3)
- [Qwen Unsloth Integration](https://qwen.readthedocs.io/en/latest/training/unsloth.html)
- [vLLM Qwen3 Reasoning Parser](https://docs.vllm.ai/en/latest/api/vllm/reasoning/qwen3_reasoning_parser/)
- [vLLM Reasoning Outputs](https://docs.vllm.ai/en/v0.10.1/features/reasoning_outputs.html)
- [vLLM Qwen3 Usage Guide (GitHub #17327)](https://github.com/vllm-project/vllm/issues/17327)
- [Qwen Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [vLLM Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [OpenPipe ART GitHub](https://github.com/OpenPipe/ART)
- [OpenPipe ART Supported Models](https://art.openpipe.ai/resources/models)
- [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [Qwen3 Blog Post](https://qwenlm.github.io/blog/qwen3/)
- [Unsloth GitHub #2428](https://github.com/unslothai/unsloth/issues/2428)
