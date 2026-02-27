# OpenPipe ART Multi-GPU Configuration Research

**Date:** 2026-02-26
**Package Version:** openpipe-art 0.5.4
**Source:** Direct source code analysis of installed package

## Architecture Overview

ART has **three service backends** for model training, each with different multi-GPU capabilities:

| Service | Training Engine | Inference Engine | Multi-GPU Support |
|---------|----------------|-----------------|-------------------|
| `UnslothService` | Unsloth + GRPOTrainer | vLLM v0 (in-process) | Single GPU only (shared GPU) |
| `DecoupledUnslothService` | Unsloth + GRPOTrainer | vLLM v1 (separate process) | vLLM can use tensor_parallel; Training on single GPU |
| `TorchtuneService` | Torchtune (distributed) | vLLM v1 (separate process) | Full multi-GPU: TP + FSDP for training, TP for inference |

## Service Selection Logic

In `LocalBackend._get_service()` (backend.py:138-174):

```python
config = get_model_config(base_model, output_dir, model._internal_config)
if config.get("torchtune_args") is not None:
    service_class = TorchtuneService           # Multi-GPU full finetuning
elif config.get("_decouple_vllm_and_unsloth", False):
    service_class = DecoupledUnslothService     # Decoupled: vLLM v1 + Unsloth
else:
    service_class = UnslothService             # Default: single GPU, vLLM v0
```

## LocalBackend Constructor

```python
class LocalBackend(Backend):
    def __init__(self, *, in_process: bool = False, path: str | None = None) -> None:
```

**Only two parameters:**
- `in_process` (bool): Run service in same process (default: False = subprocess)
- `path` (str | None): Storage directory (default: `{repo_root}/.art`)

**No direct GPU configuration on LocalBackend.** All GPU config goes through `TrainableModel._internal_config`.

## Multi-GPU Configuration Methods

### Method 1: TorchtuneService (Full Multi-GPU Training + Inference)

This is the **only way to do true multi-GPU training** in ART. It uses:
- **Training:** torchtune's `FullFinetuneRecipeDistributed` with `torch.distributed` (FSDP + optional TP)
- **Inference:** vLLM v1 `AsyncLLM` with tensor parallelism
- **GPU split:** ALL GPUs used for both (time-multiplexed via sleep/wake)

```python
model = art.TrainableModel(
    name="my-model",
    project="my-project",
    base_model="Qwen/Qwen3-32B",
    _internal_config=art.dev.InternalModelConfig(
        torchtune_args=art.dev.TorchtuneArgs(
            model="qwen3_32b",
            model_type="QWEN3",
            tensor_parallel_dim=4,       # Tensor parallelism degree for training
            context_parallel_dim=1,      # Context parallelism degree
            enable_activation_offloading=False,
            async_weight_syncing=False,
        ),
        engine_args=art.dev.EngineArgs(
            tensor_parallel_size=4,      # Tensor parallelism for vLLM inference
            gpu_memory_utilization=0.85,
        ),
    ),
)
```

**Key: TorchtuneService uses `torch.cuda.device_count()` for nproc_per_node** (torchtune/service.py:198):
```python
"--nproc-per-node",
str(torch.cuda.device_count()),  # Uses ALL available GPUs
```

Data parallelism is computed as:
```python
dp = world_size // (tp * cp)  # Data parallel = total_GPUs / (tensor_parallel * context_parallel)
```

### Method 2: DecoupledUnslothService (Multi-GPU Inference Only)

Uses vLLM v1 with tensor parallelism for inference, but Unsloth training remains single-GPU.

```python
model = art.TrainableModel(
    name="my-model",
    project="my-project",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    _internal_config=art.dev.InternalModelConfig(
        _decouple_vllm_and_unsloth=True,
        engine_args=art.dev.EngineArgs(
            tensor_parallel_size=2,      # vLLM inference across 2 GPUs
            gpu_memory_utilization=0.85,
            enable_sleep_mode=True,
        ),
        init_args=art.dev.InitArgs(
            gpu_memory_utilization=0.75,
            load_in_4bit=True,
            max_seq_length=32768,
        ),
    ),
)
```

### Method 3: Default UnslothService (Single GPU)

Default mode. Uses Unsloth's `fast_inference=True` which creates an in-process vLLM v0 engine.
Both training and inference share a single GPU via sleep/wake mechanism.

```python
model = art.TrainableModel(
    name="my-model",
    project="my-project",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    _internal_config=art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            gpu_memory_utilization=0.75,
        ),
    ),
)
```

## 8x H100 Configuration Examples

### Example A: Qwen3-32B Full Finetuning on 8x H100 (TorchtuneService)

```python
import art

backend = art.LocalBackend()

model = art.TrainableModel(
    name="qwen3-32b-agent",
    project="email-research",
    base_model="Qwen/Qwen3-32B",
    _internal_config=art.dev.InternalModelConfig(
        torchtune_args=art.dev.TorchtuneArgs(
            model="qwen3_32b",
            model_type="QWEN3",
            tensor_parallel_dim=1,              # No TP needed for 32B on 8x H100
            context_parallel_dim=1,
            enable_activation_offloading=False,
            async_weight_syncing=True,          # Async weight sync for faster iteration
        ),
        engine_args=art.dev.EngineArgs(
            tensor_parallel_size=8,             # Use all 8 GPUs for inference
            gpu_memory_utilization=0.90,
            enable_sleep_mode=True,
        ),
        trainer_args=art.dev.TrainerArgs(
            learning_rate=5e-6,
            max_grad_norm=0.1,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=2,
        ),
    ),
)

await model.register(backend)
```

With this config:
- **Inference:** 8-way tensor parallelism via vLLM v1 (all 8 H100s)
- **Training:** 8-way FSDP data parallelism via torchtune (all 8 H100s, dp = 8 / (1 * 1) = 8)
- **GPU time-multiplexing:** vLLM workers sleep during training, wake for inference

### Example B: LoRA Training on 8x H100 (DecoupledUnslothService)

```python
model = art.TrainableModel(
    name="qwen2.5-7b-agent",
    project="email-research",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    _internal_config=art.dev.InternalModelConfig(
        _decouple_vllm_and_unsloth=True,
        engine_args=art.dev.EngineArgs(
            tensor_parallel_size=8,
            gpu_memory_utilization=0.85,
            enable_sleep_mode=True,
        ),
        init_args=art.dev.InitArgs(
            gpu_memory_utilization=0.75,
            load_in_4bit=True,
            max_seq_length=32768,
        ),
        peft_args=art.dev.PeftArgs(
            r=16,
            lora_alpha=32,
        ),
        trainer_args=art.dev.TrainerArgs(
            learning_rate=5e-6,
            max_grad_norm=0.1,
        ),
    ),
)
```

Note: Training still runs on GPU 0 only. Multi-GPU benefits are only for inference throughput.

## Key EngineArgs for vLLM Multi-GPU (art.dev.EngineArgs)

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor_parallel_size` | int | Number of GPUs for tensor parallelism in vLLM |
| `pipeline_parallel_size` | int | Number of GPUs for pipeline parallelism |
| `gpu_memory_utilization` | float | Fraction of GPU memory to use (0.0-1.0) |
| `enable_sleep_mode` | bool | Enable GPU memory sharing between training and inference |
| `max_model_len` | int | Maximum sequence length |
| `enforce_eager` | bool | Disable CUDA graph compilation |
| `enable_prefix_caching` | bool | Enable prefix caching for faster inference |

## Key TorchtuneArgs for Multi-GPU Training

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | Torchtune model name (e.g., "qwen3_32b") |
| `model_type` | str | Model type for checkpointing (e.g., "QWEN3") |
| `tensor_parallel_dim` | int | TP degree for training (default: 1) |
| `context_parallel_dim` | int | CP degree for training (default: 1) |
| `enable_activation_offloading` | bool | Offload activations to CPU |
| `async_weight_syncing` | bool | Async weight sync between training and inference |

## Critical Notes

1. **No `trainer_gpu_ids` parameter exists** in ART. GPU selection is controlled via:
   - `CUDA_VISIBLE_DEVICES` environment variable
   - `torch.cuda.device_count()` (TorchtuneService auto-detects available GPUs)

2. **Time-multiplexing, NOT simultaneous:** Both DecoupledUnslothService and TorchtuneService
   use vLLM's sleep/wake mechanism. During training, vLLM workers are put to sleep to free
   GPU memory. After training completes, workers wake up and reload weights.

3. **TorchtuneService = full finetuning only** (no LoRA). It uses FSDP to shard the full model
   across all GPUs.

4. **UnslothService (default) = LoRA + single GPU**. It uses Unsloth's integrated vLLM v0 engine.

5. **DecoupledUnslothService = LoRA + multi-GPU inference**. Training is still single-GPU via
   Unsloth, but inference benefits from multi-GPU tensor parallelism.

6. **CUDA_VISIBLE_DEVICES** is the standard way to control which GPUs are used:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py  # Use GPUs 0-3
   ```

## Supported Models for TorchtuneService (Multi-GPU Full Finetuning)

Qwen2, Qwen2.5, Qwen3, Llama2/3/3.1/3.2/3.3/4, Gemma, Gemma2, Mistral, Phi3, Phi4
