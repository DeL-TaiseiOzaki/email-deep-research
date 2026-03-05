# Training Error Log Analysis

**Date**: 2026-03-04
**Node**: megatpa-a3meganodeset-4
**GPU**: NVIDIA H100 80GB HBM3 (1x)

---

## Summary

Both training jobs failed with the **exact same root cause**: vLLM's GPU memory allocation check failed because the Unsloth-patched model had already consumed ~14.6 GiB of GPU memory before vLLM tried to initialize the inference engine, leaving only ~64.6 GiB free out of 79.19 GiB. vLLM's default `gpu_memory_utilization=0.9` requires 71.27 GiB, which exceeds the available free memory.

---

## Job 1: Qwen2.5-14B Resume Training (Job ID: 3750)

### Files
- **Log**: `logs/train_qwen25_resume_3750.log`
- **Stderr**: `logs/train_qwen25_resume_3750.err`

### Configuration
- **Model**: `Qwen/Qwen2.5-14B-Instruct` (Qwen2ForCausalLM architecture)
- **wandb run**: `email-agent-008` (resume)
- **Framework**: ART (openpipe-art) + Unsloth 2026.2.1 + vLLM 0.15.1
- **Torch**: 2.9.1+cu128, CUDA Toolkit 12.8
- **Precision**: bfloat16
- **max_seq_len**: 32768
- **tensor_parallel_size**: 1, data_parallel_size: 1
- **Unsloth**: Patched 48 layers (48 QKV, 48 O, 48 MLP)
- **Entry point**: `art_e/train.py` line 201 -> `asyncio.run(run_training(MODELS[args.model]))`

### Timeline
1. **16:08:08** - Job started, pre-flight checks passed (GPU clean: 0MiB used)
2. **16:08:22** - Database generation skipped (already exists)
3. **16:08:33** - wandb initialized, resuming run `email-agent-008`
4. **16:09:22** - vLLM resolved architecture, set max_model_len=32768
5. **16:09:24** - Multiprocessing method overridden to `spawn` (CUDA already initialized)
6. **16:09:41** - vLLM EngineCore_DP0 initialized, then **FAILED**
7. **16:09:42** - Process group cleanup warning
8. **16:09:52** - Job terminated

### Error (Root Cause)
```
ValueError: Free memory on device cuda:0 (64.59/79.19 GiB) on startup is less than
desired GPU memory utilization (0.9, 71.27 GiB). Decrease GPU memory utilization or
reduce GPU memory used by other processes.
```

### Stack Trace (Condensed)
```
art_e/train.py:201 -> asyncio.run(run_training(...))
  -> train.py:133 -> model.register(backend, ...)
    -> art/model.py:797 -> backend._prepare_backend_for_training(...)
      -> art/local/backend.py:327 -> service.start_openai_server(...)
        -> art/unsloth/service.py:461 -> await self.llm
          -> art/vllm/engine.py:34 -> AsyncLLM.from_engine_args(...)
            -> vllm/v1/engine/async_llm.py:257 -> cls(...)
              -> vllm/v1/engine/core_client.py:479 -> launch_core_engines(...)
                -> vllm/v1/engine/utils.py:992 -> wait_for_engine_startup()
                  -> RuntimeError: Engine core initialization failed.

Inner cause (EngineCore_DP0 process):
  vllm/v1/worker/gpu_worker.py:235 -> request_memory(init_snapshot, self.cache_config)
    -> vllm/v1/worker/utils.py:260 -> ValueError (memory insufficient)
```

### Warnings
- `PYTORCH_CUDA_ALLOC_CONF` deprecated, use `PYTORCH_ALLOC_CONF`
- urllib3/chardet version mismatch with requests
- Unsloth import order warning (should be imported before transformers)
- Leaked semaphore resource at shutdown
- `destroy_process_group()` not called before exit

---

## Job 2: Qwen3-14B Thinking Training (Job ID: 3749)

### Files
- **Log**: `logs/train_thinking_3749.log`
- **Stderr**: `logs/train_thinking_3749.err`

### Configuration
- **Model**: `Qwen/Qwen3-14B` (Qwen3ForCausalLM architecture)
- **wandb run**: `email-agent-qwen3-14b-thinking` (resume)
- **Framework**: ART (openpipe-art) + Unsloth 2026.2.1 + vLLM 0.15.1
- **Torch**: 2.9.1+cu128, CUDA Toolkit 12.8
- **Precision**: bfloat16
- **max_seq_len**: 32768
- **tensor_parallel_size**: 1, data_parallel_size: 1
- **Unsloth**: Patched 40 layers (40 QKV, 40 O, 40 MLP)
- **Entry point**: `art_e/train.py` line 170 -> `asyncio.run(run_training(MODELS[args.model]))`

### Timeline
1. **15:39:34** - Job started, pre-flight checks passed (GPU clean: 0MiB used)
2. **15:39:50** - Database generation skipped (already exists)
3. **15:40:02** - wandb initialized, resuming run `email-agent-qwen3-14b-thinking`
4. **15:40:50** - vLLM resolved architecture, set max_model_len=32768
5. **15:40:52** - Multiprocessing method overridden to `spawn`
6. **15:41:10** - vLLM EngineCore_DP0 initialized, then **FAILED**
7. **15:41:11** - Process group cleanup warning
8. **15:41:20** - Job terminated

### Error (Root Cause)
```
ValueError: Free memory on device cuda:0 (64.6/79.19 GiB) on startup is less than
desired GPU memory utilization (0.9, 71.27 GiB). Decrease GPU memory utilization or
reduce GPU memory used by other processes.
```

### Stack Trace (Condensed)
Identical to Job 1 except:
- Entry point line: `train.py:170` (vs 201)
- Process PID: 285266 (vs 287903)
- Timestamps differ

### Warnings
Same warnings as Job 1.

---

## Common Pattern Analysis

### Root Cause: GPU Memory Conflict Between Unsloth and vLLM

Both jobs exhibit the **identical failure pattern**:

1. **GPU starts clean** (0MiB used, confirmed by nvidia-smi)
2. **Unsloth loads and patches the model** into GPU memory (~14.6 GiB consumed for the 14B parameter models)
3. **vLLM engine initialization** runs in a separate process (EngineCore_DP0) and checks available GPU memory
4. **vLLM's memory check fails** because only ~64.6 GiB is free, but 90% utilization target requires 71.27 GiB

### Memory Arithmetic
- Total GPU memory: 79.19 GiB
- Free at vLLM init: ~64.6 GiB
- Memory consumed by Unsloth model: ~14.6 GiB (79.19 - 64.6)
- vLLM required (0.9 * 79.19): 71.27 GiB
- Deficit: ~6.7 GiB

### Both Jobs Have Identical:
- Same node (megatpa-a3meganodeset-4)
- Same GPU (H100 80GB)
- Same framework versions (Unsloth 2026.2.1, vLLM 0.15.1, Torch 2.9.1)
- Same error message and stack trace path
- Same ~14.6 GiB memory consumed before vLLM init

---

## Potential Fixes

1. **Lower `gpu_memory_utilization`** in vLLM config to 0.7 or 0.8 (e.g., `gpu_memory_utilization=0.75` would require 59.4 GiB, within the ~64.6 GiB available)
2. **Use tensor parallelism** with 2+ GPUs to split model memory (SLURM script only requests GPU index 0)
3. **Use quantization** (e.g., AWQ/GPTQ 4-bit) to reduce Unsloth's model footprint
4. **Reduce `max_seq_len`** from 32768 to a smaller value to reduce KV cache requirements
5. **Ensure Unsloth shares memory** with vLLM rather than loading the model separately - this may require ART framework configuration changes
6. **Free the Unsloth model from GPU** before vLLM initialization if the framework supports sequential loading
