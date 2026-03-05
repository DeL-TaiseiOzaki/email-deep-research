# Self-hosted Qwen2.5-14B vs GPT-5.1 API: Cost & Latency Comparison

## 1. Training Overview

| Item | Value |
|:-----|:------|
| Base model | Qwen/Qwen2.5-14B-Instruct (4-bit quantized) |
| Method | LoRA (r=8, α=16) + GRPO reinforcement learning |
| Total steps | 174 steps / 5.7 hours |
| Time per step | **avg 119s** (median 115s, range 82-325s) |
| Step composition | 12 groups × 4 trajectories = 48 rollouts/step |
| Best accuracy | **91.7%** at step 158 (reward=1.213) |
| GPU | 1× GPU, tensor_parallel=1, 90% VRAM utilization |

## 2. Per-Query Token Usage (Measured)

| Metric | Mean | Median | P10 | P90 |
|:-------|-----:|-------:|----:|----:|
| Input tokens | 5,867 | 5,400 | 4,024 | 8,406 |
| Output tokens | 78 | 59 | 53 | 75 |
| Turns | 4.1 | — | — | — |
| E2E duration | 17.5s | — | — | — |

> Note: Input tokens are cumulative across all turns. E2E includes tool execution (DB search etc.).

## 3. Per-Query Cost & Latency

| # | Model | Cost | Latency | Notes |
|:-:|:------|-----:|--------:|:------|
| 1 | GPT-5.1 | $0.00811 | 1.4s | Standard pricing |
| 2 | GPT-5.1 + 80% cache | $0.00283 | 1.4s | Cache: 90% discount on input |
| 3 | GPT-5.1 Batch API | $0.00406 | async | 50% discount, not real-time |
| 4 | GPT-5-mini | $0.00162 | 1.8s | Smaller model |
| 5 | GPT-5-mini + 80% cache | $0.00057 | 1.8s | **Cheapest API option** |
| 6 | GPT-5-nano | $0.00032 | 1.0s | Accuracy unverified |
| 7 | **Self-hosted A100** | **$0.00116** | **2.8s** | $1.50/hr GPU |
| 8 | Self-hosted L4 | $0.00072 | 5.2s | $0.50/hr GPU |

```
Cost ranking (low → high):
  nano($0.0003) < mini+cache($0.0006) < L4($0.0007) < A100($0.0012)
  < mini($0.0016) < 5.1+cache($0.0028) < 5.1-nano($0.0003) < 5.1($0.0081)

Latency ranking (fast → slow):
  nano(1.0s) < GPT-5.1(1.4s) < mini(1.8s) < A100(2.8s) < L4(5.2s)
```

## 4. Monthly Cost at Scale

| Queries/month | GPT-5.1 | GPT-5.1 +cache | GPT-5.1 batch | mini +cache | A100 self |
|--------------:|--------:|---------------:|--------------:|------------:|----------:|
| 100 | $0.81 | $0.28 | $0.41 | $0.06 | $0.12 |
| 1,000 | $8.11 | $2.83 | $4.06 | $0.57 | $1.16 |
| 10,000 | $81 | $28 | $41 | $5.67 | $12 |
| 100,000 | $811 | $283 | $406 | $57 | $116 |

## 5. Break-Even: Self-hosted A100 ($1,080/mo, 24/7) vs API

| Comparison | Break-even point |
|:-----------|:-----------------|
| vs GPT-5.1 | **133K** queries/month |
| vs GPT-5.1 + cache | **381K** queries/month |
| vs GPT-5-mini | **665K** queries/month |
| vs GPT-5-mini + cache | **1.9M** queries/month |

> Below the break-even → API is cheaper. Above → self-hosted is cheaper.

## 6. Latency Breakdown

```
Actual E2E per query: 17.5s
├── LLM inference:     2-3s  (14-17%)  ← only this part changes between options
└── Tool execution:   ~15s   (83-86%)  ← DB search, email read, answer eval
```

| Scenario | GPT-5.1 | mini | A100 self | Actual (measured) |
|:---------|--------:|-----:|----------:|------------------:|
| 1 query | 1.4s | 1.8s | 2.8s | 17.5s |
| 100 serial | 138s | 175s | 277s | 1,752s |
| 100 parallel (×10) | 14s | 18s | 28s | 175s |

## 7. Decision Matrix

| Priority | Best choice | Why |
|:---------|:------------|:----|
| ⚡ Lowest latency | GPT-5.1 | 1.4s — 2× faster than A100 |
| 💰 Lowest cost (low volume) | GPT-5-mini + cache | $0.0006/query |
| 💰 Lowest cost (high volume) | Self-hosted A100 | Wins above 133K queries/month |
| 🎯 Best accuracy | Self-hosted + LoRA | Fine-tuned to 91.7% on domain data |
| 📦 Batch processing | GPT-5.1 Batch API | 50% off, no latency requirement |
| 🔧 Easiest to operate | GPT-5.1 API | Zero infra management |

## 8. Key Takeaways

1. **Tool execution dominates latency** — LLM inference is only 14-17% of E2E time. Switching inference backend saves at most ~1.4s per query.

2. **GPT-5.1 accuracy is unverified** — The fine-tuned Qwen 14B reached 91.7%. GPT-5.1 without fine-tuning may or may not match this on the email search domain.

3. **Cache is the biggest cost lever** — 80% cache hit rate cuts GPT-5.1 cost by 65% ($0.0081 → $0.0028), making it closer to self-hosted.

4. **Self-hosted has hidden costs** — GPU pricing alone doesn't include infra management, monitoring, scaling, and engineering time.

## Appendix: Pricing Reference

| Model | Input/1M | Output/1M | Cached Input/1M | Context |
|:------|:--------:|:---------:|:---------------:|:-------:|
| GPT-5.1 | $1.25 | $10.00 | $0.125 | 400K |
| GPT-5.1 Batch | $0.625 | $5.00 | $0.0625 | 400K |
| GPT-5-mini | $0.25 | $2.00 | $0.025 | 400K |
| GPT-5-nano | $0.05 | $0.40 | $0.005 | — |

Self-hosted vLLM throughput (measured):
- Prompt: avg 2,848 tokens/s (max 13,109)
- Generation: avg 109.5 tokens/s (max 190.8)
- GPU KV cache usage: avg 2.7% (max 15.8%)
