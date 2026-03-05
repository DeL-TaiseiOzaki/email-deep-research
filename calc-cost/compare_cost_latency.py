"""
Cost & Latency Comparison: Self-hosted Qwen2.5-14B (LoRA) vs GPT-5.1 API

Calculates per-query and volume-based costs and latency estimates
using actual token statistics from the ART-E training history.
"""

import json
import statistics
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Pricing (per 1M tokens)
# ---------------------------------------------------------------------------
@dataclass
class APIPricing:
    name: str
    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    cached_input_per_1m: float  # USD per 1M cached input tokens
    ttft_seconds: float  # Time to first token (seconds)
    output_tokens_per_sec: float  # Output generation speed (tokens/s)


GPT_5_1 = APIPricing(
    name="GPT-5.1",
    input_per_1m=1.25,
    output_per_1m=10.00,
    cached_input_per_1m=0.125,  # 90% discount
    ttft_seconds=0.20,  # Sub-200ms in non-reasoning mode (tool calling)
    output_tokens_per_sec=140.0,  # 120-160 t/s range, non-reasoning mode
)

GPT_5_1_BATCH = APIPricing(
    name="GPT-5.1 (Batch API)",
    input_per_1m=0.625,  # 50% batch discount
    output_per_1m=5.00,  # 50% batch discount
    cached_input_per_1m=0.0625,  # 50% of cached price
    ttft_seconds=0.0,  # Async batch - no real-time latency
    output_tokens_per_sec=0.0,  # Async batch
)

GPT_5_MINI = APIPricing(
    name="GPT-5-mini",
    input_per_1m=0.25,
    output_per_1m=2.00,
    cached_input_per_1m=0.025,  # 90% discount
    ttft_seconds=0.30,
    output_tokens_per_sec=150.0,
)

GPT_5_NANO = APIPricing(
    name="GPT-5-nano",
    input_per_1m=0.05,
    output_per_1m=0.40,
    cached_input_per_1m=0.005,
    ttft_seconds=0.15,
    output_tokens_per_sec=200.0,
)


@dataclass
class SelfHostedConfig:
    name: str
    gpu_type: str
    gpu_cost_per_hour: float  # USD/hour (cloud GPU pricing)
    num_gpus: int
    prompt_throughput: float  # tokens/s
    generation_throughput: float  # tokens/s


# Cloud GPU costs (H100 80GB — actual GPU used for training)
SELF_HOSTED_H100 = SelfHostedConfig(
    name="Qwen2.5-14B-4bit + LoRA (self-hosted)",
    gpu_type="H100 80GB",
    gpu_cost_per_hour=3.00,  # GCP A3 High on-demand ($3.00/hr)
    num_gpus=1,
    prompt_throughput=2848.0,  # From vLLM logs (measured on H100)
    generation_throughput=109.5,  # From vLLM logs (measured on H100)
)

SELF_HOSTED_H100_SPOT = SelfHostedConfig(
    name="Qwen2.5-14B-4bit + LoRA (self-hosted, spot)",
    gpu_type="H100 80GB (spot)",
    gpu_cost_per_hour=2.25,  # GCP A3 High spot/preemptible (~$2.25/hr)
    num_gpus=1,
    prompt_throughput=2848.0,  # Same H100 throughput
    generation_throughput=109.5,  # Same H100 throughput
)


# ---------------------------------------------------------------------------
# Load token statistics from training history
# ---------------------------------------------------------------------------
def load_token_stats(history_path: str) -> dict:
    lines = Path(history_path).read_text().strip().split("\n")
    data = [json.loads(line) for line in lines]

    train_entries = [d for d in data if "train/prompt_tokens" in d]

    prompt_tokens = [d["train/prompt_tokens"] for d in train_entries]
    completion_tokens = [d["train/completion_tokens"] for d in train_entries]
    num_turns = [d["train/num_turns"] for d in train_entries]
    durations = [d["train/duration"] for d in train_entries]

    return {
        "count": len(train_entries),
        "prompt_tokens": {
            "mean": statistics.mean(prompt_tokens),
            "median": statistics.median(prompt_tokens),
            "p10": sorted(prompt_tokens)[int(len(prompt_tokens) * 0.1)],
            "p90": sorted(prompt_tokens)[int(len(prompt_tokens) * 0.9)],
        },
        "completion_tokens": {
            "mean": statistics.mean(completion_tokens),
            "median": statistics.median(completion_tokens),
            "p10": sorted(completion_tokens)[int(len(completion_tokens) * 0.1)],
            "p90": sorted(completion_tokens)[int(len(completion_tokens) * 0.9)],
        },
        "num_turns": {
            "mean": statistics.mean(num_turns),
        },
        "duration": {
            "mean": statistics.mean(durations),
        },
    }


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------
def calc_api_cost(
    pricing: APIPricing,
    input_tokens: float,
    output_tokens: float,
    cache_hit_rate: float = 0.0,
) -> float:
    """Calculate cost for a single query."""
    cached_input = input_tokens * cache_hit_rate
    uncached_input = input_tokens * (1 - cache_hit_rate)

    cost = (
        uncached_input / 1_000_000 * pricing.input_per_1m
        + cached_input / 1_000_000 * pricing.cached_input_per_1m
        + output_tokens / 1_000_000 * pricing.output_per_1m
    )
    return cost


def calc_api_latency(
    pricing: APIPricing,
    input_tokens: float,
    output_tokens: float,
    num_turns: float,
) -> float:
    """Estimate total latency for a multi-turn query."""
    # Each turn: TTFT + generation time
    per_turn_generation = output_tokens / num_turns / pricing.output_tokens_per_sec
    per_turn_latency = pricing.ttft_seconds + per_turn_generation
    return per_turn_latency * num_turns


def calc_self_hosted_cost(
    config: SelfHostedConfig,
    input_tokens: float,
    output_tokens: float,
) -> float:
    """Calculate cost for a single query on self-hosted GPU."""
    prefill_time = input_tokens / config.prompt_throughput
    gen_time = output_tokens / config.generation_throughput
    total_time_seconds = prefill_time + gen_time
    cost_per_second = config.gpu_cost_per_hour * config.num_gpus / 3600
    return total_time_seconds * cost_per_second


def calc_self_hosted_latency(
    config: SelfHostedConfig,
    input_tokens: float,
    output_tokens: float,
) -> float:
    """Estimate latency for self-hosted inference."""
    prefill_time = input_tokens / config.prompt_throughput
    gen_time = output_tokens / config.generation_throughput
    return prefill_time + gen_time


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def main():
    history_path = (
        Path(__file__).parent.parent
        / ".art/email_agent/models/email-agent-008/history.jsonl"
    )
    stats = load_token_stats(str(history_path))

    avg_input = stats["prompt_tokens"]["mean"]
    avg_output = stats["completion_tokens"]["mean"]
    avg_turns = stats["num_turns"]["mean"]
    actual_duration = stats["duration"]["mean"]

    print("=" * 72)
    print("  Cost & Latency Comparison: Self-Hosted vs GPT-5.1 API")
    print("=" * 72)
    print()
    print("--- Token Statistics (from training history) ---")
    print(f"  Samples:              {stats['count']}")
    print(f"  Avg input tokens:     {avg_input:,.0f}")
    print(f"  Avg output tokens:    {avg_output:,.0f}")
    print(f"  Avg turns per query:  {avg_turns:.1f}")
    print(f"  Avg rollout duration: {actual_duration:.1f}s (measured)")
    print()

    # -----------------------------------------------------------------------
    # Per-query comparison
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("  PER-QUERY COST & LATENCY")
    print("=" * 72)

    models = [
        ("GPT-5.1", GPT_5_1),
        ("GPT-5.1 (80% cache)", GPT_5_1),
        ("GPT-5.1 Batch", GPT_5_1_BATCH),
        ("GPT-5-mini", GPT_5_MINI),
        ("GPT-5-mini (80% cache)", GPT_5_MINI),
        ("GPT-5-nano", GPT_5_NANO),
    ]

    self_hosted_configs = [
        SELF_HOSTED_H100,
        SELF_HOSTED_H100_SPOT,
    ]

    print()
    print(f"{'Model':<30} {'Cost/query':>12} {'Latency':>10} {'Notes'}")
    print("-" * 72)

    for name, pricing in models:
        cache_rate = 0.8 if "cache" in name else 0.0
        cost = calc_api_cost(pricing, avg_input, avg_output, cache_rate)
        is_batch = pricing.output_tokens_per_sec == 0.0
        if is_batch:
            latency_str = "  async  "
            notes = "50% discount, async"
        else:
            latency = calc_api_latency(pricing, avg_input, avg_output, avg_turns)
            latency_str = f"{latency:>7.1f}s"
            notes = f"cache={cache_rate:.0%}" if cache_rate > 0 else ""
        print(f"  {name:<28} ${cost:>10.5f}  {latency_str}  {notes}")

    print()
    for config in self_hosted_configs:
        cost = calc_self_hosted_cost(config, avg_input, avg_output)
        latency = calc_self_hosted_latency(config, avg_input, avg_output)
        label = f"Self-hosted ({config.gpu_type})"
        notes = f"${config.gpu_cost_per_hour:.2f}/hr GPU"
        print(f"  {label:<28} ${cost:>10.5f}  {latency:>7.1f}s  {notes}")

    # Add measured actual
    print()
    actual_cost_h100 = actual_duration * SELF_HOSTED_H100.gpu_cost_per_hour / 3600
    print(
        f"  {'Actual (measured, training)':<28} ${actual_cost_h100:>10.5f}  {actual_duration:>7.1f}s  includes tool calls"
    )

    # -----------------------------------------------------------------------
    # Volume comparison
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  VOLUME COST COMPARISON (monthly)")
    print("=" * 72)

    volumes = [100, 1_000, 10_000, 100_000]

    print()
    header = f"{'Volume':>10}"
    col_names = [
        "GPT-5.1",
        "GPT-5.1 cached",
        "GPT-5.1 batch",
        "GPT-5-mini",
        "GPT-5-mini cached",
        "H100 self",
    ]
    for c in col_names:
        header += f"  {c:>14}"
    print(header)
    print("-" * (12 + 16 * len(col_names)))

    for vol in volumes:
        costs = [
            calc_api_cost(GPT_5_1, avg_input, avg_output, 0.0) * vol,
            calc_api_cost(GPT_5_1, avg_input, avg_output, 0.8) * vol,
            calc_api_cost(GPT_5_1_BATCH, avg_input, avg_output, 0.0) * vol,
            calc_api_cost(GPT_5_MINI, avg_input, avg_output, 0.0) * vol,
            calc_api_cost(GPT_5_MINI, avg_input, avg_output, 0.8) * vol,
        ]

        # Self-hosted: GPU running time cost
        latency_per_q = calc_self_hosted_latency(
            SELF_HOSTED_H100, avg_input, avg_output
        )
        total_gpu_hours = latency_per_q * vol / 3600
        self_cost = total_gpu_hours * SELF_HOSTED_H100.gpu_cost_per_hour
        costs.append(self_cost)

        row = f"{vol:>10,}"
        for c in costs:
            if c < 1:
                row += f"  ${c:>13.4f}"
            else:
                row += f"  ${c:>13.2f}"

        print(row)

    # -----------------------------------------------------------------------
    # Break-even analysis
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  BREAK-EVEN ANALYSIS (Self-hosted H100 vs API)")
    print("=" * 72)
    print()

    gpu_monthly_cost = SELF_HOSTED_H100.gpu_cost_per_hour * 24 * 30
    print(f"  H100 monthly cost (24/7): ${gpu_monthly_cost:,.0f}")
    print()

    for name, pricing, cache in [
        ("GPT-5.1", GPT_5_1, 0.0),
        ("GPT-5.1 (80% cache)", GPT_5_1, 0.8),
        ("GPT-5-mini", GPT_5_MINI, 0.0),
        ("GPT-5-mini (80% cache)", GPT_5_MINI, 0.8),
    ]:
        api_cost_per_query = calc_api_cost(pricing, avg_input, avg_output, cache)
        if api_cost_per_query > 0:
            breakeven = gpu_monthly_cost / api_cost_per_query
            print(f"  vs {name:<26} breakeven: {breakeven:>10,.0f} queries/month")

    # -----------------------------------------------------------------------
    # Latency comparison summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  LATENCY SUMMARY")
    print("=" * 72)
    print()
    print(f"  {'Model':<30} {'Per-query':>10} {'100 serial':>12} {'100 parallel':>14}")
    print("-" * 72)

    for name, pricing in [("GPT-5.1", GPT_5_1), ("GPT-5-mini", GPT_5_MINI)]:
        lat = calc_api_latency(pricing, avg_input, avg_output, avg_turns)
        print(f"  {name:<30} {lat:>8.1f}s  {lat * 100:>10.0f}s  {lat * 10:>12.0f}s")

    for config in self_hosted_configs:
        lat = calc_self_hosted_latency(config, avg_input, avg_output)
        label = f"Self-hosted ({config.gpu_type})"
        print(f"  {label:<30} {lat:>8.1f}s  {lat * 100:>10.0f}s  {lat * 10:>12.0f}s")

    # Actual measured
    print(
        f"  {'Actual (with tool calls)':<30} {actual_duration:>8.1f}s  {actual_duration * 100:>10.0f}s  {actual_duration * 10:>12.0f}s"
    )

    print()
    print("Notes:")
    print("  - 'Per-query' = single query E2E latency (LLM inference only)")
    print("  - '100 serial' = 100 queries processed one after another")
    print("  - '100 parallel' = 100 queries with 10x concurrency")
    print("  - Actual measured includes tool execution time (DB queries, etc.)")
    print("  - GPT-5.1 latency is estimated from GPT-5/5.2 benchmarks")
    print(
        "  - Self-hosted throughput from vLLM logs on H100 80GB (batch of 10 concurrent)"
    )


if __name__ == "__main__":
    main()
