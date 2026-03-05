"""
Benchmark: GPT-5.1 vs GPT-5-mini vs ART-E (Self-hosted Qwen on H100)
Runs N test tasks per model and records per-task metrics.

Usage:
  # API models only (no GPU needed)
  uv run python calc-cost/benchmark_latency.py --models api

  # ART-E only (requires vLLM server on localhost:8000)
  uv run python calc-cost/benchmark_latency.py --models arte

  # All 3 models
  uv run python calc-cost/benchmark_latency.py --models all

  # Custom task count
  uv run python calc-cost/benchmark_latency.py --models all --limit 10
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import art
from dotenv import load_dotenv

from art_e.data.local_email_db import generate_database
from art_e.data.query_iterators import load_synthetic_queries
from art_e.project_types import ProjectPolicyConfig
from art_e.rollout import rollout

load_dotenv()

RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"

# H100 pricing for cost estimation
H100_COST_PER_HOUR = 2.86

# GPT pricing per 1M tokens
GPT_PRICING = {
    "GPT-5": {"input": 1.25, "output": 10.00},
    "GPT-5.1": {"input": 1.25, "output": 10.00},
    "GPT-5-mini": {"input": 0.25, "output": 2.00},
}


def calc_api_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = GPT_PRICING.get(model_name)
    if pricing is None:
        return 0.0
    return (
        prompt_tokens / 1e6 * pricing["input"]
        + completion_tokens / 1e6 * pricing["output"]
    )


def calc_self_hosted_cost(duration_seconds: float) -> float:
    return duration_seconds * H100_COST_PER_HOUR / 3600


async def run_single_model(
    model: art.Model,
    scenarios: list,
) -> list[dict]:
    """Run all scenarios through a single model, return per-task results."""
    results = []
    for i, scenario in enumerate(scenarios):
        print(f"  [{model.name}] Task {i + 1}/{len(scenarios)} (id={scenario.id})...")
        try:
            traj = await rollout(model, scenario)
            m = traj.metrics

            # Estimate cost
            if model.name in GPT_PRICING:
                cost = calc_api_cost(
                    model.name, m.get("prompt_tokens", 0), m.get("completion_tokens", 0)
                )
            else:
                cost = calc_self_hosted_cost(m.get("duration", 0))

            result = {
                "model": model.name,
                "scenario_id": scenario.id,
                "question": scenario.question[:80],
                # Timing
                "duration": m.get("duration", 0),
                # Tokens
                "prompt_tokens": m.get("prompt_tokens", 0),
                "completion_tokens": m.get("completion_tokens", 0),
                "total_tokens": m.get("prompt_tokens", 0)
                + m.get("completion_tokens", 0),
                # Tool calls
                "num_turns": m.get("num_turns", 0),
                # Accuracy
                "answer_correct": m.get("answer_correct", 0),
                "sources_correct": m.get("sources_correct", 0),
                "reward": traj.reward,
                # Cost
                "cost_usd": cost,
            }
            results.append(result)

            print(
                f"    -> {m.get('duration', 0):.1f}s | "
                f"turns={m.get('num_turns', 0)} | "
                f"tokens={m.get('prompt_tokens', 0)}+{m.get('completion_tokens', 0)} | "
                f"correct={m.get('answer_correct', 0)} | "
                f"cost=${cost:.5f}"
            )
        except Exception as e:
            print(f"    -> ERROR: {e}")
            results.append(
                {
                    "model": model.name,
                    "scenario_id": scenario.id,
                    "question": scenario.question[:80],
                    "duration": None,
                    "error": str(e),
                }
            )
    return results


def safe_mean(values: list) -> float:
    return statistics.mean(values) if values else 0.0


def print_summary(all_results: list[dict]) -> None:
    model_names = ["GPT-5", "GPT-5.1", "GPT-5-mini", "ART-E"]
    present = [n for n in model_names if any(r["model"] == n for r in all_results)]

    if not present:
        return

    # Header
    print(f"\n{'=' * 80}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 80}")

    # Per-model summary table
    col_w = 14
    header = f"{'Metric':<24}"
    for name in present:
        header += f"{name:>{col_w}}"
    print(header)
    print("-" * (24 + col_w * len(present)))

    def get_valid(model_name: str, key: str) -> list:
        return [
            r[key]
            for r in all_results
            if r["model"] == model_name and r.get(key) is not None
        ]

    rows = [
        ("Avg Latency (s)", "duration", ".1f"),
        ("Avg Prompt Tokens", "prompt_tokens", ",.0f"),
        ("Avg Completion Tokens", "completion_tokens", ",.0f"),
        ("Avg Total Tokens", "total_tokens", ",.0f"),
        ("Avg Tool Calls", "num_turns", ".1f"),
        ("Accuracy", "answer_correct", ".0%"),
        ("Source Accuracy", "sources_correct", ".0%"),
        ("Avg Reward", "reward", ".2f"),
        ("Avg Cost / query", "cost_usd", ".5f"),
        ("Cost / 1K queries", "cost_usd", None),  # special
    ]

    for label, key, fmt in rows:
        row = f"{label:<24}"
        for name in present:
            vals = get_valid(name, key)
            if not vals:
                row += f"{'N/A':>{col_w}}"
                continue
            avg = safe_mean(vals)
            if label == "Cost / 1K queries":
                row += f"{'${:.2f}'.format(avg * 1000):>{col_w}}"
            elif label == "Avg Cost / query":
                row += f"{'${:.5f}'.format(avg):>{col_w}}"
            elif "%" in (fmt or ""):
                row += f"{f'{avg:.0%}':>{col_w}}"
            else:
                row += f"{f'{avg:{fmt}}':>{col_w}}"
        print(row)

    n_tasks = len(get_valid(present[0], "duration"))
    print(f"\nn_tasks = {n_tasks}")

    # Per-task detail table
    print(f"\n{'=' * 80}")
    print("PER-TASK DETAIL")
    print(f"{'=' * 80}")
    scenario_ids = sorted(set(r["scenario_id"] for r in all_results))
    header = f"{'Task ID':>8}"
    for name in present:
        header += f"  {'dur(s)':>7} {'tok':>6} {'ok':>3}"
    print(header)
    print("-" * len(header))

    for sid in scenario_ids:
        row = f"{sid:>8}"
        for name in present:
            task = next(
                (
                    r
                    for r in all_results
                    if r["model"] == name and r["scenario_id"] == sid
                ),
                None,
            )
            if task and task.get("duration") is not None:
                row += f"  {task['duration']:>7.1f} {task.get('total_tokens', 0):>6} {'Y' if task.get('answer_correct') else 'N':>3}"
            else:
                err = (task or {}).get("error", "N/A")
                row += f"  {'ERR':>7} {'':>6} {'':>3}"
        print(row)


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GPT-5.1 vs GPT-5-mini vs ART-E"
    )
    parser.add_argument(
        "--models",
        choices=["api", "arte", "all"],
        default="all",
        help="Which models to benchmark",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of tasks")
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL for ART-E",
    )
    args = parser.parse_args()

    generate_database()

    # Load test scenarios (same tasks for all models)
    scenarios = load_synthetic_queries(split="test", limit=args.limit)
    print(f"Loaded {len(scenarios)} test scenarios")
    print(f"Scenario IDs: {[s.id for s in scenarios]}")

    api = art.LocalAPI()
    all_results: list[dict] = []

    # --- API models ---
    if args.models in ("api", "all"):
        for model_name, litellm_id in [
            ("GPT-5", "openai/gpt-5"),
            ("GPT-5.1", "openai/gpt-5.1"),
            ("GPT-5-mini", "openai/gpt-5-mini"),
        ]:
            print(f"\n{'=' * 50}")
            print(f"Benchmarking: {model_name}")
            print(f"{'=' * 50}")
            model = art.Model(
                name=model_name,
                project="email_agent",
                config=ProjectPolicyConfig(
                    litellm_model_name=litellm_id,
                    use_tools=True,
                    log_to_openpipe=False,
                ),
            )
            await model.register(api)
            results = await run_single_model(model, scenarios)
            all_results.extend(results)

    # --- ART-E (self-hosted, OpenPipe/art-e-008) ---
    if args.models in ("arte", "all"):
        print(f"\n{'=' * 50}")
        print("Benchmarking: OpenPipe/art-e-008 (Qwen2.5-14B + LoRA on H100)")
        print(f"{'=' * 50}")
        model = art.Model(
            name="ART-E",
            project="email_agent",
            config=ProjectPolicyConfig(
                litellm_model_name="hosted_vllm/OpenPipe/art-e-008",
                use_tools=True,
                log_to_openpipe=False,
            ),
        )
        model.base_url = args.vllm_url
        model.api_key = "default"
        await model.register(api)
        results = await run_single_model(model, scenarios)
        all_results.extend(results)

    # --- Save results ---
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to: {RESULTS_PATH}")

    # --- Print summary ---
    print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
