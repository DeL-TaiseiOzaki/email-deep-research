"""
Simple bar chart from benchmark_results.json.
Two side-by-side charts: Full-Run Latency and Cost per 1K Runs.

If ART-E data is missing from benchmark_results.json, falls back to
training history averages.

Usage:
  uv run python calc-cost/plot_simple_comparison.py
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"
HISTORY_PATH = (
    Path(__file__).parent.parent
    / ".art/email_agent/models/email-agent-008/history.jsonl"
)

H100_COST_PER_HOUR = 2.86
GPT_PRICING = {
    "GPT-5": {"input": 1.25, "output": 10.00},
    "GPT-5.1": {"input": 1.25, "output": 10.00},
    "GPT-5-mini": {"input": 0.25, "output": 2.00},
}


def load_benchmark_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    return json.loads(RESULTS_PATH.read_text())


def load_arte_from_history() -> dict:
    """Fallback: get ART-E stats from training history."""
    lines = HISTORY_PATH.read_text().strip().split("\n")
    data = [json.loads(l) for l in lines]
    train = [d for d in data if "train/prompt_tokens" in d]
    return {
        "duration": statistics.mean([d["train/duration"] for d in train]),
        "prompt_tokens": statistics.mean([d["train/prompt_tokens"] for d in train]),
        "completion_tokens": statistics.mean(
            [d["train/completion_tokens"] for d in train]
        ),
        "num_turns": statistics.mean([d["train/num_turns"] for d in train]),
        "answer_correct": None,  # not directly comparable
    }


def model_stats(results: list[dict], model_name: str) -> dict | None:
    entries = [r for r in results if r["model"] == model_name and r.get("duration")]
    if not entries:
        return None
    durations = [r["duration"] for r in entries]
    prompt_toks = [r.get("prompt_tokens", 0) for r in entries]
    comp_toks = [r.get("completion_tokens", 0) for r in entries]
    turns = [r.get("num_turns", 0) for r in entries]
    correct = [r.get("answer_correct", 0) for r in entries]

    avg_prompt = statistics.mean(prompt_toks)
    avg_comp = statistics.mean(comp_toks)

    if model_name in GPT_PRICING:
        p = GPT_PRICING[model_name]
        avg_cost = avg_prompt / 1e6 * p["input"] + avg_comp / 1e6 * p["output"]
    else:
        avg_cost = statistics.mean(durations) * H100_COST_PER_HOUR / 3600

    return {
        "duration": statistics.mean(durations),
        "prompt_tokens": avg_prompt,
        "completion_tokens": avg_comp,
        "total_tokens": avg_prompt + avg_comp,
        "num_turns": statistics.mean(turns),
        "accuracy": statistics.mean(correct),
        "cost_per_query": avg_cost,
        "cost_per_1k": avg_cost * 1000,
        "n": len(entries),
    }


def main():
    results = load_benchmark_results()

    # Build stats for each model
    stats = {}
    for name in ["GPT-5", "GPT-5.1", "GPT-5-mini", "ART-E"]:
        s = model_stats(results, name)
        if s:
            stats[name] = s

    # Fallback for ART-E from training history
    if "ART-E" not in stats:
        arte_hist = load_arte_from_history()
        avg_dur = arte_hist["duration"]
        avg_cost = avg_dur * H100_COST_PER_HOUR / 3600
        stats["ART-E"] = {
            "duration": avg_dur,
            "prompt_tokens": arte_hist["prompt_tokens"],
            "completion_tokens": arte_hist["completion_tokens"],
            "total_tokens": arte_hist["prompt_tokens"] + arte_hist["completion_tokens"],
            "num_turns": arte_hist["num_turns"],
            "accuracy": None,
            "cost_per_query": avg_cost,
            "cost_per_1k": avg_cost * 1000,
            "n": "hist",
        }

    # --- Print text summary ---
    model_order = ["GPT-5", "GPT-5.1", "GPT-5-mini", "ART-E"]
    present = [m for m in model_order if m in stats]

    print(f"\n{'=' * 72}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 72}")
    col_w = 16
    header = f"{'Metric':<26}" + "".join(f"{m:>{col_w}}" for m in present)
    print(header)
    print("-" * len(header))

    rows = [
        ("Avg Latency (s)", "duration", ".1f"),
        ("Avg Prompt Tokens", "prompt_tokens", ",.0f"),
        ("Avg Completion Tokens", "completion_tokens", ",.0f"),
        ("Avg Total Tokens", "total_tokens", ",.0f"),
        ("Avg Tool Calls (turns)", "num_turns", ".1f"),
        ("Accuracy", "accuracy", None),
        ("Cost / query ($)", "cost_per_query", None),
        ("Cost / 1K queries ($)", "cost_per_1k", ".2f"),
        ("N tasks", "n", None),
    ]

    for label, key, fmt in rows:
        row = f"{label:<26}"
        for m in present:
            val = stats[m].get(key)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            elif label == "Accuracy":
                row += f"{f'{val:.0%}':>{col_w}}"
            elif label == "Cost / query ($)":
                row += f"{'${:.5f}'.format(val):>{col_w}}"
            elif label == "N tasks":
                row += f"{str(val):>{col_w}}"
            else:
                row += f"{f'{val:{fmt}}':>{col_w}}"
        print(row)

    # -----------------------------------------------------------------------
    # Plot: 3-panel bar chart (Latency, Cost, Turns)
    # -----------------------------------------------------------------------
    models = present
    colors = ["#E8E8E8" if m != "ART-E" else "#F5A623" for m in models]

    latencies = [stats[m]["duration"] for m in models]
    cost_1k = [stats[m]["cost_per_1k"] for m in models]
    turns = [stats[m]["num_turns"] for m in models]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(models))
    bar_width = 0.55

    def style_ax(ax, values, labels, title, fmt_fn):
        bars = ax.bar(
            x, values, width=bar_width, color=colors, edgecolor="none", zorder=3
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.03,
                fmt_fn(val),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="#333333",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_ylim(0, max(values) * 1.35)
        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(bottom=False)

    style_ax(ax1, latencies, models, "Avg Latency per Query", lambda v: f"{v:.1f}s")
    style_ax(ax2, cost_1k, models, "Cost per 1K Queries", lambda v: f"${v:.2f}")
    style_ax(ax3, turns, models, "Avg Turns", lambda v: f"{v:.1f}")

    plt.tight_layout(w_pad=3)
    out_path = Path(__file__).parent / "simple_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
