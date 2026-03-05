"""
Full benchmark dashboard: 2x2 grid showing Latency, Cost, Accuracy, and Turns.
Reads from benchmark_results.json.

Usage:
  uv run python calc-cost/plot_full_dashboard.py
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"

H100_COST_PER_HOUR = 2.86
GPT_PRICING = {
    "GPT-5": {"input": 1.25, "output": 10.00},
    "GPT-5.1": {"input": 1.25, "output": 10.00},
    "GPT-5-mini": {"input": 0.25, "output": 2.00},
}


def load_results() -> list[dict]:
    return json.loads(RESULTS_PATH.read_text())


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
        "accuracy": statistics.mean(correct) * 100,
        "cost_per_1k": avg_cost * 1000,
        "n": len(entries),
    }


def make_bar(ax, models, values, colors, title, fmt, unit=""):
    """Draw a single bar chart panel."""
    x = np.arange(len(models))
    bar_width = 0.55
    bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor="none", zorder=3)

    for bar, val in zip(bars, values):
        label = f"{val:{fmt}}{unit}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.04,
            label,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, max(values) * 1.35)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(bottom=False)


def main():
    results = load_results()

    model_order = ["GPT-5", "GPT-5.1", "GPT-5-mini", "ART-E"]
    stats = {}
    for name in model_order:
        s = model_stats(results, name)
        if s:
            stats[name] = s

    present = [m for m in model_order if m in stats]
    colors = ["#E8E8E8" if m != "ART-E" else "#F5A623" for m in present]

    # --- 2x2 dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "ART-E Benchmark: GPT-5.1 vs GPT-5-mini vs ART-E (n=10)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Top-left: Latency
    make_bar(
        axes[0, 0],
        present,
        [stats[m]["duration"] for m in present],
        colors,
        "Avg Latency per Query",
        ".1f",
        "s",
    )

    # Top-right: Cost per 1K
    make_bar(
        axes[0, 1],
        present,
        [stats[m]["cost_per_1k"] for m in present],
        colors,
        "Cost per 1K Queries",
        ".2f",
        "",
    )
    # Override labels to add $ prefix
    ax = axes[0, 1]
    for child in ax.get_children():
        if isinstance(child, plt.Text) and child.get_text().replace(".", "").isdigit():
            child.set_text(f"${child.get_text()}")
    # Re-draw cost labels with $ prefix
    cost_vals = [stats[m]["cost_per_1k"] for m in present]
    x = np.arange(len(present))
    for i, val in enumerate(cost_vals):
        ax.texts[i].set_text(f"${val:.2f}")

    # Bottom-left: Accuracy
    make_bar(
        axes[1, 0],
        present,
        [stats[m]["accuracy"] for m in present],
        colors,
        "Accuracy (%)",
        ".0f",
        "%",
    )
    axes[1, 0].set_ylim(0, 120)

    # Bottom-right: Avg Turns
    make_bar(
        axes[1, 1],
        present,
        [stats[m]["num_turns"] for m in present],
        colors,
        "Avg Tool Calls (Turns)",
        ".1f",
        "",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=3, h_pad=3)
    out_path = Path(__file__).parent / "full_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()

    # --- Per-task scatter: Latency vs Tokens ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    markers = {"GPT-5.1": "s", "GPT-5-mini": "^", "ART-E": "o"}
    mcolors = {"GPT-5.1": "#AAAAAA", "GPT-5-mini": "#888888", "ART-E": "#F5A623"}

    for name in present:
        entries = [r for r in results if r["model"] == name and r.get("duration")]
        durations = [r["duration"] for r in entries]
        tokens = [r.get("total_tokens", 0) for r in entries]
        correct = [r.get("answer_correct", 0) for r in entries]
        for d, t, c in zip(durations, tokens, correct):
            edge = "#22CC22" if c else "#CC2222"
            ax2.scatter(
                t,
                d,
                marker=markers[name],
                s=100,
                c=mcolors[name],
                edgecolors=edge,
                linewidths=2,
                zorder=3,
                alpha=0.85,
            )

    # Legend entries
    for name in present:
        ax2.scatter(
            [],
            [],
            marker=markers[name],
            s=80,
            c=mcolors[name],
            edgecolors="gray",
            linewidths=1,
            label=name,
        )
    ax2.scatter(
        [],
        [],
        marker="o",
        s=60,
        c="white",
        edgecolors="#22CC22",
        linewidths=2,
        label="Correct",
    )
    ax2.scatter(
        [],
        [],
        marker="o",
        s=60,
        c="white",
        edgecolors="#CC2222",
        linewidths=2,
        label="Incorrect",
    )
    ax2.legend(fontsize=9, loc="upper left")

    ax2.set_xlabel("Total Tokens", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Latency (s)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Per-Task: Latency vs Token Usage (green=correct, red=incorrect)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    out_path2 = Path(__file__).parent / "per_task_scatter.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path2}")
    plt.close()


if __name__ == "__main__":
    main()
