"""
Clean bar chart: GPT-5.1 API vs Self-hosted Qwen2.5-14B comparison.
Two side-by-side charts: Cost and Latency.
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
history_path = (
    Path(__file__).parent.parent
    / ".art/email_agent/models/email-agent-008/history.jsonl"
)
lines = history_path.read_text().strip().split("\n")
data = [json.loads(l) for l in lines]
train = [d for d in data if "train/prompt_tokens" in d]

AVG_INPUT = statistics.mean([d["train/prompt_tokens"] for d in train])
AVG_OUTPUT = statistics.mean([d["train/completion_tokens"] for d in train])
AVG_TURNS = statistics.mean([d["train/num_turns"] for d in train])
AVG_DURATION = statistics.mean([d["train/duration"] for d in train])


def api_cost(inp_1m, out_1m, cache_rate=0.0):
    cached_1m = inp_1m * 0.1
    cached = AVG_INPUT * cache_rate
    uncached = AVG_INPUT * (1 - cache_rate)
    return (
        uncached / 1e6 * inp_1m + cached / 1e6 * cached_1m + AVG_OUTPUT / 1e6 * out_1m
    )


def api_latency(ttft, tps):
    per_turn = ttft + (AVG_OUTPUT / AVG_TURNS) / tps
    return per_turn * AVG_TURNS


def self_cost(gpu_hr, prompt_tps, gen_tps):
    t = AVG_INPUT / prompt_tps + AVG_OUTPUT / gen_tps
    return t * gpu_hr / 3600


def self_latency(prompt_tps, gen_tps):
    return AVG_INPUT / prompt_tps + AVG_OUTPUT / gen_tps


# ---------------------------------------------------------------------------
# Define models
# ---------------------------------------------------------------------------
models = [
    {
        "label": "GPT-5.1",
        "cost": api_cost(1.25, 10.0),
        "latency": api_latency(0.20, 140),
        "color": "#E74C3C",
    },
    {
        "label": "GPT-5.1\n+cache(80%)",
        "cost": api_cost(1.25, 10.0, 0.8),
        "latency": api_latency(0.20, 140),
        "color": "#F1948A",
    },
    {
        "label": "GPT-5.1\nBatch",
        "cost": api_cost(0.625, 5.0),
        "latency": None,  # async
        "color": "#D98880",
    },
    {
        "label": "GPT-5\nmini",
        "cost": api_cost(0.25, 2.0),
        "latency": api_latency(0.30, 150),
        "color": "#3498DB",
    },
    {
        "label": "GPT-5 mini\n+cache(80%)",
        "cost": api_cost(0.25, 2.0, 0.8),
        "latency": api_latency(0.30, 150),
        "color": "#85C1E9",
    },
    {
        "label": "Qwen-14B\nSelf-hosted\n(A100)",
        "cost": self_cost(1.50, 2848, 109.5),
        "latency": self_latency(2848, 109.5),
        "color": "#2ECC71",
    },
    {
        "label": "Qwen-14B\nSelf-hosted\n(L4)",
        "cost": self_cost(0.50, 1500, 60),
        "latency": self_latency(1500, 60),
        "color": "#82E0AA",
    },
]

# ===========================================================================
# FIGURE 1: Cost Comparison Bar Chart
# ===========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "GPT-5.1 API  vs  Self-hosted Qwen2.5-14B (LoRA)  :  Cost & Latency",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

# --- Cost chart ---
x = np.arange(len(models))
cost_vals = [m["cost"] * 1000 for m in models]  # in milli-dollars
cost_colors = [m["color"] for m in models]
cost_labels = [m["label"] for m in models]

bars = ax1.bar(x, cost_vals, color=cost_colors, edgecolor="white", width=0.7, zorder=3)

for bar, m in zip(bars, models):
    val = m["cost"]
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"${val:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

# Reference line for Qwen A100
qwen_cost = models[5]["cost"] * 1000
ax1.axhline(
    y=qwen_cost, color="#2ECC71", linestyle="--", alpha=0.6, linewidth=1.5, zorder=2
)
ax1.text(
    len(models) - 0.5,
    qwen_cost + 0.08,
    f"Qwen A100 = ${models[5]['cost']:.4f}",
    fontsize=8,
    color="#2ECC71",
    ha="right",
)

ax1.set_xticks(x)
ax1.set_xticklabels(cost_labels, fontsize=8)
ax1.set_ylabel("Cost per query ($)", fontsize=11)
ax1.set_title("Cost per Query", fontsize=13, fontweight="bold", pad=10)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.set_ylim(0, max(cost_vals) * 1.35)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1000:.4f}"))

# Add category labels
ax1.annotate(
    "",
    xy=(0.5, -0.18),
    xytext=(-0.5, -0.18),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-", color="#E74C3C", lw=2),
)

# --- Latency chart ---
lat_models = [m for m in models if m["latency"] is not None]
# Add actual measured
lat_models.append(
    {
        "label": "Actual E2E\n(+tool calls)",
        "latency": AVG_DURATION,
        "color": "#95A5A6",
    }
)

x2 = np.arange(len(lat_models))
lat_vals = [m["latency"] for m in lat_models]
lat_colors = [m["color"] for m in lat_models]
lat_labels = [m["label"] for m in lat_models]

bars2 = ax2.bar(x2, lat_vals, color=lat_colors, edgecolor="white", width=0.7, zorder=3)

for bar, m in zip(bars2, lat_models):
    val = m["latency"]
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{val:.1f}s",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Highlight tool execution portion on the last bar
tool_time = AVG_DURATION - statistics.mean([m["latency"] for m in lat_models[:-1]])
last_bar = bars2[-1]
ax2.bar(
    [x2[-1]],
    [tool_time],
    bottom=[AVG_DURATION - tool_time],
    color="#F39C12",
    alpha=0.5,
    edgecolor="white",
    width=0.7,
    zorder=4,
)
ax2.annotate(
    f"Tool exec\n~{tool_time:.0f}s",
    xy=(last_bar.get_x() + last_bar.get_width() / 2, AVG_DURATION - tool_time / 2),
    fontsize=8,
    ha="center",
    va="center",
    color="#E67E22",
    fontweight="bold",
)
ax2.annotate(
    f"LLM\n~{AVG_DURATION - tool_time:.0f}s",
    xy=(last_bar.get_x() + last_bar.get_width() / 2, (AVG_DURATION - tool_time) / 2),
    fontsize=8,
    ha="center",
    va="center",
    color="white",
    fontweight="bold",
)

ax2.set_xticks(x2)
ax2.set_xticklabels(lat_labels, fontsize=8)
ax2.set_ylabel("Latency per query (seconds)", fontsize=11)
ax2.set_title("Latency per Query", fontsize=13, fontweight="bold", pad=10)
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.set_ylim(0, AVG_DURATION * 1.25)

plt.tight_layout()
out_path = Path(__file__).parent / "cost_latency_bars.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close()

# ===========================================================================
# FIGURE 2: Monthly cost lines + break-even
# ===========================================================================
fig2, ax3 = plt.subplots(figsize=(12, 6))

volumes = np.array([100, 500, 1000, 5000, 10000, 50000, 100000])

scenarios = [
    ("GPT-5.1", 1.25, 10.0, 0.0, "#E74C3C", "-", 2.5),
    ("GPT-5.1 +80% cache", 1.25, 10.0, 0.8, "#E74C3C", "--", 2.0),
    ("GPT-5.1 Batch", 0.625, 5.0, 0.0, "#D98880", ":", 2.0),
    ("GPT-5-mini", 0.25, 2.0, 0.0, "#3498DB", "-", 2.5),
    ("GPT-5-mini +cache", 0.25, 2.0, 0.8, "#85C1E9", "--", 2.0),
    ("Qwen A100 (per-use)", None, None, None, "#2ECC71", "-", 2.5),
]

for name, inp, out, cache, color, ls, lw in scenarios:
    if inp is not None:
        monthly = [api_cost(inp, out, cache) * v for v in volumes]
    else:
        monthly = [self_cost(1.50, 2848, 109.5) * v for v in volumes]
    ax3.plot(volumes, monthly, label=name, color=color, linestyle=ls, linewidth=lw)

# A100 fixed cost line
ax3.axhline(y=1080, color="#2ECC71", linestyle="-.", alpha=0.7, linewidth=1.5)
ax3.fill_between(volumes, 1080, alpha=0.05, color="#2ECC71")
ax3.text(
    120,
    1250,
    "A100 24/7 fixed = $1,080/mo",
    fontsize=9,
    color="#27AE60",
    fontweight="bold",
)

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("Queries / month", fontsize=12)
ax3.set_ylabel("Monthly cost (USD)", fontsize=12)
ax3.set_title(
    "Monthly Cost at Scale: GPT-5.1 API vs Self-hosted Qwen2.5-14B",
    fontsize=14,
    fontweight="bold",
)
ax3.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax3.grid(True, alpha=0.3, which="both")
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax3.set_xlim(100, 100000)
ax3.set_ylim(0.05, 5000)

plt.tight_layout()
out2 = Path(__file__).parent / "monthly_cost_scale.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out2}")
plt.close()
