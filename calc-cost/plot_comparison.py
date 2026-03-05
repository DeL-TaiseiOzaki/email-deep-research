"""
Visualization: Self-hosted Qwen2.5-14B vs GPT-5.1 API
Generates comparison charts for cost, latency, and break-even analysis.
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams["font.family"] = [
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAGothic",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Data
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


# Models: (label, cost_per_query, latency_seconds, color, hatch)
def api_cost(inp_1m, out_1m, cache_rate=0.0, cached_1m=None):
    if cached_1m is None:
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


MODELS = [
    ("GPT-5.1", api_cost(1.25, 10.0), api_latency(0.20, 140), "#E74C3C", ""),
    (
        "GPT-5.1\n+cache",
        api_cost(1.25, 10.0, 0.8),
        api_latency(0.20, 140),
        "#E74C3C",
        "//",
    ),
    ("GPT-5.1\nbatch", api_cost(0.625, 5.0), None, "#E74C3C", ".."),
    ("GPT-5\nmini", api_cost(0.25, 2.0), api_latency(0.30, 150), "#3498DB", ""),
    (
        "GPT-5 mini\n+cache",
        api_cost(0.25, 2.0, 0.8),
        api_latency(0.30, 150),
        "#3498DB",
        "//",
    ),
    ("GPT-5\nnano", api_cost(0.05, 0.4), api_latency(0.15, 200), "#9B59B6", ""),
    (
        "Self-host\nH100",
        self_cost(3.00, 2848, 109.5),
        self_latency(2848, 109.5),
        "#2ECC71",
        "",
    ),
    (
        "Self-host\nH100 spot",
        self_cost(2.25, 2848, 109.5),
        self_latency(2848, 109.5),
        "#27AE60",
        "//",
    ),
]

labels = [m[0] for m in MODELS]
costs = [m[1] for m in MODELS]
latencies = [m[2] for m in MODELS]
colors = [m[3] for m in MODELS]
hatches = [m[4] for m in MODELS]


# ---------------------------------------------------------------------------
# Figure 1: Per-query cost bar chart
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Self-hosted Qwen2.5-14B (LoRA)  vs  GPT-5.1 API\n"
    f"avg {AVG_INPUT:.0f} input tokens + {AVG_OUTPUT:.0f} output tokens / query, {AVG_TURNS:.1f} turns",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

# --- Chart 1: Cost per query ---
ax1 = axes[0, 0]
bars = ax1.bar(
    range(len(labels)),
    [c * 1000 for c in costs],
    color=colors,
    edgecolor="white",
    linewidth=0.5,
)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)
for i, (c, bar) in enumerate(zip(costs, bars)):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"${c:.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Cost per query (× $0.001)", fontsize=10)
ax1.set_title("1. Per-Query Cost", fontsize=12, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(0, max(c * 1000 for c in costs) * 1.3)

# --- Chart 2: Latency per query ---
ax2 = axes[0, 1]
lat_labels = []
lat_vals = []
lat_colors = []
for label, _, lat, color, _ in MODELS:
    if lat is not None:
        lat_labels.append(label)
        lat_vals.append(lat)
        lat_colors.append(color)
# Add actual measured
lat_labels.append("Actual\n(+tools)")
lat_vals.append(AVG_DURATION)
lat_colors.append("#95A5A6")

barsh = ax2.barh(
    range(len(lat_labels)), lat_vals, color=lat_colors, edgecolor="white", height=0.6
)
for i, (v, bar) in enumerate(zip(lat_vals, barsh)):
    ax2.text(
        v + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.1f}s",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
    )
ax2.set_yticks(range(len(lat_labels)))
ax2.set_yticklabels(lat_labels, fontsize=8)
ax2.set_xlabel("Latency (seconds)", fontsize=10)
ax2.set_title("2. Per-Query Latency (LLM inference)", fontsize=12, fontweight="bold")
ax2.grid(axis="x", alpha=0.3)
ax2.invert_yaxis()
ax2.set_xlim(0, AVG_DURATION * 1.2)

# Add annotation for tool execution
ax2.axvspan(3, AVG_DURATION, alpha=0.08, color="orange")
ax2.annotate(
    "← tool execution time\n   (DB search, etc.)",
    xy=(10, 3),
    fontsize=8,
    color="#E67E22",
    style="italic",
)

# --- Chart 3: Monthly cost at scale ---
ax3 = axes[1, 0]
volumes = np.array([100, 500, 1000, 5000, 10000, 50000, 100000])

scenarios = [
    ("GPT-5.1", 1.25, 10.0, 0.0, "#E74C3C", "-"),
    ("GPT-5.1 +80% cache", 1.25, 10.0, 0.8, "#E74C3C", "--"),
    ("GPT-5.1 batch", 0.625, 5.0, 0.0, "#E74C3C", ":"),
    ("GPT-5-mini", 0.25, 2.0, 0.0, "#3498DB", "-"),
    ("GPT-5-mini +cache", 0.25, 2.0, 0.8, "#3498DB", "--"),
    ("Self-hosted H100", None, None, None, "#2ECC71", "-"),
]

for name, inp, out, cache, color, ls in scenarios:
    if inp is not None:
        monthly = [api_cost(inp, out, cache) * v for v in volumes]
    else:
        monthly = [self_cost(3.00, 2848, 109.5) * v for v in volumes]
    ax3.plot(volumes, monthly, label=name, color=color, linestyle=ls, linewidth=2)

# H100 24/7 line
h100_monthly = 3.00 * 24 * 30  # $2,160/mo
ax3.axhline(y=h100_monthly, color="#2ECC71", linestyle="-.", alpha=0.5, linewidth=1)
ax3.text(
    100,
    h100_monthly * 1.07,
    f"H100 24/7 = ${h100_monthly:,.0f}/mo",
    fontsize=8,
    color="#2ECC71",
)

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("Queries / month", fontsize=10)
ax3.set_ylabel("Monthly cost (USD)", fontsize=10)
ax3.set_title("3. Monthly Cost at Scale", fontsize=12, fontweight="bold")
ax3.legend(fontsize=7, loc="upper left")
ax3.grid(True, alpha=0.3, which="both")
ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# --- Chart 4: Latency breakdown (stacked) ---
ax4 = axes[1, 1]

inference_models = [
    ("GPT-5.1", api_latency(0.20, 140), "#E74C3C"),
    ("GPT-5\nmini", api_latency(0.30, 150), "#3498DB"),
    ("Self-host\nH100", self_latency(2848, 109.5), "#2ECC71"),
]

x_pos = range(len(inference_models))
inf_labels = [m[0] for m in inference_models]
inf_times = [m[1] for m in inference_models]
inf_colors = [m[2] for m in inference_models]
tool_time = AVG_DURATION - statistics.mean(inf_times)  # approximate tool time

bars_inf = ax4.bar(
    x_pos, inf_times, color=inf_colors, edgecolor="white", label="LLM inference"
)
bars_tool = ax4.bar(
    x_pos,
    [tool_time] * len(x_pos),
    bottom=inf_times,
    color="#F39C12",
    alpha=0.4,
    edgecolor="white",
    label="Tool execution",
)

for i, (inf_t, bar) in enumerate(zip(inf_times, bars_inf)):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        inf_t + tool_time + 0.3,
        f"{inf_t + tool_time:.1f}s total",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        inf_t / 2,
        f"{inf_t:.1f}s",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )

ax4.set_xticks(x_pos)
ax4.set_xticklabels(inf_labels, fontsize=8)
ax4.set_ylabel("E2E time per query (seconds)", fontsize=10)
ax4.set_title("4. E2E Latency Breakdown", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = Path(__file__).parent / "comparison_charts.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
plt.close()
