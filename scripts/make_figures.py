"""
scripts/make_figures.py

Three PNGs into runs/figures/:
  fig1_val_loss_vs_tokens.png   validation loss vs TOKENS SEEN, four runs
  fig2_memory_vs_batch.png      peak memory vs batch, with the two ceilings
  fig3_throughput_vs_batch.png  tokens/s vs batch, with the collapse at 40

Everything is plotted against TOKENS, never steps: the runs use two batch
sizes and their step counts are not comparable (6,104 vs 1,526 for the same
50M tokens).

Palette: validated categorical slots 1-4 on the light chart surface. The
validator reports a sub-3:1 contrast WARN for aqua and yellow, so the relief
rule applies and every series carries a visible direct label as well as a
legend and a distinct marker shape -- identity is never colour alone.
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
FIGS = os.path.join(RUNS, "figures")
SWEEP = os.path.join(RUNS, "batch_sweep.csv")

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SERIES = [
    # run, label, colour, marker, linewidth
    ("01_baseline_bs8",      "baseline  b8",      SLOT[0], "o", 3.4),
    ("02_euler_bs8",         "euler  b8",         SLOT[1], "s", 2.0),
    ("03_midpoint_rev_bs8",  "midpoint_rev  b8",  SLOT[2], "^", 1.7),
    ("04_midpoint_rev_bs32", "midpoint_rev  b32", SLOT[3], "D", 2.0),
]

MPS_BUDGET = 11.84
SYSTEM_RAM = 16.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "font.size": 10,
})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)


def fnum(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def load_val(run):
    p = os.path.join(RUNS, run, "val.csv")
    if not os.path.exists(p):
        return [], []
    xs, ys = [], []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        xs.append(fnum(r["tokens_seen"]) / 1e6)
        ys.append(fnum(r["val_loss"]))
    return xs, ys


def declutter(labels, min_gap):
    """Nudge near-identical label y-positions apart, preserving order."""
    order = sorted(range(len(labels)), key=lambda i: labels[i][0])
    ys = [labels[i][0] for i in order]
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < min_gap:
            ys[k] = ys[k - 1] + min_gap
    out = list(labels)
    for k, i in enumerate(order):
        out[i] = (ys[k], labels[i][1], labels[i][2])
    return out


# ---------------------------------------------------------------- figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    style(ax)
    present, ends = [], []
    for run, label, colour, marker, lw in SERIES:
        xs, ys = load_val(run)
        if not xs:
            continue
        present.append((label, colour, marker))
        ax.plot(xs, ys, color=colour, linewidth=lw, zorder=3,
                marker=marker, markersize=5, markevery=max(1, len(xs) // 8),
                markerfacecolor=colour, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=label)
        ends.append((ys[-1], label, colour))

    if ends:
        lo = min(min(load_val(r)[1]) for r, *_ in SERIES if load_val(r)[1])
        hi = max(max(load_val(r)[1]) for r, *_ in SERIES if load_val(r)[1])
        for y, label, colour in declutter(ends, (hi - lo) * 0.045):
            ax.annotate(label, xy=(51.2, y), color=colour, fontsize=9,
                        va="center", ha="left", fontweight="medium")

    ax.set_xlim(0, 68)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("tokens seen  (millions)")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss against tokens seen", color=INK,
                 fontsize=13, fontweight="semibold", loc="left", pad=14)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK2)

    # inset: the last 10M tokens, where runs 01 and 03 nearly coincide
    axi = ax.inset_axes([0.42, 0.40, 0.30, 0.34])
    style(axi)
    for run, label, colour, marker, lw in SERIES:
        xs, ys = load_val(run)
        if not xs:
            continue
        pts = [(x, y) for x, y in zip(xs, ys) if x >= 40]
        if pts:
            axi.plot([p[0] for p in pts], [p[1] for p in pts],
                     color=colour, linewidth=lw, marker=marker, markersize=4,
                     markerfacecolor=colour, markeredgecolor=SURFACE,
                     markeredgewidth=0.8)
    axi.set_title("last 10M tokens", fontsize=8, color=MUTED, loc="left", pad=4)
    axi.tick_params(labelsize=7)

    fig.text(0.008, 0.015,
             "Plotted against tokens, not steps: the runs use two batch sizes "
             "(6,104 steps at b8 vs 1,526 at b32) for the same 50M tokens.\n"
             "Runs 01 and 03 very nearly coincide, so baseline is drawn thick with "
             "midpoint_rev thin on top: where you see green inside blue,\nthe two "
             "agree. That near-identity is the reversibility result, not a plotting "
             "artifact - see the inset.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    out = os.path.join(FIGS, "fig1_val_loss_vs_tokens.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def load_sweep():
    return list(csv.DictReader(open(SWEEP, encoding="utf-8")))


def status_colour(s):
    return {"ok": GOOD, "slow": WARN, "oom": CRIT}.get(s, MUTED)


# ---------------------------------------------------------------- figure 2
def fig2():
    rows = load_sweep()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    style(ax)

    xs = [int(r["batch"]) for r in rows]
    ys = [fnum(r["peak_gb"]) for r in rows]
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    # solid only through the measured-clean region; dashed past it, so the
    # thrashing points do not read as part of one smooth scaling curve.
    ax.plot(xs[:n_ok], ys[:n_ok], color=SLOT[0], linewidth=2.0, zorder=3)
    ax.plot(xs[n_ok - 1:], ys[n_ok - 1:], color=SLOT[0], linewidth=2.0,
            linestyle=(0, (4, 3)), alpha=0.75, zorder=3)
    for r in rows:
        b, p, st = int(r["batch"]), fnum(r["peak_gb"]), r["status"]
        ax.plot([b], [p], marker="X" if st == "oom" else "o", markersize=9,
                color=status_colour(st), markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=4)
        near = min(abs(p - MPS_BUDGET), abs(p - SYSTEM_RAM)) < 1.3
        ax.annotate(f"{p:.2f}", xy=(b, p), xytext=(0, -16 if near else 9),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)

    ax.axhline(MPS_BUDGET, color=WARN, linewidth=1.6, linestyle="--", zorder=2)
    ax.annotate(f"MPS budget  {MPS_BUDGET:.2f} GB", xy=(8, MPS_BUDGET),
                xytext=(0, 5), textcoords="offset points",
                fontsize=9, color="#8a6200", ha="left")
    ax.axhline(SYSTEM_RAM, color=CRIT, linewidth=1.6, linestyle="--", zorder=2)
    ax.annotate(f"system RAM  {SYSTEM_RAM:.0f} GB", xy=(8, SYSTEM_RAM),
                xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=CRIT, ha="left")

    ax.set_xticks(xs)
    ax.set_xlabel("batch size  (context 1024)")
    ax.set_ylabel("peak MPS memory  (GB)")
    ax.set_ylim(0, 22.5)
    ax.set_title("Peak memory against batch size — midpoint_rev", color=INK,
                 fontsize=13, fontweight="semibold", loc="left", pad=14)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=GOOD, markersize=8, label="ok"),
        Line2D([], [], marker="o", linestyle="none", color=WARN, markersize=8, label="slow"),
        Line2D([], [], marker="X", linestyle="none", color=CRIT, markersize=9, label="oom"),
    ], frameon=False, loc="lower right", fontsize=9, labelcolor=INK2)

    fig.text(0.008, 0.015,
             "NOT a clean scaling curve. The 8->16 jump is +5.09 GB but 16->24 is only "
             "+0.41 GB, so these are allocator high-water marks,\nnot the model's true "
             "requirement at each batch. The batch-96 point is the peak reached before "
             "allocation failed, not a working set.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    out = os.path.join(FIGS, "fig2_memory_vs_batch.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure 3
def fig3():
    rows = [r for r in load_sweep() if fnum(r["tokens_per_s_from_median"]) ==
            fnum(r["tokens_per_s_from_median"])]
    allrows = load_sweep()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    style(ax)

    xs = [int(r["batch"]) for r in rows]
    ys = [fnum(r["tokens_per_s_from_median"]) for r in rows]
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    ax.plot(xs[:n_ok], ys[:n_ok], color=SLOT[0], linewidth=2.0, zorder=3)
    ax.plot(xs[n_ok - 1:], ys[n_ok - 1:], color=SLOT[0], linewidth=2.0,
            linestyle=(0, (4, 3)), alpha=0.75, zorder=3)
    for r in rows:
        b, t, st = int(r["batch"]), fnum(r["tokens_per_s_from_median"]), r["status"]
        ax.plot([b], [t], marker="o", markersize=9, color=status_colour(st),
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        dx, ha = ((-11, "right") if b == 40 else (0, "center"))
        ax.annotate(f"{t:,.0f}", xy=(b, t), xytext=(dx, 10 if b != 40 else 0),
                    textcoords="offset points", ha=ha,
                    va="center" if b == 40 else "bottom",
                    fontsize=8, color=INK2)

    oom = [r for r in allrows if r["status"] == "oom"]
    if oom:
        b = int(oom[0]["batch"])
        ax.plot([b], [0], marker="X", markersize=10, color=CRIT,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate("allocation failed", xy=(b, 0), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8.5, color=CRIT)

    # point at the collapse EDGE (the 32->40 segment), not the 40 marker,
    # so the arrow cannot land on that point's value label
    ax.annotate("7x collapse\nbetween 32 and 40", xy=(36.5, 5300),
                xytext=(50, 7200), fontsize=9.5, color=CRIT, ha="left",
                arrowprops=dict(arrowstyle="->", color=CRIT, linewidth=1.4,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xticks([int(r["batch"]) for r in allrows])
    ax.set_xlabel("batch size  (context 1024)")
    ax.set_ylabel("tokens / s  (from median step time)")
    ax.set_ylim(-400, 11600)
    ax.set_title("Throughput against batch size — midpoint_rev", color=INK,
                 fontsize=13, fontweight="semibold", loc="left", pad=14)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color=GOOD, markersize=8, label="ok"),
        Line2D([], [], marker="o", linestyle="none", color=WARN, markersize=8, label="slow"),
        Line2D([], [], marker="X", linestyle="none", color=CRIT, markersize=9, label="oom"),
    ], frameon=False, loc="center right", fontsize=9, labelcolor=INK2)

    fig.text(0.008, 0.015,
             "Throughput never improves with batch: 10,246 -> 9,960 -> 9,884 -> 9,488 tok/s "
             "across 8->32, monotonically down, then a 7x collapse at 40.\n"
             "Batches 48/64/80 measured only one step each before timing out; their "
             "tokens/s is a single-sample lower bound.",
             fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    out = os.path.join(FIGS, "fig3_throughput_vs_batch.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    for fn in (fig1, fig2, fig3):
        print("wrote", fn())


if __name__ == "__main__":
    main()
