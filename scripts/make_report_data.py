"""
scripts/make_report_data.py

Assembles one comparable table across all four runs, and back-fills a
mean_over_median column into experiments.csv.

WHERE tokens_per_s COMES FROM
-----------------------------
NOT from the training runs. Step 4's wall-clock throughput measured how much
swap pressure each run happened to meet, not the variant:

    run 01 wall/median = 5.26x   run 02 = 3.02x   run 03 = 1.07x

So the summary takes tokens_per_s from the clean step-5 measurements instead:

    runs 01-03 (batch 8)  <- runs/throughput_clean.csv   (200 steps, mean/med ~1.00)
    run  04    (batch 32) <- runs/batch_sweep.csv        (50 steps,  mean/med  1.010)

Every row is marked with its source. The contaminated per-run figure is kept
alongside, in tokens_per_s_from_run, so the gap is visible rather than hidden.

TWO MEASURES OF DISPERSION, AND WHY BOTH
----------------------------------------
mean_over_median_metrics is computed from each run's metrics.csv, as specified.
But metrics.csv logs every 50th step -- a 2% sample -- and it is not a reliable
contamination detector:

    run 01: metrics says 1.080, wall says 5.26  (the sample MISSED the stalls)
    run 02: metrics says 12.666, wall says 3.02 (one logged step caught a stall)
    run 03: metrics says 1.036, wall says 1.07  (agree; the run really was clean)

mean_over_median_wall = (wall_s / steps_done) / median_step_time_s uses every
step and is the figure that actually exposes step 4's contamination. Both
columns are reported; neither alone tells the truth.
"""

import csv
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
EXPERIMENTS = os.path.join(ROOT, "experiments.csv")
THROUGHPUT = os.path.join(RUNS, "throughput_clean.csv")
SWEEP = os.path.join(RUNS, "batch_sweep.csv")
SUMMARY = os.path.join(RUNS, "summary.csv")

ORDER = ["01_baseline_bs8", "02_euler_bs8", "03_midpoint_rev_bs8", "04_midpoint_rev_bs32"]


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def metrics_mean_over_median(run):
    p = os.path.join(RUNS, run, "metrics.csv")
    if not os.path.exists(p):
        return float("nan"), 0
    t = [fnum(r["step_time_s"]) for r in csv.DictReader(open(p, encoding="utf-8"))]
    t = [v for v in t if v == v]
    if not t:
        return float("nan"), 0
    med = statistics.median(t)
    return (statistics.mean(t) / med if med > 0 else float("nan")), len(t)


def main():
    exp = {r["run_name"]: r for r in csv.DictReader(open(EXPERIMENTS, encoding="utf-8"))}
    missing = [r for r in ORDER if r not in exp]
    if missing:
        print(f"WARNING: not in experiments.csv yet: {', '.join(missing)}")

    # clean throughput sources
    clean_by_variant = {r["variant"]: r for r in csv.DictReader(open(THROUGHPUT, encoding="utf-8"))}
    sweep_by_batch = {int(r["batch"]): r for r in csv.DictReader(open(SWEEP, encoding="utf-8"))}

    # ---- back-fill mean_over_median into experiments.csv -----------------
    exp_rows = list(csv.DictReader(open(EXPERIMENTS, encoding="utf-8")))
    hdr = list(exp_rows[0].keys())
    for col in ("mean_over_median_metrics", "mean_over_median_wall", "metrics_samples"):
        if col not in hdr:
            hdr.append(col)
    for r in exp_rows:
        mm, n = metrics_mean_over_median(r["run_name"])
        med = fnum(r.get("median_step_time_s"))
        steps = int(r.get("steps_done") or 0)
        wall = fnum(r.get("wall_s"))
        r["mean_over_median_metrics"] = f"{mm:.3f}" if mm == mm else ""
        r["metrics_samples"] = n
        r["mean_over_median_wall"] = (f"{(wall / steps) / med:.3f}"
                                      if steps and med and med == med else "")
    with open(EXPERIMENTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(exp_rows)
    print(f"experiments.csv: added mean_over_median_metrics / _wall for "
          f"{len(exp_rows)} run(s)\n")

    # ---- summary ---------------------------------------------------------
    out = []
    for run in ORDER:
        if run not in exp:
            continue
        e = {r["run_name"]: r for r in exp_rows}[run]
        batch = int(e["batch"])
        variant = e["variant"]

        if batch == 8 and variant in clean_by_variant:
            src = clean_by_variant[variant]
            tps, src_name = fnum(src["tokens_per_s_from_median"]), "throughput_clean.csv (batch 8, 200 steps)"
        elif batch in sweep_by_batch and variant == "midpoint_rev":
            src = sweep_by_batch[batch]
            tps, src_name = fnum(src["tokens_per_s_from_median"]), f"batch_sweep.csv (batch {batch}, 50 steps)"
        else:
            tps, src_name = float("nan"), "none"

        mm, n = metrics_mean_over_median(run)
        med = fnum(e.get("median_step_time_s"))
        steps = int(e.get("steps_done") or 0)
        wall = fnum(e.get("wall_s"))
        out.append({
            "run": run,
            "variant": variant,
            "batch": batch,
            "steps": steps,
            "final_val_loss": f"{fnum(e['final_val_loss_mean_last5']):.6f}",
            "tokens_per_s": f"{tps:.1f}" if tps == tps else "",
            "peak_gb": f"{fnum(e['peak_gb']):.2f}",
            "median_step_s": f"{med:.4f}" if med == med else "",
            "mean_over_median": f"{mm:.3f}" if mm == mm else "",
            "tokens_per_s_source": "clean measurement",
            "tokens_per_s_source_file": src_name,
            "tokens_per_s_from_run": f"{fnum(e.get('tokens_per_s')):.1f}",
            "mean_over_median_wall": (f"{(wall / steps) / med:.3f}"
                                      if steps and med == med and med else ""),
            "metrics_samples": n,
            "tokens_seen": e.get("tokens_seen", ""),
            "wall_s": e.get("wall_s", ""),
            "status": e.get("status", ""),
        })

    cols = list(out[0].keys())
    with open(SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    print("=" * 118)
    print("SUMMARY — all runs.  tokens_per_s is the CLEAN measurement, never run wall clock.")
    print("=" * 118)
    print(f"{'run':<23}{'variant':<14}{'batch':>6}{'steps':>7}{'final_val':>11}"
          f"{'tok/s':>9}{'peak_gb':>9}{'med_s':>8}{'mean/med':>10}{'src':>7}")
    print("-" * 118)
    for r in out:
        print(f"{r['run']:<23}{r['variant']:<14}{r['batch']:>6}{r['steps']:>7,}"
              f"{float(r['final_val_loss']):>11.4f}"
              f"{float(r['tokens_per_s']) if r['tokens_per_s'] else float('nan'):>9,.0f}"
              f"{float(r['peak_gb']):>9.2f}{float(r['median_step_s']):>8.4f}"
              f"{float(r['mean_over_median']):>10.3f}{'clean':>7}")
    print("-" * 118)
    print()
    print("tokens_per_s sources (none taken from the training runs' wall clock):")
    for r in out:
        print(f"  {r['run']:<23} {r['tokens_per_s_source_file']}")
    print()
    print("contamination — the same runs, measured two ways:")
    print(f"  {'run':<23}{'mean/med (metrics, 2% sample)':>32}{'mean/med (wall, every step)':>30}"
          f"{'tok/s as-run':>14}")
    for r in out:
        print(f"  {r['run']:<23}{float(r['mean_over_median']):>32.3f}"
              f"{float(r['mean_over_median_wall']) if r['mean_over_median_wall'] else float('nan'):>30.3f}"
              f"{float(r['tokens_per_s_from_run']):>14,.0f}")
    print()
    print("  metrics.csv logs every 50th step, so its ratio can miss stalls entirely")
    print("  (run 01: 1.08 sampled vs 5.26 actual) or be dominated by a single caught")
    print("  stall (run 02: 12.67 sampled vs 3.02 actual). The wall column uses every")
    print("  step and is the one that exposes step 4's contamination.")
    print("=" * 118)
    print(f"\nsaved to {SUMMARY}")


if __name__ == "__main__":
    main()
