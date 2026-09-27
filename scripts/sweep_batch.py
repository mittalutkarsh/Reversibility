"""
scripts/sweep_batch.py

PART B -- how far does midpoint_rev's batch size go, and what does the curve
look like on the way?

Sweeps batch 8, 16, 24, 32, 40, 48, 64, 80, 96 and then keeps climbing in
steps of 16 until an allocation fails outright. It deliberately CONTINUES past
a slow batch: the shape of the curve past the knee is the point, not the
location of the first slowdown.

  status ok   -- mean_over_median <= 1.5
  status slow -- mean_over_median > 1.5; a minority of steps stalled, which on
                 this machine means the allocation spilled into swap
  status oom  -- allocation failed; the sweep stops at this row

Each batch runs in a fresh subprocess so no allocator state carries over, and
the worker checkpoints its step times to disk, so a batch killed on timeout
still contributes whatever steps it managed.

tokens/s is always derived from the MEDIAN step time.
"""

import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig, get_batch                       # noqa: E402
from measure_memory import MemPoller, gb                           # noqa: E402
from measure_throughput import sysctl, vm_counters, swap_used_bytes, stats  # noqa: E402

SEED = 1337
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BIN = os.path.join(ROOT, "data", "train.bin")
CSV_PATH = os.path.join(ROOT, "runs", "batch_sweep.csv")

VARIANT = "midpoint_rev"
BLOCK_SIZE = 1024
STEPS = 50
DISCARD = 10
LR = 3e-4
GRAD_CLIP = 1.0
SLOW_RATIO = 1.5
QUIET_RATIO = 1.3
MIN_SAMPLES = 10        # below this, dispersion statistics are meaningless

# A batch killed on timeout can contribute as little as ONE step. With n=1,
# mean == median == p90 and mean_over_median is exactly 1.000 -- which reads
# like a perfectly quiet run and is the opposite of the truth. Any row with
# fewer than MIN_SAMPLES kept steps therefore reports its dispersion stats as
# nan, and is excluded from the "largest quiet batch" analysis.
TOKEN_BUDGET = 50_000_000
TIMEOUT_SEC = 900
MAX_BATCH = 256

BASE_BATCHES = [8, 16, 24, 32, 40, 48, 64, 80, 96]


def batch_schedule():
    for b in BASE_BATCHES:
        yield b
    b = BASE_BATCHES[-1] + 16
    while b <= MAX_BATCH:
        yield b
        b += 16


def worker(batch, out_path):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps"
    res = {"batch": batch, "status": "ok", "note": "", "times": []}

    poller = MemPoller()
    poller.start()
    try:
        data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
        model = GPT(GPTConfig(variant=VARIANT), verbose=False).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        g = torch.Generator().manual_seed(SEED)

        for i in range(STEPS):
            t0 = time.time()
            x, y = get_batch(data, batch, BLOCK_SIZE, device, g)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            loss.item()
            res["times"].append(time.time() - t0)
            if i % 5 == 0:
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"times": res["times"], "peak": poller.peak}, f)
                except Exception:
                    pass
    except RuntimeError as exc:
        msg = str(exc)
        low = msg.lower()
        res["status"] = "oom" if ("out of memory" in low or "insufficient" in low
                                  or "failed to allocate" in low) else "error"
        res["note"] = msg.replace("\n", " ")[:200]
    except Exception as exc:  # noqa: BLE001
        res["status"] = "error"
        res["note"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
    finally:
        poller.stop()
        res["peak_bytes"] = poller.peak

    print("RESULT " + json.dumps(res), flush=True)


def finalise(row, times, batch):
    kept = times[DISCARD:] if len(times) > DISCARD else times
    if kept:
        row.update(stats(kept, batch))
        row["steps_kept"] = len(kept)
        if row["status"] == "ok" and row["mean_over_median"] > SLOW_RATIO:
            row["status"] = "slow"
        if len(kept) < MIN_SAMPLES:
            # median_step_s stays: one real measured step is still a real
            # duration. The dispersion statistics do not survive n<MIN_SAMPLES.
            for k in ("mean_step_s", "p90_step_s", "p99_step_s", "mean_over_median"):
                row[k] = float("nan")
            row["note"] = (row.get("note", "") +
                           f" | only {len(kept)} step(s) measured; dispersion stats "
                           f"suppressed (n<{MIN_SAMPLES})").strip(" |")
    else:
        row.update({k: float("nan") for k in
                    ("median_step_s", "mean_step_s", "p90_step_s", "p99_step_s",
                     "mean_over_median", "tokens_per_s_from_median")})
        row["steps_kept"] = 0
    row["steps_for_50M_tokens"] = math.ceil(TOKEN_BUDGET / (batch * BLOCK_SIZE))
    return row


def parent():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    here = os.path.abspath(__file__)

    total_ram = int(sysctl("hw.memsize") or 0)
    vm0, page = vm_counters()
    free_start = (vm0.get("Pages free", 0) + vm0.get("Pages speculative", 0)) * page
    swap_start = swap_used_bytes()
    swapouts_start = vm0.get("Swapouts", 0)
    budget = (torch.mps.recommended_max_memory()
              if hasattr(torch.mps, "recommended_max_memory") else 0)

    print("=" * 108)
    print(f"BATCH SWEEP — {VARIANT}, {STEPS} steps each (first {DISCARD} discarded)")
    print("=" * 108)
    print(f"total RAM {gb(total_ram):.2f} GB | free at start {gb(free_start):.2f} GB | "
          f"swap used {gb(swap_start):.2f} GB | MPS budget {gb(budget):.2f} GB")
    print()

    rows = []
    swap_events = []
    for batch in batch_schedule():
        part = os.path.join(ROOT, "runs", f".sweep_{batch}.json")
        if os.path.exists(part):
            os.remove(part)

        vm_before, _ = vm_counters()
        so_before = vm_before.get("Swapouts", 0)
        sw_before = swap_used_bytes()

        print(f"  batch {batch:>4} ...", end="", flush=True)
        cmd = [sys.executable, "-u", here, "--worker", "--batch", str(batch),
               "--out", part]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line:
                r = json.loads(line[len("RESULT "):])
                row = {"batch": batch, "status": r["status"], "note": r.get("note", ""),
                       "peak_gb": gb(r.get("peak_bytes", 0))}
                times = r.get("times", [])
            else:
                row = {"batch": batch, "status": "error", "peak_gb": 0.0,
                       "note": (proc.stderr or "no RESULT").replace("\n", " ")[:200]}
                times = []
        except subprocess.TimeoutExpired:
            partial = {}
            if os.path.exists(part):
                try:
                    with open(part, encoding="utf-8") as pf:
                        partial = json.load(pf)
                except Exception:
                    pass
            times = partial.get("times", [])
            row = {"batch": batch, "status": "slow", "peak_gb": gb(partial.get("peak", 0)),
                   "note": f"exceeded {TIMEOUT_SEC}s after {len(times)}/{STEPS} steps"}
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass

        row = finalise(row, times, batch)
        row["wall_s"] = time.time() - t0

        vm_after, _ = vm_counters()
        so_delta = vm_after.get("Swapouts", 0) - so_before
        sw_delta = swap_used_bytes() - sw_before
        row["swapouts_delta"] = so_delta
        row["swap_used_delta_gb"] = gb(sw_delta) if sw_delta == sw_delta else 0.0
        if so_delta > 0:
            swap_events.append((batch, so_delta, gb(sw_delta) if sw_delta == sw_delta else 0.0))

        rows.append(row)
        print(f" {row['status']:>5}  peak {row['peak_gb']:5.2f} GB  "
              f"median {row.get('median_step_s', float('nan')):.3f}s  "
              f"mean/med {row.get('mean_over_median', float('nan')):.2f}  "
              f"({row['wall_s']:.0f}s)", flush=True)

        if row["status"] == "oom":
            print(f"\n  allocation failed at batch {batch} — stopping the sweep.")
            break

    cols = ["batch", "status", "peak_gb", "median_step_s", "mean_step_s", "p90_step_s",
            "mean_over_median", "tokens_per_s_from_median", "steps_for_50M_tokens"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["p99_step_s", "steps_kept", "wall_s",
                           "swapouts_delta", "swap_used_delta_gb", "note"])
        for r in rows:
            w.writerow([r["batch"], r["status"], f"{r['peak_gb']:.4f}",
                        f"{r.get('median_step_s', float('nan')):.5f}",
                        f"{r.get('mean_step_s', float('nan')):.5f}",
                        f"{r.get('p90_step_s', float('nan')):.5f}",
                        f"{r.get('mean_over_median', float('nan')):.3f}",
                        f"{r.get('tokens_per_s_from_median', float('nan')):.1f}",
                        r["steps_for_50M_tokens"],
                        f"{r.get('p99_step_s', float('nan')):.5f}",
                        r.get("steps_kept", 0), f"{r['wall_s']:.1f}",
                        r.get("swapouts_delta", 0),
                        f"{r.get('swap_used_delta_gb', 0.0):.3f}", r.get("note", "")])

    print()
    print("=" * 108)
    print(f"BATCH SWEEP — {VARIANT}")
    print("=" * 108)
    print(f"{'batch':>6}{'status':>8}{'peak_gb':>9}{'median_s':>10}{'mean_s':>9}"
          f"{'p90_s':>9}{'mean/med':>10}{'tok/s(med)':>12}{'steps_50M':>11}")
    print("-" * 108)
    for r in rows:
        print(f"{r['batch']:>6}{r['status']:>8}{r['peak_gb']:>9.2f}"
              f"{r.get('median_step_s', float('nan')):>10.4f}"
              f"{r.get('mean_step_s', float('nan')):>9.4f}"
              f"{r.get('p90_step_s', float('nan')):>9.4f}"
              f"{r.get('mean_over_median', float('nan')):>10.3f}"
              f"{r.get('tokens_per_s_from_median', float('nan')):>12,.0f}"
              f"{r['steps_for_50M_tokens']:>11,}")
    print("-" * 108)

    slow = [r for r in rows if r["status"] == "slow"]
    if slow:
        print()
        print(f"WARNING: mean_over_median above {SLOW_RATIO} at batch(es): "
              f"{', '.join(str(r['batch']) for r in slow)}")
        print("Those rows stalled on a minority of steps. Either the batch exceeded what")
        print("the device can hold, or the machine was not quiet; the swap lines below say which.")

    vm_end, _ = vm_counters()
    print()
    print("-" * 108)
    print(f"total system RAM            : {gb(total_ram):.2f} GB")
    print(f"free RAM at sweep start     : {gb(free_start):.2f} GB")
    print(f"swap used at sweep start    : {gb(swap_start):.2f} GB")
    print(f"swap used at sweep end      : {gb(swap_used_bytes()):.2f} GB")
    print(f"MPS budget (recommended_max): {gb(budget):.2f} GB")
    print()
    if swap_events:
        print("swapping detected: YES")
        print("  method: vm_stat 'Swapouts' counter sampled immediately before and after")
        print("          each batch's subprocess; a positive delta means the kernel paged")
        print("          memory out to disk while that batch was running.")
        for b, d, gbd in swap_events:
            print(f"    batch {b:>4}: +{d:,} swapouts, swap used {gbd:+.2f} GB")
    else:
        print("swapping detected: NO")
        print("  method: vm_stat 'Swapouts' counter sampled immediately before and after")
        print("          each batch's subprocess; no batch produced a positive delta.")
    print()
    quiet = [r for r in rows
             if r.get("steps_kept", 0) >= MIN_SAMPLES
             and r.get("mean_over_median", float("inf")) < QUIET_RATIO]
    print(f"largest batch with mean_over_median < {QUIET_RATIO} : "
          f"{max(r['batch'] for r in quiet) if quiet else 'none'}"
          f"   (rows with < {MIN_SAMPLES} measured steps excluded)")
    thin = [r for r in rows if 0 < r.get("steps_kept", 0) < MIN_SAMPLES]
    if thin:
        parts = ["batch %d (n=%d)" % (r["batch"], r["steps_kept"]) for r in thin]
        print("  excluded as under-sampled: " + ", ".join(parts))
    usable = [r for r in rows if r.get("tokens_per_s_from_median", 0) == r.get("tokens_per_s_from_median", 0)
              and r["status"] != "oom"]
    if usable:
        best = max(usable, key=lambda r: r["tokens_per_s_from_median"])
        print(f"highest tokens_per_s_from_median       : batch {best['batch']} "
              f"at {best['tokens_per_s_from_median']:,.0f} tok/s (status {best['status']})")
    print("-" * 108)
    print(f"\nsaved to {CSV_PATH}")
    print("\nThis is evidence only. No batch size is recommended here.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(int(sys.argv[sys.argv.index("--batch") + 1]),
               sys.argv[sys.argv.index("--out") + 1])
    else:
        parent()
