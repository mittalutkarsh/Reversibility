"""
scripts/measure_memory.py

Measures what reversibility actually buys: peak MPS memory and throughput for
baseline / midpoint / midpoint_rev across batch sizes.

CLEAN PEAK READINGS
-------------------
Each (variant, batch) runs in a FRESH SUBPROCESS, one config at a time. MPS
caches allocations aggressively and torch.mps.empty_cache() does not reliably
return the driver to a pristine state, so a peak measured after an earlier
config in the same process reads high and is not comparable. Restarting is the
only way to get a clean number. The CSV records this in the `isolated` column.

Peak is sampled by a background thread polling
torch.mps.driver_allocated_memory() every 100 ms for the life of the run --
that is the driver's view, which includes MPS's own cache, so it is the number
that matters for "will this batch size fit".

FAILURE HANDLING
----------------
A config that OOMs, errors, or exceeds the wall-clock budget is recorded with
that status and the scan continues. Once a variant TIMES OUT at some batch
size, larger batches for that variant are recorded as `skipped` rather than
run -- they cannot be faster, and each timeout costs the full budget. OOM does
NOT trigger skipping, because OOM is cheap and the batch size at which each
variant dies is the interesting result.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig, get_batch  # noqa: E402

SEED = 1337

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BIN = os.path.join(ROOT, "data", "train.bin")
CSV_PATH = os.path.join(ROOT, "runs", "memory_scan.csv")

VARIANTS = ("baseline", "midpoint", "midpoint_rev")
BATCHES = (8, 16, 32, 64)
BLOCK_SIZE = 1024
STEPS = 20
LR = 3e-4
POLL_SEC = 0.1
TIMEOUT_SEC = 600


class MemPoller(threading.Thread):
    """Samples driver-allocated MPS memory until told to stop.

    NOTE the attribute name: `_stop_evt`, not `_stop`. threading.Thread
    already has a `_stop` METHOD that join() calls internally, so binding an
    Event there makes every join() raise "'Event' object is not callable"
    during teardown -- after the measured work has already succeeded.

    Checkpoints the running peak to `out_path` every ~1s so that a config the
    parent has to kill on timeout still reports the peak it reached.
    """

    def __init__(self, out_path=None, state=None, interval=POLL_SEC):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.samples = 0
        self.out_path = out_path
        self.state = state
        self._stop_evt = threading.Event()
        self._since_write = 0

    def run(self):
        while not self._stop_evt.is_set():
            try:
                v = torch.mps.driver_allocated_memory()
                self.samples += 1
                if v > self.peak:
                    self.peak = v
            except Exception:
                pass
            self._since_write += 1
            if self.out_path and self._since_write >= 10:
                self._since_write = 0
                self._checkpoint()
            self._stop_evt.wait(self.interval)

    def _checkpoint(self):
        try:
            with open(self.out_path, "w", encoding="utf-8") as f:
                json.dump({"peak_bytes": self.peak,
                           "poll_samples": self.samples,
                           "steps": (self.state or {}).get("steps", 0)}, f)
        except Exception:
            pass

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)


def worker(variant, batch, out_path=None):
    """Run one config in this process and emit a RESULT json line."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps"

    result = {
        "variant": variant, "batch": batch, "status": "ok",
        "peak_bytes": 0, "tokens_per_s": 0.0, "tokens_per_s_warm": 0.0,
        "steps": 0, "wall_s": 0.0, "note": "",
    }

    poller = MemPoller(out_path=out_path, state=result)
    poller.start()
    try:
        data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
        model = GPT(GPTConfig(variant=variant), verbose=False).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        g = torch.Generator().manual_seed(SEED)

        step_times = []
        t0 = time.time()
        for _ in range(STEPS):
            ts = time.time()
            x, y = get_batch(data, batch, BLOCK_SIZE, device, g)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lv = loss.item()          # syncs
            step_times.append(time.time() - ts)
            if not np.isfinite(lv):
                result["status"] = "nonfinite"
                result["note"] = "loss went non-finite"
                break
            result["steps"] += 1
        wall = time.time() - t0

        tok = result["steps"] * batch * BLOCK_SIZE
        result["wall_s"] = wall
        result["tokens_per_s"] = tok / wall if wall > 0 else 0.0
        if len(step_times) > 1:
            warm = sum(step_times[1:])
            result["tokens_per_s_warm"] = (
                (result["steps"] - 1) * batch * BLOCK_SIZE / warm if warm > 0 else 0.0
            )
    except RuntimeError as exc:
        msg = str(exc)
        oom = "out of memory" in msg.lower() or "insufficient" in msg.lower()
        result["status"] = "oom" if oom else "error"
        result["note"] = msg.replace("\n", " ")[:160]
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["note"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:160]
    finally:
        poller.stop()
        result["peak_bytes"] = poller.peak
        result["poll_samples"] = poller.samples

    print("RESULT " + json.dumps(result), flush=True)


def gb(n):
    return n / (1024 ** 3)


def parent():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    here = os.path.abspath(__file__)

    budget = None
    if hasattr(torch.mps, "recommended_max_memory"):
        budget = torch.mps.recommended_max_memory()

    print("=" * 84)
    print("MEMORY SCAN — baseline vs midpoint vs midpoint_rev")
    print("=" * 84)
    print(f"seed          : {SEED}")
    print(f"steps/config  : {STEPS}")
    print(f"context       : {BLOCK_SIZE}")
    print(f"poll          : torch.mps.driver_allocated_memory() every {POLL_SEC * 1000:.0f} ms")
    print(f"isolation     : fresh subprocess per config (restarted between every one)")
    print(f"timeout       : {TIMEOUT_SEC}s per config")
    if budget:
        print(f"MPS budget    : {gb(budget):.2f} GB (torch.mps.recommended_max_memory)")
    print()
    print(f"{'variant':<14}{'batch':>6}{'status':>12}{'peak GB':>10}"
          f"{'tok/s':>10}{'tok/s warm':>12}{'wall s':>9}")
    print("-" * 84)

    rows = []
    timed_out_at = {}

    for variant in VARIANTS:
        for batch in BATCHES:
            if variant in timed_out_at and batch > timed_out_at[variant]:
                row = {
                    "variant": variant, "batch": batch, "status": "skipped",
                    "peak_bytes": 0, "tokens_per_s": 0.0, "tokens_per_s_warm": 0.0,
                    "steps": 0, "wall_s": 0.0,
                    "note": f"larger than batch {timed_out_at[variant]} which timed out",
                }
                rows.append(row)
                print(f"{variant:<14}{batch:>6}{'skipped':>12}{'-':>10}{'-':>10}{'-':>12}{'-':>9}")
                continue

            out_file = os.path.join(
                ROOT, "runs", f".partial_{variant}_{batch}.json")
            if os.path.exists(out_file):
                os.remove(out_file)
            cmd = [sys.executable, "-u", here, "--worker",
                   "--variant", variant, "--batch", str(batch),
                   "--out", out_file]
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=TIMEOUT_SEC)
                out = proc.stdout
                line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
                if line is None:
                    row = {
                        "variant": variant, "batch": batch, "status": "error",
                        "peak_bytes": 0, "tokens_per_s": 0.0, "tokens_per_s_warm": 0.0,
                        "steps": 0, "wall_s": time.time() - t0,
                        "note": (proc.stderr or "no RESULT line").replace("\n", " ")[:160],
                    }
                else:
                    row = json.loads(line[len("RESULT "):])
            except subprocess.TimeoutExpired:
                partial = {}
                if os.path.exists(out_file):
                    try:
                        with open(out_file, encoding="utf-8") as pf:
                            partial = json.load(pf)
                    except Exception:
                        pass
                done = partial.get("steps", 0)
                row = {
                    "variant": variant, "batch": batch, "status": "timeout",
                    "peak_bytes": partial.get("peak_bytes", 0),
                    "tokens_per_s": 0.0, "tokens_per_s_warm": 0.0,
                    "steps": done, "wall_s": time.time() - t0,
                    "note": f"exceeded {TIMEOUT_SEC}s after {done}/{STEPS} steps; "
                            f"peak is from the partial run",
                }
                timed_out_at[variant] = batch

            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                except OSError:
                    pass
            rows.append(row)
            pk = f"{gb(row['peak_bytes']):.2f}" if row["peak_bytes"] else "-"
            tps = f"{row['tokens_per_s']:,.0f}" if row["tokens_per_s"] else "-"
            tpw = f"{row['tokens_per_s_warm']:,.0f}" if row.get("tokens_per_s_warm") else "-"
            print(f"{variant:<14}{batch:>6}{row['status']:>12}{pk:>10}"
                  f"{tps:>10}{tpw:>12}{row['wall_s']:>9.1f}", flush=True)

    print("-" * 84)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "batch", "context", "steps_requested", "steps_done",
                    "status", "peak_bytes", "peak_gb", "tokens_per_s",
                    "tokens_per_s_warm", "wall_s", "isolated", "poll_ms", "note"])
        for r in rows:
            w.writerow([
                r["variant"], r["batch"], BLOCK_SIZE, STEPS, r.get("steps", 0),
                r["status"], r["peak_bytes"], f"{gb(r['peak_bytes']):.4f}",
                f"{r['tokens_per_s']:.1f}", f"{r.get('tokens_per_s_warm', 0.0):.1f}",
                f"{r['wall_s']:.2f}", "yes", int(POLL_SEC * 1000), r.get("note", ""),
            ])

    ok = {(r["variant"], r["batch"]): r for r in rows if r["status"] == "ok"}

    print()
    print("=" * 84)
    print("SUMMARY — record these")
    print("=" * 84)
    print(f"configs run          : {len(rows)}  ({sum(1 for r in rows if r['status'] == 'ok')} ok, "
          f"{sum(1 for r in rows if r['status'] == 'oom')} oom, "
          f"{sum(1 for r in rows if r['status'] == 'timeout')} timeout, "
          f"{sum(1 for r in rows if r['status'] == 'skipped')} skipped, "
          f"{sum(1 for r in rows if r['status'] == 'error')} error)")
    print(f"process restarted    : yes, between every configuration")
    if budget:
        print(f"MPS budget           : {gb(budget):.2f} GB")
    print()
    print("peak GB by batch (blank = did not complete)")
    print(f"{'variant':<14}" + "".join(f"{b:>12}" for b in BATCHES))
    print("-" * 84)
    for v in VARIANTS:
        cells = []
        for b in BATCHES:
            r = ok.get((v, b))
            cells.append(f"{gb(r['peak_bytes']):.2f}" if r else "-")
        print(f"{v:<14}" + "".join(f"{c:>12}" for c in cells))
    print()
    print("midpoint_rev vs midpoint, where both completed")
    print("-" * 84)
    any_pair = False
    for b in BATCHES:
        a, c = ok.get(("midpoint", b)), ok.get(("midpoint_rev", b))
        if a and c:
            any_pair = True
            save = 1.0 - gb(c["peak_bytes"]) / gb(a["peak_bytes"])
            slow = (a["tokens_per_s_warm"] / c["tokens_per_s_warm"]
                    if c.get("tokens_per_s_warm") else float("nan"))
            print(f"  batch {b:>3}: {gb(a['peak_bytes']):.2f} GB -> {gb(c['peak_bytes']):.2f} GB "
                  f"({save * 100:.1f}% LESS memory), {slow:.2f}x slower")
    if not any_pair:
        print("  (no batch size where both completed)")
    print()
    print(f"largest completed batch per variant")
    for v in VARIANTS:
        bs = [b for b in BATCHES if (v, b) in ok]
        print(f"  {v:<14}: {max(bs) if bs else 'none'}")
    print("=" * 84)
    print(f"\nsaved to {CSV_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.worker:
        worker(args.variant, args.batch, args.out)
    else:
        parent()


if __name__ == "__main__":
    main()
