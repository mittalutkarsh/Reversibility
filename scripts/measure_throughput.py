"""
scripts/measure_throughput.py

PART A -- clean throughput for the three variants at batch 8.

Step 4's tokens/s column was unusable: baseline's wall time was 5.26x what its
own median step time predicted, euler's 3.02x, midpoint_rev's 1.07x. That
spread measured how much swap pressure each run happened to meet, not the
variants. This measures the step itself.

METHOD
  - 200 steps per variant, first 50 discarded
  - fresh subprocess per variant, so no allocator state carries over
  - the measured step is the same shape as train.py's: forward, backward,
    grad clip, optimiser step, and a loss.item() to force the sync
  - tokens/s is derived from the MEDIAN step time, never from wall clock

mean_over_median is the honesty check. A quiet machine gives ~1.0. Anything
above 1.5 means a minority of steps stalled and the row is marked 'slow'.
"""

import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig, get_batch      # noqa: E402
from measure_memory import MemPoller, gb          # noqa: E402

SEED = 1337
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BIN = os.path.join(ROOT, "data", "train.bin")
CSV_PATH = os.path.join(ROOT, "runs", "throughput_clean.csv")

VARIANTS = ("baseline", "euler", "midpoint_rev")
BATCH = 8
BLOCK_SIZE = 1024
STEPS = 200
DISCARD = 50
LR = 3e-4
GRAD_CLIP = 1.0
SLOW_RATIO = 1.5
TIMEOUT_SEC = 900


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def vm_counters():
    """Parse vm_stat into {name: pages} plus the page size."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        return {}, 4096
    page = 4096
    first = out.splitlines()[0] if out else ""
    if "page size of" in first:
        try:
            page = int(first.split("page size of")[1].split("bytes")[0].strip())
        except Exception:
            pass
    d = {}
    for line in out.splitlines()[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip().rstrip(".")
            if v.isdigit():
                d[k.strip()] = int(v)
    return d, page


def swap_used_bytes():
    s = sysctl("vm.swapusage")
    for tok in s.split():
        if tok.endswith("M") and "used" in s:
            pass
    try:
        part = s.split("used =")[1].split()[0]
        return float(part.rstrip("M")) * 1024 * 1024
    except Exception:
        return float("nan")


def stats(times, batch):
    a = np.asarray(times, dtype=float)
    med = float(np.median(a))
    return {
        "median_step_s": med,
        "mean_step_s": float(a.mean()),
        "p90_step_s": float(np.percentile(a, 90)),
        "p99_step_s": float(np.percentile(a, 99)),
        "mean_over_median": float(a.mean() / med) if med > 0 else float("nan"),
        "tokens_per_s_from_median": (batch * BLOCK_SIZE) / med if med > 0 else float("nan"),
    }


def worker(variant):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps"
    res = {"variant": variant, "batch": BATCH, "status": "ok", "note": ""}

    poller = MemPoller()
    poller.start()
    try:
        data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
        model = GPT(GPTConfig(variant=variant), verbose=False).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        g = torch.Generator().manual_seed(SEED)

        times = []
        for _ in range(STEPS):
            t0 = time.time()
            x, y = get_batch(data, BATCH, BLOCK_SIZE, device, g)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            loss.item()                      # force the sync
            times.append(time.time() - t0)

        kept = times[DISCARD:]
        res.update(stats(kept, BATCH))
        res["steps_kept"] = len(kept)
        if res["mean_over_median"] > SLOW_RATIO:
            res["status"] = "slow"
    except RuntimeError as exc:
        msg = str(exc)
        res["status"] = "oom" if "out of memory" in msg.lower() else "error"
        res["note"] = msg.replace("\n", " ")[:160]
    finally:
        poller.stop()
        res["peak_gb"] = gb(poller.peak)

    print("RESULT " + json.dumps(res), flush=True)


def parent():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    here = os.path.abspath(__file__)

    print("=" * 96)
    print("!!  CLOSE ALL OTHER APPLICATIONS BEFORE READING THESE NUMBERS  !!")
    print("!!  Chrome, Edge, Cursor, Slack, anything holding memory.      !!")
    print("!!  Step 4 showed a 14x swing in step time purely from load.   !!")
    print("=" * 96)
    print()

    total_ram = int(sysctl("hw.memsize") or 0)
    vm, page = vm_counters()
    free_b = (vm.get("Pages free", 0) + vm.get("Pages speculative", 0)) * page
    print(f"total RAM      : {gb(total_ram):.2f} GB")
    print(f"free RAM       : {gb(free_b):.2f} GB")
    print(f"swap used      : {gb(swap_used_bytes()):.2f} GB")
    if hasattr(torch.mps, "recommended_max_memory"):
        print(f"MPS budget     : {gb(torch.mps.recommended_max_memory()):.2f} GB")
    print(f"steps          : {STEPS} per variant, first {DISCARD} discarded")
    print(f"batch          : {BATCH} x {BLOCK_SIZE}")
    print(f"isolation      : fresh subprocess per variant")
    print()

    rows = []
    for variant in VARIANTS:
        print(f"  running {variant} ...", flush=True)
        cmd = [sys.executable, "-u", here, "--worker", "--variant", variant]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
            row = json.loads(line[len("RESULT "):]) if line else {
                "variant": variant, "batch": BATCH, "status": "error",
                "note": (proc.stderr or "no RESULT")[:160], "peak_gb": 0.0}
        except subprocess.TimeoutExpired:
            row = {"variant": variant, "batch": BATCH, "status": "error",
                   "note": f"exceeded {TIMEOUT_SEC}s", "peak_gb": 0.0}
        rows.append(row)

    cols = ["variant", "batch", "median_step_s", "mean_step_s", "p90_step_s",
            "p99_step_s", "mean_over_median", "tokens_per_s_from_median",
            "peak_gb", "status"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["steps_kept", "note"])
        for r in rows:
            w.writerow([r.get("variant"), r.get("batch"),
                        f"{r.get('median_step_s', float('nan')):.5f}",
                        f"{r.get('mean_step_s', float('nan')):.5f}",
                        f"{r.get('p90_step_s', float('nan')):.5f}",
                        f"{r.get('p99_step_s', float('nan')):.5f}",
                        f"{r.get('mean_over_median', float('nan')):.3f}",
                        f"{r.get('tokens_per_s_from_median', float('nan')):.1f}",
                        f"{r.get('peak_gb', 0.0):.4f}", r.get("status"),
                        r.get("steps_kept", 0), r.get("note", "")])

    print()
    print("=" * 96)
    print("THROUGHPUT — batch 8, tokens/s from MEDIAN step time")
    print("=" * 96)
    hdr = (f"{'variant':<14}{'batch':>6}{'median_s':>10}{'mean_s':>9}{'p90_s':>9}"
           f"{'p99_s':>9}{'mean/med':>10}{'tok/s(med)':>12}{'peak_gb':>9}{'status':>8}")
    print(hdr)
    print("-" * 96)
    for r in rows:
        print(f"{r.get('variant',''):<14}{r.get('batch',''):>6}"
              f"{r.get('median_step_s',float('nan')):>10.4f}{r.get('mean_step_s',float('nan')):>9.4f}"
              f"{r.get('p90_step_s',float('nan')):>9.4f}{r.get('p99_step_s',float('nan')):>9.4f}"
              f"{r.get('mean_over_median',float('nan')):>10.3f}"
              f"{r.get('tokens_per_s_from_median',float('nan')):>12,.0f}"
              f"{r.get('peak_gb',0.0):>9.2f}{r.get('status',''):>8}")
    print("-" * 96)

    slow = [r for r in rows if r.get("status") == "slow"]
    if slow:
        print()
        print("WARNING: mean_over_median above "
              f"{SLOW_RATIO} for: {', '.join(r['variant'] for r in slow)}")
        print("The machine may not have been quiet. Close other applications and rerun;")
        print("these rows understate the throughput the variant is capable of.")
    else:
        print()
        print(f"All rows have mean_over_median <= {SLOW_RATIO}: the machine was quiet.")
    print(f"\nsaved to {CSV_PATH}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(sys.argv[sys.argv.index("--variant") + 1])
    else:
        parent()
