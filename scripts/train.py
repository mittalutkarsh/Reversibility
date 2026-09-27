"""
scripts/train.py

One full training run. Everything except --variant is identical across runs,
so the three runs are comparable by construction.

    python3 -u scripts/train.py --variant baseline     --batch 16 --name 01_baseline_bs16
    python3 -u scripts/train.py --variant euler        --batch 16 --name 02_euler_bs16
    python3 -u scripts/train.py --variant midpoint_rev --batch 16 --name 03_midpoint_rev_bs16

TOKEN BUDGET
------------
50,000,000 is not divisible by batch*context = 16,384. 3,052 steps is the
smallest count that reaches the budget: 50,003,968 tokens, 3,968 over (0.008%).
Every run overshoots identically, so the comparison is unaffected. config.json
and experiments.csv record the exact figure rather than claiming 50,000,000.

DETERMINISM
-----------
The train stream is driven by a generator seeded from SEED, and every variant
draws the same number of batches per step, so all three runs see the SAME data
in the SAME order. Validation uses a separate generator so it cannot perturb
that stream. config.json records the first 10 token ids of the first batch;
if those differ between runs, the data order was not held fixed.

EULER trains with standard autograd. Step 2 proved its inverse is wrong, so it
gets no reversible backward -- but its loss trajectory is still worth having.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig, get_batch  # noqa: E402

SEED = 1337

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BIN = os.path.join(ROOT, "data", "train.bin")
VAL_BIN = os.path.join(ROOT, "data", "val.bin")
RUNS_DIR = os.path.join(ROOT, "runs")
EXPERIMENTS_CSV = os.path.join(ROOT, "experiments.csv")

TOKEN_BUDGET = 50_000_000
BLOCK_SIZE = 1024

# optimiser -- fixed for all runs, not tuned per variant
LR_MAX = 3e-4
LR_MIN = 3e-5
WARMUP_STEPS = 100
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

LOG_EVERY = 50
VAL_EVERY = 200
VAL_BATCHES = 20          # fixed set, identical every run
VAL_MICRO = 4             # evaluate in micro-batches; see build_val_batches
# Validation set size is FROZEN in absolute tokens, independent of the
# training batch. 20 x 8 x 1024 = 163,840 tokens -- exactly what runs 01-03
# consumed at batch 8. Previously this scaled with the training batch, which
# at batch 32 would have validated on 655,360 tokens (more than val.bin holds)
# and made run 04's loss incomparable with the others.
VAL_TOKENS = VAL_BATCHES * 8 * BLOCK_SIZE
WARMUP_EXCLUDE = 50       # steps excluded from the tokens/s figure
POLL_SEC = 0.1


class MemPoller(threading.Thread):
    """Peak driver-allocated MPS memory, sampled every POLL_SEC.

    The Event is `_stop_evt`, NOT `_stop`: threading.Thread already has a
    `_stop` method that join() calls internally.
    """

    def __init__(self, interval=POLL_SEC):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.samples = 0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                v = torch.mps.driver_allocated_memory()
                self.samples += 1
                if v > self.peak:
                    self.peak = v
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)


def gb(n):
    return n / (1024 ** 3)


def lr_at(step, total_steps):
    """Linear warmup then cosine decay LR_MAX -> LR_MIN."""
    if step < WARMUP_STEPS:
        return LR_MAX * (step + 1) / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    prog = min(1.0, max(0.0, prog))
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * prog))


def build_val_batches(batch, device):
    """A FIXED set of validation batches -- same offsets in every run.

    Evaluated in micro-batches of VAL_MICRO rather than at the training batch
    size. A batch-16 eval forward pushed the MPS allocator cache from 9.58 GB
    to 10.08 GB, past what the 11.84 GB budget absorbs, and the run never
    recovered: 1.4 s/step before the first validation, ~25 s/step after. The
    micro-batches keep the eval peak below the training peak.

    The METRIC IS UNCHANGED. Every micro-batch has the same token count, so
    the mean of the micro-batch losses equals the mean over the same token set
    that batch-sized grouping would give. Equal sizes are what makes
    mean-of-means exact here.

    The set does NOT depend on `batch`: VAL_TOKENS is absolute, and the
    generator is seeded identically, so every run -- at any training batch
    size -- validates on the same 40 micro-batches at the same offsets. That
    is what makes final_val_loss comparable across runs 01-04.
    """
    val = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    g = torch.Generator().manual_seed(SEED + 1)   # separate from the train stream
    micro = VAL_MICRO
    n_micro = VAL_TOKENS // (micro * BLOCK_SIZE)
    return [get_batch(val, micro, BLOCK_SIZE, device, g) for _ in range(n_micro)]


@torch.no_grad()
def evaluate(model, val_batches):
    model.eval()
    total = 0.0
    for x, y in val_batches:
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    # Hand the eval's allocator cache back. Without this the driver footprint
    # ratchets up at every validation and never comes down.
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
    return total / len(val_batches)


def append_experiment_row(row):
    header = list(row.keys())
    exists = os.path.exists(EXPERIMENTS_CSV)
    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="override step count (smoke tests only)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokens_per_step = args.batch * BLOCK_SIZE
    total_steps = args.steps or math.ceil(TOKEN_BUDGET / tokens_per_step)
    total_tokens = total_steps * tokens_per_step

    run_dir = os.path.join(RUNS_DIR, args.name)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 78)
    print(f"TRAIN — {args.name}")
    print("=" * 78)
    print(f"variant        : {args.variant}")
    print(f"device         : {device}   precision: fp32")
    print(f"batch x context: {args.batch} x {BLOCK_SIZE} = {tokens_per_step:,} tokens/step")
    print(f"steps          : {total_steps:,}  -> {total_tokens:,} tokens")
    print(f"lr             : {LR_MAX} -> {LR_MIN} cosine, {WARMUP_STEPS} warmup")
    print(f"adamw          : betas={BETAS} wd={WEIGHT_DECAY} clip={GRAD_CLIP}")
    print(f"val            : every {VAL_EVERY} steps, {VAL_TOKENS:,} fixed tokens "
          f"in micro-batches of {VAL_MICRO} (same set for every run)")
    print()

    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_batches = build_val_batches(args.batch, device)

    model = GPT(GPTConfig(variant=args.variant), verbose=True).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX,
                            betas=BETAS, weight_decay=WEIGHT_DECAY)
    g = torch.Generator().manual_seed(SEED)

    # Fetch batch 0 up front so its token ids can go into config.json, which
    # must exist before the first step. It is then used AS step 0, so the data
    # order is unchanged.
    x0, y0 = get_batch(train_data, args.batch, BLOCK_SIZE, device, g)
    first_ids = [int(v) for v in x0[0, :10].tolist()]

    config = {
        "run_name": args.name,
        "variant": args.variant,
        "seed": SEED,
        "device": device,
        "precision": "fp32",
        "batch_size": args.batch,
        "context": BLOCK_SIZE,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "token_budget_requested": TOKEN_BUDGET,
        "total_tokens_actual": total_tokens,
        "token_overshoot": total_tokens - TOKEN_BUDGET,
        "lr_max": LR_MAX, "lr_min": LR_MIN, "warmup_steps": WARMUP_STEPS,
        "schedule": "linear warmup then cosine",
        "adamw_betas": list(BETAS),
        "weight_decay": WEIGHT_DECAY,
        "weight_decay_applies_to": "all parameters (single param group)",
        "grad_clip": GRAD_CLIP,
        "dropout": 0.0,
        "h": GPTConfig().h, "blend": GPTConfig().blend,
        "params": model.num_params(),
        "val_every": VAL_EVERY, "val_batches": VAL_BATCHES,
        "val_micro_batch": VAL_MICRO,
        "val_tokens": VAL_TOKENS,
        "log_every": LOG_EVERY,
        "tokens_per_s_excludes_first_steps": WARMUP_EXCLUDE,
        "first_batch_first_10_token_ids": first_ids,
        "torch_version": torch.__version__,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"config.json written. first 10 token ids: {first_ids}")
    print()

    poller = MemPoller()
    poller.start()

    metrics_f = open(os.path.join(run_dir, "metrics.csv"), "w", newline="", encoding="utf-8")
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow(["step", "tokens_seen", "train_loss", "lr", "grad_norm",
                        "step_time_s", "mem_current_gb", "mem_driver_gb", "mem_peak_gb"])
    val_f = open(os.path.join(run_dir, "val.csv"), "w", newline="", encoding="utf-8")
    val_w = csv.writer(val_f)
    val_w.writerow(["step", "tokens_seen", "val_loss", "wall_s"])

    status = "aborted"
    note = ""
    step_times = []
    val_points = []
    last_train_loss = float("nan")
    steps_done = 0
    t_start = time.time()

    try:
        for step in range(total_steps):
            lr = lr_at(step, total_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            ts = time.time()
            x, y = (x0, y0) if step == 0 else get_batch(
                train_data, args.batch, BLOCK_SIZE, device, g)

            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            lv = loss.item()          # syncs
            dt = time.time() - ts
            step_times.append(dt)
            last_train_loss = lv
            steps_done = step + 1
            tokens_seen = steps_done * tokens_per_step

            if not math.isfinite(lv):
                status = "diverged"
                note = f"non-finite loss at step {step}"
                print(f"\nDIVERGED: loss {lv} at step {step}. Curves saved; not restarting.")
                break

            if step % LOG_EVERY == 0 or steps_done == total_steps:
                metrics_w.writerow([
                    step, tokens_seen, f"{lv:.6f}", f"{lr:.8f}",
                    f"{float(gnorm):.6f}", f"{dt:.4f}",
                    f"{gb(torch.mps.current_allocated_memory()):.4f}",
                    f"{gb(torch.mps.driver_allocated_memory()):.4f}",
                    f"{gb(poller.peak):.4f}",
                ])
                metrics_f.flush()
                el = time.time() - t_start
                eta = (total_steps - steps_done) * (el / steps_done)
                print(f"step {step:>5}/{total_steps}  loss {lv:.4f}  lr {lr:.2e}  "
                      f"gn {float(gnorm):.2f}  {dt:.2f}s  peak {gb(poller.peak):.2f}GB  "
                      f"eta {eta / 60:.0f}m", flush=True)

            if (step + 1) % VAL_EVERY == 0 or steps_done == total_steps:
                vl = evaluate(model, val_batches)
                val_points.append(vl)
                val_w.writerow([step, tokens_seen, f"{vl:.6f}", f"{time.time() - t_start:.1f}"])
                val_f.flush()
                print(f"           val {vl:.4f}  (point {len(val_points)})", flush=True)

        else:
            status = "completed"

        if status == "aborted" and steps_done == total_steps:
            status = "completed"

    except KeyboardInterrupt:
        status = "interrupted"
        note = "KeyboardInterrupt"
        print("\ninterrupted", flush=True)
    except Exception as exc:  # noqa: BLE001
        status = "aborted"
        note = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
        print(f"\nABORTED: {note}", flush=True)
    finally:
        wall = time.time() - t_start
        poller.stop()
        try:
            metrics_f.close()
            val_f.close()
        except Exception:
            pass

        warm = step_times[WARMUP_EXCLUDE:] if len(step_times) > WARMUP_EXCLUDE else step_times
        median_step = statistics.median(warm) if warm else float("nan")
        tok_warm = len(warm) * tokens_per_step
        secs_warm = sum(warm)
        tps = tok_warm / secs_warm if secs_warm > 0 else float("nan")
        final_val = (statistics.mean(val_points[-5:])
                     if len(val_points) >= 1 else float("nan"))

        row = {
            "run_name": args.name,
            "variant": args.variant,
            "status": status,
            "batch": args.batch,
            "context": BLOCK_SIZE,
            "steps_done": steps_done,
            "steps_planned": total_steps,
            "tokens_seen": steps_done * tokens_per_step,
            "seed": SEED,
            "final_val_loss_mean_last5": f"{final_val:.6f}",
            "val_points": len(val_points),
            "best_val_loss": f"{min(val_points):.6f}" if val_points else "",
            "last_train_loss": f"{last_train_loss:.6f}",
            "median_step_time_s": f"{median_step:.4f}",
            "tokens_per_s": f"{tps:.1f}",
            "peak_gb": f"{gb(poller.peak):.4f}",
            "wall_s": f"{wall:.1f}",
            "params": model.num_params(),
            "lr_max": LR_MAX, "lr_min": LR_MIN,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "note": note,
        }
        append_experiment_row(row)

        print()
        print("=" * 78)
        print("SUMMARY — record these")
        print("=" * 78)
        print(f"run                  : {args.name}")
        print(f"variant              : {args.variant}")
        print(f"status               : {status}")
        print(f"steps                : {steps_done:,} / {total_steps:,}")
        print(f"tokens seen          : {steps_done * tokens_per_step:,}")
        print(f"final val (mean last5): {final_val:.4f}   over {len(val_points)} points")
        print(f"best val             : {min(val_points):.4f}" if val_points else "best val : -")
        print(f"last train loss      : {last_train_loss:.4f}")
        print(f"median step time     : {median_step:.3f}s   (excl. first {WARMUP_EXCLUDE})")
        print(f"tokens/s             : {tps:,.0f}   (excl. first {WARMUP_EXCLUDE} steps)")
        print(f"peak memory          : {gb(poller.peak):.2f} GB")
        print(f"wall clock           : {wall / 60:.1f} min")
        print(f"note                 : {note or '-'}")
        print("=" * 78)
        print(f"\nrun dir: {run_dir}")
        print(f"experiments.csv row appended (status={status})")


if __name__ == "__main__":
    main()
