"""
scripts/smoke_train.py

200 steps on the BASELINE variant only, to confirm the model learns at all
before any reversible machinery is wired up. Not the real training loop: no
schedule, no eval, no checkpointing, no logging infrastructure.

Expectation: step 0 loss near ln(8192) = 9.011 (uniform over the vocab), then
a clear drop. Asserts the first loss is in [8.5, 9.5] -- outside that band the
init is wrong and nothing downstream is worth running.

BATCH SHAPE
-----------
Effective batch is 16 x 1024 as specified, reached as 2 gradient-accumulation
micro-batches of 8 x 1024. This is arithmetic-identical to a single batch of
16 for the optimiser (the micro-losses are averaged before the step), but it
avoids a hard memory cliff on this 16 GB M1 Pro:

    batch 16 x 1024 :  4.14 s/step   3,962 tok/s   1.94 GB MPS alloc
    batch  8 x 1024 :  0.62 s/step  13,308 tok/s   0.92 GB MPS alloc

Batch 8 is 3.4x faster PER TOKEN, so batch 16 is not compute-bound -- it is
spilling. Measured on a machine whose swap was already ~26 GB deep from other
applications, so the cliff may sit elsewhere on an idle machine.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig, get_batch  # noqa: E402

SEED = 1337

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BIN = os.path.join(ROOT, "data", "train.bin")

VARIANT = "baseline"
STEPS = 200
MICRO_BATCH = 8
ACCUM = 2
EFFECTIVE_BATCH = MICRO_BATCH * ACCUM   # 16, as specified
BLOCK_SIZE = 1024
LR = 3e-4
LOG_EVERY = 20

FIRST_LOSS_MIN = 8.5
FIRST_LOSS_MAX = 9.5


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()

    print("=" * 72)
    print("SMOKE TRAIN — baseline only")
    print("=" * 72)
    print(f"seed         : {SEED}")
    print(f"device       : {device}")
    print(f"variant      : {VARIANT}")
    print(f"steps        : {STEPS}")
    print(f"micro-batch  : {MICRO_BATCH} x {BLOCK_SIZE}, accum {ACCUM}")
    print(f"effective    : {EFFECTIVE_BATCH} x {BLOCK_SIZE} = "
          f"{EFFECTIVE_BATCH * BLOCK_SIZE:,} tokens/step")
    print(f"optimiser    : AdamW lr={LR}")
    print(f"precision    : fp32")
    print()

    data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    print(f"train.bin    : {len(data):,} tokens", flush=True)

    model = GPT(GPTConfig(variant=VARIANT)).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)

    print()
    losses = []
    first_loss = None
    t0 = time.time()

    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(ACCUM):
            x, y = get_batch(data, MICRO_BATCH, BLOCK_SIZE, device, g)
            _, loss = model(x, y)
            # average over micro-batches so the gradient matches a single
            # batch of EFFECTIVE_BATCH
            (loss / ACCUM).backward()
            step_loss += loss.item() / ACCUM
        opt.step()

        losses.append(step_loss)

        if step == 0:
            first_loss = step_loss
            print(f"step {step:>4}  loss {step_loss:.4f}   (ln(8192) = 9.011)", flush=True)
            assert FIRST_LOSS_MIN <= step_loss <= FIRST_LOSS_MAX, (
                f"first loss {step_loss:.4f} outside "
                f"[{FIRST_LOSS_MIN}, {FIRST_LOSS_MAX}] -- init is wrong, stop here"
            )
        elif (step + 1) % LOG_EVERY == 0:
            el = time.time() - t0
            print(f"step {step + 1:>4}  loss {step_loss:.4f}   "
                  f"{el:.0f}s  "
                  f"{(step + 1) * EFFECTIVE_BATCH * BLOCK_SIZE / el:,.0f} tok/s",
                  flush=True)

    elapsed = time.time() - t0
    last20 = sum(losses[-20:]) / len(losses[-20:])

    print()
    print("=" * 72)
    print("SUMMARY — record these")
    print("=" * 72)
    print(f"variant              : {VARIANT}")
    print(f"device / precision   : {device} / fp32")
    print(f"params               : {model.num_params():,}")
    print(f"steps                : {STEPS}")
    print(f"effective batch      : {EFFECTIVE_BATCH} x {BLOCK_SIZE} "
          f"({MICRO_BATCH} x {BLOCK_SIZE}, accum {ACCUM})")
    print(f"tokens seen          : {STEPS * EFFECTIVE_BATCH * BLOCK_SIZE:,}")
    print(f"first loss           : {first_loss:.4f}  (band [{FIRST_LOSS_MIN}, {FIRST_LOSS_MAX}])")
    print(f"final loss           : {losses[-1]:.4f}")
    print(f"mean last 20 steps   : {last20:.4f}")
    print(f"min loss             : {min(losses):.4f}")
    print(f"drop first -> final  : {first_loss - losses[-1]:.4f}")
    print(f"wall clock           : {elapsed:.1f}s  ({elapsed / STEPS:.2f}s/step)")
    print(f"throughput           : {STEPS * EFFECTIVE_BATCH * BLOCK_SIZE / elapsed:,.0f} tok/s")
    print(f"all losses finite    : {all(np.isfinite(losses))}")
    print("=" * 72)


if __name__ == "__main__":
    main()
