"""
scripts/check_gradients.py

THE NEW GATE. The reversible backward is only worth having if it produces the
same gradients as ordinary autograd. This runs both on one real batch and
compares every parameter tensor.

  A: variant "midpoint"      -- stores every state, standard autograd
  B: variant "midpoint_rev"  -- stores the final pair, reconstructs the rest

Both models are built from the same SEED, so their parameters are identical
bit for bit before the backward; any difference in the gradients comes from
the backward itself.

RELATIVE DIFFERENCE
-------------------
Reported per tensor as  max|gA - gB| / max|gA|  -- normalised by the TENSOR's
scale, not elementwise. An elementwise ratio explodes meaninglessly wherever
gA has a near-zero entry, which every gradient tensor has, and would say
nothing about whether the backward is correct.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig, get_batch  # noqa: E402

SEED = 1337

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_BIN = os.path.join(ROOT, "data", "train.bin")
OUT_PATH = os.path.join(ROOT, "runs", "gradient_check.txt")

BATCH_SIZE = 8
BLOCK_SIZE = 1024

REL_TOL = 1e-4          # gate: every tensor must be under this
LOSS_TOL = 1e-5         # losses must agree to fp32 precision


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()

    def close(self):
        self.f.close()


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run_variant(variant, device, x, y):
    """Fresh model from SEED, one forward+backward, return (loss, {name: grad})."""
    torch.manual_seed(SEED)
    model = GPT(GPTConfig(variant=variant), verbose=False).to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    _, loss = model(x, y)
    loss.backward()

    grads = {}
    for name, p in model.named_parameters():
        grads[name] = None if p.grad is None else p.grad.detach().clone()
    return loss.item(), grads, model


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tee = Tee(OUT_PATH)
    sys.stdout = tee
    try:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        device = pick_device()

        print("=" * 78)
        print("GRADIENT CHECK — midpoint (autograd) vs midpoint_rev (reconstructing)")
        print("=" * 78)
        print(f"seed        : {SEED}")
        print(f"device      : {device}")
        print(f"precision   : fp32")
        print(f"batch       : {BATCH_SIZE} x {BLOCK_SIZE} from data/train.bin")
        print(f"rel tol     : {REL_TOL:.0e} on max|gA-gB| / max|gA|, every tensor")
        print()

        data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
        g = torch.Generator().manual_seed(SEED)
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device, g)

        loss_a, grads_a, model_a = run_variant("midpoint", device, x, y)
        loss_b, grads_b, model_b = run_variant("midpoint_rev", device, x, y)

        # sanity: identical starting parameters
        pa = dict(model_a.named_parameters())
        pb = dict(model_b.named_parameters())
        param_delta = max(
            (pa[n].detach() - pb[n].detach()).abs().max().item() for n in pa
        )

        print("-" * 78)
        print("losses")
        print("-" * 78)
        print(f"  midpoint      : {loss_a:.10f}")
        print(f"  midpoint_rev  : {loss_b:.10f}")
        print(f"  abs diff      : {abs(loss_a - loss_b):.3e}   (tol {LOSS_TOL:.0e})")
        print(f"  params identical before backward: max delta {param_delta:.3e}")
        loss_ok = abs(loss_a - loss_b) <= LOSS_TOL
        print(f"  losses match  : {loss_ok}")
        print()

        names = list(grads_a.keys())
        assert set(names) == set(grads_b.keys()), "parameter name mismatch"

        print("-" * 78)
        print(f"{'parameter':<34}{'max|gA|':>12}{'max abs diff':>14}{'max rel diff':>14}")
        print("-" * 78)

        rows = []
        for n in names:
            ga, gb = grads_a[n], grads_b[n]
            if ga is None or gb is None:
                print(f"{n:<34}{'MISSING GRAD':>40}")
                rows.append((n, float("nan"), float("nan"), float("inf")))
                continue
            scale = ga.abs().max().item()
            abs_d = (ga - gb).abs().max().item()
            rel_d = abs_d / scale if scale > 0 else (0.0 if abs_d == 0 else float("inf"))
            rows.append((n, scale, abs_d, rel_d))
            flag = "" if rel_d < REL_TOL else "   <-- OVER TOL"
            print(f"{n:<34}{scale:>12.3e}{abs_d:>14.3e}{rel_d:>14.3e}{flag}")

        print("-" * 78)
        worst = max(rows, key=lambda r: r[3])
        max_rel = worst[3]
        max_abs = max(r[2] for r in rows)
        n_over = sum(1 for r in rows if not (r[3] < REL_TOL))
        grads_ok = n_over == 0
        passed = grads_ok and loss_ok

        print()
        print("=" * 78)
        print("SUMMARY — record these")
        print("=" * 78)
        print(f"device / precision    : {device} / fp32")
        print(f"batch                 : {BATCH_SIZE} x {BLOCK_SIZE}")
        print(f"parameter tensors     : {len(rows)}")
        print(f"loss (midpoint)       : {loss_a:.10f}")
        print(f"loss (midpoint_rev)   : {loss_b:.10f}")
        print(f"loss abs diff         : {abs(loss_a - loss_b):.3e}")
        print(f"max abs grad diff     : {max_abs:.3e}")
        print(f"max rel grad diff     : {max_rel:.3e}   (tol {REL_TOL:.0e})")
        print(f"worst tensor          : {worst[0]}")
        print(f"tensors over tol      : {n_over} of {len(rows)}")
        print(f"GATE                  : {'PASS' if passed else 'FAIL'}")
        print("=" * 78)

        if not passed:
            print()
            print("FAIL — per-tensor breakdown is above. Tolerance was NOT adjusted.")
            print("Stopping here rather than proceeding to the memory scan.")
        print(f"\nsaved to {OUT_PATH}")
        return 0 if passed else 2
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
