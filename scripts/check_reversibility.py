"""
scripts/check_reversibility.py

THE GATE. Does the reversible rule actually reconstruct the forward
activations, in the precision we intend to train in?

Method, for euler and midpoint separately:
  1. forward pass on one real batch from data/train.bin, keeping every state
  2. walk backward from the final state(s) using the inverse rule, CHAINED --
     each reconstructed state is built from the previously reconstructed one,
     never from a stored ground-truth state, so errors compound the way they
     would in a real reversible backward pass
  3. max |error| per layer, plus whether the error grows on the way down
  4. the whole thing twice: fp32 and bf16

WHY EULER IS EXPECTED TO FAIL
-----------------------------
The forward rule is p[l+1] = p[l] + h*f(p[l]). Solving for p[l]:

    p[l] = p[l+1] - h*f(p[l])
                        ^^^^^ the unknown appears inside f

That is implicit. There is no closed-form inverse. Recovering p[l] exactly
would need a fixed-point or Newton iteration per layer -- an iterative
approximation with its own tolerance, not an inverse, and it would cost more
forward evaluations than the activation memory it saves. So this script uses
the naive explicit inverse

    p_hat[l] = p[l+1] - h*f(p[l+1])

which is a *different* map (it is one step of backward Euler's explicit
cousin). Its error is O(h^2) per layer and compounds across layers. The
numbers below are what that costs. No fixed-point iteration is used to paper
over it, and h is left at 0.25 rather than shrunk to flatter the result.

WHY MIDPOINT WORKS
------------------
p[l+1] = p[l-1] + 2h*f(p[l])  inverts to  p[l-1] = p[l+1] - 2h*f(p[l]).
f is evaluated at p[l], which the backward walk already holds. Exact, up to
floating-point rounding. The cost is carrying two adjacent states instead of
one.
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
OUT_PATH = os.path.join(ROOT, "runs", "reversibility_check.txt")

BATCH_SIZE = 8
BLOCK_SIZE = 1024

FP32_TOL = 1e-4     # pass threshold for fp32 max error

# Growth criterion. The original 10x-per-step tolerance was too loose to
# discriminate: euler grows a steady ~1.4x per layer and was waved through as
# "not growing", so only the magnitude check was doing any work. The rule now
# flags a RUN of monotonic growth rather than any single step.
ULP = {"fp32": 2.0 ** -23, "bf16": 2.0 ** -8}
GROWTH_RUN = 3          # >= this many consecutive increases to count as a run
GROWTH_FACTOR = 3.0     # total growth across that run must exceed this
GROWTH_FLOOR_ULP = 8    # errors below this many ULP are rounding noise


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


def growth_verdict(errs, ulp):
    """Flag sustained monotonic growth in the reconstruction error.

    Errors are first clamped UP to an 8-ULP floor, so a walk that merely
    accumulates a few rounding units cannot be flagged -- at that scale the
    "growth" is the float grid, not the rule. Above the floor, a run of
    GROWTH_RUN or more consecutive increases whose total growth exceeds
    GROWTH_FACTOR is a real signal that the inverse is wrong: a correct
    inverse accumulates roundoff, a wrong one multiplies error per layer.

    Returns (ok, runs, flagged, floor) where each run is
    (start_idx, end_idx, n_increases, total_growth).
    """
    floor = GROWTH_FLOOR_ULP * ulp
    e = [max(v, floor) for v in errs]
    runs = []
    i = 0
    while i < len(e) - 1:
        j = i
        while j < len(e) - 1 and e[j + 1] > e[j]:
            j += 1
        if j > i:
            runs.append((i, j, j - i, e[j] / e[i]))
            i = j + 1
        else:
            i += 1
    flagged = [r for r in runs if r[2] >= GROWTH_RUN and r[3] > GROWTH_FACTOR]
    return len(flagged) == 0, runs, flagged, floor


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------- backward walks
@torch.no_grad()
def reconstruct_euler(model, states, cos, sin):
    """Naive explicit inverse. NOT an exact inverse -- see module docstring.

    Returns list of (layer_index, reconstructed_state) in walk order.
    """
    h = model.cfg.h
    out = []
    p_hat = states[-1]                       # only the final state is "kept"
    for l in range(len(model.blocks) - 1, -1, -1):
        # exact would be p[l] = p[l+1] - h*f(p[l]); f is evaluated at the
        # unknown, so we substitute p_hat[l+1] for it and accept the error.
        p_hat = p_hat - h * model.blocks[l](p_hat, cos, sin)
        out.append((l, p_hat))
    return out


@torch.no_grad()
def reconstruct_midpoint(model, states, cos, sin):
    """Exact inverse: p[l-1] = p[l+1] - 2h*f(p[l]).

    Seeded with the final ADJACENT PAIR (p[L], p[L-1]); that pair is what a
    reversible stack would keep instead of every activation.
    """
    h = model.cfg.h
    L = len(model.blocks)
    out = []
    p_next = states[L]       # p[L]
    p_cur = states[L - 1]    # p[L-1]
    # block l maps (p[l-1], p[l]) -> p[l+1] for l = 1..L-1; invert in reverse.
    for l in range(L - 1, 0, -1):
        p_prev = p_next - 2.0 * h * model.blocks[l](p_cur, cos, sin)
        out.append((l - 1, p_prev))
        p_next, p_cur = p_cur, p_prev
    return out


@torch.no_grad()
def run_check(variant, dtype, device, idx, dtype_name):
    torch.manual_seed(SEED)
    model = GPT(GPTConfig(variant=variant), verbose=False).to(device=device, dtype=dtype)
    model.eval()

    p0 = model.embed(idx)
    cos, sin = model.rope(idx.shape[1], p0.device, p0.dtype)
    _, states = model.run_stack(p0, cos, sin)

    if variant == "euler":
        recon = reconstruct_euler(model, states, cos, sin)
        exact = False
    else:
        recon = reconstruct_midpoint(model, states, cos, sin)
        exact = True

    rows = []
    for layer_idx, p_hat in recon:
        truth = states[layer_idx].float()
        err = (p_hat.float() - truth).abs().max().item()
        scale = truth.abs().max().item()
        rows.append((layer_idx, err, err / scale if scale > 0 else float("nan")))

    errs = [r[1] for r in rows]
    max_err = max(errs)
    growth_ok, runs, flagged, floor = growth_verdict(errs, ULP[dtype_name])
    ratio = errs[-1] / errs[0] if errs[0] > 0 else float("inf")

    print(f"  {variant} / {dtype_name} — reconstructing {len(rows)} states, "
          f"chained from the final {'pair' if variant == 'midpoint' else 'state'}")
    print(f"  {'walk':<6}{'recovers p[l]':<15}{'max |err|':>14}{'rel err':>14}")
    print("  " + "-" * 60)
    for step, (layer_idx, err, rel) in enumerate(rows):
        print(f"  {step:<6}{'p[' + str(layer_idx) + ']':<15}{err:>14.3e}{rel:>14.3e}")
    print("  " + "-" * 60)
    print(f"  max error over all layers : {max_err:.3e}")
    print(f"  error ratio last/first    : {ratio:.3e}")
    print(f"  8-ULP floor ({dtype_name})        : {floor:.3e}"
          f"   ({sum(1 for v in errs if v > floor)}/{len(errs)} layers above it)")
    if runs:
        longest = max(runs, key=lambda r: (r[2], r[3]))
        print(f"  longest monotonic run     : {longest[2]} consecutive increase(s), "
              f"{longest[3]:.2f}x total")
    else:
        print(f"  longest monotonic run     : none above the floor")
    print(f"  growth verdict            : "
          f"{'not growing' if growth_ok else 'GROWING'}"
          f"   (flag at >={GROWTH_RUN} increases and >{GROWTH_FACTOR}x)")
    print()

    return {
        "variant": variant,
        "dtype": dtype_name,
        "rows": rows,
        "max_err": max_err,
        "growth_ok": growth_ok,
        "exact": exact,
    }


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tee = Tee(OUT_PATH)
    sys.stdout = tee

    try:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        device = pick_device()

        print("=" * 72)
        print("REVERSIBILITY CHECK")
        print("=" * 72)
        print(f"seed        : {SEED}")
        print(f"device      : {device}")
        print(f"batch       : {BATCH_SIZE} x {BLOCK_SIZE} tokens from data/train.bin")
        print(f"h / blend   : {GPTConfig().h} / {GPTConfig().blend}  (not tuned)")
        print(f"fp32 tol    : {FP32_TOL:.0e}")
        print(f"growth rule : flag >={GROWTH_RUN} consecutive increases totalling "
              f">{GROWTH_FACTOR}x, floored at {GROWTH_FLOOR_ULP} ULP")
        print()

        data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
        g = torch.Generator().manual_seed(SEED)
        idx, _ = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device, g)
        print(f"batch token ids: min {int(idx.min())}, max {int(idx.max())}, shape {tuple(idx.shape)}")
        print()

        results = []
        for dtype, dtype_name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            print("=" * 72)
            print(f"{dtype_name.upper()}")
            print("=" * 72)
            for variant in ("euler", "midpoint"):
                results.append(run_check(variant, dtype, device, idx, dtype_name))

        # ---- verdicts ------------------------------------------------
        print("=" * 72)
        print("VERDICT (fp32 is the gate; bf16 is reported for the record)")
        print("=" * 72)
        verdicts = {}
        for r in results:
            if r["dtype"] != "fp32":
                continue
            ok = r["max_err"] < FP32_TOL and r["growth_ok"]
            verdicts[r["variant"]] = ok
            why = []
            if r["max_err"] >= FP32_TOL:
                why.append(f"max err {r['max_err']:.3e} >= {FP32_TOL:.0e}")
            if not r["growth_ok"]:
                why.append("per-layer error grows on the way down")
            tail = "" if ok else "   <- " + "; ".join(why)
            print(f"  {r['variant']:<10} fp32  {'PASS' if ok else 'FAIL'}{tail}")

        print()
        if not verdicts.get("euler", False):
            print("  euler FAILS by construction, not by tuning. p[l+1] = p[l] + h*f(p[l])")
            print("  inverts to p[l] = p[l+1] - h*f(p[l]), where f is evaluated at the")
            print("  unknown state. No closed-form inverse exists. The naive explicit")
            print("  substitution used above is an approximation, and its error is what")
            print("  is tabulated. Shrinking h would shrink the error without making the")
            print("  map invertible, so h was left at 0.25.")
        print()

        by = {(r["variant"], r["dtype"]): r for r in results}
        print("=" * 72)
        print("SUMMARY — record these")
        print("=" * 72)
        print(f"device                       : {device}")
        print(f"batch                        : {BATCH_SIZE} x {BLOCK_SIZE}")
        print(f"h / blend                    : {GPTConfig().h} / {GPTConfig().blend}")
        print()
        print(f"{'variant':<10}{'dtype':<8}{'max |err|':>14}{'growing?':>12}{'verdict':>10}")
        print("-" * 72)
        for variant in ("euler", "midpoint"):
            for dn in ("fp32", "bf16"):
                r = by[(variant, dn)]
                v = ("PASS" if (r["max_err"] < FP32_TOL and r["growth_ok"]) else "FAIL") if dn == "fp32" else "-"
                print(f"{variant:<10}{dn:<8}{r['max_err']:>14.3e}"
                      f"{('no' if r['growth_ok'] else 'yes'):>12}{v:>10}")
        print("-" * 72)
        print(f"euler exactly invertible     : False (implicit inverse, see above)")
        print(f"midpoint exactly invertible  : True  (needs an adjacent state pair)")
        f32 = by[("midpoint", "fp32")]["max_err"]
        b16 = by[("midpoint", "bf16")]["max_err"]
        print(f"midpoint bf16 / fp32 error   : {b16 / f32:.1f}x worse -> train the")
        print(f"                               reversible stack in fp32")
        print(f"GATE (midpoint fp32)         : {'PASS' if verdicts.get('midpoint') else 'FAIL'}")
        print("=" * 72)
        print(f"\nsaved to {OUT_PATH}")
    finally:
        sys.stdout = sys.__stdout__
        tee.close()


if __name__ == "__main__":
    main()
