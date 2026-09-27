"""
environment_check.py

Confirms this Mac can train the 20M-param model:
  - MPS backend present and usable
  - a 50-step toy training loop produces finite losses
  - whether torch.autocast("mps", dtype=torch.bfloat16) is safe to use

Run first, before prepare_data.py. Record the summary block at the end.
"""

import platform
import subprocess
import sys

import torch
import torch.nn as nn

TOY_STEPS = 50


def sysctl(key):
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def describe_machine():
    chip = sysctl("machdep.cpu.brand_string")
    mem_bytes = sysctl("hw.memsize")
    try:
        ram_gb = int(mem_bytes) / (1024**3)
    except ValueError:
        ram_gb = float("nan")
    return chip, ram_gb


class ToyModel(nn.Module):
    """Small stand-in for the real model: same shapes, no training code reused."""

    def __init__(self, d_model=512, n_classes=8192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.head(self.net(x))


def toy_loop(device, autocast_dtype=None, steps=TOY_STEPS, seed=0):
    """Run `steps` optimiser steps on random data.

    Returns (losses, error_string_or_None).
    """
    torch.manual_seed(seed)
    model = ToyModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    batch, d_model, n_classes = 16, 512, 8192
    losses = []

    try:
        for _ in range(steps):
            x = torch.randn(batch, d_model, device=device)
            y = torch.randint(0, n_classes, (batch,), device=device)

            if autocast_dtype is not None:
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    logits = model(x)
                    loss = loss_fn(logits, y)
            else:
                logits = model(x)
                loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach().float().item())
    except Exception as exc:  # noqa: BLE001 - we want to report any backend failure
        return losses, f"{type(exc).__name__}: {exc}"

    return losses, None


def all_finite(losses):
    import math

    return len(losses) > 0 and all(math.isfinite(v) for v in losses)


def main():
    chip, ram_gb = describe_machine()

    print("=" * 72)
    print("ENVIRONMENT CHECK")
    print("=" * 72)
    print(f"python            : {sys.version.split()[0]}")
    print(f"torch             : {torch.__version__}")
    print(f"platform          : {platform.platform()}")
    print(f"machine           : {platform.machine()}")
    print(f"chip              : {chip}")
    print(f"total RAM         : {ram_gb:.1f} GB")
    print()

    mps_built = torch.backends.mps.is_built()
    mps_avail = torch.backends.mps.is_available()
    print(f"mps.is_built()    : {mps_built}")
    print(f"mps.is_available(): {mps_avail}")

    if not mps_avail:
        print()
        print("MPS is NOT available. Training would fall back to CPU and be very slow.")
        print("Stopping here — resolve the torch/macOS install before continuing.")
        sys.exit(1)

    device = "mps"

    # ---- fp32 toy loop -------------------------------------------------
    print()
    print("-" * 72)
    print(f"fp32 toy training loop ({TOY_STEPS} steps, random data)")
    print("-" * 72)
    fp32_losses, fp32_err = toy_loop(device, autocast_dtype=None)
    if fp32_err:
        print(f"ERROR: {fp32_err}")
    else:
        print(f"first loss        : {fp32_losses[0]:.4f}")
        print(f"last loss         : {fp32_losses[-1]:.4f}")
        print(f"min / max loss    : {min(fp32_losses):.4f} / {max(fp32_losses):.4f}")
    fp32_ok = fp32_err is None and all_finite(fp32_losses)
    print(f"all losses finite : {fp32_ok}")

    # ---- bf16 autocast toy loop ---------------------------------------
    print()
    print("-" * 72)
    print(f'bf16 autocast toy loop — torch.autocast("mps", torch.bfloat16)')
    print("-" * 72)
    bf16_losses, bf16_err = toy_loop(device, autocast_dtype=torch.bfloat16)
    if bf16_err:
        print(f"ERROR: {bf16_err}")
        bf16_ok = False
    else:
        print(f"first loss        : {bf16_losses[0]:.4f}")
        print(f"last loss         : {bf16_losses[-1]:.4f}")
        print(f"min / max loss    : {min(bf16_losses):.4f} / {max(bf16_losses):.4f}")
        bf16_ok = all_finite(bf16_losses)
        if not bf16_ok:
            bad = [i for i, v in enumerate(bf16_losses) if v != v or abs(v) == float("inf")]
            print(f"non-finite at step(s): {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    print(f"all losses finite : {bf16_ok}")

    if bf16_ok:
        recommendation = 'bfloat16 autocast works — torch.autocast("mps", dtype=torch.bfloat16) is safe.'
    else:
        recommendation = "bfloat16 autocast FAILED on this machine — use fp32 for training."

    # ---- summary -------------------------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY — record these")
    print("=" * 72)
    print(f"torch version      : {torch.__version__}")
    print(f"chip               : {chip}")
    print(f"total RAM (GB)     : {ram_gb:.1f}")
    print(f"mps available      : {mps_avail}")
    print(f"fp32 {TOY_STEPS}-step loop : {'PASS' if fp32_ok else 'FAIL'}"
          + (f" (final loss {fp32_losses[-1]:.4f})" if fp32_losses else ""))
    print(f"bf16 autocast      : {'PASS' if bf16_ok else 'FAIL'}"
          + (f" (final loss {bf16_losses[-1]:.4f})" if bf16_ok and bf16_losses else ""))
    print(f"RECOMMENDED DTYPE  : {'bfloat16 autocast' if bf16_ok else 'fp32'}")
    print()
    print(recommendation)
    print("=" * 72)

    sys.exit(0 if fp32_ok else 1)


if __name__ == "__main__":
    main()
