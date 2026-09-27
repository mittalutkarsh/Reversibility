"""
verify_data.py

Checks the output of prepare_data.py before any training happens:
  - exact token counts in tokens.bin / train.bin / val.bin
  - train.bin + val.bin really is tokens.bin, in order
  - every id is inside the vocab
  - bytes per token (sampled by decoding, expect 3.3 - 4.3)
  - 3 random 200-token slices decoded so you can read them

All .bin files are read with np.memmap. Run last.
"""

import argparse
import os
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

TOKENIZER_PATH = os.path.join(DATA, "tokenizer.json")
TOKENS_BIN = os.path.join(DATA, "tokens.bin")
TRAIN_BIN = os.path.join(DATA, "train.bin")
VAL_BIN = os.path.join(DATA, "val.bin")

VOCAB_SIZE = 8192
TOTAL_TOKENS = 50_000_000
VAL_TOKENS = 500_000
TRAIN_TOKENS = TOTAL_TOKENS - VAL_TOKENS

DTYPE = np.uint16
BPT_FLOOR, BPT_CEIL = 3.3, 4.3

# bytes-per-token is estimated by decoding this many random windows.
BPT_WINDOWS = 100
BPT_WINDOW_TOKENS = 20_000

SAMPLE_SLICES = 3
SAMPLE_TOKENS = 200


def human(n):
    return f"{n / 1024 / 1024:.1f} MB"


def open_mm(path, expected_tokens, label):
    if not os.path.exists(path):
        sys.exit(f"FATAL: missing {path} — run prepare_data.py first")
    mm = np.memmap(path, dtype=DTYPE, mode="r")
    n = mm.shape[0]
    assert n == expected_tokens, (
        f"{label}: expected {expected_tokens:,} tokens, found {n:,}"
    )
    on_disk = os.path.getsize(path)
    assert on_disk == expected_tokens * 2, (
        f"{label}: expected {expected_tokens * 2:,} bytes on disk, found {on_disk:,}"
    )
    print(f"  {label:<10} {n:>12,} tokens  {human(on_disk):>10}  OK")
    return mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="seed for the random slices")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("VERIFY DATA")
    print("=" * 72)

    if not os.path.exists(TOKENIZER_PATH):
        sys.exit(f"FATAL: missing {TOKENIZER_PATH} — run prepare_data.py first")
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    vocab = tok.get_vocab_size()
    assert vocab == VOCAB_SIZE, f"tokenizer vocab is {vocab}, expected {VOCAB_SIZE}"

    # ---- counts ---------------------------------------------------------
    print()
    print("token counts")
    print("-" * 72)
    tokens = open_mm(TOKENS_BIN, TOTAL_TOKENS, "tokens.bin")
    train = open_mm(TRAIN_BIN, TRAIN_TOKENS, "train.bin")
    val = open_mm(VAL_BIN, VAL_TOKENS, "val.bin")
    assert train.shape[0] + val.shape[0] == tokens.shape[0]

    # ---- the split is the real split, not a reshuffle -------------------
    print()
    print("split integrity")
    print("-" * 72)
    assert np.array_equal(train[:TRAIN_TOKENS], tokens[:TRAIN_TOKENS]), \
        "train.bin is not the first 49,500,000 tokens of tokens.bin"
    assert np.array_equal(val[:], tokens[TRAIN_TOKENS:]), \
        "val.bin is not the last 500,000 tokens of tokens.bin"
    print("  train.bin == tokens.bin[:49,500,000]      OK")
    print("  val.bin   == tokens.bin[49,500,000:]      OK")

    # ---- ids in range ---------------------------------------------------
    lo, hi = int(tokens.min()), int(tokens.max())
    assert 0 <= lo and hi < VOCAB_SIZE, f"token ids out of range: [{lo}, {hi}]"
    n_distinct = int(np.unique(np.asarray(tokens[::97])).size)
    print(f"  id range [{lo}, {hi}] within vocab {VOCAB_SIZE}   OK")
    print(f"  distinct ids in a 1/97 sample: {n_distinct:,}")

    # ---- bytes per token -------------------------------------------------
    print()
    print(f"bytes per token (decoding {BPT_WINDOWS} x {BPT_WINDOW_TOKENS:,} random tokens)")
    print("-" * 72)
    total_bytes = 0
    total_tokens = 0
    starts = rng.integers(0, TOTAL_TOKENS - BPT_WINDOW_TOKENS, size=BPT_WINDOWS)
    for s in starts:
        ids = np.asarray(tokens[s:s + BPT_WINDOW_TOKENS]).tolist()
        text = tok.decode(ids, skip_special_tokens=False)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
    bpt = total_bytes / total_tokens
    print(f"  sampled tokens : {total_tokens:,}")
    print(f"  sampled bytes  : {total_bytes:,}")
    print(f"  bytes/token    : {bpt:.3f}   (expect {BPT_FLOOR} - {BPT_CEIL})")

    in_band = BPT_FLOOR <= bpt <= BPT_CEIL
    print(f"  within band    : {in_band}")

    # ---- readable samples ------------------------------------------------
    print()
    print(f"{SAMPLE_SLICES} random {SAMPLE_TOKENS}-token slices (seed {args.seed})")
    print("=" * 72)
    for i in range(SAMPLE_SLICES):
        s = int(rng.integers(0, TOTAL_TOKENS - SAMPLE_TOKENS))
        ids = np.asarray(tokens[s:s + SAMPLE_TOKENS]).tolist()
        text = tok.decode(ids, skip_special_tokens=False)
        print()
        print(f"--- slice {i + 1}: tokens[{s:,} : {s + SAMPLE_TOKENS:,}] ---")
        print(text)
    print()

    # ---- summary ---------------------------------------------------------
    sizes = {p: os.path.getsize(p) for p in (TOKENS_BIN, TRAIN_BIN, VAL_BIN, TOKENIZER_PATH)}
    print("=" * 72)
    print("SUMMARY — record these")
    print("=" * 72)
    print(f"vocab size        : {vocab}")
    print(f"dtype             : {np.dtype(DTYPE).name}")
    print(f"bytes per token   : {bpt:.3f}")
    print(f"tokens.bin        : {TOTAL_TOKENS:,} tokens   {human(sizes[TOKENS_BIN])}   ({sizes[TOKENS_BIN]:,} bytes)")
    print(f"train.bin         : {TRAIN_TOKENS:,} tokens   {human(sizes[TRAIN_BIN])}   ({sizes[TRAIN_BIN]:,} bytes)")
    print(f"val.bin           : {VAL_TOKENS:,} tokens   {human(sizes[VAL_BIN])}   ({sizes[VAL_BIN]:,} bytes)")
    print(f"tokenizer.json    : {sizes[TOKENIZER_PATH]:,} bytes")
    print(f"all assertions    : PASS")
    print("=" * 72)

    if bpt < BPT_FLOOR:
        print()
        print(f"STOP: bytes per token is {bpt:.3f}, below the {BPT_FLOOR} floor.")
        print("The vocab is too small for this text — get more raw data before training.")
        sys.exit(2)
    if bpt > BPT_CEIL:
        print()
        print(f"NOTE: bytes per token is {bpt:.3f}, above the expected {BPT_CEIL} ceiling.")
        print("Not a failure, but worth a look before training.")


if __name__ == "__main__":
    main()
