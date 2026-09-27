"""
prepare_data.py

Builds the training data for the 20M-param model:
  1. download ~250 MB of the TinyStories train split (HTTP Range, not the 1.9 GB file)
  2. train a byte-level BPE tokenizer at vocab 8192  -> data/tokenizer.json
  3. tokenise the slice into one flat uint16 stream
  4. trim to EXACTLY 50,000,000 tokens          -> data/tokens.bin
  5. split 49,500,000 / 500,000                 -> data/train.bin, data/val.bin

Run after environment_check.py, before verify_data.py.
"""

import os
import sys
import time

import numpy as np
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# ---------------------------------------------------------------- config
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
RAW_MB = 250
RAW_BYTES = RAW_MB * 1024 * 1024
RAW_PATH = os.path.join(DATA, f"TinyStories-train-{RAW_MB}MB.txt")

VOCAB_SIZE = 8192
EOT = "<|endoftext|>"

TOTAL_TOKENS = 50_000_000
VAL_TOKENS = 500_000
TRAIN_TOKENS = TOTAL_TOKENS - VAL_TOKENS  # 49,500,000

TOKENIZER_PATH = os.path.join(DATA, "tokenizer.json")
TOKENS_BIN = os.path.join(DATA, "tokens.bin")
TRAIN_BIN = os.path.join(DATA, "train.bin")
VAL_BIN = os.path.join(DATA, "val.bin")

DTYPE = np.uint16  # vocab 8192 fits comfortably in uint16
BPT_FLOOR = 3.3    # below this, the vocab is too small for this text


def human(n_bytes):
    return f"{n_bytes / 1024 / 1024:.1f} MB"


# ------------------------------------------------------------- 1. download
def download_slice():
    """Fetch the first RAW_BYTES of the train split via an HTTP Range request."""
    if os.path.exists(RAW_PATH) and os.path.getsize(RAW_PATH) >= RAW_BYTES:
        print(f"[1/5] raw slice already present: {RAW_PATH} ({human(os.path.getsize(RAW_PATH))})")
        return

    print(f"[1/5] downloading first {RAW_MB} MB of TinyStories train split ...")
    headers = {"Range": f"bytes=0-{RAW_BYTES - 1}"}
    tmp = RAW_PATH + ".part"
    got = 0
    t0 = time.time()
    with requests.get(URL, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if got % (25 << 20) < (1 << 20):
                    print(f"        {human(got)} / {human(RAW_BYTES)}")
    os.replace(tmp, RAW_PATH)
    print(f"        done: {human(got)} in {time.time() - t0:.1f}s -> {RAW_PATH}")


def load_documents():
    """Read the slice, drop the truncated trailing story, split into documents."""
    with open(RAW_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # The byte-range cut lands mid-story; keep only complete documents.
    cut = text.rfind(EOT)
    if cut == -1:
        sys.exit(f"FATAL: no {EOT} separator found in {RAW_PATH}")
    text = text[:cut]

    docs = [d.strip() for d in text.split(EOT)]
    docs = [d for d in docs if d]
    slice_bytes = len(text.encode("utf-8"))
    print(f"        complete documents in slice : {len(docs):,}")
    print(f"        text in slice               : {human(slice_bytes)} "
          f"(headroom; only part is consumed — see 'text consumed' in the summary)")
    return docs


# ------------------------------------------------------------ 2. tokenizer
def train_tokenizer(docs):
    if os.path.exists(TOKENIZER_PATH):
        tok = Tokenizer.from_file(TOKENIZER_PATH)
        if tok.get_vocab_size() == VOCAB_SIZE:
            print(f"[2/5] tokenizer already present: {TOKENIZER_PATH} (vocab {tok.get_vocab_size()})")
            return tok
        print(f"[2/5] existing tokenizer has vocab {tok.get_vocab_size()} != {VOCAB_SIZE}, retraining")

    print(f"[2/5] training byte-level BPE, vocab_size={VOCAB_SIZE} ...")
    t0 = time.time()

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(docs, trainer=trainer, length=len(docs))
    tok.save(TOKENIZER_PATH)
    print(f"        trained in {time.time() - t0:.1f}s, vocab {tok.get_vocab_size()} -> {TOKENIZER_PATH}")
    return tok


# ------------------------------------------------------------- 3. tokenise
def tokenise(tok, docs):
    """Encode documents into one flat stream, stopping once TOTAL_TOKENS is reached.

    Each document is followed by the <|endoftext|> id so the model sees story
    boundaries. Returns (array, chars_consumed, docs_consumed).
    """
    eot_id = tok.token_to_id(EOT)
    if eot_id is None:
        sys.exit(f"FATAL: {EOT} missing from tokenizer vocab")

    print(f"[3/5] tokenising to {TOTAL_TOKENS:,} tokens (eot id = {eot_id}) ...")
    t0 = time.time()

    parts = []
    n_tokens = 0
    chars = 0
    docs_used = 0
    BATCH = 10_000

    for start in range(0, len(docs), BATCH):
        batch = docs[start:start + BATCH]
        encs = tok.encode_batch(batch, add_special_tokens=False)

        for doc, enc in zip(batch, encs):
            ids = enc.ids
            parts.append(np.asarray(ids, dtype=DTYPE))
            parts.append(np.asarray([eot_id], dtype=DTYPE))
            n_tokens += len(ids) + 1
            chars += len(doc) + len(EOT)
            docs_used += 1
            if n_tokens >= TOTAL_TOKENS:
                break

        print(f"        {n_tokens:,} tokens from {docs_used:,} docs "
              f"({time.time() - t0:.0f}s)")
        if n_tokens >= TOTAL_TOKENS:
            break

    if n_tokens < TOTAL_TOKENS:
        sys.exit(
            f"FATAL: only {n_tokens:,} tokens from the {RAW_MB} MB slice, "
            f"need {TOTAL_TOKENS:,}. Increase RAW_MB and rerun."
        )

    arr = np.concatenate(parts)
    print(f"        encoded {len(arr):,} tokens in {time.time() - t0:.1f}s")
    return arr, chars, docs_used


# ----------------------------------------------------------- 4/5. write out
def write_bins(arr, chars_consumed):
    # Exact trim happens here and nowhere else.
    trimmed = arr[:TOTAL_TOKENS]
    assert trimmed.shape[0] == TOTAL_TOKENS, trimmed.shape
    assert trimmed.dtype == DTYPE

    print(f"[4/5] writing {TOTAL_TOKENS:,} tokens -> {TOKENS_BIN}")
    trimmed.tofile(TOKENS_BIN)

    # Read back through memmap (never np.load) and split.
    print(f"[5/5] splitting {TRAIN_TOKENS:,} train / {VAL_TOKENS:,} val")
    mm = np.memmap(TOKENS_BIN, dtype=DTYPE, mode="r")
    assert mm.shape[0] == TOTAL_TOKENS, mm.shape

    mm[:TRAIN_TOKENS].tofile(TRAIN_BIN)
    mm[TRAIN_TOKENS:].tofile(VAL_BIN)
    del mm

    # bytes-per-token measured over exactly the text that produced these tokens.
    bpt = chars_consumed / len(arr)
    return bpt


def main():
    os.makedirs(DATA, exist_ok=True)

    download_slice()
    docs = load_documents()
    tok = train_tokenizer(docs)
    arr, chars, docs_used = tokenise(tok, docs)
    bpt = write_bins(arr, chars)

    sizes = {p: os.path.getsize(p) for p in (TOKENS_BIN, TRAIN_BIN, VAL_BIN, TOKENIZER_PATH)}

    print()
    print("=" * 72)
    print("SUMMARY — record these")
    print("=" * 72)
    print(f"raw slice           : {RAW_PATH}")
    print(f"raw slice size      : {human(os.path.getsize(RAW_PATH))}")
    print(f"documents used      : {docs_used:,} of {len(docs):,} in slice")
    print(f"text consumed       : {human(chars)}")
    print(f"vocab size          : {tok.get_vocab_size()}")
    print(f"bytes per token     : {bpt:.3f}   (expect {BPT_FLOOR} - 4.3)")
    print(f"tokens.bin          : {TOTAL_TOKENS:,} tokens, {human(sizes[TOKENS_BIN])}")
    print(f"train.bin           : {TRAIN_TOKENS:,} tokens, {human(sizes[TRAIN_BIN])}")
    print(f"val.bin             : {VAL_TOKENS:,} tokens, {human(sizes[VAL_BIN])}")
    print(f"tokenizer.json      : {sizes[TOKENIZER_PATH] / 1024:.0f} KB")
    print(f"dtype               : {np.dtype(DTYPE).name}")
    print("=" * 72)

    if bpt < BPT_FLOOR:
        print()
        print(f"STOP: bytes per token is {bpt:.3f}, below the {BPT_FLOOR} floor.")
        print("The vocab is too small for this text — you need more raw data")
        print("(raise RAW_MB) or a larger vocab. Files were written, but do not")
        print("train on them until this is resolved.")
        sys.exit(2)


if __name__ == "__main__":
    main()
