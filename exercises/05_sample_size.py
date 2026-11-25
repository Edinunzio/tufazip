"""Does sample size matter? Run rungs 1-4 on prefixes of the same text and compare.

The claim to test: a model that has seen little text predicts badly, and DIFFERENT
kinds of model suffer by different amounts as the sample shrinks. Counting tables
learn instantly but run out of context; neural nets start slow and then keep improving.

It loads the real rungs from their own files (nothing is copied), encodes the first
N bytes of sample.txt for several N, and prints one table of bits/byte vs size.
Encode only: the round trips were already proven by each rung's own run.

Two columns for rung 2 (order 3, and order 5) show "context helps, then runs out":
a longer context needs more data before its counts mean anything.

Caveats (say them when you share the numbers):
  * Every sample is a PREFIX of the same book, so small samples are just chapter 1.
    A different 2 KB slice would score a bit differently.
  * sample.txt is only 39,785 bytes. This shows the TREND. Whether the Transformer
    overtakes the count table needs more text than we have here.
  * Rung 4's learning rate is untuned.

Run (from the repo root; about 15 minutes, almost all of it rung 4):
    docker run --rm -v "$PWD":/work tufazip-learn python exercises/05_sample_size.py
"""
import importlib.util
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SIZES = [2_000, 5_000, 10_000, 20_000, None]   # None = the whole file


def load(filename):
    """Import an exercise by path (names starting with a digit can't be `import`ed)."""
    spec = importlib.util.spec_from_file_location(
        filename.removesuffix(".py"), ROOT / "exercises" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(1)   # same setting rung 4 uses for its own runs
    r1 = load("01_order0_coder.py")
    r2 = load("02_ngram_mixing.py")
    r3 = load("03_tiny_neural_numpy.py")
    r4 = load("04_tiny_transformer.py")

    # Each rung's encode() returns something different; each adapter returns just the
    # compressed size in bytes.
    rungs = [
        ("1: order-0",       lambda d: len(r1.encode(d)[0])),
        ("2: n-gram o3",     lambda d: len(r2.encode(d, 3))),
        ("2: n-gram o5",     lambda d: len(r2.encode(d, 5))),
        ("3: tiny net",      lambda d: len(r3.encode(d)[0])),
        ("4: Transformer",   lambda d: len(r4.encode(d)[0])),
    ]

    full = (ROOT / "exercises" / "sample.txt").read_bytes()
    rows = []
    for size in SIZES:
        data = full if size is None else full[:size]
        row = [f"{len(data):,}"]
        for name, run in rungs:
            t0 = time.time()
            blob_len = run(data)
            bpb = blob_len * 8 / len(data)
            row.append(f"{bpb:.3f}")
            print(f"  {len(data):>6,} bytes  {name:<15} {bpb:.3f} bits/byte "
                  f"({time.time() - t0:.0f}s)", flush=True)
        rows.append(row)

    header = ["bytes"] + [name for name, _ in rungs]
    print("\nbits/byte (lower is better; raw text = 8.000)\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")

    # --- Things to try ---
    # 1. Add 1_000 to SIZES. Which rung is hurt most at the very start?
    # 2. Which column improves the most from the smallest size to the largest? Is it the
    #    one that is best at the end?
    # 3. Look for a crossing: at what size does one rung stop being ahead of another?
    #    If you can't find one, what would you need to look for it?
    # 4. Replace `full[:size]` with a slice from the middle of the file
    #    (`full[20_000:20_000 + size]`). Do the small-sample numbers move?


if __name__ == "__main__":
    main()
