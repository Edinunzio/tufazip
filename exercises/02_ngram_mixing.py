"""Rung 2: context models. Predict the next byte from the previous k bytes.

Rung 1 asked "how common is each byte overall?". Now we ask "how common is each
byte *after these last k bytes*?" After "th", 'e' is very likely. After "q",
'u' is. That context is where almost all of the compression comes from, and it is
the job a Transformer does with up to 192 tokens of context in tufazip (V2_CONFIG
ctx_len in nn.py).

Mechanics:
  * For each order k = 0..max_order, keep a table: last-k-bytes -> counts[256].
  * Blend the orders with Witten-Bell interpolation. Start from "uniform", then for
    k = 0, 1, 2, ...:

        p_k(s) = ( count_k[s] + u_k * p_{k-1}(s) ) / ( n_k + u_k )

    where n_k = total count seen in this context and u_k = how many DISTINCT bytes
    have followed it. A context that has only ever been followed by one byte
    (u small) is trusted. One that has been followed by many different bytes
    (u large) leans on the lower order. A context never seen before (n = 0)
    contributes nothing, so we just keep the lower order's answer.

The coder, FREQ_TOTAL and the 'cumulative table with a floor of 1' are the same
recipe as make_freq_table() in nn.py: probabilities in, integer cumsum out.

Run (from the repo root; takes a minute, since the coder is pure Python):
    docker run --rm -v "$PWD":/work tufazip-learn python exercises/02_ngram_mixing.py
"""
import io
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from arithmetic_coder import (ArithmeticDecoder, ArithmeticEncoder,  # noqa: E402
                              BitInputStream, BitOutputStream)

STATE_BITS = 32
FREQ_TOTAL = 10_000_000  # same resolution nn.py hands to the coder


class KeepOpenBytesIO(io.BytesIO):
    def close(self):  # let BitOutputStream pad+close without losing getvalue()
        pass


class ContextModel:
    """Blends orders 0..max_order. predict() and update() are the whole interface:
    exactly the shape of 'run the model, then learn from the true symbol' in tufazip."""

    def __init__(self, max_order):
        self.max_order = max_order
        # tables[k][context_bytes] -> int64 array of 256 counts
        self.tables = [dict() for _ in range(max_order + 1)]

    @staticmethod
    def _context(history, k):
        return bytes(history[len(history) - k:]) if k else b""

    def predict(self, history):
        p = np.full(256, 1.0 / 256)  # "order -1": know nothing, everything equal
        for k in range(self.max_order + 1):
            if k > len(history):
                break  # not enough history yet for this order
            counts = self.tables[k].get(self._context(history, k))
            if counts is None:
                break  # never seen this context, so no longer one can have either
            n = counts.sum()
            u = np.count_nonzero(counts)
            p = (counts + u * p) / (n + u)
        return p

    def update(self, history, symbol):
        """Learn from `symbol`, which followed `history`. Call AFTER coding it."""
        for k in range(self.max_order + 1):
            if k > len(history):
                break
            table = self.tables[k]
            ctx = self._context(history, k)
            if ctx not in table:
                table[ctx] = np.zeros(256, dtype=np.int64)
            table[ctx][symbol] += 1


def to_freqs(p):
    """Probabilities -> cumulative integer table. Mirrors make_freq_table in nn.py:
    scale, round, floor at 1 (so no byte is ever uncodable), cumsum."""
    freq = np.maximum(np.round(p * FREQ_TOTAL).astype(np.int64), 1)
    return np.cumsum(freq)


def encode(data, max_order):
    model = ContextModel(max_order)
    out = KeepOpenBytesIO()
    out.write(len(data).to_bytes(4, "big"))
    bits = BitOutputStream(out)
    encoder = ArithmeticEncoder(STATE_BITS, bits)

    history = bytearray()
    for byte in data:
        freqs = to_freqs(model.predict(history))
        encoder.write(freqs, byte)
        model.update(history, byte)
        history.append(byte)
    encoder.finish()
    bits.close()
    return out.getvalue()


def decode(blob, max_order):
    n = int.from_bytes(blob[:4], "big")
    decoder = ArithmeticDecoder(STATE_BITS, BitInputStream(io.BytesIO(blob[4:])))
    model = ContextModel(max_order)

    history = bytearray()
    for _ in range(n):
        freqs = to_freqs(model.predict(history))  # decoder sees only past bytes
        byte = decoder.read(freqs)
        model.update(history, byte)
        history.append(byte)
    return bytes(history)


def main():
    data = (ROOT / "exercises" / "sample.txt").read_bytes()
    n = len(data)
    print(f"input: {n:,} bytes (8.000 bits/byte); rung 1 order-0 got ~4.638\n")
    print("max order | compressed bytes | bits/byte | ratio | contexts stored")
    print("----------+------------------+-----------+-------+----------------")

    results = {}
    for order in range(0, 6):
        blob = encode(data, order)
        results[order] = blob
        # Rebuild the model once more just to count its table entries for display.
        model, hist = ContextModel(order), bytearray()
        for b in data:
            model.update(hist, b)
            hist.append(b)
        stored = sum(len(t) for t in model.tables)
        print(f"{order:>9} | {len(blob):>16,} | {len(blob) * 8 / n:>9.3f} | "
              f"{n / len(blob):>4.2f}x | {stored:>14,}")

    best = min(results, key=lambda k: len(results[k]))
    print(f"\nbest: order {best}. Verifying the round trip for that one...")
    restored = decode(results[best], best)
    print(f"round trip identical: {restored == data}")
    assert restored == data

    # --- Things to try ---
    # 1. Why does the "contexts stored" column explode while compression stops
    #    improving? (Long contexts almost never repeat in 40 KB: sparse tables.)
    # 2. Replace u_k with a constant (say 1.0) in predict(). Better or worse?
    # 3. Train on the first half of sample.txt, then compress the second half
    #    without updating. How much worse is it? That gap is why tufazip ships
    #    pretrained weights AND keeps learning online.
    # 4. Real "context mixing" (PAQ, cmix) replaces this fixed blend with weights
    #    that are *learned* to favour whichever order is predicting well. A small
    #    neural net doing exactly that is rung 3's idea.


if __name__ == "__main__":
    main()
