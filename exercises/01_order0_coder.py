"""Rung 1: an adaptive order-0 model + the repo's arithmetic coder.

The idea behind every rung of this ladder (and tufazip itself):

    compressed size  ~=  sum over symbols of  -log2( P(symbol | what came before) )

A *model* supplies the probabilities. The *arithmetic coder* turns them into bits.
They are independent: here the model is a byte-frequency counter; in tufazip it is
a 12-layer Transformer. The coder doesn't know or care.

"Order-0" means the model ignores context: P(byte) is just how often that byte
has appeared so far. It knows "e" is common and "z" is rare, nothing more.

Run (from the repo root):
    docker run --rm -v "$PWD":/work tufazip-learn python exercises/01_order0_coder.py
"""
import io
import math
import sys
from pathlib import Path

import numpy as np

# exercises/ is a subfolder, so add the repo root to the import path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from arithmetic_coder import (ArithmeticDecoder, ArithmeticEncoder,  # noqa: E402
                              BitInputStream, BitOutputStream)

STATE_BITS = 32  # width of the coder's low/high registers; nn.py uses 32 too


class KeepOpenBytesIO(io.BytesIO):
    """BitOutputStream.close() pads the last byte, then closes the underlying
    stream. A closed BytesIO can't give us .getvalue(), so ignore the close."""

    def close(self):
        pass


def encode(data: bytes):
    """Return (compressed_bytes, ideal_bits)."""
    # Start every byte value at count 1, not 0. A symbol with zero frequency
    # would occupy zero width on the probability line and could never be coded.
    counts = np.ones(256, dtype=np.int64)

    out = KeepOpenBytesIO()
    out.write(len(data).to_bytes(4, "big"))  # decoder must know when to stop
    bits = BitOutputStream(out)
    encoder = ArithmeticEncoder(STATE_BITS, bits)

    ideal_bits = 0.0
    for byte in data:
        freqs = np.cumsum(counts)  # cumulative table: freqs[s] = end of s's slice
        ideal_bits += -math.log2(counts[byte] / freqs[-1])  # what this symbol *should* cost
        encoder.write(freqs, byte)
        counts[byte] += 1  # learn from the symbol AFTER coding it
    encoder.finish()
    bits.close()
    return out.getvalue(), ideal_bits


def decode(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    decoder = ArithmeticDecoder(STATE_BITS, BitInputStream(io.BytesIO(blob[4:])))

    # The decoder runs the SAME model, in the SAME order, seeing only symbols it has
    # already decoded. If it ever drifted from the encoder by even one count, every
    # later symbol would decode to garbage. This is why tufazip needs bit-exact
    # hardware: its model is a neural net, and floating-point noise would be drift.
    counts = np.ones(256, dtype=np.int64)
    out = bytearray()
    for _ in range(n):
        freqs = np.cumsum(counts)
        byte = decoder.read(freqs)
        out.append(byte)
        counts[byte] += 1
    return bytes(out)


def static_entropy_bits_per_byte(data: bytes) -> float:
    """Shannon entropy of the file's overall byte distribution: the best any
    order-0 model could do if it were handed the final frequencies for free."""
    freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = freq[freq > 0] / len(data)
    return float(-(p * np.log2(p)).sum())


def main():
    data = (ROOT / "exercises" / "sample.txt").read_bytes()
    blob, ideal_bits = encode(data)
    restored = decode(blob)

    n = len(data)
    print(f"input                 : {n:>8,} bytes  (8.000 bits/byte)")
    print(f"static entropy        : {static_entropy_bits_per_byte(data):>8.3f} bits/byte  (order-0 floor)")
    print(f"ideal adaptive cost   : {ideal_bits / n:>8.3f} bits/byte  (sum of -log2 p)")
    print(f"actual compressed     : {len(blob):>8,} bytes  ({len(blob) * 8 / n:.3f} bits/byte, "
          f"ratio {n / len(blob):.2f}x)")
    print(f"round trip identical  : {restored == data}")
    assert restored == data, "decode(encode(x)) != x"

    # --- Things to try (edit and re-run) ---
    # 1. Why is "actual" a few bytes above "ideal"? (4-byte header, flush, rounding
    #    to whole bytes, and integer rounding inside the coder.)
    # 2. Change `np.ones` to `np.full(256, 10)`. Does it help or hurt, and why?
    #    (A bigger starting count = slower to learn, but safer for rare bytes.)
    # 3. Shuffle the bytes of sample.txt before encoding. Does the size change?
    #    It shouldn't, and that's the point: order-0 can't see order or context.
    #    Rung 2 fixes that.


if __name__ == "__main__":
    main()
