"""Rung 3: a tiny neural predictor in plain numpy, with hand-written backprop.

Same pipeline as rungs 1 and 2. Only the thing that produces the probabilities changes:

    last 3 bytes --embed--> 48 numbers --W1,tanh--> 64 numbers --W2--> 256 logits --softmax--> P(next byte)

Why a neural net at all? Rung 2's tables had no idea that 'e' and 'a' are alike
(both vowels), so every context had to be seen verbatim to be useful. Here each byte
gets a learned vector (its embedding); bytes that behave alike get similar vectors,
and the net can generalise to contexts it has never seen. A Transformer is this idea
taken much further (nn.py: tok_emb -> 12 blocks -> head).

Learning ("online", like tufazip's decoder): after coding each byte we nudge every
weight to make that byte a little more likely. The decoder makes the SAME nudges,
so both sides stay in sync. We also replay a few random earlier bytes each step,
a small version of tufazip's "retrain" on already-decoded data.

Run (from the repo root):
    docker run --rm -v "$PWD":/work tufazip-learn python exercises/03_tiny_neural_numpy.py
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

CTX = 3      # how many previous bytes the net sees
EMB = 16     # numbers per byte embedding
HID = 64     # hidden layer width
LR = 0.01    # learning rate: size of each nudge. 0.1 overshoots and gets WORSE over
             # time (6.9 bits/byte); 0.003 is too timid (3.84); 0.01-0.03 work (~3.4)
REPLAY = 2   # extra random earlier bytes re-learned after each new byte


class KeepOpenBytesIO(io.BytesIO):
    def close(self):
        pass


# ---------------------------------------------------------------------------
# The network. Shapes in comments; N = CTX * EMB = 48.
# ---------------------------------------------------------------------------
class TinyNet:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.1, (256, EMB))                         # embeddings
        self.W1 = rng.normal(0, 1 / math.sqrt(CTX * EMB), (CTX * EMB, HID))
        self.b1 = np.zeros(HID)
        self.W2 = rng.normal(0, 0.1 / math.sqrt(HID), (HID, 256))
        self.b2 = np.zeros(256)
        self.params = {"E": self.E, "W1": self.W1, "b1": self.b1,
                       "W2": self.W2, "b2": self.b2}

    def forward(self, ctx):
        """ctx: array of CTX byte values. Returns (probabilities, cache for backward)."""
        x = self.E[ctx].reshape(-1)           # (N,)   look up + concatenate embeddings
        z = x @ self.W1 + self.b1             # (HID,) weighted sums
        h = np.tanh(z)                        # (HID,) squash to (-1, 1): the nonlinearity
        logits = h @ self.W2 + self.b2        # (256,) one raw score per possible next byte
        e = np.exp(logits - logits.max())     # subtract max: avoids overflow, same result
        p = e / e.sum()                       # softmax: scores -> probabilities summing to 1
        return p, (ctx, x, h, p)

    def backward(self, cache, target):
        """Gradients of loss = -log p[target] with respect to every parameter.
        Work backwards through the forward pass, applying the chain rule one layer
        at a time."""
        ctx, x, h, p = cache
        dlogits = p.copy()
        dlogits[target] -= 1.0                # softmax + cross-entropy: (p - onehot)
        dW2 = np.outer(h, dlogits)            # (HID,256)
        db2 = dlogits
        dh = self.W2 @ dlogits                # (HID,) how much each hidden unit mattered
        dz = dh * (1.0 - h * h)               # tanh'(z) = 1 - tanh(z)^2
        dW1 = np.outer(x, dz)                 # (N,HID)
        db1 = dz
        dx = self.W1 @ dz                     # (N,) gradient flowing into the embeddings
        dE = np.zeros_like(self.E)
        np.add.at(dE, ctx, dx.reshape(CTX, EMB))   # add (not assign): a byte can repeat in ctx
        return {"E": dE, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply(self, grads, lr):
        for name, g in grads.items():
            self.params[name] -= lr * g       # step downhill: the "nudge"


def context_at(history, j):
    """The CTX bytes before position j, left-padded with 0 near the start."""
    ctx = np.zeros(CTX, dtype=np.int64)
    window = history[max(0, j - CTX):j]
    if len(window):
        ctx[CTX - len(window):] = np.frombuffer(bytes(window), dtype=np.uint8)
    return ctx


# ---------------------------------------------------------------------------
# Predict/learn interface, the same shape as ContextModel in rung 2.
# ---------------------------------------------------------------------------
class NeuralModel:
    def __init__(self):
        self.net = TinyNet()
        self.rng = np.random.default_rng(1)  # replay choices: identical on both sides
        self._cache = None

    def predict(self, history):
        p, self._cache = self.net.forward(context_at(history, len(history)))
        return p

    def update(self, history, symbol):
        """Learn from `symbol`, which followed `history`. Call AFTER coding it."""
        self.net.apply(self.net.backward(self._cache, symbol), LR)
        for _ in range(REPLAY):               # re-study random already-known bytes
            if len(history) == 0:
                break
            j = int(self.rng.integers(0, len(history)))
            p, cache = self.net.forward(context_at(history, j))
            self.net.apply(self.net.backward(cache, history[j]), LR)


def to_freqs(p):
    """Probabilities -> cumulative integer table (same recipe as make_freq_table)."""
    freq = np.maximum(np.round(p * FREQ_TOTAL).astype(np.int64), 1)
    return np.cumsum(freq)


def encode(data):
    model = NeuralModel()
    out = KeepOpenBytesIO()
    out.write(len(data).to_bytes(4, "big"))
    bits = BitOutputStream(out)
    encoder = ArithmeticEncoder(STATE_BITS, bits)

    history, costs = bytearray(), []
    for byte in data:
        freqs = to_freqs(model.predict(history))
        costs.append(-math.log2((freqs[byte] - (freqs[byte - 1] if byte else 0)) / freqs[-1]))
        encoder.write(freqs, byte)
        model.update(history, byte)
        history.append(byte)
    encoder.finish()
    bits.close()
    return out.getvalue(), costs


def decode(blob):
    n = int.from_bytes(blob[:4], "big")
    decoder = ArithmeticDecoder(STATE_BITS, BitInputStream(io.BytesIO(blob[4:])))
    model = NeuralModel()
    history = bytearray()
    for _ in range(n):
        byte = decoder.read(to_freqs(model.predict(history)))
        model.update(history, byte)
        history.append(byte)
    return bytes(history)


# ---------------------------------------------------------------------------
# Trust-but-verify: compare hand-written backprop with brute-force numerics.
# ---------------------------------------------------------------------------
def gradient_check():
    net = TinyNet(seed=42)
    ctx, target = np.array([104, 101, 32]), 116  # "he " -> 't'

    def loss():
        p, _ = net.forward(ctx)
        return -math.log(p[target])

    p, cache = net.forward(ctx)
    analytic = net.backward(cache, target)
    rng, worst, eps = np.random.default_rng(0), 0.0, 1e-5
    for name, param in net.params.items():
        for _ in range(6):
            idx = tuple(int(rng.integers(0, s)) for s in param.shape)
            old = param[idx]
            param[idx] = old + eps
            up = loss()
            param[idx] = old - eps
            down = loss()
            param[idx] = old
            numeric = (up - down) / (2 * eps)  # slope measured by wiggling the weight
            denom = max(abs(numeric), abs(analytic[name][idx]), 1e-8)
            worst = max(worst, abs(numeric - analytic[name][idx]) / denom)
    print(f"gradient check: worst relative error {worst:.1e}  "
          f"({'OK' if worst < 1e-4 else 'BAD'})")
    assert worst < 1e-4


def main():
    gradient_check()
    data = (ROOT / "exercises" / "sample.txt").read_bytes()
    n = len(data)
    print(f"\ninput: {n:,} bytes. Reference from earlier rungs: "
          f"order-0 4.610, order-3 2.652 bits/byte")
    blob, costs = encode(data)
    restored = decode(blob)

    print(f"compressed: {len(blob):,} bytes = {len(blob) * 8 / n:.3f} bits/byte "
          f"({n / len(blob):.2f}x)")
    print("\nlearning curve (bits/byte over each quarter of the file):")
    q = n // 4
    for i in range(4):
        part = costs[i * q:(i + 1) * q if i < 3 else n]
        print(f"  quarter {i + 1}: {sum(part) / len(part):.3f}")
    print(f"\nround trip identical: {restored == data}")
    assert restored == data

    # --- Things to try ---
    # 1. Set REPLAY = 0. How much does the extra re-studying help?
    # 2. Change LR to 0.01 and to 1.0. Too small learns slowly; too big overshoots.
    # 3. Set CTX = 1, then 6. Compare with rung 2's orders. Which wins at each size?
    # 4. Why does the first quarter cost more than the last? What would tufazip's
    #    pretrained weights.bin do to that first-quarter number?


if __name__ == "__main__":
    main()
