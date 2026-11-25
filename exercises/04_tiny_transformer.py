"""Rung 4 (Part 1): a tiny causal Transformer, mirroring nn.py.

Same pipeline as rungs 1-3. Only the thing that produces the probabilities changes.
Rung 3 saw the last 3 bytes as one fixed bundle. Here attention lets every position
look back over up to CTX bytes and choose how much to listen to each (the five steps
in attention_by_hand.py), and a stack of blocks refines that.

    bytes -> tok_emb + pos_emb
          -> [ RMSNorm -> Attention (+residual) -> RMSNorm -> SwiGLU (+residual) ]
          -> RMSNorm -> head -> 256 logits -> softmax -> P(next byte)

This file is nn.py shrunk down:

                     nn.py (V2)      here
    vocabulary       word tokens     256 bytes
    layers           12              1
    d_model          1024            64
    heads            8               4
    context          192 tokens      32 bytes

Left out on purpose (stability tricks that only matter at 12 layers): QK-norm,
"peri" norm placement, block_schedule, learned_relative attention.
One deliberate difference: nn.py uses RoPE to tell the model about order; here a
plain learned position embedding does the same job and is easier to read. RoPE is
covered in docs/NN_WALKTHROUGH.md.

PART 1: the model classes, plus two checks (does it have the claimed shape, and does
        the causal mask hold?).
PART 2: the encoder/decoder loop that turns these probabilities into bits, learning
        online exactly like rung 3: after coding each byte, nudge the weights so that
        byte was a little more likely. The decoder makes the SAME nudges. Unlike rung
        3, one update trains on the whole window (a Transformer predicts at every
        position at once, about CTX training signals per step), and the optimizer is
        Adam with gradient clipping, as in nn.py, instead of plain SGD.

Run (from the repo root):
    docker run --rm -v "$PWD":/work tufazip-learn python exercises/04_tiny_transformer.py
"""
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from arithmetic_coder import (ArithmeticDecoder, ArithmeticEncoder,  # noqa: E402
                              BitInputStream, BitOutputStream)

VOCAB = 256     # one entry per possible byte
CTX = 32        # how many previous bytes attention can look back over
D_MODEL = 64    # width of the vector each position carries (nn.py: 1024)
N_HEAD = 4      # attention heads side by side (nn.py: 8)
D_INNER = 176   # SwiGLU hidden width: ~8/3 * D_MODEL rounded to a multiple of 16
                # (nn.py's comment at line ~747 explains the 8/3: SwiGLU has 3
                # matrices where a plain ReLU feed-forward has 2)


# ---------------------------------------------------------------------------
# The model. Shapes in comments: B = batch, T = sequence length, D = D_MODEL.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Rescale each position's vector so its average square is 1, then apply a
    learned per-number gain. Keeps numbers from drifting huge or tiny as they pass
    through layers. (nn.py line 125; identical.)"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)          # mean of squares, per position
        return x * torch.rsqrt(var + self.eps) * self.weight


class Attention(nn.Module):
    """Steps 1-5 of attention_by_hand.py, run by N_HEAD heads at once."""

    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        # One matrix makes q, k and v for every head at once: D -> 3*D.
        # In attention_by_hand.py we wrote q, k, v down by hand; this is where a
        # real model gets them. (nn.py line 162.)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # Mixes the heads' outputs back into one D-sized vector. (nn.py line 163.)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                                # x: [B, T, D]
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)            # each [B, T, D]
        # Cut D into N_HEAD slices of d_head, and move heads next to the batch so each
        # head does its own attention: [B, T, D] -> [B, N_HEAD, T, d_head].
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # Scores (Q @ K.T), scaling by sqrt(d_head), causal mask, softmax, weights @ V:
        # the whole of attention_by_hand.py Steps 1-5 in one call. (nn.py line 189.)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # glue the heads back
        return self.proj(out)


class SwiGLU(nn.Module):
    """The per-position "thinking" step. Attention moves information BETWEEN
    positions; this transforms it WITHIN each position. (nn.py line 220; identical.)

    gate and up both widen D -> D_INNER. silu(gate) acts as a soft on/off switch on
    `up`, element by element; down narrows back to D."""

    def __init__(self, d_model, d_inner):
        super().__init__()
        self.gate = nn.Linear(d_model, d_inner, bias=False)
        self.up = nn.Linear(d_model, d_inner, bias=False)
        self.down = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Attention then SwiGLU, each with a residual connection (`x = x + ...`): the
    layer adds a correction to what was already there instead of replacing it, so
    information (and gradients) can flow straight through many layers."""

    def __init__(self, d_model, n_head, d_inner):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_head)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_inner)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab=VOCAB, n_layer=1, d_model=D_MODEL, n_head=N_HEAD,
                 d_inner=D_INNER, ctx=CTX):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)   # byte -> vector (rung 3's E)
        # Attention cannot tell "ab" from "ba" by itself (it is a weighted average,
        # and averages ignore order), so we add a learned vector per position.
        self.pos_emb = nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_inner) for _ in range(n_layer)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # Tie input and output embeddings (nn.py line 295): the vector that
        # represents byte 'e' going in is the one scored against coming out.
        # Halves those parameters and ties "what e means" to "when e comes next".
        # (nn.py supports this, but V2_CONFIG sets untie_embeddings=True: with a
        # 4097-word vocabulary and 1024 wide vectors it pays for separate ones.
        # With only 256 bytes, tying is the better trade.)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # Small starting weights: all 256 logits start nearly equal, so the
        # untrained model predicts nearly uniformly (nn.py line 313).
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):                           # idx: [B, T] ints in 0..255
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)     # [B, T, D]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))                # [B, T, VOCAB] logits


# ---------------------------------------------------------------------------
# Part 1 checks: does the model have the shape and behaviour we claim?
# ---------------------------------------------------------------------------
def bits_per_byte(model, idx):
    """Average -log2 P(true next byte) over every position of `idx` ([B, T])."""
    logits = model(idx[:, :-1])                       # predict byte t+1 from bytes 0..t
    nats = F.cross_entropy(logits.reshape(-1, VOCAB), idx[:, 1:].reshape(-1))
    return nats.item() / math.log(2)                  # nats -> bits


def part1_checks():
    torch.manual_seed(0)
    model = TinyTransformer().eval()
    data = (ROOT / "exercises" / "sample.txt").read_bytes()
    idx = torch.tensor(list(data[:CTX]), dtype=torch.long)[None]   # [1, CTX]

    n_params = sum(p.numel() for p in model.parameters())  # tied weight counted once
    print(f"parameters: {n_params:,}   (nn.py V2 is ~160 million)")

    logits = model(idx)
    print(f"input {tuple(idx.shape)} -> logits {tuple(logits.shape)}"
          f"   (one 256-way prediction per position)")

    # CHECK 1: untrained model ~ uniform, so ~log2(256) = 8 bits/byte.
    bpb = bits_per_byte(model, idx)
    print(f"untrained bits/byte: {bpb:.3f}   (uniform guessing = {math.log2(VOCAB):.3f})")
    assert abs(bpb - 8.0) < 0.1

    # CHECK 2: causal mask. Change the LAST byte; every earlier position's logits
    # must be untouched, since they may not see the future.
    changed = idx.clone()
    changed[0, -1] = (changed[0, -1] + 1) % VOCAB
    with torch.no_grad():
        a, b = model(idx), model(changed)
    same_before = torch.allclose(a[:, :-1], b[:, :-1])
    moved_last = not torch.allclose(a[:, -1], b[:, -1])
    print(f"causal: earlier positions unchanged = {same_before}, "
          f"last position moved = {moved_last}")
    assert same_before and moved_last


# ---------------------------------------------------------------------------
# PART 2: from probabilities to bits.
# ---------------------------------------------------------------------------
STATE_BITS = 32
FREQ_TOTAL = 10_000_000  # same resolution nn.py hands to the coder
LR = 3e-3        # Adam step size. (nn.py uses 4e-5 for a 160M-parameter model; a
                 # 69k-parameter model on 8 KB can afford a much bigger step.)
CLIP = 1.0       # cap on the size of one update, so a surprising byte can't wreck it
REPLAY = 2       # extra random earlier windows re-studied after each new byte
BOS = 0          # start-of-text marker: byte 0 never occurs in the sample text
N_BYTES = 8000   # how much of sample.txt to code. None = the whole 39,785 bytes
                 # (about 5x slower: every byte costs a forward+backward pass or three)


class KeepOpenBytesIO(io.BytesIO):
    def close(self):
        pass


def seq_ending_at(history, end):
    """The last CTX+1 values of [BOS, history[0], ..., history[end-1]].

    As a model INPUT we use the last CTX of these; the same list shifted by one is the
    TARGET (what each position should have predicted). The window slides as we go, so
    position 0 of the window is whichever byte is CTX back."""
    return (bytes([BOS]) + bytes(history[max(0, end - CTX):end]))[-(CTX + 1):]


def as_ids(values):
    return torch.tensor(list(values), dtype=torch.long)[None]    # [1, T]


class TransformerModel:
    """predict / update, the same interface as rungs 2 and 3."""

    def __init__(self):
        torch.manual_seed(0)           # encoder and decoder must start IDENTICAL
        self.net = TinyTransformer()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=0.0)
        self.rng = np.random.default_rng(1)   # replay choices: identical on both sides
        self._logits = None

    def predict(self, history):
        """P(next byte) as 256 probabilities. Keeps the graph for update()."""
        ids = as_ids(seq_ending_at(history, len(history))[-CTX:])
        self._logits = self.net(ids)[0]                          # [T, 256]
        # The newest position's logits predict the byte we are about to code.
        return torch.softmax(self._logits[-1].detach().double(), -1).numpy()

    def _learn(self, logits, targets):
        loss = F.cross_entropy(logits, targets)       # mean -log P(true byte), per position
        self.opt.zero_grad()
        loss.backward()                               # autograd = rung 3's backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()

    def update(self, history, symbol):
        """Learn from `symbol`, which followed `history`. Call AFTER coding it."""
        grown = bytes(history) + bytes([symbol])
        seq = seq_ending_at(grown, len(grown))
        # Reuse predict()'s logits: its input was exactly seq[:-1]. Target at each
        # position is simply the next byte of the window.
        self._learn(self._logits, as_ids(seq[1:])[0])
        for _ in range(REPLAY):                       # re-study random known bytes
            if len(history) == 0:
                break
            j = int(self.rng.integers(0, len(history)))
            seq = seq_ending_at(history, j + 1)
            self._learn(self.net(as_ids(seq[:-1]))[0], as_ids(seq[1:])[0])


def to_freqs(p):
    """Probabilities -> cumulative integer table (same recipe as rungs 2-3)."""
    freq = np.maximum(np.round(p * FREQ_TOTAL).astype(np.int64), 1)
    return np.cumsum(freq)


def encode(data):
    model = TransformerModel()
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
    model = TransformerModel()
    history = bytearray()
    for _ in range(n):
        byte = decoder.read(to_freqs(model.predict(history)))
        model.update(history, byte)
        history.append(byte)
    return bytes(history)


def main():
    torch.set_num_threads(1)   # one thread: tiny model, and reproducible float sums
    part1_checks()
    data = (ROOT / "exercises" / "sample.txt").read_bytes()
    if N_BYTES is not None:
        data = data[:N_BYTES]
    n = len(data)
    print(f"\ninput: {n:,} bytes. References on the FULL 39,785 bytes: order-0 4.638, "
          f"order-3 2.652, rung-3 net 3.364 bits/byte")

    t0 = time.time()
    blob, costs = encode(data)
    t1 = time.time()
    restored = decode(blob)
    t2 = time.time()

    print(f"compressed: {len(blob):,} bytes = {len(blob) * 8 / n:.3f} bits/byte "
          f"({n / len(blob):.2f}x)   [encode {t1 - t0:.0f}s, decode {t2 - t1:.0f}s]")
    print("\nlearning curve (bits/byte over each quarter of the input):")
    q = n // 4
    for i in range(4):
        part = costs[i * q:(i + 1) * q if i < 3 else n]
        print(f"  quarter {i + 1}: {sum(part) / len(part):.3f}")
    print(f"\nround trip identical: {restored == data}")
    assert restored == data

    # --- Things to try ---
    # 1. Set REPLAY = 0. How much does re-studying help, compared with rung 3?
    # 2. Try LR = 3e-2 and 3e-4. Which side overshoots, and which learns too slowly?
    # 3. Set CTX = 4 (and N_HEAD = 1). Attention now sees about what rung 3 saw. Does
    #    the Transformer's extra machinery still help?
    # 4. Set N_BYTES = None. Does the full file land closer to rung 2's 2.652?
    # 5. Read the learning curve against the rung-2 note "context helps, then runs
    #    out": what would tufazip's pretrained weights.bin do to quarter 1?


if __name__ == "__main__":
    main()
