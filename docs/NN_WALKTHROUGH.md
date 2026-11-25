# `nn.py`, line by line

A guided read of the real tufazip model and codec. It assumes you have run (or at least
skimmed) the exercises, especially `04_tiny_transformer.py`, which is this file's model
shrunk down. Line numbers refer to `nn.py` as of this fork's `master`.

**Status: Part A of D is written.** The file is 1,738 lines, so the guide is split:

| Part | Lines | What it covers | Status |
|---|---|---|---|
| **A** | 1-120, 122-336 | Imports, config, benchmark helpers, **the model** | written (below) |
| B | 339-640 | The packed-weights loader (`weights.bin`) | not yet written |
| C | 641-855 | Helpers: LR schedules, frequency table, CLI arguments | not yet written |
| D | 859-1737 | `main()`: the online encode/decode loop, retraining, checkpoints | not yet written |

Where Part A touches something that only `main()` can settle, it says so instead of
guessing.

---

# Part A1: the top of the file (lines 1-120)

## Lines 1-7: the docstring says what the file is

> V2_CONFIG holds the fixed settings for the verified enwik9 bitstream. weights.bin
> supplies the Cycle2 initialization; encoding and decoding then reproduce the same
> online learning steps. Deterministic decoding requires the matching two-B200
> hardware and pinned runtime documented in README.md.

Three things to take from it:

1. **The model is not trained in advance and then frozen.** It starts from the weights
   in `weights.bin` and **keeps learning while it codes**, in both directions. This is
   the same online-learning idea as rungs 3-4.
2. **Decoding must replay the encoder's learning steps exactly.** That is why the
   settings are *fixed* (`V2_CONFIG`).
3. **It needs matching hardware.** Same reason as the rungs' determinism warning, at
   much higher stakes: one flipped bit in a GPU float and the rest of a 7.5-day decode
   is garbage.

## Lines 8-10: determinism switches, set *before* importing torch

```python
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")  # deterministic custom kernels
```

These come first on purpose: the libraries read them when they load. `setdefault`
means a value you set yourself wins. The comments give the intent (deterministic matrix
multiplies and kernels). Nothing in this file is "just" performance work: the same
numbers have to come out on every run.

## Lines 12-35: imports worth noticing

| Import | What it tells you |
|---|---|
| `torch.distributed`, `DistributedDataParallel` | Two GPUs, working in parallel (`'distributed': True`) |
| `SummaryWriter` (TensorBoard) | Training curves for the researchers. A do-nothing stand-in, `NullSummaryWriter` (line 696), exists, so logging looks optional |
| `arithmetic_coder` | **The same file every exercise reuses.** Rungs 1-4 and tufazip share it |
| `checkpoint_utils` (`REQUEUE_EXIT_CODE`, `atomic_torch_save`, `stop_requested`, ...) | A 7.5-day job must survive being stopped and restarted. Part D covers how |
| `zipfile` | Used for the size benchmark in lines 80-119, below |

## Lines 37-74: `V2_CONFIG`, the fixed settings

A plain dict. What each entry means, grouped:

**Data and vocabulary**

| Key | Value | Meaning |
|---|---|---|
| `raw_bytes` | 1,000,000,000 | The input: 1 GB (`enwik9`) |
| `tokenizer` | `'word'` | Text is split into **word-ish tokens**, not bytes (that is what `preprocess.c` does) |
| `vocab_size` | 4,097 | Number of distinct tokens. Rung 4 had 256 (bytes) |
| `vocab_bytes` | 34,421 | Size in bytes of the vocabulary file `enwik9.voc` (checked: matches the file) |

**Model shape** (compare with `04_tiny_transformer.py`)

| Key | Real | Rung 4 |
|---|---|---|
| `n_layer` | 12 | 1 |
| `d_model` | 1024 | 64 |
| `n_head` | 8 | 4 |
| `d_inner` | 2816 | 176 |
| `ctx_len` | 192 | 32 |
| `dropout` | 0.25 | none |
| `position_type` | `'rope'` | learned position embedding |
| `norm_style` | `'peri'` | plain pre-norm |
| `untie_embeddings` | True | tied |
| `dtype` | `'bf16'` | float32 |

**Online learning and training** (the meaning of these is settled in `main()`, Part D;
I list them so you recognise them, not to explain them early)
`lr`, `min_lr_ratio`, `warmup`, `clip`, `adam_eps`, `beta1`, `beta2`, `batch_size`,
`block_len`, `online_loss`, `project_only_targets`, `retrain_*`, `checkpoint_every_blocks`,
`log_interval`, `seed`.

Two that are worth knowing now:

- **`beta1: 0.0`** means Adam is run **without momentum** (the optimizer's memory of
  past gradient direction is switched off). My rung 4 used Adam's default (0.9). This is
  one of the places the toy and the real thing differ.
- **`seed: 1111`** pins the random initialization and every random choice. As in the
  rungs (`np.random.default_rng(1)`, `torch.manual_seed(0)`), anything random has to be
  replayable by the decoder.

## Lines 77-78: two constants

```python
LOG2 = math.log(2.0)
FREQ_TOTAL = 10_000_000  # frequency-table resolution handed to the arithmetic coder
```

`LOG2` converts between natural-log "nats" (what PyTorch's loss reports) and bits, the
same `nats / math.log(2)` I used in rung 4's `bits_per_byte`. `FREQ_TOTAL` is the
resolution of the probability table given to the coder: probabilities are scaled to
integers summing to about 10 million. **Rungs 2-4 copied this exact number** so their
precision matched the real thing.

## Lines 80-119: the "total size" benchmark helpers

```python
# Files that must ship with this script to decompress its output, used for the
# Matt Mahoney "total size" benchmark (compressed bytes + DEFLATE-9 zip of these).
```

A decompressor is useless without its code, so the benchmark scores **the compressed
file plus the zipped source of everything needed to decode it**. Otherwise you could
hide all the information inside a giant decompressor.

- `DECOMPRESSOR_COMMON` is `nn.py`, `arithmetic_coder.py`, `checkpoint_utils.py`.
- `decompressor_files(tokenizer, model_backend)` (line 95) adds the tokenizer source
  (`preprocess.c` for `'word'`, otherwise `tokenizers.py`) and extra files for
  non-Transformer backends (Mamba, KDA). With the default settings it returns the common
  three plus `preprocess.c`.
- `decompressor_zip_size(files, extra_files)` (line 111) zips them into memory at
  **DEFLATE level 9** and returns the byte count.

The practical consequence: **code size counts**, so extra source files cost score.

**Open question for Part B:** `weights.bin` is only 4.3 MB (4,334,741 bytes), yet the
model has roughly 160 million parameters, which is only ~0.2 bits per parameter. So it
can't simply be the full model's weights. The names `load_depth_prior` and
`depth_block_mapping` (lines 454 and 515) suggest a smaller-depth model that gets
expanded to 12 layers. That is a guess until Part B reads the code.

---

# Part A2: the model (lines 122-336)

The architecture, bottom up. The model takes token ids `[batch, seq]` and returns
logits `[batch, seq, vocab]`. Shapes below use `B` batch, `T` sequence length,
`D = d_model = 1024`, `H = n_head = 8`, `d_head = D/H = 128`.

## `RMSNorm` (lines 125-134)

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        return normed.type_as(x) * self.weight
```

**Why:** numbers passing through many layers drift huge or tiny; normalizing keeps them
in a range where learning is stable.

**What:** for each position's vector, compute the **mean of squares** (`var`), divide by
its square root (`rsqrt` is "1 / sqrt"), then multiply by a learned per-number gain.
`eps` stops a division by zero. Compared with the more familiar LayerNorm, there is no
mean subtraction and no bias: simpler and cheaper.

**The one difference from my rung 4 version:** `x.float()` upcasts to float32 for the
arithmetic, then `type_as(x)` casts back to the working type. The real model runs in
**bf16** (16-bit floats), which is too coarse for squaring and averaging. Doing the norm
in float32 protects accuracy where it matters most.

## RoPE: `build_rope_cache` and `apply_rope` (lines 137-149)

**Why:** attention alone can't tell "ab" from "ba". Rung 4 added a learned vector per
position. The real model uses **RoPE** (rotary position embedding) instead, which
injects position by *rotating* the query and key vectors.

```python
def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos(), emb.sin()  # each [seq_len, head_dim]
```

Line by line, for `head_dim = 128`:

1. `torch.arange(0, head_dim, 2)` is `0, 2, 4, ..., 126`: **64 values**, one per *pair*
   of numbers in a head. Dividing by `head_dim` and using `base ** (...)` gives 64
   different rotation speeds, `1 / 10000^(2i/128)`. The first pair spins at about
   **1 radian per position**; the last at about **0.0001** (very slowly).
2. `torch.outer(positions, inv_freq)` gives `[seq_len, 64]`: the angle each pair has
   rotated by at each position (position × speed).
3. `torch.cat([angles, angles])` repeats it to `[seq_len, 128]` so it lines up with all
   128 numbers of a head (details in `apply_rope` below).
4. Returns `cos` and `sin` of those angles.

Fast pairs tell nearby positions apart; slow pairs tell far-apart positions apart,
like the hands of a clock: seconds, minutes, hours.

```python
def apply_rope(x, cos, sin):
    # x: [batch, n_head, seq, head_dim]; cos/sin: [seq, head_dim]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
```

- `x.chunk(2)` splits the 128 numbers into a first half `x1` and a second half `x2`.
  Number `i` of the first half is **paired** with number `i` of the second half.
- `rotated = cat([-x2, x1])` is the "rotate by 90 degrees" version of each pair.
- `x * cos + rotated * sin` is the standard 2D rotation: for each pair
  `(x1, x2) -> (x1·cos − x2·sin, x1·sin + x2·cos)`.

**Why this gives *relative* position (the clever bit):** rotating `q` at position `m`
and `k` at position `n`, then taking their dot product, gives a result that depends
only on `n − m`, not on `m` or `n` separately (because a rotation's transpose is the
reverse rotation). So attention scores see "how far back" without any extra learned
table. At position 0 the angle is 0 (`cos = 1`, `sin = 0`), so the vector is unchanged.

## `Attention` (lines 152-217)

**Setup (lines 153-178):**

```python
self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)   # makes q, k, v together
self.proj = nn.Linear(d_model, d_model, bias=False)      # mixes the heads back
self.q_norm = RMSNorm(self.d_head) if qk_norm else nn.Identity()
self.k_norm = RMSNorm(self.d_head) if qk_norm else nn.Identity()
```

- `qkv` and `proj` are identical to rung 4.
- **`q_norm` / `k_norm` (QK-norm):** a separate RMSNorm over each head's `q` and `k`,
  each of size `d_head = 128`. The comment (lines 164-166) explains the aim: bound the
  attention scores so they can't spike, which removes the need for "logit
  soft-capping" and tolerates a higher learning rate. Rung 4 left this out because it
  matters with depth.
- Lines 169-178 define an alternative `learned_relative` position scheme. **V2_CONFIG
  uses RoPE, so that branch isn't used;** skip it on a first read.

**`forward` (lines 180-217), RoPE path:**

```python
batch, seq, d_model = x.shape
q, k, v = self.qkv(x).split(d_model, dim=2)                          # 3 x [B,T,D]
q = q.view(batch, seq, self.n_head, self.d_head).transpose(1, 2)     # [B,H,T,d_head]
k = k.view(...).transpose(1, 2)
v = v.view(...).transpose(1, 2)
q, k = self.q_norm(q), self.k_norm(k)                                # QK-norm
q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)              # rotate
out = F.scaled_dot_product_attention(
    q, k, v, is_causal=True,
    dropout_p=self.dropout if self.training else 0.0)
...
out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)    # glue heads
return self.proj(out)
```

The order matters: **make q, k, v, split into heads, normalize q and k, *then* rotate
them** (lines 186-188). `v` is never normalized or rotated: values are *content*, not
positions.

Line 189 is the call that `attention_by_hand.py` checks against PyTorch: scores,
scaling by `sqrt(d_head)`, the causal mask (`is_causal=True`) and softmax, in one
fused function. **New compared with the rungs:** `dropout_p` randomly zeroes some
attention weights, but only when `self.training` is true. `V2_CONFIG` has `dropout: 0.25`,
so whether the real codec runs in training mode is something `main()` has to settle.
I'll mark it in Part D rather than assume.

Lines 192-215 are the `learned_relative` alternative, not covered here.

## `SwiGLU` (lines 220-228)

```python
self.gate = nn.Linear(d_model, d_inner, bias=False)
self.up   = nn.Linear(d_model, d_inner, bias=False)
self.down = nn.Linear(d_inner, d_model, bias=False)

def forward(self, x):
    return self.down(F.silu(self.gate(x)) * self.up(x))
```

Identical to rung 4. Attention moves information **between** positions; this
transforms it **within** each position. `silu(x)` is `x · sigmoid(x)`, a smooth
on/off switch. `silu(gate(x)) * up(x)` lets the "gate" path decide, number by number,
how much of the "up" path to let through; `down` narrows back to `D`.
`d_inner = 2816` is about 8/3 of 1024 rounded up to a multiple of 256 (the 8/3 comes
from SwiGLU having three matrices where a plain feed-forward has two; the comment is at
line ~747).

## `Block` (lines 231-259)

One layer: attention, then SwiGLU, each with a residual connection (`x = x + ...`).

```python
def forward(self, x, cos, sin):
    if self.norm_style == "peri":
        x = x + self.drop(
            self.attn_post_norm(
                self.attn(self.attn_pre_norm(x), cos, sin)))
        x = x + self.drop(
            self.ffn_post_norm(self.ffn(self.ffn_pre_norm(x))))
    else:
        ...
```

**"peri" = norm *around* each sub-layer:** normalize the input (`attn_pre_norm`), run
attention, normalize the *output* (`attn_post_norm`), apply dropout, and **then** add it
to the residual stream. The same pattern repeats for the feed-forward. V2_CONFIG uses
`'peri'`.

- Rung 4 only normalized the input of each sub-layer ("pre-norm").
- The `else` branch (lines 254-258) is the original Transformer "post-norm": add first,
  then normalize the sum.
- **Why the extra norm:** it keeps each layer's contribution to the residual stream a
  similar size, which helps a 12-layer stack train stably. (The `Transformer` comment at
  lines 282-283 says the same about the embedding.)

## `Transformer` (lines 262-336)

**Constructor, in order of what it builds:**

| Lines | Builds | Notes |
|---|---|---|
| 269-280 | `block_schedule` | Which block to run at each depth. Default `range(n_layer)` = each block once. Validation (275-279): indices in range, every block used. Lets a block run more than once (a "looped" model); V2 doesn't |
| 281 | `tok_emb` | token id to 1024-vector. 4,097 x 1024 |
| 284-285 | `emb_norm` | RMSNorm on the embedding ("so the first block sees the same well-conditioned input every other block sees") |
| 286 | `drop` | dropout on the embedding |
| 287-290 | `blocks` | the 12 `Block`s |
| 291-292 | `norm` | final RMSNorm |
| 293 | `head` | `Linear(d_model, vocab_size)`: hidden vector to one score per token |
| 294-295 | weight tying | `self.head.weight = self.tok_emb.weight` **when `tie_emb`**. V2_CONFIG sets `untie_embeddings=True`, so presumably the real run builds them separately; the argument plumbing is in `main()` (Part D) and I haven't verified it yet |
| 297-303 | RoPE tables | computed once; `register_buffer(..., persistent=False)` stores them with the model (moves to GPU with it) but **keeps them out of saved weights**, since they're recomputable |

**Initialization (lines 305-317):**

```python
self.apply(self._init_weights)           # every Linear and Embedding: normal(0, 0.02)
residual_std = 0.02 / math.sqrt(2 * len(self.block_schedule))
for name, param in self.named_parameters():
    if name.endswith("proj.weight") or name.endswith("down.weight"):
        nn.init.normal_(param, mean=0.0, std=residual_std)
```

All weights start at standard deviation 0.02 (the same as rung 4). Then the two matrices
that **write into the residual stream** (`attn.proj` and `ffn.down`) are made smaller by
`1 / sqrt(2 × depth)`. Every block adds two contributions to the stream, so with `L`
blocks there are `2L` additions, and shrinking each keeps the sum's size roughly
constant regardless of depth. The comment (lines 306-307) notes it scales by *executed*
depth, so a looped model starts at the same scale as the unlooped one it will become.

**Important:** per the file's docstring, this random init is only the *fallback*.
`weights.bin` supplies the real starting weights (Part B).

**`forward` (lines 319-336):**

```python
x = self.drop(self.emb_norm(self.tok_emb(idx)))        # ids -> normalized vectors
for block_index in self.block_schedule:
    x = self.blocks[block_index](x, cos, sin)          # 12 times
x = self.norm(x)
if output_positions is not None:
    x = x[:, output_positions]                         # keep only the positions we need
return self.head(x)                                    # -> logits [B, T', vocab]
```

Differences from rung 4's `forward`:

- **No position embedding is added.** RoPE acts inside attention, on `q` and `k`
  (the `cos, sin` slices `[:seq]` are passed down to every block).
- **`output_positions`** (lines 329-335). The final projection to 4,097 scores is large,
  and during online coding only the **newest** position's prediction is ever used (earlier
  ones were already coded). The comment says "replay may prepend context whose logits are
  never used". Slicing the hidden states *before* `head` skips those rows entirely, and
  the earlier layers are untouched. Rung 4 computed all 32 positions' logits and used
  them all for training; the real codec can't afford to.

## Summary: the model in one picture

```
token ids [B,T]
  -> tok_emb -> emb_norm -> dropout                                     [B,T,1024]
  -> 12 x Block( peri-norm Attention[8 heads, QK-norm, RoPE, causal] + residual
                 peri-norm SwiGLU(1024 -> 2816 -> 1024)              + residual )
  -> final norm
  -> (optionally keep only the newest position)
  -> head -> 4,097 logits -> softmax -> P(next token)
```

Parameter count, from the config: per block about 3.1M (qkv) + 1.0M (proj) + 8.7M
(SwiGLU) = 12.8M; times 12 = **154M**, plus 4.2M for each embedding table (one if
tied, two if untied). So about **158-163M** in total. This is arithmetic from the
config, not a measurement.

---

*Parts B-D follow once Part A is reviewed.*
