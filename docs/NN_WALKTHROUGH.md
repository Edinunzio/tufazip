# `nn.py`, line by line

A guided read of the real tufazip model and codec. It assumes you have run (or at least
skimmed) the exercises, especially `04_tiny_transformer.py`, which is this file's model
shrunk down. Line numbers refer to `nn.py` as of this fork's `master`.

**Status: complete. Parts A, B, C, D1 and D2 are all written.** The file is 1,738 lines, so the guide is split:

| Part | Lines | What it covers | Status |
|---|---|---|---|
| **A** | 1-120, 122-336 | Imports, config, benchmark helpers, **the model** | written (below) |
| **B** | 339-640 | The packed-weights loader (`weights.bin`) | written (below) |
| **C** | 641-855 | Helpers: LR schedules, frequency table, CLI arguments | written (below) |
| **D1** | 859-1254, 1429-1574 | `main()`: setup, then **the core online encode/decode loop** | written (below) |
| **D2** | 1255-1427, 1577-1737 | `main()`: `retrain`, checkpoints and requeues, the finish | written (below) |

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

A note on `weights.bin`: it is only 4.3 MB (4,334,741 bytes), yet the model has about
162 million parameters, so it can't simply hold that model's weights. Part B (below)
opens it up and shows what it really is: a **2-block, 1-bit** model that is "untied"
into 12 blocks.

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
but `main()` keeps the model in **eval mode while coding** and switches to training
mode only during retraining (Part D1), so dropout is active only in the retraining
passes.

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
| 294-295 | weight tying | `self.head.weight = self.tok_emb.weight` **when `tie_emb`**. The real run is **untied**: `weights.bin` records `tie_embeddings: false` for both its source and target, and `load_depth_prior` (Part B) refuses to load unless the geometry matches exactly |
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

Parameter count, from the config: per block 3,145,728 (qkv) + 1,048,576 (proj) +
8,650,752 (SwiGLU) + 4,352 (norms) = 12,849,408; times 12 = 154,192,896. Add two
untied 4,097 x 1,024 embedding tables (8,390,656) and two top-level norms (2,048) for
**162,585,600** in total. This is arithmetic, but it is cross-checked: the same formula
with 2 blocks gives 34,091,520, exactly the parameter count in `weights.bin`'s header
(Part B).

---

---

# Part B: the packed-weights loader (lines 339-640)

**The question this part answers:** the model has ~162.6 million parameters, but
`weights.bin` is only 4.3 MB. How does the one become the other?

## What `weights.bin` actually contains (measured, not guessed)

I parsed the file's header (a small JSON block, described below). Findings:

| Field | Value | Meaning |
|---|---|---|
| magic | `QPI1` | Archive type: a "prior" meant to *initialize* the model |
| `initialization.kind` | `loop-untying` | See "Loop-untying", below |
| source model | `n_layer: 2`, otherwise identical geometry | **Only 2 blocks are stored** |
| target model | `n_layer: 12` | What it is expanded into |
| `block_schedule` | `[0,1,0,1,0,1,0,1,0,1,0,1]` | Target block `i` copies source block `schedule[i]` |
| `initialization_seed` | 1111 | Matches `V2_CONFIG['seed']` |
| `quantizer` | `learned-symmetric-row-fp16-scale` | Each row is stored as signs plus one fp16 scale |
| `policy` | `uniform1` | Every large matrix is stored at **1 bit** per number |
| entries | 26 tensors | 2 blocks x 11 tensors each, plus `tok_emb`, `head`, 2 norms (the exact set is enforced; see below) |
| stored parameters | 34,091,520 | 34,080,768 at 1 bit, plus 10,752 at 8 bits (the norm weights) |
| file size | 4,334,741 bytes | about 4.26 MB of sign bits plus scales and metadata |

So: **34 million numbers, each stored as a single sign bit** (plus one fp16 scale per
row, and a few 8-bit norm gains), which is what makes 4.3 MB possible. The 2-block
model is then copied into 12 block slots.

`★` Two consequences worth stating plainly:

1. The starting point is a *coarse* one (signs only), not a polished full-precision
   model. It is a head start, not a finished model. The online learning of Part D does
   the rest.
2. The same 4.3 MB serves both the encoder and the decoder, which is part of what makes
   the scheme work: both sides begin from identical numbers.

## The file format (lines 339-440)

```
+--------+----------------+-----------------------+----------------------------+
| magic  | metadata_len   | metadata (JSON)       | for each entry, in order:  |
| 4 B    | 8 B, little-   | `metadata_len` bytes  |   scales (fp16)  [if bits<16]
|        | endian uint64  |                       |   values (packed bits)     |
+--------+----------------+-----------------------+----------------------------+
```

Line by line of `read_tensor_archive(path)`:

- **Lines 362-364:** read 4 bytes; accept only `QPR1`, `QPI1`, `LRA1`, `QLA1`. (Only
  `QPI1` is used by this file's loader; I don't know what the other three are for.)
- **Lines 365-372:** read an 8-byte length (`struct.unpack("<Q", ...)`, little-endian
  unsigned 64-bit), then that many bytes of JSON, and parse them.
- **Lines 376-399:** for each metadata `entry`, **validate before trusting**:
  non-empty unique name, a valid shape, `bits` in `{1, 2, 4, 8, 16}`,
  `value_bytes` equals what `_packed_size` computes, and `scale_count` is the first
  dimension (one scale **per row**) for 2D+ tensors, or 1 for vectors, and 0 at 16 bits.
- **Lines 400-407, 16-bit entries:** just read `float16` values and widen to float32.
- **Lines 408-432, quantized entries:** read the scales, read the packed values,
  unpack, convert to signed numbers, multiply by the row scale.
- **Line 433:** wrap each as a `torch` float32 tensor.
- **Lines 436-437:** **reject trailing bytes**: any extra data after the last tensor is
  an error.

**Why so strict?** This code decides the starting weights that *both* sides must agree
on bit for bit. A truncated or altered file should fail loudly rather than quietly
produce slightly different weights and garbage output hours into a decode.

The docstring (line 360) says it all: "Decode an archive into **the exact float32
tensors seen by inference**." Everything downstream sees ordinary float32, so nothing
else in the file needs to know about bit packing.

### Packing and unpacking (lines 343-357)

```python
def _packed_size(num_values, bits):
    if bits == 16:
        return num_values * 2
    return (num_values * bits + 7) // 8      # round UP to whole bytes
```

`(n * bits + 7) // 8` is "ceiling divide by 8" in integer maths: 1,000 one-bit values
need 125 bytes, 1,001 need 126.

```python
def _unpack_codes(payload, count, bits):
    packed = np.frombuffer(payload, dtype=np.uint8)
    if bits == 8:
        return packed[:count].copy()
    per_byte = 8 // bits                                   # 1 bit -> 8 values/byte
    shifts = np.arange(per_byte, dtype=np.uint16) * bits   # 0,1,2,...,7 for 1 bit
    values = (
        (packed.astype(np.uint16)[:, None] >> shifts[None, :])
        & ((1 << bits) - 1))                               # keep only `bits` bits
    return values.reshape(-1)[:count].astype(np.uint8)
```

For 1-bit values: each byte is expanded into 8 values by shifting right by 0, 1, ..., 7
and masking with `1`. The first value in a byte is the **lowest** bit, matching the
header's `"bit_packing": "little-endian-within-byte"`. `[:count]` drops the padding bits
in the final byte.

### From codes to numbers (lines 419-432)

```python
if bits == 1:
    signed = codes.astype(np.int16) * 2 - 1                  # 0 -> -1, 1 -> +1
else:
    signed = codes.astype(np.int16) - ((1 << (bits - 1)) - 1)
...
value = signed.astype(np.float32).reshape(shape) * scales.reshape(scale_shape)
```

For **1 bit**, a stored `0` means `-1` and a stored `1` means `+1`. Then each **row** is
multiplied by its own scale (`scale_shape = (rows, 1, ...)`). So a 1-bit matrix is
`sign pattern x one number per row`. For wider formats (2, 4, 8 bits) the code is
re-centred around zero (`code - (2^(bits-1) - 1)`) before scaling. This file uses only
1-bit (the large matrices) and 8-bit (the norms).

## Fingerprints: `sha256_file` and `model_state_sha256` (lines 443-451, 503-512)

- `sha256_file` hashes a file in 1 MB chunks. Used on `weights.bin` and on the
  vocabulary.
- `model_state_sha256` hashes every tensor's **name, shape, dtype and exact bytes**, in
  sorted-name order. Two models match under this hash only if every single float is
  identical.

These are the tools of the determinism theme: not "are they close?" but "are they
byte for byte the same?". Both sides can compute them and compare.

## `depth_block_mapping` and `parse_block_schedule` (lines 454-483)

Two ways to grow a shallow model into a deeper one:

**Depth-growth** (`depth_block_mapping(source_depth, target_depth)`):

```python
return [index * target_depth // source_depth for index in range(source_depth)]
```

Spread the source blocks evenly through the target. For 2 -> 12 that is
`[0 * 12 // 2, 1 * 12 // 2] = [0, 6]`: source block 0 becomes target block 0, source
block 1 becomes target block 6, and **the other 10 target blocks are new**.

**Loop-untying** (`parse_block_schedule`): a shallow model was trained by running its
blocks *repeatedly* (a "recurrent" or looped model), e.g. blocks `0,1,0,1,...`. This
function parses such a schedule from a string like `"0,1,0,1"` (or a list) and checks
it is valid: non-empty, every index below the source depth, and **every source block
used at least once** (lines 479-482, the same checks `Transformer.__init__` does).
"Untying" means giving each loop iteration **its own copy** of the weights, which can
then drift apart as learning continues.

`weights.bin` uses **loop-untying**. The depth-growth code path exists in the same
function but is not what this archive uses.

## `transformer_geometry` (lines 486-500)

Just builds a dict of the ten shape settings (`vocab_size`, `n_layer`, `d_model`,
`n_head`, `d_inner`, `ctx_len`, `tie_embeddings`, `position_type`, `qk_norm`,
`norm_style`). It is a **description of the model's shape** that can be compared for
equality with the one stored in the archive.

## `load_depth_prior` (lines 515-623)

The function that puts it all together. It reads in four phases.

**1. Check that this archive is the right one for this run (lines 517-568).**
It raises an error unless **all** of these hold:

| Check | Why it matters |
|---|---|
| a vocabulary file was given and exists | the weights are tied to a vocabulary |
| archive magic is `QPI1` and it has `initialization` metadata | right kind of archive |
| `kind` is `depth-growth` or `loop-untying` | known recipes only |
| archive's **target geometry** equals this run's expected geometry | the model being built must be exactly the one the archive was made for |
| source geometry equals expected geometry on every key **except** `n_layer` | only the *depth* may differ |
| depth-growth: stored `block_mapping` equals the canonical mapping; loop-untying: schedule is valid and **its length equals the target depth** (12) | no custom mappings |
| archive's `initialization_seed` equals `--seed` | the run's seed matches |
| archive's `vocab_sha256` equals the hash of the vocabulary file | **right weights for the right vocabulary** |

Because the geometry check includes `tie_embeddings`, and the archive says `false`,
this also settles the earlier question: **with this archive the real run is untied.**

**2. Build the lookup (lines 541-563).** For loop-untying with schedule
`[0,1,0,1,...]`, `target_sources = {0: 0, 1: 1, 2: 0, 3: 1, ...}`: target block `i`
takes its weights from source block `schedule[i]`.

**3. Verify the archive has exactly the right tensors (lines 570-586).**
`root_names` are tensors outside any block (`tok_emb`, `head`, norms). `block_suffixes`
are the per-block names (`attn.qkv.weight`, `ffn.gate.weight`, ...), taken from block 0
of the *target* model. The archive must contain exactly the root names plus those
suffixes for **source blocks 0..1**, with nothing missing and nothing extra.

**4. Copy the weights in (lines 588-611), under `torch.no_grad()`.** Root tensors are
copied directly. Then for every target block, each tensor is copied from its source
block (a shape check on every one). `.to(dtype)` converts from float32 to the model's
working type.

For **depth-growth only** (lines 603-611): target blocks that have no source block get
their `attn.proj.weight` and `ffn.down.weight` set to **zero**. These are exactly the
two matrices that write into the residual stream (the same two Part A's initialization
shrinks). Zeroing them makes a new block add nothing, so it starts as an identity
function and the model initially behaves like the shallow one. This path is not used by
`weights.bin`.

**5. Return an identity record (lines 613-623):** archive size and SHA-256, vocabulary
SHA-256, kind, source depth, mapping, schedule, and `initial_state_sha256`: the hash of
the whole model right after loading. `main()` logs these values (lines 1045-1054) and
stores the two archive digests in every checkpoint, where `validate_checkpoint_prior`
(below) checks them on resume.

## `validate_checkpoint_prior` (lines 626-638)

When resuming a stopped job from a checkpoint, the run must start from **the exact same
initial prior** the original did. This function compares the checkpoint's recorded
archive digest and vocabulary digest with the current invocation's and raises an error
if either differs. A 7.5-day decode that resumed from the wrong starting weights would
silently produce garbage; this makes that impossible.

---

## Part B in one picture

```
weights.bin (4.3 MB)
  [QPI1][len][JSON: geometry, schedule, seeds, hashes][26 tensors: signs + row scales]
        |
        v  read_tensor_archive      -> float32 tensors  (2 blocks + tok_emb/head/norms)
        v  load_depth_prior         -> checks: geometry, seed, vocab hash, tensor set
        v                              copies source block schedule[i] into target block i
   Transformer, 12 blocks, 162.6M params  (blocks 0,2,4,... identical; 1,3,5,... identical)
        |
        v  online learning (Part D) makes all 12 diverge
```

**Why this design:** the rules of the contest count the size of what you ship. A
12-block full-precision model would be hundreds of megabytes. Shipping a 2-block,
1-bit version (4.3 MB) and expanding it at load time keeps the shipped size small. The
price is a rougher starting point, which the online learning has to make up.
(The accounting half is confirmed in Part D1: `main()` zips `weights.bin` together with
the code and charges it to the benchmark total. Why it was designed this way is my
reading; the file itself only states the mechanism.)

---

---

# Part C: helpers and command-line arguments (lines 641-855)

Small functions, then one very long `argparse` block. The helpers are short but each
one connects to something earlier; the argument list is the most compact **catalogue of
every knob** in the system, and it settles several questions Part A left open.

## `load_tokens` (lines 644-646)

```python
def load_tokens(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=">u2").astype(np.int64)
```

The tokenized input (what `preprocess.c` produces from the text) is a plain file of
**2-byte numbers**. `">u2"` means big-endian ("`>`") unsigned 16-bit integer. 16 bits
holds up to 65,535, comfortably more than the 4,097-token vocabulary. `astype(np.int64)`
widens to the type PyTorch's embedding lookup wants. So the input to the model is just
a long array of token numbers, as in the rungs' arrays of byte values.

## `lr_at`: warmup, then cosine decay (lines 649-656)

```python
def lr_at(step, peak_lr, min_lr, warmup, total_steps):
    if step < warmup:
        return peak_lr * (step + 1) / warmup               # ramp up linearly
    if step >= total_steps:
        return min_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```

**Why:** the learning rate is the size of each nudge (the lesson from rung 3: too big
overshoots, too small is slow). Starting at full size can wreck a fresh model, so the
rate **ramps up** during `warmup` steps, then **glides down** along half a cosine wave
to `min_lr`. The cosine is a smooth curve from 1 to 0: `0.5 * (1 + cos(pi * progress))`
is 1 at the start (`progress = 0`), 0.5 in the middle, 0 at the end.

With the real settings (`lr = 4e-5`, `warmup = 200`; assuming `min_lr = lr x
min_lr_ratio = 1e-5` (confirmed at line 1209); and an **illustrative**
`total_steps = 10,000`, since the real value is `file_len // batch_size`, line 1208),
I ran the function:

| step | lr |
|---|---|
| 0 | 2.0e-07 (just starting) |
| 99 | 2.0e-05 (halfway up) |
| 199-200 | 4.0e-05 (peak) |
| 5,000 | 2.5e-05 |
| 7,500 | 1.5e-05 |
| 9,999+ | 1.0e-05 (floor) |

## `parse_piecewise_lr` and `piecewise_lr_at` (lines 659-686)

A second, more flexible schedule, used for the **retraining** passes (see below).
The syntax is `lr0[,step1,lr1,step2,lr2,...]`: a starting rate, then pairs of
"by this step, be at this rate", with **straight lines between the points**. There must
be an **odd** number of comma-separated parts (line 662) and the steps must strictly
increase (line 670); I confirmed both errors by feeding it bad input.

V2's retrain schedule is `1.6e-4,13000,1.6e-4,93000,7.9e-5,163000,4.0e-5,1911300,1.3e-5`,
which parses to five points. Ran against it:

| global step | lr | |
|---|---|---|
| 0 - 13,000 | 1.60e-04 | flat |
| 53,000 | 1.20e-04 | sliding down |
| 93,000 | 7.90e-05 | |
| 163,000 | 4.00e-05 | |
| 1,000,000 | 2.7e-05 | |
| 1,911,300 and beyond | 1.30e-05 | stays at the last value |

So the retrain rate **starts 4x higher than the online peak** (1.6e-4 vs 4e-5) and
decays by about 12x. It is indexed by a **global step counter** (the argument's help
text says so), so each new retrain call continues the schedule rather than restarting.

## `make_freq_table` (lines 689-693)

```python
def make_freq_table(logits):
    """Cumulative integer frequency table per row, as the arithmetic coder wants."""
    prob = F.softmax(logits.detach(), dim=-1)
    freq = torch.clamp(torch.round(prob * FREQ_TOTAL).to(torch.int32), min=1)
    return torch.cumsum(freq, dim=-1).cpu().numpy()
```

**This is the bridge from the model to the arithmetic coder.** Four steps:

1. `softmax`: logits to probabilities (they sum to 1).
2. `prob * FREQ_TOTAL` then `round`: scale to integers summing to about 10,000,000.
3. `clamp(min=1)`: **every token gets at least a width of 1**, so even an
   "impossible" token can still be coded (it just costs a lot). This is the same
   "floor of 1" reason rung 1 started every count at 1.
4. `cumsum` along the last axis: turn "width per token" into "where each token's slice
   *ends*", the form the coder wants. Then `.cpu().numpy()`: it leaves the GPU.

**Rungs 2-4 each have their own `to_freqs` with exactly this recipe** (NumPy rather than
PyTorch). `logits.detach()` cuts the gradient connection, since this is for coding, not
learning. The "per row" in the docstring means it handles many positions at once.

## `NullSummaryWriter` (lines 696-709)

Four methods that do nothing: `add_scalar`, `add_hparams`, `flush`, `close`. Its
docstring: "Rank-local no-op writer; only rank zero owns durable logs." With two GPUs
(two "ranks"), only rank 0 writes TensorBoard logs; presumably the other gets this
stand-in with the same interface, so the rest of the code can call `writer.add_scalar(...)` without
checking which rank it is.

## `parse_args` (lines 712-853)

### The key trick: the last three lines

```python
p.set_defaults(**V2_CONFIG,
               vocab_file=os.path.join(os.path.dirname(__file__), "enwik9.voc"),
               initial_prior=os.path.join(os.path.dirname(__file__), "weights.bin"))
```

Every `add_argument` has a default (`--d_model` is 512, `--dropout` is 0.0,
`--ctx_len` is 512, `--lr` is 5e-4, ...), **and then `set_defaults` overwrites them with
`V2_CONFIG`** (`d_model` 1024, `dropout` 0.25, `ctx_len` 192, `lr` 4e-5, ...) and points
the vocabulary and prior at `enwik9.voc` and `weights.bin` next to the script. So:

- Running with just `--input` gets the **v2 bitstream's exact settings**.
- Any flag you pass on the command line still overrides these (`set_defaults` only
  changes defaults, not what the command line says).
- The defaults you see in `add_argument(...)` lines are **historical or generic and are
  not what v2 uses.** When reading, check `V2_CONFIG` for the real value.
- `--untie_embeddings` is a `store_true` flag, and `V2_CONFIG` sets it `True`. There is
  **no flag that turns it back off**, so from the command line the embeddings can't
  be re-tied.

### What each group of arguments does

**I/O**

| Argument | Meaning |
|---|---|
| `--input` (required) | Tokenized input (or the compressed file with `--decompress`) |
| `--output` | Output file; defaults under `--work_dir` |
| `--decompress` | Decode instead of encode |
| `--no_coding` | "skip the arithmetic coder; just train and report bits/token": measures how well the model predicts **without** producing a file |
| `--max_tokens` | Limit input tokens, for quick tests (0 = all) |
| `--raw_bytes` | Original file size, enabling **bits/byte** logging (the enwik benchmark's metric) |
| `--vocab_bytes` | "serialized vocabulary size for separate benchmark-total metrics" |
| `--vocab_file` | Vocabulary whose SHA-256 a `QPI1` prior pins (Part B) |
| `--initial_prior` | "**transmitted** QPI1 shallow-model initialization archive" (`weights.bin`). **Confirmed in Part D1:** it is zipped together with the code and charged to the benchmark total |
| `--external_decompressor_bytes` | For non-Transformer backends only: bytes of external runtime dependencies; "-1 = unaccounted" |

**Model** (real values are in `V2_CONFIG`; see Part A)
`--n_layer --d_model --n_head --d_inner --ctx_len --dropout --untie_embeddings
--position_type --no_qk_norm --norm_style`, plus `--vocab_size` (default 16,388; V2 uses
4,097).

Two comments in the source are worth reading in full:

- Lines 747-751 explain `d_inner`: `ceil(8/3 * d_model / 256) * 256`, because SwiGLU has
  three matrices where a ReLU feed-forward has two, so `4 x 2/3 = 8/3` keeps the
  parameter count equal. (Confirms what Part A said.)
- Lines 753-757 explain a longer context: "the Python arithmetic coder dominates wall
  time, so the ~4x extra compute per step is mostly absorbed". **The coder is the
  bottleneck, not the neural net**, so a larger context is nearly free. They also say
  Bellard's NNCP v2 effectively used ~1024.

**Other architectures:** `--model_backend` can be `transformer` (V2), `mamba2`,
`mamba3` or `kda`, with their own `--ssm_*`, `--mamba3_*` and `--kda_*` flags. These are
alternative model designs; **this walkthrough does not cover them.** The files they need
(`ssm_models.py`, `kda_models.py`, `tokenizers.py`) are not in this repository.

**Online learning and training** (the part that most differs from the rungs)

| Argument | Default | V2 | What it does |
|---|---|---|---|
| `--online_loss` | `window` | **`code`** | `window` "repeats all context targets (legacy)"; `code` "trains only the newly coded target, matching NNCP's online exposure" |
| `--project_only_targets` | off | on | "apply the vocabulary head only to supervised positions; requires `--online_loss code`". This is why `Transformer.forward` has `output_positions` (Part A) |
| `--lr` | 5e-4 | 4e-5 | Peak online learning rate |
| `--min_lr_ratio` | 0.1 | 0.25 | Floor as a fraction of peak |
| `--warmup` | 200 | 200 | Steps of linear ramp-up |
| `--clip` | 0.25 | 0.25 | Gradient clipping size |
| `--beta1` | 0.0 | 0.0 | Adam momentum: **off** |
| `--beta2` | 0.9999 | 0.9999 | Adam's long-memory average of squared gradients |
| `--adam_eps` | 1e-8 | 1e-9 | Adam's small constant for stability |
| `--batch_size` | 64 | 64 | |
| `--block_len` | 500,000 | 500,000 | "tokens per block; streams are batched within a block" |

The comment at lines 789-790 explains why the default peak LR was raised from 3e-4 to
5e-4: QK-norm and peri-norm make the model tolerate a higher rate.

**Retraining** ("extra passes over the already-coded prefix (valid because both the
compressor and decompressor hold it)", comment at lines 805-806). **This is the real
version of the "replay" in rungs 3 and 4.** Both sides have already seen the earlier
tokens, so both can re-study them without any extra information.

| Argument | V2 | Meaning |
|---|---|---|
| `--retrain_period` | 1 | Retrain every N symbols, checked only at **block boundaries**, so the real period rounds up to a multiple of `block_len` (500,000). With 1 it happens at **every block boundary** |
| `--retrain_len` | 35,000,000 | Train on at most this many **trailing** symbols of the coded prefix |
| `--retrain_epochs` | 1 | Passes over that window |
| `--retrain_early_epochs`, `--retrain_early_until` | unset (0) | Optionally use a different epoch count early on |
| `--retrain_lr` | the piecewise schedule above | |
| `--retrain_update_len` | 64 | New targets per optimizer update (0 = `ctx_len`) |
| `--retrain_batch_size` | 32 | Independent retraining streams (0 = `batch_size`) |

**Runtime and infrastructure**

| Argument | Meaning |
|---|---|
| `--cpu`, `--dtype` (bf16 / fp32) | Force CPU; working number format (V2 uses bf16) |
| `--distributed` | "run a deterministic single-node DDP codec under torchrun; **world size is part of the codec and must match on decode/resume**" |
| `--work_dir`, `--run_name`, `--run_dir`, `--log_interval` | Where logs and outputs go, and how often progress is reported |
| `--checkpoint_dir` | "persistent directory for latest.pt; **an existing checkpoint is resumed automatically**" |
| `--checkpoint_every_blocks` | How often to save (V2: every 4 blocks) |
| `--stop_after_blocks` | Hidden from `--help`; presumably for testing |
| `--seed` | 1111 |

The `--distributed` line is the most important one for anyone hoping to reproduce this:
the **number of GPUs is itself part of the algorithm**: the help text says "world size
is part of the codec and must match on decode/resume". Part D1 confirms one concrete
reason: each rank reseeds its random generators with `seed + rank`, so the dropout
masks used in retraining depend on the number of ranks. A second, probable reason (my
inference) is that distributed training splits the batch across processes and
combines their gradients, which changes the order of floating-point additions, so a
different world size would give slightly different numbers than the encoder had. That is
why `nn.py`'s docstring points to the README's "two-B200 hardware" requirement, not just
"some GPUs".

`--run_dir` is described as "stable across Slurm requeues", and
`--checkpoint_every_blocks` mentions "a Slurm stop request": this was run on a shared
cluster whose scheduler can stop and restart jobs, which is what `checkpoint_utils` and
`REQUEUE_EXIT_CODE` (Part A imports) are for.

## What differs between rung 4 and the real training setup

This table is the main payoff of Part C: **the exercise taught the idea but not these
settings.**

| | Rung 4 | Real (V2) |
|---|---|---|
| What each update trains on | whole window, every position (the legacy `window`-style) | **only the newly coded target** (`online_loss code`) |
| Adam momentum (`beta1`) | 0.9 (default) | **0.0** |
| Adam `beta2` | 0.999 (default) | 0.9999 |
| Gradient clip | 1.0 | 0.25 |
| Online LR | 3e-3, constant | 4e-5 peak, warmup then cosine to 1e-5 |
| Replay | 2 random earlier windows per byte | **every 500,000 tokens: re-train on up to 35M trailing tokens** at a decaying LR |
| Dropout | none | 0.25, but **only during retraining** (coding runs in eval mode) |
| Number format | float32 | bf16 |
| GPUs | 1 CPU thread | 2 GPUs, and the count is part of the codec |

**Correction to something I said earlier in this project:** when writing rung 4 I said
`nn.py` trains on the whole window like my toy does. That was wrong for V2: V2 sets
`online_loss = 'code'`, which trains **only on the newly coded target**. The whole-window
style my exercise uses is the "legacy" mode here. My toy's choice was a reasonable one
for a tiny model, but it is not what the real codec does.

---

---

# Part D1: `main()`, setup and the core coding loop (lines 859-1254, 1429-1574)

`main()` is 880 lines and nearly half the file. It does one job, the same one every
rung did: **predict, code, learn, repeat.** Most of its length is infrastructure
(two GPUs, resumable checkpoints, logging). D1 covers the setup and then the loop that
*is* the codec. D2 covers `retrain`, checkpoints and the finish.

**If you read only one thing in this file, read "The core loop" below** and compare it
with `encode()` in `04_tiny_transformer.py`. They are the same shape.

## Setup, in order

### 1. Validate the arguments (lines 860-880)

A run of `if ... raise ValueError(...)` checks. The ones that tell you something:
`--project_only_targets` requires `--online_loss code` (the head can only be sliced when
training targets one position); the early-replay pair must both be set or both be 0;
`--initial_prior` is Transformer-only and the file must exist. These fail fast, before
any GPU time is spent.

### 2. Devices and ranks (lines 882-914)

- `WORLD_SIZE` is set by `torchrun`. More than one process **requires** `--distributed`.
- `--distributed` requires CUDA and at least two ranks. It initializes **NCCL** (the
  GPU-to-GPU communication library), reads `rank` and `LOCAL_RANK`, and pins each
  process to its own GPU.
- Otherwise: one process, on CUDA if present, else CPU (or `--cpu`).
- `is_main = rank == 0`. Many things (logging, the arithmetic coder, file writes) happen
  **only on rank 0**.
- `amp_enabled`: bf16 "autocast" is on only on CUDA with `--dtype bf16`. Inside it,
  matrix maths run in bf16, with the precision-sensitive bits (like `RMSNorm`'s
  float32 upcast from Part A) protected.

### 3. Determinism (lines 910-914)

```python
torch.use_deterministic_algorithms(True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(args.seed)
```

`use_deterministic_algorithms(True)` makes PyTorch **raise an error** rather than use any
operation that isn't bit-for-bit repeatable. With `CUBLAS_WORKSPACE_CONFIG` (Part A,
line 9), this is the code's answer to "the decoder must make the same floats".
Everything random is seeded from `seed = 1111`.

### 4. Run directory and logging (lines 916-946)

A run directory is named from a timestamp plus the key settings. Under `--distributed`,
rank 0 picks the name and **broadcasts it** so every rank agrees, then all wait at a
`barrier`. `log()` prints and appends to `log.txt`, **on rank 0 only**. The TensorBoard
writer is a real `SummaryWriter` on rank 0 and a `NullSummaryWriter` (Part C) elsewhere,
which confirms what Part C guessed.

### 5. Build the model, then load the prior (lines 948-994)

```python
model = Transformer(
    args.vocab_size, args.n_layer, args.d_model, args.n_head,
    args.d_inner, args.ctx_len, args.dropout,
    tie_emb=not args.untie_embeddings,
    position_type=args.position_type,
    qk_norm=not args.no_qk_norm,
    norm_style=args.norm_style)
```

This is Part A's `Transformer`, built from the (V2) arguments. Facts this settles:

- **Embeddings are untied**: `tie_emb=not args.untie_embeddings` and V2 sets
  `untie_embeddings=True`. (Confirmed, closing the question from Part A.)
- **No `block_schedule` is passed**, so the default `range(n_layer)` applies: each of the
  12 blocks runs once.
- The other branches (lines 957-981) build Mamba or KDA models from files that aren't in
  this repository.

Then the prior (lines 982-990):

```python
expected_geometry = transformer_geometry(..., not args.untie_embeddings, ...)
prior_info = load_depth_prior(model, args.initial_prior, args.vocab_file,
                              expected_geometry, args.seed)
```

`load_depth_prior` (Part B) overwrites the random initialization with the expanded
`weights.bin`, on the CPU, **before** the model moves to the GPU (line 991). The
geometry is built from the *current arguments*, so any mismatch with the archive (say, a
different `d_model`) raises an error here.

```python
model_core = model.to(device)
# Gradients work in eval mode. Keeping online coding in eval mode lets a
# matched profile enable dropout only during prefix retraining, as NNCP does.
model_core.eval()
```

**This answers the dropout question from Part A.** Coding runs in **eval mode**, so
dropout is **off** while coding; `retrain` (D2) switches to `model.train()`, turns
dropout on for its passes, and switches back. PyTorch computes gradients fine in eval
mode, which is why online learning still works. The comment says this mirrors NNCP.

### 6. Wrap in DistributedDataParallel (lines 995-1006)

```python
model = DistributedDataParallel(model_core, device_ids=[local_rank],
                                output_device=local_rank, broadcast_buffers=False)
np.random.seed(args.seed + rank)
torch.manual_seed(args.seed + rank)
torch.cuda.manual_seed(args.seed + rank)
```

DDP is PyTorch's "several GPUs, one model" wrapper: each GPU processes part of the batch,
and the gradients are averaged between them so all copies stay identical. The key detail
is the **reseeding**. The comment explains: initialization has to be identical on every
rank *before* DDP synchronizes, then each rank gets its own stream `seed + rank`, "so
replay dropout [is] independent while remaining exactly reproducible on encode, decode,
and resume." That is a concrete reason **the number of ranks is part of the codec**: the
dropout masks used in retraining are a function of the rank layout.

### 7. Two optimizers (lines 1007-1013)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                             betas=(args.beta1, args.beta2), eps=args.adam_eps)
# separate optimizer so retrain's Adam moments don't perturb the online-coding ones
retrain_optimizer = torch.optim.Adam(model.parameters(),
                                     lr=retrain_lr_schedule[0][1],
                                     betas=(args.beta1, args.beta2), eps=args.adam_eps)
```

Adam (not AdamW) with `betas = (0.0, 0.9999)` and `eps = 1e-9`, as Part C's table
said. There are **two** optimizers over the same weights: one for the online coding steps
and one for retraining. Adam keeps running statistics about past gradients; keeping them
separate stops retraining's statistics from contaminating the online ones. (The comment
at line 1009 says exactly this.)

### 8. Logging the model size (lines 1015-1026)

`n_params` counts `model_core.parameters()`, and `emb_params` counts `tok_emb` plus
`head` only if they are not the same tensor. By the arithmetic in Part A the log should
say 162,585,600 parameters (154,194,944 non-embedding). That is arithmetic, not a log I
have seen: the real model is never run in this fork.

### 9. Benchmark accounting (lines 1027-1065)

This **answers the open question from Parts A-C: does `weights.bin` count?** Yes.

```python
base_decompressor_bytes = decompressor_zip_size(base_dfiles)             # code only
local_decompressor_bytes = decompressor_zip_size(dfiles, extra_files=(args.initial_prior,))
prior_transmission_bytes = local_decompressor_bytes - base_decompressor_bytes
```

The zip of the decompressor's code (`nn.py`, `arithmetic_coder.py`, `checkpoint_utils.py`,
`preprocess.c`; from Part A's `decompressor_files`) is measured once without and once
**with `weights.bin` added**, at DEFLATE level 9. The difference is the prior's cost:
`prior_transmission_bytes`. My expectation (not measured) is that 1-bit sign data is close to random, so DEFLATE
shrinks it little and the prior costs roughly its raw 4.3 MB.

The final score (D2) is: **compressed payload + vocabulary size + zipped decompressor
(with prior)**. For Transformers `dependency_accounted` is always true; Mamba and KDA
runs print a warning that their runtime isn't charged.

### 10. The checkpoint signature (lines 1067-1100)

`config_signature(args, excluded=...)` (in `checkpoint_utils.py`) turns every setting
**that affects predictions** into one sorted JSON string. It is a plain string, not a
hash, and resuming requires the saved string to match exactly. The `excluded` set lists what does *not*: input and output paths, log and
run directories, checkpoint settings, and accounting. The comments explain two:
paths may differ between encoder and decoder machines ("QPI1 contents are pinned
separately by the archive and vocabulary SHA-256 values"), and a few newer options are
stored separately so older checkpoints stay resumable.

### 11. Resuming (lines 1101-1133)

If `--checkpoint_dir` has a saved checkpoint, `main()` refuses to continue unless **every**
one of these matches: checkpoint version, config signature, target-projection mode,
early-replay policy, input file size, **world size**, and the prior's two digests
(`validate_checkpoint_prior`, Part B). Only then does it load the model weights **and
both optimizers' states**. This is Part A's determinism theme applied to resuming a
multi-day job.

### 12. Input and output files, and the arithmetic coder (lines 1135-1200)

**Encoding:**

```python
file_data = load_tokens(args.input).copy()            # the whole token array
out_file.write(original_len.to_bytes(4, byteorder="big"))   # 4-byte token count
bit_out = BitOutputStream(out_file)
encoder = ArithmeticEncoder(32, bit_out)              # 32-bit coder, as in the rungs
```

The same arrangement as every rung: a **4-byte big-endian count** first (so the decoder
knows when to stop), then the arithmetic coder. Only rank 0 owns the file and the coder.

**Decoding:** read the 4-byte count, set up an `ArithmeticDecoder`, and create
`file_data = np.zeros(original_len)`, to be **filled in as tokens are decoded**.
Under `--distributed`, rank 0 broadcasts the length to the other ranks. (The
`decoded_prefix` code is only for resuming, which D2 covers.)

### 13. Padding, blocks and streams (lines 1202-1213)

```python
file_len = ((original_len + batch - 1) // batch) * batch     # round up to a multiple of 64
block_len = max(batch, (args.block_len // batch) * batch)    # 499,968 for V2
total_steps = file_len // batch
min_lr = args.lr * args.min_lr_ratio                         # 1e-5 for V2
```

- The token array is **padded with zeros** up to a multiple of `batch_size` (64). Those
  few extra tokens are coded too; the decoder writes only `original_len` tokens back out.
- A **block** is 500,000 tokens rounded down to a multiple of 64: **499,968**.
- `total_steps = file_len // 64` is the count of online learning steps; it's what
  `lr_at` (Part C) uses to place the end of the cosine, and `min_lr = lr * min_lr_ratio`
  confirms the assumption made in Part C.

**What the "64 streams" means is the one idea you need for the loop.** Each block is
cut into **64 consecutive pieces**, `block_stride = 499,968 / 64 = 7,812` tokens each.
The model reads those 64 pieces **in parallel**, one token at a time:

```
block (499,968 tokens)
  |-- stream 0:  tokens [        0 ..  7,811]
  |-- stream 1:  tokens [    7,812 .. 15,623]
  |-- stream 2:  ...
  |-- stream 63: ...

step t:  predict token t of ALL 64 streams in one batch,
         code them one after another, then learn from all 64 at once
```

So the model is effectively reading **64 different passages of text at the same time**.
Each stream's context is **only its own earlier tokens** (never another stream's), and
it starts fresh at the top of each block.

*Why do it this way? (my reading; the code only says "streams are batched within a
block")* A GPU is far more efficient doing 64 predictions in one go than 64 predictions
one at a time. The price is that a stream can't see the text just before its start,
which costs a little compression.

### 14. Splitting the batch across GPUs (lines 1231-1250)

`batch` and `retrain_batch` must be divisible by `world_size`. With 2 ranks, each GPU
takes **32 of the 64 streams** (`local_batch`, `local_batch_start`, `local_batch_end`).

---

## The core loop (lines 1429-1574)

This is the part to compare with the rungs. Here is the skeleton, then each piece.

```python
while block_start < file_len:                    # one pass per 499,968-token block
    if retrain is due: retrain(...)              # D2
    block_stride = cur_block_len // batch        # 7,812
    for stream_pos in range(block_stride):       # one step: one token from each of 64 streams
        ...build the input...
        ...forward pass...
        ...arithmetic-code 64 symbols...
        ...learn from them...
    ...write the block (decode), maybe checkpoint (D2)...
```

### Step 1: build the input (lines 1439-1446)

```python
window = min(stream_pos, ctx_len)
if window == 0:
    inp = np.zeros((local_batch, 1), dtype=np.int64)   # dummy BOS for the first token
else:
    offsets = stream_pos - window + np.arange(window)
    inp = file_data[stream_base[:, None] + offsets[None, :]]
```

For each stream: the **previous `window` tokens**, at most `ctx_len = 192`. At the very
first position there is nothing before it, so the input is a single dummy token `0`, the
same start-of-text trick my rung 4 used (`BOS = 0`). The input is
`[local_batch, window]`: the model sees the context **before** the token it is about to
code, never the token itself.

### Step 2: set the learning rate (lines 1448-1450)

`lr_at(train_step, args.lr, min_lr, args.warmup, total_steps)` from Part C, written
into the optimizer **before every step**. (In the code, one training step is one 64-token
step through the loop.)

### Step 3: forward pass (lines 1452-1459)

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
    if args.project_only_targets:
        logits = model(inp_t, output_positions=slice(-1, None))
    else:
        logits = model(inp_t)
code_logits = logits[:, -1, :].float()
```

`slice(-1, None)` is the `output_positions` option from Part A: **project only the last
position** through the 4,097-way head. The newest position's logits predict the next
token; they are converted to float32 (`.float()`) for the coding step.

### Step 4: arithmetic-code the 64 symbols (lines 1461-1481)

```python
if is_main:
    freq = make_freq_table(global_code_logits)       # Part C
    for i in range(batch):                           # 64 symbols, one at a time
        if decode:
            file_data[global_code_pos[i]] = decoder.read(freq[i])
        else:
            encoder.write(freq[i], int(file_data[global_code_pos[i]]))
```

- Under `--distributed`, each rank's logits are first **gathered onto rank 0**
  (`dist.gather`), because there is **one** arithmetic coder and it lives on rank 0.
- `make_freq_table` turns the `[64, 4097]` logits into 64 cumulative tables.
- Then a plain Python `for` loop codes (or decodes) the 64 symbols **in order**.
  Compare with the rungs: `encoder.write(freqs, byte)` and `decoder.read(freqs)`. It is
  the same call, 64 times per step.
- This single-threaded, pure-Python loop is the bottleneck the `--ctx_len` comment in Part
  C pointed at: the GPUs wait while one process does arithmetic.

**Why interleaving 64 streams is safe for decoding:** at step `t`, every stream's
prediction depends only on *earlier* steps, so the decoder can compute all 64
predictions before decoding any of the 64 symbols, and decode them in the same order
the encoder wrote them.

### Step 5: the targets (lines 1483-1495)

When encoding, the targets are simply `file_data[code_pos]`. When decoding they are the
symbols just decoded. Under `--distributed`, rank 0 **broadcasts** the freshly decoded
tokens to the other rank (lines 1483-1493), since only rank 0 ran the decoder but both
ranks need the targets to learn from.

### Step 6: count the bits (lines 1497-1503)

```python
code_nats = F.cross_entropy(code_logits, code_targets, reduction="sum").detach()
...
n_bits += code_nats.item() / LOG2
n_symbols += batch
```

The cost of a token is `-ln P(token)` ("nats"); dividing by `LOG2` gives bits. This is
the same "size ~ sum of -log2(p)" idea from your session 1 notes. It is the **model's
own estimate**; the true compressed size is measured at the end (D2) from the actual
file, and the two should agree to within a few bytes.

### Step 7: learn (lines 1505-1523)

```python
if args.online_loss == "code":
    loss = F.cross_entropy(code_logits, code_targets)     # mean over the 64 new targets
else:
    ...legacy: every position in the window...
optimizer.zero_grad(set_to_none=True)
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)   # 0.25
optimizer.step()
train_step += 1
```

With V2's `online_loss = 'code'`, the loss is **only on the 64 tokens just coded** (the
comment: "trains only the new symbol, like NNCP v2, so each online target has
approximately equal exposure"). Then: backpropagate, clip the gradient to norm 0.25, one
Adam step. **One optimizer step per 64 coded tokens.** The decoder repeats exactly this
after decoding the same 64 tokens, so both sides' weights stay identical.

### Step 8: log, and write the block (lines 1527-1574)

Every `log_interval` symbols (100,000) it prints a row and writes TensorBoard scalars.
The columns (set up at lines 1426-1428) are step, symbols so far, compressed bytes so
far (`n_bits / 8`), bits per token and bits per byte (local to the interval, and
cumulative), throughput in kilo-symbols per second, and the learning rate.
`bpb_cum` is the headline number, **comparable with the "bits/byte" in your rung table**
(it is `bits per token / bytes per token`, using the whole file's average token length;
the bits/byte columns print `nan` unless `--raw_bytes` is given).

When decoding, at the end of each block, rank 0 writes the block's tokens to the output
file as big-endian 16-bit numbers (the same format `load_tokens` reads, Part C).

---

## Rung 4's loop next to the real one

| | `04_tiny_transformer.py` `encode()` | `nn.py` `main()` loop |
|---|---|---|
| Unit of work | **one** byte | **64** tokens (one from each of 64 streams) |
| Context | last 32 bytes, with a start marker | last 192 tokens of the *same stream*, with a start marker |
| Predict | `model.predict(history)` | `model(inp)`, project the last position, `.float()` |
| Frequency table | `to_freqs(p)` (NumPy) | `make_freq_table(logits)` (PyTorch), one row per stream |
| Code | `encoder.write(freqs, byte)` | the same call, 64 times, in a Python loop on rank 0 |
| Cost accounting | `-log2` of the integer slice | cross-entropy in nats, divided by `LOG2` |
| Learn | all window positions, 1 Adam step per byte, then 2 replays | **new targets only**, 1 Adam step per 64 tokens, **plus** big retrain passes at block ends (D2) |
| Mode | default (train) | **eval** while coding; train only in retraining |
| Determinism | same container, 1 thread | `use_deterministic_algorithms`, fixed seeds per rank, same GPU layout |

---

# Part D2: `retrain`, checkpoints and the finish (lines 1255-1427, 1577-1737)

D1 covered the loop that codes tokens. D2 covers the three things around it: the
big **retraining passes** that happen between blocks, the **checkpoint and restart**
machinery that lets a multi-day job survive being stopped, and what happens at the end.
The helpers `main()` uses from `checkpoint_utils.py` (145 lines) are covered here too.

## `retrain(file_data, file_pos)` (lines 1255-1334)

**Why it exists (the docstring):** "Extra training passes over the already-coded prefix
`file_data[:file_pos]`. Reads only data both sides already hold at this point, so the
weights it produces are reproducible during decompression and the codec stays valid.
Runs on a separate optimizer, leaving the online-coding Adam state untouched."

This is the real version of the "replay" in your rungs 3 and 4. Between blocks, both
sides re-study the text they have already coded, **for free**: the decoder has it too.
(How much of the final compression this contributes is not something the code states or
that I have measured.)

It is called at the top of the main loop (lines 1430-1432), whenever
`block_start - last_retrain_pos >= retrain_period`. With V2's `retrain_period = 1`, that
means **at every block boundary except the very first** (`block_start = 0`).

### Choosing what to study (lines 1262-1268)

```python
lo = max(0, file_pos - args.retrain_len)                  # at most 35,000,000 tokens back
stride = (file_pos - lo) // retrain_batch                 # retrain_batch = 32
if stride < 2:
    return                                                # too little text yet
global_stream_base = lo + retrain_arange_batch * stride
stream_base = global_stream_base[local_retrain_start:local_retrain_end]
n_targets = stride - 1
```

The window `[lo, file_pos)` (the most recent up-to-35M coded tokens) is cut into **32
equal streams**, each `stride` tokens long. Each stream is a contiguous passage, and
under `--distributed` each GPU takes half the streams (16 each). `n_targets = stride - 1`
because the last token of a stream has no "next token" inside it.

### One epoch (lines 1280-1323)

```python
model.train()                                             # dropout ON
for epoch in range(retrain_epochs):
    for start in range(0, n_targets, update_len):         # update_len = 64
        length = min(update_len, n_targets - start)
        context_start = max(0, start - (ctx_len - length)) if args.retrain_update_len else start
        prefix_len = start - context_start
        cols = context_start + np.arange(prefix_len + length)
        inp = file_data[stream_base[:, None] + cols[None, :]]
        tgt = file_data[stream_base[:, None] + (start + np.arange(length))[None, :] + 1]
```

For each stream, **march along it 64 targets at a time**. Each update feeds the model the
64 tokens being trained on **plus up to 128 earlier tokens as context** (`ctx_len`
192 minus 64; `context_start` is that arithmetic). The comment (lines 1284-1285): "Each
target is optimized exactly once, while earlier known tokens provide up to ctx_len
positions of conditioning." So every token in the window is a training target **once per
epoch**, with a full context in front of it.

```python
target_logits = model(inp_t, output_positions=slice(prefix_len, prefix_len + length))
loss = F.cross_entropy(target_logits.float().reshape(-1, vocab_size), tgt_t.reshape(-1))
retrain_optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
current_retrain_lr = piecewise_lr_at(retrain_train_step, retrain_lr_schedule)
for group in retrain_optimizer.param_groups:
    group["lr"] = current_retrain_lr
retrain_optimizer.step()
retrain_train_step += 1
```

- `output_positions=slice(prefix_len, prefix_len + length)` is the same "only project
  the positions we need" idea as the coding loop, but for a **range** instead of the
  last position. The context positions are used for conditioning and never scored.
- The loss is cross-entropy on those 64 targets per stream; then the familiar
  backprop, clip (0.25), and one step of the **separate `retrain_optimizer`** at the
  rate from Part C's piecewise schedule (indexed by `retrain_train_step`, a counter that
  keeps growing across all retrains and is saved in checkpoints).
- **Dropout is on** (`model.train()`) for these passes only. That is the one place
  `dropout = 0.25` and the per-rank seeding from D1 come into play.

At the end, `model.eval()` restores coding mode (line 1325), the loss is averaged
across GPUs (`all_reduce`) and logged as an `R ...` line with `bits/tok` over the
window, plus TensorBoard scalars under `retrain/`.

### How big is one retrain?

Arithmetic from V2's settings, for a **full** 35M-token window and 32 streams:
`stride = 35,000,000 // 32 = 1,093,750`, so about 1.09 million targets per stream, at 64
per update gives **about 17,090 optimizer steps per retrain call**. The 500,000-token
block that triggers it needed only 7,812 online steps. Early in a file the window is
smaller (the first retrain, at 499,968 tokens, is about 245 steps), and it fills up after
roughly 70 blocks. This is arithmetic from the arguments, not a measured run.

**Why this is allowed:** every index it reads is below `file_pos`, so it only touches
tokens the decoder has already produced. Same weights on both sides.

`★` What this means for sample size: by the arithmetic above, the real compressor
performs **more optimizer steps in each retrain than in the block that triggers it**
(17,090 vs 7,812 for a full window). It is the same trick as my replay in rung 4,
scaled up, and it means the model is not one trained once on the data: it keeps
re-reading its own past.

## Resuming: restoring the counters (lines 1336-1361)

On a fresh run all counters start at zero (`n_symbols`, `n_bits`, `train_step`,
`block_start`, `last_retrain_pos`). On a resume they come from the checkpoint, and the
random number generators are restored (`restore_rng_state`), **per rank** when
distributed (lines 1346-1350). `start_time` is back-dated by the saved
`elapsed_seconds`, so the throughput numbers ignore time spent waiting in the queue.

## `checkpoint_utils.py`: the helpers `main()` uses

| Helper | What it does |
|---|---|
| `atomic_torch_save(value, path)` | Writes to a temporary file, **`fsync`s** it to disk, `os.replace`s it over the real name, then `fsync`s the directory. Retries up to 3 times. A reader can **never see a half-written `latest.pt`**, even if the machine dies mid-save |
| `capture_rng_state` / `restore_rng_state` | Save and restore Python's, NumPy's, PyTorch's and (if present) every CUDA device's random generator. Needed because dropout in retraining is random, and the decoder must draw the same numbers |
| `config_signature` | The sorted-JSON settings string used to reject an incompatible resume (D1) |
| `load_checkpoint` | `torch.load(path, map_location=device, weights_only=False)`. See the note below |
| `flush_and_sync`, `mark_done`, `stop_requested` | Flush a file to disk; write a `DONE` marker file; and report whether a file named **`STOP_REQUESTED`** exists in the checkpoint directory |

`REQUEUE_EXIT_CODE = 75`.

**A safety note** (since you asked to be told about these): `weights_only=False` means
loading a checkpoint can run arbitrary Python code embedded in the file (that is how
`pickle` works). It is fine for checkpoints your own run wrote, but **never resume from
a checkpoint you didn't create or don't trust.**

## `save_resume_checkpoint(reason)` (lines 1363-1423)

What gets saved, so a restarted job can continue **bit for bit**:

| Saved | Why it is needed |
|---|---|
| `model` weights | The learned state |
| `optimizer` and `retrain_optimizer` states | Adam's running statistics, for both |
| RNG states (per rank) | Same random draws for dropout after the restart |
| Arithmetic coder state: **`low`, `high`** (and `num_underflow` when encoding, `code` when decoding) | Everything the coder needs to continue. (This is the "state is just two integers" idea from your rung 1 notes: `arithmetic_coder.py` lines 123-137 and 178-188) |
| Bit stream state: file position and the partly-filled byte | Where the compressed output stands |
| For decoding: `output_position` | How much plain output has already been written |
| Counters: `n_symbols`, `n_bits`, `train_step`, `retrain_train_step`, `block_start`, `last_retrain_pos`, elapsed time | Resume exactly where it stopped |
| Identity: config string, prior archive and vocabulary SHA-256, world size, input size, `original_len`, `file_len` | Refuse to resume with anything that differs (D1, step 11) |

How it runs: every rank captures its RNG, rank 0 **gathers** them (`dist.gather_object`),
synchronizes CUDA, then the other ranks wait at a barrier. Rank 0 flushes the output file
to disk (`flush_and_sync`) **before** writing the checkpoint, so the output and the
checkpoint agree, then calls `atomic_torch_save`.

**When resuming an encode, the output file is trimmed back to the checkpoint:**
`BitOutputStream.load_state_dict` does `seek(position)` then `truncate(position)`
(`arithmetic_coder.py` lines 377-382), discarding anything written after the checkpoint.
The decode path does the equivalent for its output file (lines 1168-1173). So a crash
between checkpoints can't leave stray bytes behind.

## The tail of the loop: blocks, checkpoints, stopping (lines 1570-1601)

After each block:

```python
if decode and is_main:
    out_file.write(file_data[block_start:end].astype(">u2").tobytes())   # write the block
block_start += cur_block_len
completed_blocks += 1

requested_stop = args.checkpoint_dir and stop_requested(args.checkpoint_dir)
test_stop = args.stop_after_blocks and completed_blocks >= args.stop_after_blocks
periodic = args.checkpoint_dir and completed_blocks % args.checkpoint_every_blocks == 0
if block_start < file_len and (periodic or requested_stop or test_stop):
    save_resume_checkpoint(reason)
if block_start < file_len and (requested_stop or test_stop):
    log("stopping cleanly for Slurm requeue")
    ... close files, tear down the process group ...
    return REQUEUE_EXIT_CODE
```

- **Checkpoints happen only at block boundaries** (about every 500,000 tokens), every
  `checkpoint_every_blocks` blocks (V2: 4, so about every 2 million tokens), and never
  after the last block (`block_start < file_len`).
- A **stop request is polled once per block**: if a file named `STOP_REQUESTED` appears
  in the checkpoint directory, the job saves a checkpoint and exits **cleanly** with
  code **75**. The wording in the source ("Slurm requeue") says that exit code is what a
  Slurm job script uses to resubmit the job. That script is not in this repository, so I
  am reading intent from the comments, not from running code.
- `--stop_after_blocks` (the hidden flag from Part C) triggers the same exit after N
  blocks: it is the **test hook** for exercising the stop-and-resume path without
  waiting for a real stop.
- `sys.exit(main())` (line 1738) passes that exit code to the shell.

The cost of this design: up to **one block plus its retrain** of work can be lost if a
job dies between checkpoints. The benefit is that a 7.5-day decode can be stopped and
restarted any number of times and still produce exactly the same bytes.

## The finish (lines 1603-1734)

```python
if encoder is not None:
    encoder.finish()          # flush the coder's final bits
    bit_out.close()           # pad the last byte with zeros, close the file
```

Then it measures **two different sizes**, and both are reported:

| Number | From | Meaning |
|---|---|---|
| `modeled_payload_bytes = n_bits / 8` | the sum of the model's own `-log2 P` (D1, step 6) | What the model *says* it costs |
| `actual_compressed_bytes` | `os.path.getsize` of the real output (or input when decoding) | What the file really is |

They should agree to within a few bytes, since the coder reaches almost exactly the
theoretical cost (your session 1 insight about the coder landing within ~3 bytes of the
ideal).

The headline figure:

```python
benchmark_total_bytes = actual_compressed_bytes + args.vocab_bytes + decompressor_bytes
benchmark_total_bpb   = benchmark_total_bytes * 8 / args.raw_bytes
```

= **compressed payload + vocabulary (34,421 bytes) + zipped decompressor (the code, plus
`weights.bin`)**, divided by the original 1,000,000,000 bytes. The code calls this the
"benchmark total" (the Matt Mahoney size metric Part A mentioned).

Then: write TensorBoard hyperparameters and results, log one machine-readable line,

```
FINAL_RESULT {...json: symbols, modeled_bits, modeled_bits_per_byte, actual_compressed_bytes,
              vocab_bytes, prior_archive_bytes, prior_transmission_bytes, initial_prior_sha256,
              benchmark_total_bytes, benchmark_total_bits_per_byte, elapsed_seconds, ...}
```

and a human summary (`done in ... s (... kS/s), ... bits/token, ... bits/byte`). Finally
`mark_done` writes a `DONE` file in the checkpoint directory (presumably so a job
script can tell a finished run from an interrupted one; no script here reads it), and
`main()` returns `0`.

---

# The whole file in one picture

```
                 weights.bin (1-bit, 2 blocks)            input tokens (2-byte, 4,097 vocab)
                        |                                       |
                        v  Part B: read_tensor_archive,         |
                        v  load_depth_prior (checks, copy)      |
        Transformer, 12 blocks, 162.6M params   <--- Part A     |
                        |                                       v
                        +----------------->  D1: 64 streams x (predict 1 token each)
                                                  |
                                    make_freq_table (Part C) -> arithmetic coder (rank 0)
                                                  |
                              compressed bytes <--+
                                                  |
                              learn on the 64 new tokens (1 Adam step, eval mode)
                                                  |
        every 500K tokens:  D2: retrain on up to 35M earlier tokens (dropout on,
                                separate optimizer, decaying LR)
        every 4 blocks:     D2: checkpoint (atomic), resumable bit for bit
        at the end:         D2: payload + vocab + zip(code + weights.bin) = the score
```

## What this walkthrough does **not** cover

- **`preprocess.c`** (Bellard's tokenizer, in C): how text becomes the 4,097-token
  stream.
- **The inner workings of `arithmetic_coder.py`** beyond its `state_dict` methods.
  Rung 1 builds the idea up from scratch, and you can read that file next.
- **The Mamba and KDA backends**, the `learned_relative` position mode, and the
  `ssm_*` / `kda_*` flags: alternative paths whose source files aren't in this repo.
- **The Slurm job scripts and the README's run instructions.** I haven't read them, so
  claims about the cluster are inferred from comments.
- **Any measured run of the real model.** Every figure here that isn't read straight from
  a file (e.g. 162,585,600 parameters, about 17,090 retrain steps) is arithmetic from the
  settings.

*End of the walkthrough.*
