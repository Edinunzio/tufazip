"""Attention, small enough to do on paper.

Three tokens, hand-picked numbers, no learning. Every step prints, so you can
follow along with a calculator and see where each number comes from.

WHY attention exists: in rung 3 the net saw a fixed bundle of the last 3 bytes.
Attention lets each position LOOK BACK over everything before it and decide, per
position, how much to listen to each earlier token. tufazip does this over its last
192 tokens (V2_CONFIG ctx_len) in each of 12 layers.

The vocabulary of attention:
  query (q): "what am I looking for?"       one per token
  key   (k): "what do I offer to a search?"  one per token
  value (v): "what I hand over if chosen"    one per token
In a real model q, k, v are made from each token's vector by learned matrices
(nn.py Attention: `self.qkv = nn.Linear(d_model, 3*d_model)`). Here we skip that
and just write q, k, v down.

Run (from the repo root):
    docker run --rm -v "$PWD":/work tufazip-learn python exercises/attention_by_hand.py
"""
import numpy as np
import torch
import torch.nn.functional as F

np.set_printoptions(precision=3, suppress=True)

# Three tokens (rows), each described by 4 numbers (columns).
Q = np.array([[2, 0, 0, 0],
              [0, 2, 0, 0],
              [2, 2, 0, 0]], dtype=float)
K = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [1, 1, 0, 0]], dtype=float)
V = np.array([[1, 0],
              [0, 1],
              [1, 1]], dtype=float)
D = Q.shape[1]  # size of q and k vectors; used in the scaling step

print("Q (what each token is looking for):\n", Q)
print("K (what each token offers):\n", K)
print("V (what each token hands over):\n", V)

# STEP 1: match scores. How well does token i's query match token j's key?
# A dot product is large when two vectors point the same way: a similarity score.
scores = Q @ K.T
print("\nSTEP 1  scores = Q @ K.T   (row i, column j = q_i . k_j)\n", scores)

# STEP 2: scale by sqrt(D). Dot products grow with vector length; unscaled, softmax
# would saturate into near-0/near-1 and learn badly. nn.py: d_head = 1024/8 = 128.
scaled = scores / np.sqrt(D)
print("\nSTEP 2  scaled = scores / sqrt(D)   (here sqrt(4) = 2)\n", scaled)

# STEP 3: causal mask. Token i may only look at tokens 0..i, never forward. This is
# non-negotiable for compression: the decoder doesn't have the future yet.
# (nn.py passes is_causal=True.) -inf becomes weight 0 after softmax, since e^-inf = 0.
mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
masked = np.where(mask, -np.inf, scaled)
print("\nSTEP 3  masked (future positions -> -inf)\n", masked)

# STEP 4: softmax each row into weights that are >= 0 and sum to 1: "how much do I
# listen to each earlier token". The same softmax turns logits into probabilities at
# the model's output.
exp = np.exp(masked - masked.max(axis=1, keepdims=True))
weights = exp / exp.sum(axis=1, keepdims=True)
print("\nSTEP 4  weights = softmax(masked), row by row\n", weights)
print("        row sums:", weights.sum(axis=1))

# STEP 5: blend the values. Each token's output is a weighted average of the values of
# the tokens it listens to.
out = weights @ V
print("\nSTEP 5  output = weights @ V\n", out)

# CHECK: PyTorch's built-in attention, the very call in nn.py (Attention.forward).
ref = F.scaled_dot_product_attention(
    torch.tensor(Q)[None, None], torch.tensor(K)[None, None],
    torch.tensor(V)[None, None], is_causal=True)[0, 0].numpy()
print("\nPyTorch scaled_dot_product_attention agrees:", np.allclose(out, ref))
assert np.allclose(out, ref)

# --- Things to try ---
# 1. Row 0 can only see itself. What must its weights be, and why?
# 2. Change Q[2] to [0, 0, 0, 0]. What do row 2's weights become? (All matches are
#    equal, so it listens to everyone equally.)
# 3. Delete STEP 3's masking. Does token 0's output change? That's "peeking at the
#    future"; a compressor that did this could never be decoded.
# 4. Real models run many of these side by side ("heads"), each with its own q/k/v
#    matrices. nn.py: n_head=8, d_head=128. Each head can learn to look for something
#    different (the previous word, the matching bracket, ...), and their outputs are
#    concatenated and mixed by `self.proj`.
