# Session Logs

A running diary of the learning effort on this fork. **Newest session first.**
To add a session, insert it directly below the marker line, above the previous entry.

<!-- NEW SESSIONS GO BELOW THIS LINE -->

---

## Session 1: 2026-11-23 to 2026-11-24 (logged 2026-11-24 17:06 EST)

**Goal:** learn transformers, neural networks and compression from the tufazip repo.
Be able to (a) explain how attention works, (b) build a toy compressor with
assistance, and (c) read `nn.py` line by line. Show the result off after a few days off.

### What was done, in order

1. **Read the repo before asking questions.** Findings that shaped everything:
   - tufazip v2 compresses `enwik9` (1 GB of Wikipedia) to ~101 MB with a 12-layer
     Transformer that predicts the next word-token, plus an arithmetic coder.
   - The real decoder needs **2x NVIDIA B200 GPUs**, CUDA 13, and ~7.5 days, and the
     1 GB input is not in the repo. It cannot be run here, so it never is.
   - Files: `nn.py` (model + codec), `arithmetic_coder.py`, `preprocess.c` (Bellard's
     tokenizer), `weights.bin` (4.3 MB quantized starting weights), `enwik9.voc` (vocab).
2. **Scoped the work.** Docker was first dismissed, then brought back as a small
   CPU-only image for the exercises only. Decisions: full 4-rung toy compressor,
   Markdown docs in the repo, remote `origin` -> the fork, `upstream` -> Tufalabs,
   work on branch `learn`, `master` untouched. No PRs to Tufalabs.
3. **Plan written and approved** (plan file under `~/.claude/plans/`). Approval was
   deliberately not treated as a go-ahead; each step waited for an explicit "go".
4. **Created the skeleton** (`mkdir`/`touch` only), then the git setup.
5. **Docker:** wrote `Dockerfile.learn` + `.dockerignore`; user built it. Verified:
   torch 2.15.0+cpu, numpy 2.5.5.
6. **Sample text:** considered Vonnegut (Project Gutenberg has only two short stories),
   chose *Pride and Prejudice*. Gutenberg was down for maintenance (HTTP 503), so the
   text came from the mirror `gutenberg.pglaf.org`. `exercises/sample.txt` = first
   39,785 bytes.
7. **Housekeeping:** added `.DS_Store` to `.gitignore` (standing preference, saved to
   memory); saved a project memory so a new session can resume.
8. **Built and ran the exercises** (all verified in the container, all round-trip exactly):

   | Rung | Model | bits/byte | ratio |
   |---|---|---|---|
   | raw | none | 8.000 | 1.00x |
   | 1 | adaptive order-0 byte counts | 4.638 | 1.72x |
   | 2 | context n-gram, best at order 3 | 2.652 | 3.02x |
   | 3 | tiny neural net (numpy, hand-written backprop) | 3.364 | 2.38x |

   Also `attention_by_hand.py` (3 tokens, hand-picked numbers; agrees with PyTorch's
   `scaled_dot_product_attention`, the call in `nn.py:189`).
9. **Walked through rung 1 interactively** in five steps (ruler, cost, encoder,
   decoder, reading the output), then began the attention walkthrough.
10. **Side question** from a non-technical friend ("are compressed files a formula? will
    a USB drive hold the world's library?") answered in plain language.

### Commits on `learn` (nothing pushed yet)

| Hash | What |
|---|---|
| `4a2b7f1` | Docker image + sample text |
| `41ef559` | Rung 1 + `.gitignore` |
| `4633fb8` | Rung 2 |
| `ac10161` | Rung 3 |

**Uncommitted:** `exercises/attention_by_hand.py`, this file, and the still-empty
skeleton files (`docs/ONBOARDING.md`, `docs/NN_WALKTHROUGH.md`,
`exercises/04_tiny_transformer.py`).

### Where the walkthrough stopped

Attention steps 1-3 done (match scores, scaling by sqrt(D), causal mask). Next is
**Step 4: softmax**. For token 1 the row is `[0, 1, -inf]`; the exponentials are
`1, 2.718, 0`, so the weights should come out `[0.269, 0.731, 0]`. Then Step 5 blends
the values with those weights.

### Insights (the ideas worth keeping)

- **Compression is prediction.** Size ~ the sum of `-log2(p)` over every symbol. A
  model supplies the probabilities; the arithmetic coder turns them into bits. They
  are independent, so rungs 1-3 and tufazip itself differ only in *where the
  probabilities come from*.
- **A bit is one yes/no answer;** a symbol's cost is how many two-way splits are needed
  to land in its slice of the 0-to-1 line. Likely symbols are cheap.
- **Total cost = the cost of the final window's width,** because widths multiply and
  logs turn multiplying into adding. That is why the coder lands within ~3 bytes of
  the theoretical ideal.
- **The coder's window zooms, it doesn't slide.** State is just two integers
  (`low`, `high`); settled leading bits are shifted out and the rest is scaled up.
- **Decoder and encoder must learn identically.** Any drift, even one count, turns
  everything after it to garbage. A neural net's "counts" are floats, which is why
  tufazip pins exact hardware and library versions.
- **Context helps, then runs out.** Order 3 was best on a 40 KB file; orders 4-5 got
  worse as tables became sparse. Neural nets sidestep that by generalizing through
  embeddings, but need more data to get going.
- **Learning rate matters a lot.** At 0.1 the tiny net got *worse* over time (6.9
  bits/byte); at 0.01 it improved steadily. Replaying earlier bytes helped too
  (3.85 -> 3.36), a small version of tufazip's `retrain`.
- **Attention = each token looks back and chooses how much to listen to each earlier
  token.** The causal mask is required for compression: the decoder does not have the
  future.

### Lessons (process)

- **Look before asking.** Reading the repo first exposed the B200 requirement before
  any Docker work was wasted on it.
- **Explain the why before the steps.** Steps without a reason felt like ritual. Giving
  the purpose first (and a concrete toy example) is what unlocked each concept.
- **Check unfamiliar notation.** Interval notation `[0, 0.6)` was mistaken for a bug.
  It is the same half-open idea as Python's `range(0, 3)`. Naming it removed the
  confusion.
- **Small predictions beat re-reading.** One-number questions worked far better than
  asking for intuition ("roughly how many bits?") before the mental model existed.
- **Verify, don't assume.** Rung 3's first run passed the gradient check and the
  round trip yet compressed *worse* than rung 1. Correct code can still be badly
  tuned; the learning curve exposed it.
- **Tooling hiccups seen:** Project Gutenberg 503 (use an official mirror); Docker
  Desktop briefly not seeing a just-created file in the mounted folder (rerun fixed it).
- **One wrong turn of mine:** I introduced the `[0, 1)` window without saying where it
  came from, which caused a round of confusion. Define a starting point before using it.

### Open items / next session

1. Finish attention steps 4-5, then commit `attention_by_hand.py` and this log.
2. Rung 4: tiny causal-attention Transformer (`exercises/04_tiny_transformer.py`),
   mirroring `nn.py` (RMSNorm, attention, SwiGLU, residual blocks).
3. Write `docs/ONBOARDING.md` and `docs/NN_WALKTHROUGH.md` (the line-by-line `nn.py` guide).
4. Add friend-friendly docs (see the recommendations given at the end of this session).
5. `git push origin learn` when ready (not yet pushed); build the image on the Mac mini.
