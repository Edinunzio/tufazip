# Session Logs

A running diary of the learning effort on this fork. **Newest session first.**
To add a session, insert it directly below the marker line, above the previous entry.

<!-- NEW SESSIONS GO BELOW THIS LINE -->

---

## Session 2: 2026-11-25

**Goal:** finish the attention walkthrough, build rung 4 (the tiny Transformer), then turn
the project into something that can be **shared with an engineer and with a layperson**,
with the point that **sample size matters** backed by data. Then read the real `nn.py`
line by line (goal (c) from session 1).

### What was done, in order

1. **Finished attention steps 4-5** from `attention_by_hand.py`. Softmax by hand
   (`e^x`, divide by the row total), then the weighted blend of values. Two predictions:
   token 0's weights `[1, 0, 0]` (right), token 2's output `[0.788, 0.788]` (first guess
   `[1.1, 0.788]`; the cause of the 1.1 was not established. Each output number only picks
   up the weights of the rows where V has a 1 in that column).
2. **Rung 4: tiny causal Transformer** (`exercises/04_tiny_transformer.py`), written in
   two parts, each run in the container before moving on. Decisions: learned position
   embedding (RoPE is in the walkthrough instead), PyTorch autograd instead of
   hand-written backprop, Adam with window-wide loss and 2 replays per byte.
   - Part 1: the model, 68,800 parameters. Checks: untrained model scores 7.980
     bits/byte (about uniform), and the causal mask holds.
   - Part 2: online encode/decode loop. First run on 8 KB: 3.919 bits/byte. Full file
     (7 minutes each way): **3.064**, round trip exact.
3. **Workflow diagram** (Mermaid), saved into `docs/ONBOARDING.md`.
4. **Sample-size experiment** (`exercises/05_sample_size.py`): rungs 1-4 on the first
   2 KB to the full 40 KB. About 14 minutes.
5. **Docs for sharing:** `docs/EXPLAINER.md` (plain language), `docs/ONBOARDING.md`
   (engineer), `docs/README.md` (index). First push of `learn` to the fork.
6. **`docs/NN_WALKTHROUGH.md`**: all of `nn.py` (1,738 lines) in five parts, written and
   pushed one at a time. Included reading `weights.bin`'s header directly.

### Results

Full sample (39,785 bytes), bits per byte:

| Rung | Model | bits/byte | ratio |
|---|---|---|---|
| raw | none | 8.000 | 1.00x |
| 1 | adaptive order-0 counts | 4.638 | 1.72x |
| 2 | n-gram, order 3, blended | **2.652** | **3.02x** |
| 3 | tiny net (numpy) | 3.364 | 2.38x |
| 4 | tiny Transformer (PyTorch) | 3.064 | 2.61x |

By sample size (first N bytes of the same file):

| bytes | order-0 | n-gram o3 | n-gram o5 | tiny net | Transformer |
|---|---|---|---|---|---|
| 2,000 | 5.120 | **3.800** | 3.912 | 5.304 | 5.188 |
| 5,000 | 4.922 | **3.363** | 3.502 | 4.523 | 4.294 |
| 10,000 | 4.822 | **3.092** | 3.236 | 4.026 | 3.774 |
| 20,000 | 4.707 | **2.840** | 2.977 | 3.669 | 3.377 |
| 39,785 | 4.638 | **2.652** | 2.758 | 3.364 | 3.064 |

Rung 4's learning curve over the full file, by quarter: 3.77, 2.99, 2.80, 2.70.

**What the data supports:** every model improves with more text; the neural nets improve
most (Transformer -2.12 bits/byte from 2 KB to 40 KB, n-gram o3 -1.15); neural nets start
*behind* simple counting (at 2 KB both are worse than order-0); and the Transformer's gap
to n-gram o3 shrinks at every size (1.39, 0.93, 0.68, 0.54, 0.41).
**What it does not support:** no crossover was observed. The count table won at every
size we measured. Rung 4's learning rate was not tuned, and each sample is a prefix of
one book.

### Commits on `learn` this session (all pushed)

| Hash | What |
|---|---|
| `a577552` | Rung 4 part 1 (model) |
| `38614a8` | Rung 4 part 2 (coding loop); 3.064 bits/byte |
| `a241d40` | Sample-size experiment, with the table |
| `b5ab750` | EXPLAINER, ONBOARDING, docs index |
| `b0120d2` | Walkthrough part A (config, model) |
| `ad24643` | Walkthrough part B (`weights.bin` loader) |
| `f26ebe9` | Walkthrough part C (helpers, CLI) |
| `f47b8c0` | Walkthrough part D1 (`main()` setup and loop) |
| `a0b31f6` | Walkthrough part D2 (retrain, checkpoints, finish); doc fixes |

**Uncommitted:** this entry.

### Insights (the ideas worth keeping)

- **Start-up cost.** A smarter learner pays more up front. The Transformer lost to plain
  letter-counting at 2 KB and was still catching up at 40 KB. This is why tufazip does
  not start from nothing.
- **Same pipeline, new box.** Rungs 1-4 and tufazip differ only in where the
  probabilities come from; the arithmetic coder never sees the model.
- **The coder's whole state is a few integers** (`low`, `high`, plus an underflow counter
  when encoding), which is why a checkpoint of it is tiny.
- **`weights.bin` is not a trained 12-layer model.** It is a 2-block model stored at
  **1 bit per weight** (34,091,520 parameters, 4.3 MB), copied into 12 block slots at load
  time ("loop-untying"). The full model is 162,585,600 parameters. It is charged to the
  score: zipped together with the code.
- **The real codec reads 64 passages at once.** Each 499,968-token block is cut into 64
  streams; one batched prediction per step, 64 symbols coded in a Python loop on rank 0,
  one Adam step per 64 tokens, on the newly coded tokens only.
- **Retraining is the real "replay."** At every block boundary both sides re-study up to
  35M already-coded tokens (about 17,090 optimizer steps when the window is full, vs 7,812
  online steps per block), with dropout on and a separate optimizer.
- **Dropout is off while coding** (eval mode) and on only in retraining.
- **The number of GPUs is part of the codec.** One concrete reason: each rank reseeds
  with `seed + rank`, so retraining's dropout masks depend on the rank layout.
- **RoPE** makes attention depend on *distance* between tokens, not absolute position,
  by rotating `q` and `k` by a position-dependent angle.

### Lessons (process)

- **Measure the claim, then state it.** "Sample size matters" started as a hunch. One
  small script turned it into a table, and the table showed the claim was subtler: the
  start-up cost is real, but we never saw a crossover.
- **Say what the data does not show**, in the doc itself. The EXPLAINER and ONBOARDING
  both carry a "what we did not show" block.
- **Read the source before describing it.** Most of my errors this session were guesses
  I then had to retract (below). The fix each time was to read the code or parse the
  file.
- **A running log of corrections is worth keeping.** Mine:
  - I gave token 0's masked score as 0.5; it is 1 (still `[1, 0, 0]`).
  - I said "nn.py V2 is ~100 million+" parameters; it is 162,585,600.
  - I said `nn.py` trains on the whole window like rung 4; V2 uses
    `online_loss = 'code'` (new tokens only).
  - I tied `weights.bin`'s small size to code-size pressure; it is really a 1-bit,
    2-block prior.
  - I called `config_signature` a hash; it is a sorted JSON string.
  - Smaller overreaches removed before they were committed: "biggest source of
    learning", "the number the contest scores", the purpose of the `DONE` file.
- **Do what you said you would.** Stated plans ("commit after seeing the number",
  "commit and push C, then start D1") held. Twice I said I would draft in chat first
  (the EXPLAINER, and this entry) and wrote the file directly instead, saying so each
  time; it was fine here, but it is a place the plan and the action drifted apart.
- **Verify before reporting.** I parsed `weights.bin`'s header (a bug in my first
  script, a `seek` after the file closed, was fixed on the second try) and ran the LR
  functions on V2's real numbers instead of describing them.
- **Terminology** (from you, for your global log): "2.718 is an exponential" -> 2.718 is
  the number **e** (Euler's number); the **exponential function** is `e^x`, and 2.718 is
  `e^1`.
- **Tooling:** none broke this session. Rung 4 on the full file takes about 7 minutes per
  direction on this laptop, one thread.

### Open items / next session

1. **Review before sharing.** `SESSION_LOGS.md` is public on the fork. Read it, and look at
   `ONBOARDING.md` on GitHub to confirm the Mermaid diagram renders (never viewed).
2. **Rung 4 docstring tweak** (not yet approved): "Adam with gradient clipping, as in
   nn.py" is true but could mislead, because the real Adam has `beta1 = 0`.
3. **Optional experiments:** a learning-rate sweep for rung 4; and the full novel
   (~700 KB) on the Mac mini, the only way to test whether the Transformer ever
   overtakes the count table.
4. **Not covered in the walkthrough:** `preprocess.c` (the tokenizer), the inner workings
   of `arithmetic_coder.py`, the Mamba/KDA backends, and any Slurm scripts.
5. Build the Docker image on the Mac mini.

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
