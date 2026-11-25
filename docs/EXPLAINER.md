# How does a computer shrink a file? (A plain-language explainer)

No maths or programming needed. If a word is new, it's explained where it first appears.

## What this project is

[tufazip](https://github.com/Tufalabs/tufazip) is a real, record-chasing program that
shrinks a 1 GB chunk of Wikipedia down to roughly 100 MB. It uses a small artificial
"brain" (a neural network) to do it. It needs two very expensive graphics cards and
about a week of computing to run once, so nobody runs it casually.

This fork is a **study project**. To understand how it works, we built four tiny
versions of the same idea, each a little smarter than the last, and ran them on one
short book (the first ~40,000 letters of *Pride and Prejudice*). Everything runs on an
ordinary laptop in minutes.

## The one big idea: shrinking is guessing

Imagine playing 20 Questions, but about the *next letter* of a sentence.

You see: `The cat sat on the m`

Almost certainly the next letter is `a` (as in "mat"). If you're that sure, writing
down "it was the likely one" takes almost no space. If the next letter had been a `z`,
you'd need more words to say so.

That's all compression is, at heart:

> **Letters you can predict are cheap to write down. Letters that surprise you are
> expensive.**

So a better compressor is just **a better guesser**. The "zip" machinery (called an
*arithmetic coder*) is a standard, unchanging part that turns a guess into actual
saved space. The only thing that changes between a weak and a strong compressor is
**who is doing the guessing**.

A note on units: the cost of a letter is measured in **bits**. One bit is one yes/no
answer. Plain text normally uses 8 bits per letter, so **8 means "no shrinking at
all"**, and smaller is better.

## The ladder: four guessers, each smarter than the last

| Step | Guesser | In plain words | Bits per letter | Smaller by |
|---|---|---|---|---|
| 0 | none | store every letter as-is | 8.00 | 1x (nothing) |
| 1 | counting | "'e' is common, 'z' is rare" | 4.64 | 1.7x |
| 2 | short memory | "after `th`, an `e` is likely" | **2.65** | **3.0x** |
| 3 | tiny neural net | learns patterns from the last 3 letters | 3.36 | 2.4x |
| 4 | tiny "Transformer" | looks back over the last 32 letters and learns which ones matter | 3.06 | 2.6x |

(These are results on the whole 40,000-letter sample.)

The Transformer is the same family of idea as the one behind modern chatbots. The key
trick is **attention**: for each new letter, it asks "which of the earlier letters
should I pay the most attention to?", and it learns this on its own.

## The surprise: the fancy one did not win

Look at the table again. The cleverest guesser (step 4) did **not** beat the simple
"short memory" one (step 2). That surprised us at first. The explanation is the most
useful thing in this whole project.

## Why sample size matters so much

Here is the same experiment run on smaller and smaller amounts of text. Each model
starts knowing nothing and learns **as it reads**. Same book, but we stop early.

| Letters read | Counting | Short memory | Transformer |
|---|---|---|---|
| 2,000 | 5.12 | **3.80** | 5.19 |
| 5,000 | 4.92 | **3.36** | 4.29 |
| 10,000 | 4.82 | **3.09** | 3.77 |
| 20,000 | 4.71 | **2.84** | 3.38 |
| 39,785 | 4.64 | **2.65** | 3.06 |

(Bits per letter. Lower is better. The best in each row is in bold.)

Three things stand out:

1. **Everyone improves the more they read.** Even simple counting does, because it has
   more evidence about which letters are common.
2. **The clever models start badly.** At 2,000 letters the Transformer (5.19) is
   actually *worse* than just counting letters (5.12). It's like hiring a brilliant
   new employee: for the first few days they're slower than the person who just
   follows a checklist, because they're still learning the job.
3. **The clever models catch up faster.** The gap between the Transformer and the
   short-memory guesser shrinks every time we double the text (1.39, then 0.93, 0.68,
   0.54, 0.41 bits per letter). The Transformer improved by about 2.1 bits per letter
   over the experiment; the short-memory guesser improved by about 1.1.

**A good picture for it:** a simple method is a checklist, which works on day one but
has a ceiling. A learning method is an apprentice, who is poor at first but keeps
getting better. *How long you let the apprentice practise decides who wins.*

### What we did **not** show

We have to be careful here:

- The short-memory guesser was ahead at **every** size we tried. We never saw the
  Transformer actually overtake it. We only saw the gap closing.
- We can't say *when* it would overtake, or promise that it would. Our sample is just
  40,000 letters, and testing further needs a much longer text.
- Each smaller sample is just the opening of the same book, so it isn't a perfectly
  fair test of "any 2,000 letters".
- We didn't tune the Transformer's settings. A better-tuned one might do better.

So the honest summary is: **"with little data, simple beats clever; with more data,
clever closes the gap fast."** That is a measured trend, not a measured winner.

## Why the real tufazip cheats (in a good way)

The real program faces the same start-up problem, so it doesn't start from nothing.

- It **ships with a pre-taught brain** (a 4.3 MB file of "starting knowledge"). That
  removes most of the slow beginning we saw above.
- Its job is **enormous**: 1 GB of text, so a learner has plenty of time to pay off.
- Its brain is far bigger: about **160 million** adjustable numbers, against about
  **69 thousand** in our Transformer.

Our toy shows the *start-up cost* in miniature. tufazip is the case where the
apprentice has had a head start and a very long time to practise.

## One fragile detail worth knowing

To *un*-shrink a file, the decoder has to replay the guesser **exactly** as it was
during shrinking, making the same guess at every single step. Even a tiny difference
(a rounding wobble in a decimal number) and everything after it turns to garbage. This
is why the real tufazip insists on specific hardware and software versions.

## A few quick questions

**Is a compressed file a formula?**
Not really: it's a record of how surprised the guesser was at each step. Where the
text was predictable, hardly anything was written down.

**Can compression shrink anything?**
No. It only works when there is something to predict. Truly random data can't be
shrunk at all, and well-predicted data (like English) can shrink a lot. tufazip
reports about a 10x reduction on Wikipedia text; our toy ranges from 1.7x to 3x.

**Do I need to run any of this?**
No. This page and the tables above are the story. If you're curious, the engineers'
guide is in [ONBOARDING.md](ONBOARDING.md).
