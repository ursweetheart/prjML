# Design

## Context

- The repo has A.1 (`scorer.py`, spec `choice-scorer`, with four cyclic shifts and a position-bias measure) and A.3 (`quantizer.py`, spec `group-quantizer`). It has no conformal code and no speculative-decoding code.
- On 26 November 2026 the Colab notebook `Ai_Scorer.ipynb` ran these, all saved with outputs: a full-data scorer (15,827 questions, four shifts, Qwen2.5-0.5B), conformal prediction and APS on its probabilities, the repo quantizer on Qwen2.5-0.5B, and speculative decoding on toy and real models. The notebook's scorer has its own prompt (no subject line), different from `scorer.format_prompt`.
- Requirements for the new modules are in `specs/conformal-prediction/spec.md` and `specs/speculative-decoding/spec.md`. See proposal.md for motivation.

## Goals / Non-Goals

**Goals:**
- Conformal and speculative-decoding code in the repo, with self-tests that repeat the checks that passed in Colab.
- The Colab results recorded in the repo, so the report can cite them and a reader can see how they were produced.

**Non-Goals:**
- A KV cache or any speed measurement for speculative decoding (D.4).
- Replacing the notebook's scorer with `scorer.py`, or changing either one.
- Fixing `quantize_model` for models with tied embeddings (see Open Questions).

## Decisions

**Conformal functions take an `(N, 4)` probability array in the original option order, a gold-index array and a boolean calibration mask.** They know nothing about models, so the same code runs on any scorer's cache and on simulated data. Alternative: a class that wraps a scorer. Rejected: it would tie conformal code to Hugging Face and make the simulated self-check awkward.

**The random generator is passed in.** The adaptive score needs one uniform number per question; passing a seeded `numpy.random.Generator` makes runs reproducible, which the assignment requires, and lets callers use one generator across the calibration and test draws as the notebook does.

**APS is the randomized version of Romano et al. (2020).** The non-randomized version includes the option's own probability in its score and over-covers. The randomized one gives coverage close to `1 - alpha`: in the Colab run the mean coverage over five splits was 0.9005 at alpha 0.1 against 0.90. Alternative: the non-randomized score as the default. Rejected: coverage would sit above target for a four-option problem, where each option carries much probability.

**The threshold is infinite when `ceil((n+1)(1-alpha)) > n`.** That is the standard conservative rule and it keeps the guarantee for tiny calibration sets such as a single subject.

**Speculative decoding takes two callables that return a NumPy `(len, V)` array of next-token distributions, one row per input position.** This lets the same loop run on toy models with exactly known distributions (needed for an exact chi-square test) and on Hugging Face models through a thin wrapper. Alternative: take Hugging Face models directly. Rejected: the exact check needs a vocabulary small enough to enumerate every 3-token continuation.

**No KV cache.** Each target call re-reads the whole sequence, so the loop is slow. Correctness is what A.4 asks for; speed is measured in D.4 with a cached implementation or Hugging Face's assisted generation, and this module stays the reference to compare against.

**Count rejections and report `a = accepted / (accepted + rejected)`.** The Colab run printed `accepted / drafted` as 0.43 for a run of 44 drafted tokens, but draft tokens after a rejection are discarded unjudged. Using 0.43 in the formula of B.3, `(1 - a^(γ+1)) / (1 - a)` with γ = 4, gives 1.73 tokens per call, while 2.73 were observed. If every one of the 11 calls had ended in a rejection, `a` would be 19/30 = 0.63 and the formula gives 2.45, which is close to the observed 2.73; the exact value needs the rejection count.

**Evidence is the committed notebook plus self-tests.** The notebook keeps the real-model checks (they need a GPU); the self-tests repeat the toy and the simulated checks on the CPU in about a minute each, so a reader can rerun them without a GPU.

## Risks / Trade-offs

- [The new modules were written from the notebook code and not executed by the author before the commit, because the user asked that nothing run on their machine] → the algorithms are the ones whose checks passed in Colab, and the tasks end with the user running both self-tests. Treat the change as unverified until they pass.
- [The notebook's probabilities come from its own prompt, not from `scorer.format_prompt`, so the conformal numbers in the notebook may differ from those the repo scorer would give] → the report must say which scorer produced the cache. Rerunning the cache with the repo scorer is a separate run, not part of this change.
- [Conformal coverage is marginal, not per subject] → the notebook shows per-subject coverage from 0.829 (History) to 0.993 (Mathematics); D.3 reports it as is.
- [The chi-square tests are statistical, so a correct implementation fails about once in a thousand runs] → the tests use fixed seeds, and the self-test threshold of 0.001 is stated in the spec.

## Open Questions

- `quantize_model` replaces `lm_head`'s weight, and on models with tied embeddings (Qwen2.5-0.5B) that also changes the embedding table, although its docstring says embeddings stay in full precision. The notebook avoids it by skipping `lm_head`. This is a bug fix for D.4, not part of this change.
- Whether the final D.3 numbers should come from the repo scorer's prompt. Deferred until Part D.
