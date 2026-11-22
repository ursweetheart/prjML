# Design

## Context

See `proposal.md`. Constraints: `transformers` and `torch` are already dependencies; the development GPU has 4 GB (Qwen2.5-0.5B fits in FP16, 3B models do not), so the scorer must not assume a large GPU and must keep memory flat by batching by length. Benchmark records come from `data/processed/vnhsge.jsonl` with fields `subject`, `question`, `option_a` to `option_d`, `answer`, `split`.

## Goals / Non-Goals

**Goals:** one small module the Day 3 cache script can call per model; scores that `lm-eval` reproduces on the same prompts.
**Non-Goals:** caching to disk, conformal sets, ECE, the 3B-class runs, option-text likelihood scoring.

## Decisions

**Score the answer letters, not the option text.** After the prompt the next token is compared across ` A`, ` B`, ` C`, ` D` and renormalised over the four. This is what `lm-eval` does for multiple choice, it needs one forward pass per prompt instead of four, and it gives a proper distribution over the options for conformal prediction. Alternative: sum the log-likelihood of each option's text. Rejected: length-biased, four passes, and the probabilities do not form a distribution over answers.

**Left padding, read the last position, batches sorted by length.** With left padding the last position is the real last token for every row, so one indexing step gives the logits; sorting by length keeps padding small. Alternative: one prompt per call. Rejected: too slow for 15,000 questions times four permutations.

**Letter tokens are looked up once and asserted to be single tokens.** Each of ` A` to ` D` must encode to exactly one token for the model's tokenizer; the scorer raises otherwise instead of scoring a prefix.

**Cyclic shifts for the permutations.** With shift k, the option shown at position i is original option `(i + k) % 4`. Four shifts place each option at each position once, which is enough to separate position bias from content and keeps the cost at four passes. Alternative: all 24 permutations. Rejected: six times the cost for little extra information.

**The agreement check feeds `lm-eval` our finished prompt.** The check writes the 100 prompts to a JSONL file and defines an `lm-eval` task whose text is exactly that field, with choices ` A` to ` D`. The check therefore tests the scoring path (tokenization, padding, log-softmax, argmax) against the library, not whether two prompt templates agree. Alternative: re-implement the prompt as a Jinja template. Rejected: any template difference would be blamed on the scorer.

**Dependency `lm-eval` as a development dependency; the model for the check is `Qwen/Qwen2.5-0.5B`.** It is open (no token) and fits the 4 GB GPU.

## Risks / Trade-offs

- [Padding changes scores] → the invariance check runs in float32, where a padding bug shows as a difference above 1e-4. In float16 the first run measured 0.011 between batch sizes 1 and 8 (float32: 1e-5, no predicted letter changed), which is rounding noise, so float16 is only reported.
- [`lm-eval` and the installed `transformers` do not agree on versions] → try the dev dependency first; if the resolver refuses, run the check through `uv run --with lm-eval` and record that in the notes.
- [Accuracy on 100 questions moves by a whole point for one flipped question] → identical prompts make flips rare; the criterion is at most one point, so one flip passes and two fail.
- [Long English reading passages exceed memory in a batch] → batch by a token budget, not a count, and move to CPU if needed for the check.
