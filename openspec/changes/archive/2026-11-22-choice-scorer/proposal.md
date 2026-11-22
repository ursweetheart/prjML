# Proposal

## Why

Every Part D number (accuracy, McNemar, conformal sets, ECE) comes from the probability a model gives to each answer option A–D, so the scorer is the base of the critical path: the log-probability cache of Day 3 is just this scorer run over the benchmark. It is task A.1 and was due on Day 2; nothing implements it yet, and the assignment asks for a check against a standard library and for option permutation to measure position bias.

## What Changes

- Add a scorer that turns a benchmark record into a fixed prompt, runs a Hugging Face causal language model, and returns the log-probability of the letters A, B, C and D renormalised over those four.
- Add option permutation: four cyclic shifts so each option appears at each position once, with the scores mapped back to the original option order, and a measure of position bias.
- Add a check command that scores 100 test-split questions with this scorer and with `lm-eval` on the identical prompts, and fails if the accuracies differ by more than one point.
- Add `lm-eval` as a development dependency.

Working assumptions: a letter-based prompt (the model is asked for the answer letter after the options), as `lm-eval` does for multiple choice; Qwen2.5-0.5B is the model for the check because the development GPU has 4 GB. The three 3B-class models are run on Colab on Day 3.

## Capabilities

### New Capabilities

- `choice-scorer`: scoring multiple-choice benchmark records with a causal language model: prompt, per-option log-probabilities, option permutation, position bias, and agreement with a standard evaluation library.

### Modified Capabilities

None.

## Impact

- New code: `src/machinelearningproject/scorer.py` and `src/machinelearningproject/check_scorer.py`, run with `uv run python -m machinelearningproject.check_scorer`.
- New dev dependency: `lm-eval`. The check downloads `Qwen/Qwen2.5-0.5B` (about 1 GB) from Hugging Face.
- Downstream: the Day 3 cache, A.2 conformal prediction and D.2/D.3 read the `(N, 4)` log-probability arrays this scorer returns.
- Out of scope: running the 3B-class models on the full benchmark, the cache file format, calibration and conformal code.
