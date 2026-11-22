# Tasks

## 1. Scorer

- [x] 1.1 Create `src/machinelearningproject/scorer.py` with the prompt formatter and the cyclic-shift permutation (shift the options, map scores back), with a `_self_test()` for the layout, determinism, each option at each position once, and the mapping-back scenario; verify `uv run python -c "from machinelearningproject.scorer import _self_test; _self_test()"` passes
- [x] 1.2 Add the letter-probability function (left padding, batches sorted by length and capped by a token budget, last-position logits, the four letter tokens asserted to be single tokens, renormalised log-softmax over them) and the position-bias report, with an assert for the biased-model scenario on fake logits; verify the self-test passes

## 2. Check against lm-eval

- [x] 2.1 Add `lm-eval` as a dev dependency (`uv add --dev lm-eval`, or record the `uv run --with` fallback in the notes if the resolver refuses) and verify `uv run python -c "import lm_eval"` works
- [x] 2.2 Create `src/machinelearningproject/check_scorer.py`: sample 100 test-split records with a fixed seed, score them with `Qwen/Qwen2.5-0.5B`, check normalisation and batch-size invariance (1 vs 8), write the prompts to a JSONL file and run `lm-eval` on them with letter choices, compare accuracies and exit non-zero if they differ by more than 0.01; verify the command prints both accuracies and exits 0
- [x] 2.3 Run the four permutations on the same 100 questions and print the position-bias table; verify the mapped-back scores have no NaN and record the accuracies for each shift, both accuracies and the bias table in `notes.md`

## 3. Documentation

- [x] 3.1 Add a "Scorer check" section to `README.md` with the command and what it downloads; verify the command runs as written from a fresh shell
