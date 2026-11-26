# Tasks

## 1. Conformal prediction

- [x] 1.1 Create `src/machinelearningproject/conformal.py` with the plain and randomized APS scores, the calibration threshold (infinite when the rank exceeds `n`), the prediction sets with the hit flags, and functions for coverage, mean set size and the fraction of empty sets; take the scores, gold indices, a calibration mask, alpha and a seeded generator as inputs; verify the file parses
- [x] 1.2 Add `_self_test()` with asserts for the spec scenarios: plain scores `0.5, 0.7, 0.85, 0.95`, adaptive scores `0, 0.5, 0.8, 0.95` at random number 0, threshold `0.70` for the ten-score example at alpha 0.2, an infinite threshold for 3 scores at alpha 0.1, the set `{A, B}` in the worked example, identical sets for the same seed, and mean size 4 with coverage 1 for an infinite threshold; verify `uv run python -c "from machinelearningproject.conformal import _self_test; _self_test()"` passes
- [x] 1.3 Add the simulated coverage check to `_self_test()`: 10,000 calibration and 10,000 test questions with probabilities from a uniform Dirichlet distribution and answers drawn from them; verify both scores give coverage between 0.88 and 0.92 at alpha 0.1

## 2. Speculative decoding

- [x] 2.1 Create `src/machinelearningproject/speculative.py` with the draft-and-verify loop over two callables that return a `(len, V)` NumPy array, returning the output and the statistics (target calls, drafted, accepted, rejected) with the per-token acceptance rate, trimming to the requested number of new tokens and raising an error that names both sizes when the vocabularies differ; verify the file parses
- [x] 2.2 Add `_self_test()` with asserts for the scenarios that need no sampling at scale: identical draft and target accept every token and add `gamma + 1` tokens per round, the output has exactly the requested length, a size mismatch raises an error containing both numbers, and a forced rejection counts one accepted, one rejected and the rest drafted only; verify `uv run python -c "from machinelearningproject.speculative import _self_test; _self_test()"` passes the fast part
- [x] 2.3 Add the statistical checks to `_self_test()` on toy models of 5 tokens with a fixed seed: 60,000 three-token continuations tested by chi-square against the exact target distribution (p above 0.001), the draft-only control (p below 1e-6), and the single-token acceptance rate within 0.01 of `1 - TV(p, q)`; verify they pass and note the run time in `notes.md`

## 3. Record the Colab run

- [x] 3.1 Copy the saved notebook to `notebooks/ai_scorer.ipynb` with its outputs, from the Drive file `Ai_Scorer.ipynb`; verify the file parses as JSON and its A.2, A.3 and A.4 cells have outputs
- [x] 3.2 Write `notes.md` in this change with the figures from the notebook that the report will cite: the five-split coverage and set sizes, the per-subject coverage, the A.3 error table (relative error, ratio to `step^2/12`, packed size), and the A.4 chi-square p-values, acceptance rates and the full-loop statistics; state which scorer and prompt produced the probabilities and that the new modules were not run by the author
- [x] 3.3 Add a short section to `README.md` that names the two modules, the notebook, and the two self-test commands; verify the commands are written exactly as in tasks 1.2 and 2.2
