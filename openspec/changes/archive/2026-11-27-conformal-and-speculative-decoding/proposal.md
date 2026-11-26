# Proposal

## Why

Part A of the assignment asks for four implementations. A.1 (option scorer) and A.3 (group-wise quantizer) are in the repo; A.2 (split conformal prediction and APS) and A.4 (speculative decoding with a check that the output distribution is unchanged) exist only as cells in a Colab notebook. The Colab run on 26 November 2026 also produced results that the repo does not record: the scorer over all 15,827 questions with four option shifts, conformal sets at two significance levels, the A.3 error on Qwen2.5-0.5B, and the A.4 checks on toy and real models. Part D reads A.2 directly, so it must be code in the repo with its own checks, not notebook cells.

## What Changes

- Add a conformal module: the scores `1 - p(y|x)` and randomized APS, the calibration threshold at the `ceil((n+1)(1-alpha))`-th smallest score, and prediction sets with their coverage and mean size. It reads an `(N, 4)` array of option probabilities, so it works on any scorer's output.
- Add a speculative-decoding module: the draft-and-verify loop over two models that share a vocabulary, with accept/reject statistics, and checks that the output follows the target distribution and that the acceptance rate equals `1 - TV(p, q)`.
- Record the Colab notebook, with its outputs, in `notebooks/`, and write the figures that the report will cite into the change's notes.
- Count rejections in the speculative-decoding statistics. The notebook printed `accepted / drafted` (0.43) as the acceptance rate, which understates the per-token rate `a` of proof B.3 because tokens drafted after a rejection are never judged.

Out of scope: the 2025 or any other post-cutoff question set, a KV cache for speculative decoding, and any change to the scorer or the quantizer.

## Capabilities

### New Capabilities

- `conformal-prediction`: nonconformity scores, calibration threshold and prediction sets for four-option questions, with the empirical coverage and set-size measures used to check the guarantee.
- `speculative-decoding`: draft-and-verify generation with a small and a large model sharing a vocabulary, its statistics, and the checks of distribution and acceptance rate.

### Modified Capabilities

None.

## Impact

- New code: `src/machinelearningproject/conformal.py` and `src/machinelearningproject/speculative.py`, each with a `_self_test()`. NumPy and SciPy only, both already dependencies.
- New file: `notebooks/ai_scorer.ipynb` with the Colab outputs.
- `README.md` gets a short section on the two modules and how to run their self-tests.
- Downstream: D.3 (conformal sets per subject at alpha 0.05 and 0.1) and D.4 (speedup) call these modules.
