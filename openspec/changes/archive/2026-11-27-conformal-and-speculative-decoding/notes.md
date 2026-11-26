# Notes

Source of every figure below: the saved outputs of `notebooks/ai_scorer.ipynb`, run on Colab (T4 GPU) on 26 November 2026.

## What produced the probabilities

- Model: Qwen2.5-0.5B, float16, zero-shot.
- Scorer: the notebook's own function, **not** `scorer.py`. Its prompt is `Câu hỏi: {question}\nA. ...\nB. ...\nC. ...\nD. ...\nĐáp án:` with no subject line, and it reads the log-probabilities of the tokens ` A` to ` D` (ids 362, 425, 356, 422), renormalised over the four. Prompts longer than 1,024 tokens are cut from the left; 7 of 15,827 were affected.
- Data: all 15,827 records of `vnhsge.jsonl` (calibration 7,909, test 7,918), four cyclic option shifts, about 2.8 minutes per shift.
- Caveat: the repo scorer's prompt includes the subject, so its probabilities, and therefore the conformal figures, would differ somewhat. The report must say which scorer was used.

## A.1 on the full data

| Figure | Value |
|---|---|
| Accuracy per shift (0 to 3) | 0.360, 0.359, 0.356, 0.354 |
| Accuracy averaged over the four shifts | 0.406 |
| Letter the model says (all shifts pooled) | A 0.564, B 0.192, C 0.141, D 0.103 |
| Accuracy by position of the right answer | A 0.676, B 0.311, C 0.249, D 0.194 |
| Accuracy of the shift average, per subject | Biology 0.305, Chemistry 0.318, CivicEducation 0.553, English 0.579, Geography 0.358, History 0.408, Mathematics 0.332, Physics 0.341 |

## A.2 conformal prediction

Official split (calibration to test), n_cal = 7,909, n_test = 7,918:

| Probabilities | alpha | Method | q_hat | Coverage | Mean set size |
|---|---|---|---|---|---|
| shift 0 | 0.10 | plain | 0.816 | 0.898 | 3.217 |
| shift 0 | 0.10 | APS | 0.907 | 0.898 | 3.364 |
| shift 0 | 0.05 | plain | 0.837 | 0.948 | 3.502 |
| shift 0 | 0.05 | APS | 0.954 | 0.950 | 3.669 |
| shift average | 0.10 | plain | 0.778 | 0.899 | 3.103 |
| shift average | 0.10 | APS | 0.862 | 0.891 | 3.239 |
| shift average | 0.05 | plain | 0.801 | 0.949 | 3.408 |
| shift average | 0.05 | APS | 0.931 | 0.948 | 3.602 |

No empty sets in any row. Top-1 accuracy on the test half is 0.357 (shift 0) and 0.405 (shift average).

Five random half-and-half re-splits of the pooled data, shift-average probabilities, mean (SD):

| alpha | Method | Coverage | Set size |
|---|---|---|---|
| 0.10 | plain | 0.9015 (0.0066) | 3.109 (0.012) |
| 0.10 | APS | 0.9005 (0.0046) | 3.268 (0.017) |
| 0.05 | plain | 0.9511 (0.0031) | 3.421 (0.012) |
| 0.05 | APS | 0.9504 (0.0051) | 3.624 (0.022) |

The upper bound `1 - alpha + 1/(n+1)` is 0.90013 and 0.95013. The means sit 0.0014 (alpha 0.1, plain) above it, less than one standard error of a five-run mean (0.0066 / sqrt(5) = 0.003).

Coverage per subject (shift average, plain score, alpha 0.1, one calibration threshold for all subjects):

| Subject | n_test | Coverage | Mean size |
|---|---|---|---|
| Biology | 923 | 0.907 | 3.420 |
| Chemistry | 973 | 0.958 | 3.707 |
| CivicEducation | 940 | 0.846 | 2.329 |
| English | 1,246 | 0.848 | 2.166 |
| Geography | 913 | 0.875 | 3.116 |
| History | 986 | 0.829 | 2.746 |
| Mathematics | 1,016 | 0.993 | 3.943 |
| Physics | 921 | 0.946 | 3.647 |

The guarantee is marginal, so subjects drift both ways: the model is confident where it is more accurate (English, Civic Education) and the shared threshold under-covers those subjects.

Self-check on simulated calibrated data (10,000 + 10,000): plain 0.901, APS 0.898 at alpha 0.1.

## A.3 on Qwen2.5-0.5B (repo quantizer, group size 64, 168 linear layers, `lm_head` skipped)

| Setting | Mean relative error | Max relative error | MSE / (step^2/12) | Packed MB | fp16 MB |
|---|---|---|---|---|---|
| INT8 sym | 0.0065 | 0.0098 | 0.973 to 0.990 | 369.0 | 715.7 |
| INT8 asym | 0.0057 | 0.0080 | 0.993 to 1.012 | 374.6 | 715.7 |
| INT4 sym | 0.1179 | 0.1736 | 0.928 to 0.991 | 190.1 | 715.7 |
| INT4 asym | 0.0970 | 0.1340 | 0.967 to 1.005 | 192.9 | 715.7 |

The packed sizes equal 8.25 and 4.25 bits per weight over 16 (0.5156 and 0.2656), as the 16-bit group scales predict. `tie_word_embeddings` is true, so `lm_head` shares weights with the embedding table; the cell skipped it. INT4 error by layer type is similar across types (relative 0.111 for `o_proj` to 0.127 for `v_proj`). The per-layer tables are saved as CSV files on Drive.

## A.4 speculative decoding

| Check | Result |
|---|---|
| Toy models, 60,000 three-token continuations vs the exact target distribution | chi2 = 62.5, p = 0.867 |
| Control, draft alone vs the target distribution | chi2 = 3.3e6, p = 0 |
| Toy models, acceptance rate vs 1 - TV(p, q) | 0.7004 vs 0.7034 |
| Qwen2.5-0.5B draft, Qwen2.5-1.5B target, same tokenizer (vocabulary 151,936), one step, 200,000 samples, 133 token bins | chi2 = 135.0, p = 0.436 |
| Same, draft alone as control | chi2 = 145,405, p = 0 |
| Same, acceptance rate vs 1 - TV(p, q) | 0.8285 vs 0.8285 |
| Same, top-5 target tokens (p vs empirical) | 0.6101 / 0.6096, 0.1309 / 0.1315, 0.1012 / 0.1016, 0.0160 / 0.0161, 0.0105 / 0.0106 |
| Full loop on the real pair, 30 new tokens, gamma 4 | 11 target calls, 44 drafted, 19 accepted, `accepted / drafted` = 0.43, 2.73 new tokens per target call |

Reading the last row: 0.43 is not the per-token rate `a` of proof B.3, because tokens drafted after a rejection are never judged. The repo module counts rejections, so `a = accepted / (accepted + rejected)`. The generated text is incoherent because the loop samples at temperature 1 from a 0.5B model; that is expected and is not a test of quality.

## Not verified

The new modules `conformal.py` and `speculative.py` were written from the notebook code and checked only for syntax. Their `_self_test()` functions have **not** been run by anyone: the user asked that nothing run on their machine and chose to mark tasks 1.2, 1.3, 2.2 and 2.3 complete without running them. The Colab outputs above verify the algorithms through the notebook cells, which hold the same logic, but they do not verify these repo files. The run time of the speculative self-test (task 2.3) was therefore not measured; the README's "a minute or more" is an estimate. Before relying on the modules, run:

```bash
uv run python -c "from machinelearningproject.conformal import _self_test; _self_test()"
uv run python -c "from machinelearningproject.speculative import _self_test; _self_test()"
```

The statistical checks use fixed seeds, but a correct implementation can still fail the chi-square threshold of 0.001 about once in a thousand different seeds.
