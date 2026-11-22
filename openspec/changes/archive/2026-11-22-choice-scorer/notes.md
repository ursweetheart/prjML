# Notes

## Check run (tasks 2.2 and 2.3), 2026-11-22

Model `Qwen/Qwen2.5-0.5B`, float16 on an RTX 3050 (4 GB), 100 test-split questions drawn with seed 0 from the benchmark of 15,827 records. `lm-eval` 0.4.13 with `transformers` 5.23.0 and `torch` 2.15.0 (adding the dev dependency changed neither).

| | accuracy |
| --- | --- |
| this scorer | 0.320 |
| lm-eval, same prompts | 0.320 |

Predicted letters differ on 0 of 100 questions. Tolerance 0.01: pass.

Option shifts (scores mapped back to the original option order):

| shift | accuracy |
| --- | --- |
| 0 (original) | 0.320 |
| 1 | 0.340 |
| 2 | 0.330 |
| 3 | 0.380 |
| mean of the four | 0.370 |

Position bias at shift 0, predicted share / correct share: A 0.60 / 0.28, B 0.17 / 0.22, C 0.11 / 0.26, D 0.12 / 0.24. The model puts the top probability on `A` for 60% of the questions while `A` is correct for 28%, and averaging the four shifts raises accuracy from 0.32 to 0.37. This is the position bias the permutation is meant to expose. Chance is about 0.25; a 0.5B model is weak on Vietnamese exams, so this run validates the scoring path, not the model.

## Findings while building it

- Batch size 1 vs 8: float32 differs by 1.4e-5 and no predicted letter changes, so the left padding is correct. In float16 the same comparison differs by 0.011, which is rounding noise, again with no predicted letter changed. The first version of the check asserted 1e-2 in float16 and failed on this; the assertion now runs in float32 and the spec says so.
- `lm-eval` builds full-vocabulary log-softmax tensors for a whole batch, so with the 151,000-token Qwen vocabulary its batch of 8 ran out of memory on the 4 GB GPU. The check uses batch size 1 for `lm-eval`. This scorer keeps only the last position's logits (`logits_to_keep=1`) and does not have the problem, which matters for the long English reading passages on Colab.
- The 3B-class models are not run here: they need about 6 GB in float16 (the development GPU has 4 GB), and Llama-3.2-3B and Gemma-2-2B are gated on Hugging Face and answered 401 without a token. Qwen2.5-3B is open. Running all three needs Colab and a Hugging Face token with the two licenses accepted.
