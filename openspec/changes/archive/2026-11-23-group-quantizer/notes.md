# Notes

## Check on the trained tiny models (tasks 3.1 and 3.2), 2026-11-22

Group size 64, weights of every `nn.Linear` quantized, embeddings and LayerNorm in FP32, validation loss measured on the same 20 fixed windows as in `tiny-transformer` (the FP32 numbers, 1.5566 and 1.5887, match that change's notes).

| model | setting | validation loss | change | packed size vs FP32 | measured MSE / mean(step^2/12) |
| --- | --- | --- | --- | --- | --- |
| target | FP32 | 1.5566 | | | |
| target | INT8 sym | 1.5567 | +0.0001 | 25.8% | 0.98 .. 0.99 |
| target | INT8 asym | 1.5569 | +0.0002 | 26.2% | 0.99 .. 1.01 |
| target | INT4 sym | 1.5650 | +0.0084 | 13.3% | 0.97 .. 0.99 |
| target | INT4 asym | 1.5635 | +0.0069 | 13.5% | 0.99 .. 1.01 |
| draft | FP32 | 1.5887 | | | |
| draft | INT8 sym | 1.5891 | +0.0004 | 25.8% | 0.96 .. 1.00 |
| draft | INT8 asym | 1.5890 | +0.0003 | 26.2% | 0.99 .. 1.01 |
| draft | INT4 sym | 1.6261 | +0.0374 | 13.3% | 0.97 .. 1.00 |
| draft | INT4 asym | 1.6055 | +0.0168 | 13.5% | 0.98 .. 1.02 |

What it shows:

- **B.4 holds on real weights.** The measured mean squared error is within 4% of the mean of step^2/12 in every layer and setting (ratio 0.96 to 1.02), not only on the Gaussian test matrix. The per-layer relative error at INT4 is about 10% to 13%.
- **INT8 is free, INT4 costs something small.** INT8 moves the loss by at most 0.0004. INT4 moves it by 0.007 to 0.008 for the target and 0.017 to 0.037 for the draft.
- **The smaller model is more sensitive.** The draft loses 4 to 5 times as much loss at INT4 as the target, which fits a model with less redundancy. This matters for D.4: quantizing the draft hurts the acceptance rate more than quantizing the target.
- **Asymmetric is better at INT4, equal at INT8.** For the draft, INT4 asymmetric loses 0.017 and symmetric 0.037; for the target 0.007 and 0.008. Asymmetric uses the full integer range when a group's values are not centred on zero, and it costs one zero point per group (13.5% against 13.3% of FP32).
- **The packed size follows the formula.** INT8 about 26% and INT4 about 13% of FP32, the excess over 25% and 12.5% being the 16-bit scales (and zero points).

Largest single-weight error at INT4 symmetric: 0.061 in `blocks.3.mlp.2` of the target, about twice the other layers' 0.03 to 0.04; that layer's weights have a heavier tail.

## Finding while building it

The first asymmetric version clamped the zero point to the integer range and clipped the top of all-positive groups; the half-step self-test caught it. Fix: force 0 into each group's range, recorded in `design.md`.

## Not done

Comparison with `bitsandbytes` and the latency and memory runs on Qwen are D.4 work (Day 4) and need a GPU with more than 4 GB for the 3B models.
