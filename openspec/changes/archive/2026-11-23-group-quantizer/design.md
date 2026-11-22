# Design

## Context

See `proposal.md`. The model to test on is the trained tiny target (4 layers, width 128) and draft; the quantizer works on any 2-D `torch` weight, so the same functions apply to Qwen later. `torch` is already a dependency; there is no `pytest`, so `_self_test()` as in the other modules.

## Goals / Non-Goals

**Goals:** a quantizer small enough to read and prove properties about; numbers per layer for the report.
**Non-Goals:** speed, bit packing, activation quantization, calibration data, comparing with `bitsandbytes` (Day 4).

## Decisions

**Groups of consecutive input features within each output row; scale per group.** The weight of `nn.Linear` is (out, in); reshaping to (out, in / group, group) gives one scale per group by a max over the last axis. Alternative: per-tensor or per-row scales. Rejected: A.3 asks for group-wise, and finer groups are what makes INT4 usable.

**Symmetric: `scale = max|w| / qmax`, integers in `[-qmax-1, qmax]` with `qmax = 2^(b-1) - 1`. Asymmetric: `scale = (max(hi, 0) - min(lo, 0)) / (2^b - 1)`, integer zero point `round(-min(lo, 0) / scale)`, integers in `[0, 2^b - 1]`.** Standard definitions, so the error is at most half a step in both modes. Zero is forced into each group's range: without it an all-positive group needs a negative zero point, which a stored integer in `[0, 2^b - 1]` cannot hold, and the first version of this code clipped the top of such groups (the half-step test caught it). The price is a coarser step for a group far from zero, which weights rarely are. It also makes constant groups exact without a special case. Alternative: clip outliers for a smaller step. Rejected: it adds a hyperparameter and breaks the Δ/2 bound that B.4 proves.

**Reconstruction is stored unpacked in `int8` tensors, size is accounted by formula.** Packing two INT4 values per byte matters only for the memory number in D.4, which the formula gives exactly; kernels that use packed integers are out of scope. Alternative: real packing. Rejected for now: code and a round-trip test for no new result.

**Whole-model quantization is "fake": dequantized weights written back into a model copy.** Accuracy and loss under quantization are what D.4 needs; speed is measured separately with `bitsandbytes`. Only `nn.Linear` weights are touched, which is where nearly all parameters sit and what A.3 means by layers.

**Error properties are tested against the formulas of B.4.** `mean(Δ²/12)` computed from the actual group scales is compared with the measured mean squared error on Gaussian weights; the dot-product bound is checked per row. Gaussian weights make the within-group rounding error close to uniform, which is the assumption of B.4.

## Risks / Trade-offs

- [The Δ²/12 prediction is off for small groups or heavy tails] → the check uses Gaussian weights with 64-element groups and a 10% tolerance; a larger gap on real weights is a finding to report, not a failure of the quantizer.
- [The tiny model's width 64 draft needs group size 64 or less] → the check uses group sizes that divide every layer (32 and 64) and the error names both numbers otherwise.
