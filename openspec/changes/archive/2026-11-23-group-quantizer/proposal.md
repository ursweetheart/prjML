# Proposal

## Why

A.3 asks for post-training weight quantization at 8 and 4 bits, symmetric and asymmetric, group-wise, with the reconstruction error per layer, and Part D must use it in the pipeline (D.4: accuracy, ECE, latency and memory at FP16, INT8 and INT4). It was due on Day 2 and nothing implements it. The tiny Transformer's trained weights now exist, so it can be tested there first, as the plan says, and its error claims are exactly the statements to prove in B.4 (mean squared error Δ²/12, and the dot-product bound Δ/2·‖x‖₁).

## What Changes

- Add a group-wise weight quantizer for 2-D weights: 8 or 4 bits, symmetric or asymmetric, with a configurable group size along the input dimension, and the matching dequantizer.
- Add whole-model "fake" quantization: every `nn.Linear` weight of a model copy is replaced by its dequantized value, with a per-layer reconstruction-error report; embeddings and LayerNorm stay in full precision.
- Report the storage a packed representation would need, so D.4 can state memory without implementing bit packing.
- Add a check command that quantizes the trained tiny target and draft, prints the per-layer errors and the validation loss at FP32, INT8 and INT4 in both modes, and verifies numerically the two B.4 claims.

Working assumptions: groups run along the input dimension of each weight row (as in common INT4 schemes); the default group size is 64; scales are kept in float32 in the code and counted as 16-bit in the storage report; weights are stored unpacked in `int8` tensors in memory.

## Capabilities

### New Capabilities

- `group-quantizer`: group-wise post-training weight quantization (8/4 bits, symmetric/asymmetric): quantize and dequantize, per-layer reconstruction error, storage accounting, and the error properties B.4 proves.

### Modified Capabilities

None.

## Impact

- New code: `src/machinelearningproject/quantizer.py` and `src/machinelearningproject/check_quantizer.py`, run with `uv run python -m machinelearningproject.check_quantizer`.
- No new dependencies; the check reads `data/processed/tiny/` (see the Tiny Transformer section of the README).
- Downstream: D.4 applies the same quantizer to the pretrained models; the comparison with `bitsandbytes` and the latency and memory runs are not part of this change.
- Out of scope: activation quantization, outlier handling, GPTQ-style calibration, real bit packing, integer matmul kernels.
