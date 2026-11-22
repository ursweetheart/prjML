# Tasks

## 1. Quantizer

- [x] 1.1 Create `src/machinelearningproject/quantizer.py` with `quantize` and `dequantize` for 2-D weights (8/4 bits, symmetric/asymmetric, group size along the input dimension), handling zero and constant groups and rejecting a group size that does not divide the input features, plus a `_self_test()` for the round-trip, integer range, half-step, more-bits-less-error, zero-group and group-size scenarios; verify `uv run python -c "from machinelearningproject.quantizer import _self_test; _self_test()"` passes
- [x] 1.2 Add the storage-size function and the two B.4 checks (mean squared error against the mean of Δ²/12 within 10% on a 512 by 512 Gaussian matrix, and the dot-product bound per row), with asserts in the self-test; verify the self-test passes

## 2. Model-level quantization

- [x] 2.1 Add `quantize_model` (copy of the model with every `nn.Linear` weight replaced by its dequantized value, one report row per layer with RMSE, max absolute error and relative Frobenius error) with asserts that the original is unchanged, the row count equals the linear-layer count and 8-bit logits are closer to the original than 4-bit on a small random model; verify the self-test passes

## 3. Check on the tiny models

- [x] 3.1 Create `src/machinelearningproject/check_quantizer.py`: load the trained tiny target and draft, quantize at 8 and 4 bits in both modes with group size 64, print the per-layer error table for the target at 4 bits and, for each model and setting, the validation loss, its change against FP32 and the packed size against the FP32 size; verify the command runs and the 8-bit loss differs from FP32 by less than the 4-bit loss does
- [x] 3.2 Record the table, the Δ²/12 comparison and any surprise in `notes.md`, and add a "Quantizer check" section to `README.md`; verify the documented command runs as written from a fresh shell
