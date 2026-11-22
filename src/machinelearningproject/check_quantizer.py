"""A.3 check on the trained tiny models: per-layer error, validation loss and size at INT8 and INT4.

Run: uv run python -m machinelearningproject.check_quantizer   (needs data/processed/tiny, see README)
Spec: openspec/specs/group-quantizer/spec.md (while in flight: openspec/changes/group-quantizer/)
"""

import sys

import torch
import torch.nn as nn

from machinelearningproject.quantizer import dequantize, quantize, quantize_model
from machinelearningproject.tiny_transformer import Tokenizer, load_model
from machinelearningproject.train_tiny import CORPUS, OUT, VAL_FRACTION, estimate

GROUP = 64
SETTINGS = [(8, True), (8, False), (4, True), (4, False)]


def label(bits, symmetric):
    return f"INT{bits} {'sym ' if symmetric else 'asym'}"


def uniform_model_ratio(model, bits, symmetric):
    """Measured mean squared error over the predicted mean of step^2 / 12 (B.4), per linear layer."""
    ratios = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            t = quantize(mod.weight.data, bits, GROUP, symmetric)
            mse = (mod.weight.data - dequantize(t)).pow(2).mean().item()
            ratios.append(mse / (t.scale**2 / 12).mean().item())
    return min(ratios), max(ratios)


def main():
    tok = Tokenizer.load(OUT / "vocab.json")
    data = torch.tensor(tok.encode(CORPUS.read_text(encoding="utf-8")))
    val = data[int(len(data) * (1 - VAL_FRACTION)):]
    ok = True
    for kind in ("target", "draft"):
        model = load_model(OUT / f"{kind}.pt")
        base = estimate(model, val, 128, "cpu")
        fp32 = sum(m.weight.numel() * 4 for m in model.modules() if isinstance(m, nn.Linear))
        print(f"\n{kind}: FP32 validation loss {base:.4f}, linear weights {fp32 / 1e6:.2f} MB")
        print(f"  {'setting':11} {'val loss':>9} {'change':>8} {'packed MB':>10} {'vs FP32':>8}   mse / (step^2/12) over layers")
        change = {}
        for bits, sym in SETTINGS:
            qmodel, rows = quantize_model(model, bits, GROUP, sym)
            loss = estimate(qmodel, val, 128, "cpu")
            packed = sum(r["bytes"] for r in rows)
            lo, hi = uniform_model_ratio(model, bits, sym)
            change[(bits, sym)] = loss - base
            print(f"  {label(bits, sym):11} {loss:9.4f} {loss - base:+8.4f} {packed / 1e6:10.2f} {packed / fp32:8.1%}   {lo:.2f} .. {hi:.2f}")
            if kind == "target" and (bits, sym) == (4, True):
                table = rows
        for sym in (True, False):  # eight bits must hurt less than four, in both modes
            if not abs(change[(8, sym)]) < abs(change[(4, sym)]):
                ok = False
                print(f"  FAIL: INT8 {'sym' if sym else 'asym'} changes the loss by {change[(8, sym)]:+.4f}, not less than INT4's {change[(4, sym)]:+.4f}")
        if kind == "target":
            print(f"\nper-layer error of the target at {label(4, True)}, group size {GROUP}:")
            print(f"  {'layer':22} {'shape':>12} {'rmse':>9} {'max abs':>9} {'relative':>9}")
            for r in table:
                print(f"  {r['name']:22} {str(r['shape']):>12} {r['rmse']:9.5f} {r['max_abs']:9.5f} {r['rel']:9.2%}")
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
