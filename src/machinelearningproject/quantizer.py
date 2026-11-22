"""A.3: group-wise post-training weight quantization, 8 or 4 bits, symmetric or asymmetric.

Spec: openspec/specs/group-quantizer/spec.md (while in flight: openspec/changes/group-quantizer/)
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Quantized:
    q: torch.Tensor  # (out, in): int8 for symmetric, uint8 for asymmetric (8-bit asymmetric needs 0..255)
    scale: torch.Tensor  # (out, groups) float32; the step size of each group
    zero: torch.Tensor | None  # (out, groups) uint8, asymmetric only
    bits: int
    group_size: int


def _grouped(w, group_size):
    out_f, in_f = w.shape
    if in_f % group_size:
        raise ValueError(f"group size {group_size} does not divide {in_f} input features")
    return w.reshape(out_f, in_f // group_size, group_size)


def quantize(w, bits=4, group_size=64, symmetric=True):
    """Quantize a (out, in) weight; each group of group_size consecutive inputs in a row has its own scale."""
    g = _grouped(w.float(), group_size)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = g.abs().amax(-1, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # all-zero group: any scale reconstructs 0
        q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax).to(torch.int8)
        return Quantized(q.reshape(w.shape), scale.squeeze(-1), None, bits, group_size)
    qmax = 2**bits - 1
    # 0 is kept inside [lo, hi] so the zero point is an integer in 0..qmax; without it an all-positive group would need a
    # negative zero point and its top would be clipped. The cost is a wider step for groups far from zero.
    lo, hi = g.amin(-1, keepdim=True).clamp(max=0), g.amax(-1, keepdim=True).clamp(min=0)
    scale = (hi - lo) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # all-zero group
    zero = torch.round(-lo / scale)  # in 0..qmax because lo <= 0 <= hi
    q = torch.clamp(torch.round(g / scale + zero), 0, qmax).to(torch.uint8)
    return Quantized(q.reshape(w.shape), scale.squeeze(-1), zero.squeeze(-1).to(torch.uint8), bits, group_size)


def dequantize(t):
    out_f, in_f = t.q.shape
    q = t.q.float().reshape(out_f, in_f // t.group_size, t.group_size)
    if t.zero is not None:
        q = q - t.zero.float().unsqueeze(-1)
    return (q * t.scale.unsqueeze(-1)).reshape(out_f, in_f)


def packed_bytes(t):
    """Bytes a packed representation needs: integers at their bit width, 16-bit scales, and zero points at the bit width."""
    groups = t.scale.numel()
    bits = t.q.numel() * t.bits + groups * 16 + (groups * t.bits if t.zero is not None else 0)
    return bits / 8


def quantize_model(model, bits=4, group_size=64, symmetric=True):
    """Copy of the model with every nn.Linear weight replaced by its dequantized value, and one error row per layer.

    Embeddings, biases and LayerNorm stay in full precision; the original model is not touched.
    """
    qmodel, rows = copy.deepcopy(model), []
    for name, mod in qmodel.named_modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data.clone()
            t = quantize(w, bits, group_size, symmetric)
            wq = dequantize(t)
            err = wq - w.float()
            mod.weight.data = wq.to(w.dtype)
            rows.append({"name": name, "shape": tuple(w.shape), "rmse": err.pow(2).mean().sqrt().item(),
                         "max_abs": err.abs().max().item(), "rel": (err.norm() / w.float().norm()).item(),
                         "bytes": packed_bytes(t), "fp32_bytes": w.numel() * 4})
    return qmodel, rows


def _self_test():
    torch.manual_seed(0)
    step = lambda t: t.scale.repeat_interleave(t.group_size, dim=1)  # per-element step size

    # symmetric 4-bit round trip: shape, integer range, half-step bound
    w = torch.randn(64, 256)
    t = quantize(w, 4, 64, True)
    wq = dequantize(t)
    assert wq.shape == w.shape and t.q.min() >= -8 and t.q.max() <= 7
    assert ((w - wq).abs() <= step(t) / 2 + 1e-6).all()
    # asymmetric 8-bit on all-positive values: 0..255 and the same half-step bound
    wp = torch.rand(32, 128) + 1
    tp = quantize(wp, 8, 64, False)
    assert tp.q.dtype == torch.uint8 and ((wp - dequantize(tp)).abs() <= step(tp) / 2 + 1e-6).all()
    # more bits, less error, in both modes
    for sym in (True, False):
        e8 = (w - dequantize(quantize(w, 8, 64, sym))).pow(2).mean()
        e4 = (w - dequantize(quantize(w, 4, 64, sym))).pow(2).mean()
        assert e8 < e4

    # degenerate groups: zero rows, constant groups (even large ones) and a group size that does not divide
    for sym in (True, False):
        wz = torch.randn(4, 128)
        wz[0] = 0
        wz[1, :64] = 300.0
        wz[2, :64] = -3.0
        out = dequantize(quantize(wz, 8, 64, sym))
        assert torch.isfinite(out).all() and (out[0] == 0).all()
        assert torch.allclose(out[1, :64], wz[1, :64]) and torch.allclose(out[2, :64], wz[2, :64], atol=1e-5)
    try:
        quantize(torch.randn(4, 100), 4, 64)
        raise AssertionError("group size 64 was accepted for 100 inputs")
    except ValueError as e:
        assert "100" in str(e) and "64" in str(e)

    # B.4: mean squared error is the mean of step^2 / 12, and the dot-product error obeys sum of step/2 * ||x||_1
    big, x = torch.randn(512, 512), torch.randn(512)
    for sym in (True, False):
        tb = quantize(big, 4, 64, sym)
        mse = (big - dequantize(tb)).pow(2).mean().item()
        predicted = (tb.scale**2 / 12).mean().item()
        assert abs(mse - predicted) / predicted < 0.10, (sym, mse, predicted)
        l1 = x.reshape(-1, 64).abs().sum(-1)  # L1 norm of the input per group
        bound = (tb.scale / 2) @ l1
        assert ((big @ x - dequantize(tb) @ x).abs() <= bound + 1e-4).all()

    # storage: 128x128 at 4 bits, group 64, symmetric = 8192 bytes of integers + 512 bytes of scales
    assert packed_bytes(quantize(torch.randn(128, 128), 4, 64, True)) == 8192 + 512

    # model level: one row per Linear, original untouched, embeddings and norms untouched, 8-bit closer than 4-bit
    net = nn.Sequential(nn.Embedding(50, 64), nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 32))
    before = [p.clone() for p in net.parameters()]
    q8, rows8 = quantize_model(net, 8, 64)
    q4, rows4 = quantize_model(net, 4, 64)
    assert len(rows8) == 2 and all(torch.equal(a, b) for a, b in zip(before, net.parameters()))
    assert torch.equal(q4[0].weight, net[0].weight) and torch.equal(q4[2].weight, net[2].weight)
    ids = torch.randint(0, 50, (4, 10))
    ref = net(ids)
    assert (q8(ids) - ref).abs().max() < (q4(ids) - ref).abs().max()
