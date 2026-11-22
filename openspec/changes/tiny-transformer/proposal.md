# Proposal

## Why

A.4 (speculative decoding) and A.3/B.5 (quantization, KV cache) must be checked against an exactly known output distribution, which needs a vocabulary of about 100 characters, not the 150,000 tokens of the pretrained models. The project proposal calls this the tiny Transformer test bed (Topic 9 A.1–A.2), and nothing in the repo implements it yet. It was planned for Day 2 and is now late; the A.3 quantizer and the A.4 speculative decoder both need its weights and its KV cache.

## What Changes

- Add a minimal decoder-only Transformer in PyTorch, character-level, written without `nn.MultiheadAttention` or `F.scaled_dot_product_attention`: scaled dot-product attention with a causal mask, multi-head attention, sinusoidal position encoding, a pre-norm decoder block, and a KV cache.
- Add a character tokenizer built from the training text, shared by the target and draft models.
- Add a training script that trains a larger target and a smaller draft model on Truyện Kiều and saves their weights and the shared vocabulary.
- Add a generation function that samples text with or without the KV cache; the two must give the same logits.
- Check the attention against `F.scaled_dot_product_attention` in the self-test, as the assignment asks for a comparison with the standard library.

Working assumptions: the Truyện Kiều text (public domain, 1820) is supplied as a plain-text file at `data/raw/truyen_kieu.txt`, the same way the VNHSGE clone is supplied; the code does not download anything. Model sizes and training steps are constants in the module, with the target at 4 layers, width 128, 4 heads and the draft at 2 layers, width 64, 2 heads.

## Capabilities

### New Capabilities

- `tiny-transformer`: the character-level decoder-only Transformer pair (target and draft): its components, the KV-cache contract that speculative decoding and the quantizer rely on, the shared tokenizer, and the saved artifacts.

### Modified Capabilities

None.

## Impact

- New code: `src/machinelearningproject/tiny_transformer.py` (model and tokenizer) and `src/machinelearningproject/train_tiny.py` (training and sampling), run with `uv run python -m machinelearningproject.train_tiny`.
- New files, git-ignored under `data/`: `data/raw/truyen_kieu.txt` (input) and `data/processed/tiny/` (weights and vocabulary).
- No new dependencies: `torch` is already in `pyproject.toml`.
- Downstream: A.3 reads the weights of these models; A.4 calls the model with a KV cache and uses the target/draft pair, so the forward signature becomes their contract.
- Out of scope: the quantizer, speculative decoding itself, and the pretrained-model code.
