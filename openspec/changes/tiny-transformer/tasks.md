# Tasks

## 1. Attention and positions

- [x] 1.1 Create `src/machinelearningproject/tiny_transformer.py` with scaled dot-product attention and the causal mask, plus a `_self_test()` that checks the three attention scenarios in the spec (no look-ahead, rows sum to 1 with a zero upper triangle, agreement with `F.scaled_dot_product_attention(is_causal=True)` at atol 1e-5); verify `uv run python -c "from machinelearningproject.tiny_transformer import _self_test; _self_test()"` passes
- [x] 1.2 Add the multi-head wrapper (one combined Q/K/V `nn.Linear`, reshape to heads, output projection) and the fixed sinusoidal position table as a non-trainable buffer, with self-test asserts that the table is deterministic and has no parameters and that a multi-head output keeps shape (B, T, width); verify the self-test passes

## 2. Model and KV cache

- [x] 2.1 Add the pre-norm decoder block, the full model (embedding, positions, blocks, final LayerNorm, output head) with `forward(ids, cache=None)` returning `(logits, new_cache)` and the maximum-length check, plus asserts for the logits shape `(2, 16, 100)` and the error on an over-long sequence; verify the self-test passes
- [x] 2.2 Implement the per-layer `(k, v)` cache with the offset causal mask and position offset, plus asserts that one-character-at-a-time logits equal the full-pass logits at every position (atol 1e-4) and that the cache grows by one position per step in every layer; verify the self-test passes

## 3. Tokenizer and model sizes

- [x] 3.1 Add the character tokenizer (sorted distinct characters, `encode`, `decode`, JSON save and load) with asserts for the round trip and the error naming an unknown character; verify the self-test passes
- [x] 3.2 Add the target and draft size constants and a builder for each, with an assert that both share the vocabulary size and context length and that the draft has fewer parameters; verify the self-test passes

## 4. Training and sampling

- [x] 4.1 Create `src/machinelearningproject/train_tiny.py`: read `data/raw/truyen_kieu.txt` (exit non-zero with a message naming the file if it is missing), build the tokenizer, hold out the last 10% for validation, train with a fixed seed and a `--steps` argument, print training and validation loss, and save the best-validation weights and the vocabulary under `data/processed/tiny/` through a temporary file and rename; verify with a short 50-step run on a few lines of text that the files appear, and that running without the corpus exits non-zero and writes nothing
- [x] 4.2 Add sampling with temperature and a local seeded `torch.Generator`, with and without the KV cache, and a save/reload check; verify with asserts that two runs with the same seed give the same text, that the cached and uncached texts are identical, and that reloaded weights give identical logits

## 5. Real run and documentation

- [x] 5.1 Place the Truyện Kiều text at `data/raw/truyen_kieu.txt` (the source and edition are recorded in the change's notes), train the target and the draft with the default constants, and verify the final validation loss of each is below ln V and that a printed 300-character sample is plausible Vietnamese
- [x] 5.2 Add a "Tiny Transformer" section to `README.md` with where to put the corpus, the training command `uv run python -m machinelearningproject.train_tiny` and the output location; verify the command runs as written from a fresh shell

## 6. Make the target clearly stronger than the draft

- [x] 6.1 Move dropout into the per-model size settings (target 0.3, draft 0.1), add a per-model weight decay (target 0.1, draft 0.01, the AdamW default) and evaluate every 100 steps, leaving the draft's settings otherwise unchanged; verify both self-tests pass and a 50-step run on a few lines still writes the files
- [x] 6.2 Retrain only the target (`--models target`), at most four runs, adjusting dropout, weight decay and learning rate between runs; record each run's settings and best validation loss in `notes.md`; verify the saved target's best validation loss is at least 0.05 lower than the saved draft's (1.589), or stop after four runs and record the best result and why the gap was not reached
