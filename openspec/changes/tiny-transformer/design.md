# Design

## Context

See `proposal.md` for why. Constraints that shape the approach:

- `torch` and `numpy` are already dependencies; `pytest` is not installed, and the converter checks itself with a `_self_test()` that runs on every use. The same pattern fits here.
- The corpus is small (Truyện Kiều is about 3,250 lines, roughly 100,000 characters), so the models overfit quickly and a held-out part is needed to see that.
- Downstream code reads the model, not this change's scripts: the A.3 quantizer walks its weights, and the A.4 speculative decoder calls it one step at a time with a cache and also reads the target's softmax. The forward signature and cache layout are therefore the interface to keep stable.
- Requirements are in `specs/tiny-transformer/spec.md`.

## Goals / Non-Goals

**Goals:**
- Two small files, readable top to bottom in one sitting, so every team member can explain the model at the defense.
- Attention written out with `matmul`, `softmax` and a mask; the library's attention is used only as a reference in the self-test.

**Non-Goals:**
- Training tricks beyond dropout and weight decay (learning-rate schedules, mixed precision, gradient accumulation), weight tying, rotary or learned positions, grouped-query attention.
- Choosing draft training steps to hit specific acceptance rates; the step count is a command-line argument and B.3/C.3 sets the values later.

## Decisions

**One module for the model and tokenizer (`tiny_transformer.py`), one script for training and sampling (`train_tiny.py`).** The model is imported by the quantizer and the speculative decoder, which must not pull in the training code. Alternative: a package with one file per component. Rejected: six files for about 150 lines.

**Pre-norm decoder block: `x + Attn(LN(x))`, then `x + MLP(LN(x))`, with a final LayerNorm.** The original paper uses post-norm; pre-norm trains reliably without warmup, which matters because we skip schedules. Alternative: post-norm as in the paper. Rejected: needs warmup to be stable.

**Linear, LayerNorm and Embedding from `torch.nn` are allowed; only attention is hand-written.** The assignment forbids built-in attention modules, not layers. Heads are formed by reshaping one combined Q/K/V projection to (B, H, T, d_head), so the quantizer sees ordinary `nn.Linear` weights, which is what A.3 targets.

**Cache is a list with one `(k, v)` pair per layer, each of shape (B, H, T_cached, d_head). `forward(ids, cache=None)` returns `(logits, new_cache)`.** Without a cache the call is a normal full pass. With a cache, `ids` holds only the new characters, positions start at the cached length, and the causal mask is offset so new positions see all cached ones. Alternative: a stateful cache object inside the model. Rejected: the speculative decoder must roll back rejected draft tokens, which is trivial with a returned value (slice or drop it) and error-prone with hidden state.

**Position encoding is a fixed table `sin/cos(pos / 10000^(2i/d))` computed once and stored as a non-trainable buffer.** Matches the paper's formula and the spec; it is not a parameter, so the quantizer never touches it.

**Sizes are module constants.** Target: 4 layers, width 128, 4 heads. Draft: 2 layers, width 64, 2 heads. Context length 128, dropout 0.1, AdamW at lr 3e-3, batch 64, 3,000 steps. These are starting values meant to train in minutes on CPU or a free GPU. The first run showed one thing they must change (next decision); beyond that there is no tuning pass.

**The target is regularised more than the draft: dropout 0.3 and weight decay 0.1 for the target, 0.1 and 0.01 for the draft, evaluation every 100 steps.** In the first run (`notes.md`) the target overfit after step 500 and ended with a validation loss of 1.611 against the draft's 1.589. B.3/C.3 control the acceptance rate by training the draft for more or fewer steps, which only works if the target is clearly the stronger model. Criterion: saved target validation loss at least 0.05 below the saved draft's. Capped at four runs on the target alone; the draft stays as it is, so its 1.589 is the fixed reference. Alternatives: a smaller draft (changes a model other work may already use) or more data (the corpus is already the whole poem). Rejected both.

**Validation split: the last 10% of the text, taken contiguously.** Random character windows from the whole text would leak neighbouring lines between train and validation. Alternative: random split. Rejected for that leakage.

**Tokenizer: sorted distinct characters of the whole corpus, saved as JSON next to the weights.** Fitting the vocabulary on the full text, validation part included, keeps one vocabulary for both models and avoids unknown characters at validation. It does not leak labels, since a vocabulary only lists symbols.

**Sampling takes a seed and uses a local `torch.Generator`.** This makes "same seed, same text" independent of other random calls and lets the cached and uncached samplers be compared exactly.

**Self-check: `_self_test()` in `tiny_transformer.py`, run at the start of training.** It covers the three attention scenarios (no look-ahead, weights sum to 1 with a zero upper triangle, agreement with the library at atol 1e-5), cached versus full logits at atol 1e-4, the tokenizer round trip and the unknown-character error. Alternative: add pytest. Deferred, as in the converter.

## Risks / Trade-offs

- [Overfitting on about 100,000 characters: validation loss rises while training loss falls] → report both losses, keep dropout at 0.1, save the best-validation weights; the models only need to beat the ln V uniform baseline and write plausible Vietnamese, not generalise.
- [Cached and full logits differ by more than rounding if the mask offset is wrong] → the self-test checks every position, not only the last.
- [The draft and target end up too similar or too different for the acceptance-rate experiments] → the step count and sizes are constants and arguments; B.3/C.3 can retrain the draft without touching the model code.
- [The corpus file is not supplied] → training stops with an error naming the file; the README states where to put it. Obtaining the text is a manual step because the source and its license are not something the code should guess.

## Open Questions

- Which edition of the text to use (diacritics and line-break conventions differ between editions). It changes the vocabulary size by a few characters at most, so it can be settled when the file is added.
