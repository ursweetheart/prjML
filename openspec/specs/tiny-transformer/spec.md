# tiny-transformer Specification

## Purpose
A small character-level decoder-only language model pair (a larger target and a smaller draft) written from scratch, used as the controlled test bed for quantization and speculative decoding, where the exact output distribution can be computed over a vocabulary of about 100 characters.

## Requirements

### Requirement: Causal self-attention
The model SHALL compute scaled dot-product attention with a causal mask, so the output at position t depends only on inputs at positions 0 to t. The attention SHALL be implemented without the library's built-in attention modules and SHALL match the library's reference attention numerically.

#### Scenario: No information flows from the future
- **WHEN** the key and value inputs at the last position are replaced by other values
- **THEN** the attention outputs at all earlier positions are unchanged

#### Scenario: Weights form a distribution over visible positions
- **WHEN** attention is computed for a sequence of length T
- **THEN** each row of weights sums to 1 and every weight above the diagonal is exactly 0

#### Scenario: Agreement with the standard implementation
- **WHEN** the same random queries, keys and values are given to this attention and to the library's causal scaled dot-product attention
- **THEN** the two outputs agree within an absolute tolerance of 1e-5

### Requirement: Position encoding without parameters
The model SHALL add sinusoidal position encodings to the character embeddings, with no learned position parameters, so the encoding is fixed by the model width and position alone.

#### Scenario: Encoding is deterministic
- **WHEN** the encoding is computed twice for the same width and length
- **THEN** both results are identical and no trainable parameter belongs to it

### Requirement: Next-character logits
The model SHALL map a batch of character ids of shape (B, T) to logits of shape (B, T, V), where V is the vocabulary size, and SHALL reject a sequence longer than the model's maximum context length.

#### Scenario: Output shape
- **WHEN** a batch of 2 sequences of 16 ids is given to a model with vocabulary size 100
- **THEN** the logits have shape (2, 16, 100)

#### Scenario: Sequence too long
- **WHEN** a sequence longer than the maximum context length is given
- **THEN** the model raises an error and returns no logits

### Requirement: KV cache for incremental decoding
The model SHALL accept and return a per-layer cache of keys and values so that decoding one new character reuses earlier computation. Logits from cached decoding SHALL equal the logits of a full forward pass over the same characters, and the cache SHALL expose its key and value tensors for reading.

#### Scenario: Cached and full decoding agree
- **WHEN** a sequence is decoded one character at a time with the cache and, separately, in one full forward pass
- **THEN** the logits at every position agree within an absolute tolerance of 1e-4

#### Scenario: Cache grows by one position per step
- **WHEN** one character is decoded with a cache that holds n positions
- **THEN** the returned cache holds n + 1 positions in every layer

### Requirement: One shared character tokenizer
The tokenizer SHALL be built once from the training text as the sorted set of its distinct characters, and the target and draft models SHALL use this same tokenizer. Encoding followed by decoding SHALL return the original text, and a character outside the vocabulary SHALL raise an error.

#### Scenario: Round trip
- **WHEN** a text made of vocabulary characters is encoded and decoded
- **THEN** the result equals the original text

#### Scenario: Unknown character
- **WHEN** a text contains a character that is not in the vocabulary
- **THEN** encoding raises an error naming that character

### Requirement: Target and draft pair
The project SHALL provide two models with the same vocabulary and maximum context length, where the draft has fewer layers and a smaller width than the target.

#### Scenario: Compatible sizes
- **WHEN** the target and the draft are created
- **THEN** their vocabulary sizes are equal and the draft has fewer parameters than the target

### Requirement: Reproducible training and saved artifacts
Training SHALL read the corpus from `data/raw/truyen_kieu.txt`, use a fixed random seed, and save each model's weights and the shared vocabulary under `data/processed/tiny/`. A model loaded from the saved files SHALL give the same logits as the model that was saved. Training SHALL stop with an error if the corpus file is missing.

#### Scenario: Training lowers the loss
- **WHEN** a model is trained on the corpus
- **THEN** the final validation loss is lower than the loss of a uniform guess over the vocabulary, which is ln V

#### Scenario: Save and reload
- **WHEN** a trained model is saved and loaded back
- **THEN** both give identical logits for the same input

#### Scenario: Missing corpus
- **WHEN** `data/raw/truyen_kieu.txt` does not exist
- **THEN** training exits with a non-zero status and a message naming the missing file, and writes no weights

### Requirement: Sampling
The system SHALL generate text from a trained model by sampling the next character from its softmax distribution, with a temperature, and SHALL produce the same text for the same seed. Generated text SHALL contain only vocabulary characters.

#### Scenario: Seeded sampling is repeatable
- **WHEN** text is sampled twice with the same prompt, seed and temperature
- **THEN** the two texts are identical

#### Scenario: Cache does not change the sample
- **WHEN** the same text is sampled with and without the KV cache using the same seed
- **THEN** the two texts are identical
