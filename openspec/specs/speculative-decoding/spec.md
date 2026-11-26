# speculative-decoding Specification

## Purpose

Generates text with a small draft model and a large target model that share a vocabulary, verifying the draft's tokens with one target pass per round, and checks by experiment that the output follows the target's distribution.

## Requirements

### Requirement: Draft and verify loop
Each round SHALL draw a fixed number of draft tokens from the draft model, score all of them with one call of the target model, and accept each draft token with probability `min(1, p/q)`, where `p` and `q` are the target's and the draft's probabilities of that token. At the first rejection the round SHALL stop and draw the token from the normalised positive part of `p - q`. When every draft token is accepted the round SHALL add one more token drawn from the target.

#### Scenario: All drafts accepted
- **WHEN** the draft and the target give the same distribution and 4 tokens are drafted
- **THEN** each round adds 5 tokens and uses one target call

#### Scenario: Rejection
- **WHEN** the first draft token is rejected
- **THEN** the round adds exactly one token, drawn from the residual distribution, and discards the other draft tokens

### Requirement: Exact number of new tokens
The output SHALL contain the prefix followed by exactly the requested number of new tokens, cutting any surplus from the end of the last round.

#### Scenario: Surplus tokens
- **WHEN** 3 new tokens are requested and the last round produces 5
- **THEN** the output ends after the third new token

### Requirement: Shared vocabulary is required
The module SHALL raise an error naming both sizes when the draft and the target return distributions of different lengths.

#### Scenario: Different vocabularies
- **WHEN** the draft returns 5 probabilities per position and the target returns 6
- **THEN** the call fails with an error that contains both numbers

### Requirement: Output follows the target distribution
On two toy models whose sequence distribution is known exactly, the empirical distribution of 3-token continuations from 60,000 runs SHALL pass a chi-square test against the target's exact distribution at significance 0.001. As a control, samples drawn from the draft alone SHALL fail the same test at significance 1e-6.

#### Scenario: Speculative output
- **WHEN** 60,000 continuations are produced with the loop and tested against the target distribution
- **THEN** the p-value is above 0.001

#### Scenario: Control
- **WHEN** 60,000 samples drawn from the draft alone are tested against the target distribution
- **THEN** the p-value is below 1e-6

### Requirement: Acceptance rate equals one minus total variation
On the same toy models, with one draft token per round, the measured fraction of accepted draft tokens SHALL be within 0.01 of `1 - TV(p, q)`, where TV is half the sum of absolute differences of the first-step distributions.

#### Scenario: Measured acceptance
- **WHEN** 60,000 single-token rounds are run
- **THEN** the accepted fraction differs from `1 - TV(p, q)` by less than 0.01

### Requirement: Statistics count rejections
The result SHALL report the number of target calls, drafted tokens, accepted tokens and rejected tokens, and the per-token acceptance rate `accepted / (accepted + rejected)`, which excludes draft tokens that were never judged because an earlier one was rejected.

#### Scenario: Never-judged drafts
- **WHEN** a round drafts 4 tokens and the second is rejected
- **THEN** the round adds 1 accepted token and 1 rejected token to the counts, and the 2 later draft tokens count as drafted only
