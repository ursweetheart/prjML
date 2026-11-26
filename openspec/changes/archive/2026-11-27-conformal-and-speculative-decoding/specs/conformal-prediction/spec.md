# Spec Delta

## Purpose

Turns the option probabilities of a multiple-choice scorer into prediction sets with a coverage guarantee, using split conformal prediction with two nonconformity scores, so that Part D can report sets, coverage and set size per subject.

## ADDED Requirements

### Requirement: Two nonconformity scores
For an `(N, 4)` array of option probabilities, the module SHALL give an `(N, 4)` array of scores, one per option. The first score SHALL be `1 - p`. The second, adaptive score (APS) SHALL be the probability mass of the options ranked above the option plus a uniform random number in [0, 1) times the option's own probability.

#### Scenario: Plain score
- **WHEN** the probabilities of a question are `0.5, 0.3, 0.15, 0.05`
- **THEN** the plain scores are `0.5, 0.7, 0.85, 0.95`

#### Scenario: Adaptive score without randomness
- **WHEN** the probabilities are `0.5, 0.3, 0.15, 0.05` and the random number is 0
- **THEN** the adaptive scores are `0, 0.5, 0.8, 0.95`

### Requirement: Calibration threshold
The threshold SHALL be the `ceil((n+1)(1-alpha))`-th smallest of the `n` scores of the correct answers on the calibration questions. When that rank exceeds `n`, the threshold SHALL be infinite, so every option is kept.

#### Scenario: Ten calibration scores
- **WHEN** ten calibration scores `0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.50, 0.60, 0.70, 0.90` are given with alpha 0.2
- **THEN** the threshold is `0.70`

#### Scenario: Too few calibration questions
- **WHEN** only 3 calibration scores are given with alpha 0.1
- **THEN** the threshold is infinite

### Requirement: Prediction sets
For each test question the prediction set SHALL contain every option whose score is at most the threshold. The result SHALL include, for each test question, whether the correct answer is in its set.

#### Scenario: Worked example
- **WHEN** the threshold is `0.70` and a question has probabilities `0.5, 0.3, 0.15, 0.05`
- **THEN** the set is options A and B

### Requirement: Coverage on calibrated probabilities
When the probabilities are exactly the distribution the answers are drawn from, and calibration and test questions are exchangeable, the empirical coverage on the test questions SHALL be within 0.02 of `1 - alpha` at alpha 0.1, for both scores, in the module's self-check on 10,000 calibration and 10,000 test questions.

#### Scenario: Self-check on simulated data
- **WHEN** the self-check draws probabilities from a uniform Dirichlet distribution and the answers from those probabilities
- **THEN** both scores give a coverage between 0.88 and 0.92 at alpha 0.1

### Requirement: Reproducible sets
With the same random seed, the same inputs SHALL give identical prediction sets. The plain score SHALL not depend on the seed.

#### Scenario: Same seed
- **WHEN** the adaptive procedure is run twice with the same seed
- **THEN** the two sets of prediction sets are identical

### Requirement: Set size and empty sets are reported
Alongside coverage, the module SHALL provide the mean set size and the fraction of empty sets, since the plain score can return an empty set.

#### Scenario: All options kept
- **WHEN** the threshold is infinite
- **THEN** the mean set size is 4 and the coverage is 1
