# choice-scorer Specification

## Purpose
Scores multiple-choice questions with a causal language model by the probability it gives each option letter, with option permutation to measure position bias, so that accuracy, calibration and conformal prediction all read one consistent set of option probabilities.

## Requirements

### Requirement: Fixed prompt per record
The scorer SHALL build the prompt from a record's subject, question and four options in one fixed format that ends with the answer cue, and the same record and option order SHALL always give the same prompt text.

#### Scenario: Prompt layout
- **WHEN** a record with subject `Geography`, a question and options a to d is formatted
- **THEN** the prompt contains the subject, the question, the four options on separate lines labelled `A.` to `D.`, and ends with the answer cue

#### Scenario: Deterministic
- **WHEN** the same record is formatted twice
- **THEN** both prompts are identical strings

### Requirement: Option probabilities over A to D
For each prompt the scorer SHALL return four log-probabilities, one per option letter, taken from the model's next-token distribution at the end of the prompt and renormalised over those four letters, so that they sum to 1 as probabilities and contain no NaN. Results SHALL not depend on the batch size.

#### Scenario: Normalised
- **WHEN** a batch of prompts is scored
- **THEN** each row exponentiates to values that sum to 1 within 1e-5 and none is NaN or infinite

#### Scenario: Batch size does not matter
- **WHEN** the same prompts are scored in float32 with batch size 1 and with batch size 8
- **THEN** the log-probabilities agree within 1e-4 for every prompt (in float16 the two differ by rounding noise of about 1e-2 and the check only reports it)

### Requirement: Option permutation
The scorer SHALL support four cyclic shifts of the option order, such that each option appears at each position exactly once across the four shifts, and SHALL return the scores in the original option order, so a model that ignores position gives the same scores for every shift.

#### Scenario: Every option visits every position
- **WHEN** the four shifts are generated for one record
- **THEN** each original option appears once at each of the positions A, B, C and D

#### Scenario: Mapping back
- **WHEN** scores computed under shift k are mapped back
- **THEN** the score of the original option i is the score of the letter at the position where option i was placed under shift k

### Requirement: Position bias measure
The scorer SHALL report position bias as, for each letter, the fraction of questions where that letter had the highest probability under the original order, next to the fraction of questions whose correct answer is that letter.

#### Scenario: Uniform answers, biased model
- **WHEN** a model always puts the highest probability on `A`
- **THEN** the reported share for `A` is 1.0 and the shares for `B`, `C` and `D` are 0.0

### Requirement: Agreement with a standard library
On a sample of 100 test-split records, with the prompts identical, the accuracy of the scorer SHALL differ from the accuracy computed by `lm-eval` on the same model by at most one percentage point, and the check SHALL exit non-zero otherwise.

#### Scenario: Same model, same prompts
- **WHEN** the check scores 100 seeded test-split records with Qwen2.5-0.5B and runs `lm-eval` on the same prompts
- **THEN** the two accuracies differ by at most 0.01 and the command exits with status 0
