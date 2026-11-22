# Proposal

## Why

Every Part D experiment (scoring, conformal prediction, McNemar, quantization runs) reads the benchmark, but the benchmark does not exist yet: VNHSGE is cloned to `data/raw/VNHSGE` in its own format, with the four options inside the question text, no exam type, no split and some broken items. This is task 3 of Day 1 in `project_plan_one_week.md` and sits on the critical path.

## What Changes

- Add a converter that reads the Vietnamese JSON files of VNHSGE and writes one benchmark file, `data/processed/vnhsge.jsonl`, in the record schema of the proposal's Data plan.
- Parse the four options A–D out of the question text, including the formats that a line-start check misses (option A on the question line, all options on one line).
- Read `exam_type` (official, illustrative, practice) and `year` from the file name; practice exams get year `unknown (≤2023)`.
- Drop items that have an image in the question, an empty or invalid answer key, Literature essays, or options that cannot be parsed; write every dropped item with its reason to a log file.
- Remove duplicate questions, keeping the copy from the most authoritative exam type.
- Assign a reproducible calibration/test split, stratified by subject and exam type, with a fixed seed.
- Fail loudly when the output breaks the schema (answer not A–D, empty option, duplicate id).

Working assumption, recorded in the proposal's Data plan: the instructor accepts the practice exams, so they are included.

## Capabilities

### New Capabilities

- `benchmark-dataset`: building the Vietnamese multiple-choice benchmark file from raw sources: record schema, item filtering, deduplication, provenance, and the calibration/test split. The 2025 post-cutoff exams and paraphrased variants will extend this capability later.

### Modified Capabilities

None.

## Impact

- New code: `src/machinelearningproject/convert_vnhsge.py`, run with `uv run python -m machinelearningproject.convert_vnhsge`.
- New files (git-ignored under `data/`): `data/processed/vnhsge.jsonl` and `data/processed/vnhsge_dropped.jsonl`.
- No new dependencies: standard library only.
- Downstream: A.1 scoring, A.2 conformal and all Part D runs read `vnhsge.jsonl`; its field names become the contract they code against.
