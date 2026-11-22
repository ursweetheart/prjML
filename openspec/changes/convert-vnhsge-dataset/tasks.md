# Tasks

## 1. Parsing

- [x] 1.1 Create `src/machinelearningproject/convert_vnhsge.py` with the option parser (last `A.` followed by `B.`, `C.`, `D.`) and `Câu N` stripping, plus `_self_test()` asserts for every parsing scenario in the spec (one per line, one line, A on the stem line, label inside option text, `A. B. C. C.` rejected, numbered stem); verify `uv run python -c "from machinelearningproject.convert_vnhsge import _self_test; _self_test()"` passes
- [x] 1.2 Add exam type and year detection from the file name, with self-test asserts for `MET_Math_OE_2021_101`, `MET_Civ_2022_301`, `MET_Bio_IE_2020` and `Phy_41`; verify the self-test passes

## 2. Conversion

- [x] 2.1 Read every JSON file under `data/raw/VNHSGE/Dataset/VNHSGE-V/JSON format` in sorted order and build records in the spec's schema, with `id` prefixed `vnhsge:`, `source_url` at the clone's commit and `retrieved_at` as its date; verify a printed sample record has all 16 fields
- [x] 2.2 Drop image, answer, Literature and options items and remove duplicates (official > illustrative > practice), collecting each dropped item with its reason; verify the dropped counts per reason are printed and the `image` count is 726 (about 887 after task 5.1)
- [x] 2.3 Assign the 50/50 calibration/test split per (subject, exam_type) with seed 0 over id-sorted records; verify each group's calibration count is half its size rounded down

## 3. Output and validation

- [x] 3.1 Validate every record (answer in A–D, non-empty stem and options, unique ids) before writing, then write `data/processed/vnhsge.jsonl` and `data/processed/vnhsge_dropped.jsonl` through a temp file and rename; verify that editing one record to an empty option in a quick local test makes the script exit non-zero and leaves the old file unchanged
- [x] 3.2 Print the summary (kept per subject and exam type, dropped per reason, split totals) and expose `main()` for `uv run python -m machinelearningproject.convert_vnhsge`; verify the command runs and reports roughly 16,000 kept records

## 4. Verification

- [x] 4.1 Run the conversion twice and verify `sha256sum data/processed/vnhsge.jsonl` is identical both times
- [x] 4.2 Verify the options-recovery result: fewer than 100 items dropped with reason `options` (241 failed the old line-start check, about 153 are layout-only)
- [x] 4.3 Hand-check 30 random kept records against their source JSON (stem, four options, answer) and record the result in the change's notes; verify zero mismatches, or fix the parser and rerun
- [x] 4.4 Add a "Data" section to `README.md` with the clone command and the conversion command; verify both commands run as written from a fresh shell

## 5. Figures embedded in the question text

- [x] 5.1 Extend the image check to also drop items whose `Question` contains `\includegraphics` (reason `image`), and add a `_self_test()` assert for an item with empty `Image_Question` and an embedded figure; verify the self-test passes
- [x] 5.2 Rerun the conversion; verify `grep -c includegraphics data/processed/vnhsge.jsonl` prints 0, kept records drop by about 161 (to about 15,867), two runs give the same `sha256sum`, and each group's calibration count is still half its size rounded down
