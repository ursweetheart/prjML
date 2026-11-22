# Design

## Context

Facts from inspecting `data/raw/VNHSGE/Dataset/VNHSGE-V/JSON format` (clone at commit `04c845d`, 2026-10-30):

- 406 JSON files under `eval/`, `test/`, `train/`, one folder per subject; each file is a list of items with keys `ID`, `Image_Question`, `Question`, `Choice`, `Image_Answer`, `Explanation`. The 30 Literature items use other keys (`Câu hỏi`, `Đáp án`, ...) and are essays.
- `ID` is unique across all files.
- The options live inside `Question`. A line-start check finds all four labels in 16,085 usable items. Of the 241 failures: 101 have option A on the stem line, 52 have all options on one tab-separated line, about 80 have real typos (`C.` twice, `D` without a dot, missing labels).
- File names: `MET_*_OE_<year>_*` official, `MET_*_IE_<year>` illustrative, `MET_Civ_2022_30x` official (exam codes 301–304), everything else practice with no year.

Requirements are in `specs/benchmark-dataset/spec.md`.

## Goals / Non-Goals

**Goals:**
- One script, standard library only, that rebuilds the benchmark from the raw clone in seconds.
- Recover the 153 items lost only to option layout; drop the true typos rather than guess.

**Non-Goals:**
- Fixing typo items by hand (can be done later by editing an override file; not needed to reach the target size).
- The 2025 post-cutoff exams, paraphrased and permuted variants, and difficulty labels: later changes extend `benchmark-dataset`.
- Unicode normalization beyond NFC and whitespace trimming; LaTeX is kept as written.

## Decisions

**Single module `src/machinelearningproject/convert_vnhsge.py`, run with `python -m`.** The project has no data layer yet; one module with a `main()` is enough. Alternative: a package with parser/split/io modules. Rejected: three files for ~150 lines.

**Option parsing: last `A.` followed by `B.`, `C.`, `D.` in order.** A label is `[ABCD][.)]` at the start of the text or after whitespace, `?`, `.` or `:`, or a bare letter plus space at a line start. Found during implementation: allowing any non-word character before a label broke 24 math items (`+C.` is the integration constant, `(B)` a name), so `+` and `(` are excluded; the final rule gives identical results to the whitespace-only rule on all 16,855 items that rule could parse, and recovers 112 more. Scanning backwards for the last `A` that has `B`, `C`, `D` after it handles line-separated, one-line and "A on the stem line" layouts with one rule, and passages that mention `A.` earlier in the text. Alternatives: line-based split (misses 153 items); trying several regexes per layout (more code, same result). Known limit: a label-shaped token inside an option, such as `anh C. B.`, still parses correctly as long as the real labels appear in order, which matches the inspected cases; the hand check (task 4.3) catches the rest.

**Strip `Câu N` from the stem.** The number carries no information, and leaving it in makes the same question in two exams look different, which breaks deduplication. The pattern is case-insensitive, the separator after the number is optional, and English `Question N` is covered too: the first version matched only `Câu N:` / `Câu N.`, and 2,998 of 15,867 kept records (2,894 `CÂU N:`, 30 `Câu N)`, 74 with no separator) plus 139 English items still carried the prefix. Only the first number is removed, so a stem such as `Câu 6 Câu “Con trâu...` keeps its second `Câu`.

**Duplicate key: whitespace-collapsed stem plus four options, keep official > illustrative > practice.** Keeping the official copy preserves the year and the stronger exam type for D.1. Comparing options too avoids merging two questions that share a stem but differ in options.

**Split: per (subject, exam_type) group, `random.Random(0)` shuffle over records sorted by id, first half calibration.** Sorting before shuffling makes the result independent of file-system order, so reruns are byte-identical. 50/50 is the usual split-conformal choice; it is a constant in the module and easy to change. The VNHSGE `train/test/eval` folders are ignored, as the proposal's Data plan says.

**Figures inside the question text count as images.** 161 items (Physics 136, Chemistry 25, all practice) have an empty `Image_Question` but embed `\includegraphics{...}` in `Question`; a text-only model sees the file name, not the figure, and would guess. They are dropped with the same reason `image`, so one rule covers every figure item, as the proposal's Data plan says. Alternative: keep them and evaluate image items with a vision-language model. Rejected: Topic 12 evaluates language models, A.1 scores option text, and a VLM track would need its own pipeline and at least two VLMs for McNemar. The datasheet notes that results cover text-only questions and that Physics loses the most items.

**Validate all records before writing; write to a temp file and rename.** This meets "invalid output stops the conversion" and never leaves a half-written benchmark.

**Built-in self-check instead of a test framework.** pytest is not installed; a `_self_test()` with asserts on the real-shaped strings from the spec scenarios runs at the start of every conversion. Alternative: add pytest as a dev dependency. Deferred until there is more than one module to test.

## Risks / Trade-offs

- [Parser accepts a wrong split on an unseen layout] → hand-check 30 random records against the source; the summary counts must land near 16,000 kept.
- [Practice exams are later rejected by the instructor] → `exam_type` lets downstream code filter them out without rerunning the converter.
- [Upstream VNHSGE changes after a new clone] → `source_url` pins the commit; the split changes only if the input changes.
