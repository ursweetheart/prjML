# Spec Delta

## Purpose

Builds the Vietnamese multiple-choice benchmark that every evaluation and efficiency experiment reads, with a fixed record schema, documented filtering, provenance and a reproducible calibration/test split.

## ADDED Requirements

### Requirement: Benchmark records follow the record schema
The conversion SHALL write one JSON object per line to `data/processed/vnhsge.jsonl`. Each record SHALL have the fields `id`, `subject`, `year`, `exam`, `exam_type`, `level`, `question`, `option_a`, `option_b`, `option_c`, `option_d`, `answer`, `source_url`, `retrieved_at`, `split` and `variant_of`.

#### Scenario: A valid source item is converted
- **WHEN** a VNHSGE item has no question image, a `Choice` of A, B, C or D, and four parseable options
- **THEN** the output contains one record with all schema fields, `answer` equal to the source `Choice`, the question stem without the options, and each option text without its label

#### Scenario: Fields with no source value
- **WHEN** a record is written
- **THEN** `level` and `variant_of` are `null`

### Requirement: Item ids are stable and unique
Each record's `id` SHALL be the VNHSGE item id prefixed with `vnhsge:`, and no two records SHALL share an id.

#### Scenario: Id format
- **WHEN** the source item has `ID` `Phy_41_1`
- **THEN** the record's `id` is `vnhsge:Phy_41_1`

### Requirement: Exam type and year come from the source file name
`exam_type` SHALL be `illustrative` for files whose name contains `_IE_`, `official` for other files whose name starts with `MET_` and contains a year, and `practice` otherwise. `year` SHALL be the four-digit year in the file name, or the string `unknown (≤2023)` when there is none.

#### Scenario: Official exam
- **WHEN** the source file is `MET_Math_OE_2021_101.json` or `MET_Civ_2022_301.json`
- **THEN** `exam_type` is `official` and `year` is `2021` or `2022`

#### Scenario: Illustrative exam
- **WHEN** the source file is `MET_Bio_IE_2020.json`
- **THEN** `exam_type` is `illustrative` and `year` is `2020`

#### Scenario: Practice exam
- **WHEN** the source file is `Phy_41.json`
- **THEN** `exam_type` is `practice` and `year` is `unknown (≤2023)`

### Requirement: Options are parsed from the question text
The conversion SHALL split the question text into a stem and four options. A label is `A.`–`D.` (or with `)`) at the start of the text or after whitespace, `?`, `.` or `:`, or a bare letter followed by a space at the start of a line. Labels must appear in order A, B, C, D; the conversion SHALL use the last `A` that is followed by the other three and gives a non-empty stem and four non-empty options.

#### Scenario: One option per line
- **WHEN** the text is a stem followed by four lines `A. x`, `B. y`, `C. z`, `D. w`
- **THEN** the options are `x`, `y`, `z`, `w`

#### Scenario: All options on one line
- **WHEN** the text ends with `\tA. Trung Quốc. \tB. Nhật Bản. \tC. Pháp. \tD. Ấn Độ.`
- **THEN** the options are `Trung Quốc.`, `Nhật Bản.`, `Pháp.`, `Ấn Độ.`

#### Scenario: Option A on the stem line
- **WHEN** the text is `... tăng chủ yếu do A. x\nB. y\nC. z\nD. w`
- **THEN** the stem ends before `A.` and the options are `x`, `y`, `z`, `w`

#### Scenario: A label inside option text
- **WHEN** an option's text contains a name such as `ông A.` after the real option A
- **THEN** that occurrence is part of the option text and is not used as a label

#### Scenario: No space before a label
- **WHEN** the text is `... vô sinh?A. Động vật.\nB. Nhiệt độ.\nC. Thực vật.\nD. Vi sinh vật.`
- **THEN** the stem is `... vô sinh?` and option A is `Động vật.`

#### Scenario: Label without a dot
- **WHEN** an option line is `A wonderful` or `B containing`
- **THEN** it is read as option A or B

#### Scenario: Math that looks like a label
- **WHEN** an option ends with the integration constant `+C.`, or contains `(A)` or `(B)`
- **THEN** that text stays in the option and is not used as a label

#### Scenario: Labels out of order
- **WHEN** the options are labelled `A.`, `C.`, `B.`, `D.` in that order
- **THEN** the item is dropped with reason `options`

### Requirement: Question numbering is removed from the stem
The stem SHALL NOT start with the exam's question number, in any letter case and with or without a separator (`Câu 12:`, `CÂU 12:`, `Câu 12 :`, `Câu 12.`, `Câu 12)`, `Câu 12` followed by a space or newline, `Question 12:`).

#### Scenario: Uppercase or separator-less number
- **WHEN** the question text starts with `CÂU 24:Hạn chế của nguồn lao động`, `Câu 6 Câu “Con trâu đi trước”` or `Question 6: the medicine`
- **THEN** the stem starts with `Hạn chế của nguồn lao động`, `Câu “Con trâu đi trước”` and `the medicine` respectively

#### Scenario: Numbered stem
- **WHEN** the question text starts with `Câu 1:  Dao động của một vật`
- **THEN** the stem starts with `Dao động của một vật`

### Requirement: Unusable items are dropped and logged
The conversion SHALL drop an item when it has a non-empty `Image_Question` or its `Question` text contains `\includegraphics`, when `Choice` is not exactly one of A, B, C, D, when it is a Literature item, or when four non-empty options cannot be parsed. Every dropped item SHALL be written to `data/processed/vnhsge_dropped.jsonl` with its source id, file and reason.

#### Scenario: Image question
- **WHEN** an item has a non-empty `Image_Question`
- **THEN** it is absent from the benchmark and logged with reason `image`

#### Scenario: Figure embedded in the question text
- **WHEN** an item has an empty `Image_Question` but its `Question` contains `\includegraphics{...}`
- **THEN** it is absent from the benchmark and logged with reason `image`

#### Scenario: Typo in labels
- **WHEN** an item's options are labelled `A.`, `B.`, `C.`, `C.`
- **THEN** it is absent from the benchmark and logged with reason `options`

#### Scenario: Missing answer key
- **WHEN** an item's `Choice` is empty
- **THEN** it is absent from the benchmark and logged with reason `answer`

### Requirement: Duplicate questions are removed
Two items SHALL count as duplicates when their stems and four options are equal after collapsing whitespace. Only one copy SHALL be kept: the one from the exam type ranked highest in official, illustrative, practice; ties keep the first in sorted file-path order. Removed copies SHALL be logged with reason `duplicate`.

#### Scenario: Same question in an official and a practice exam
- **WHEN** the same question appears in `MET_His_OE_2021_301.json` and `His_12.json`
- **THEN** the benchmark keeps the record from the official exam and logs the other as `duplicate`

### Requirement: Calibration and test split is reproducible and stratified
Every kept record SHALL have `split` set to `calibration` or `test`. Within each (subject, exam_type) group, half of the records (rounded down) SHALL be `calibration` and the rest `test`, chosen by a seeded random shuffle with seed 0. Running the conversion twice on the same input SHALL produce byte-identical output.

#### Scenario: Group sizes
- **WHEN** a (subject, exam_type) group has 201 records
- **THEN** 100 records are `calibration` and 101 are `test`

#### Scenario: Rerun
- **WHEN** the conversion is run twice on the same raw data
- **THEN** both runs write identical `vnhsge.jsonl` files

### Requirement: Provenance is recorded
`source_url` SHALL point to the source file in the VNHSGE GitHub repository at the cloned commit, and `retrieved_at` SHALL be that commit's date.

#### Scenario: Provenance fields
- **WHEN** the clone is at commit `04c845d…` dated 2026-10-30
- **THEN** `source_url` is `https://github.com/Xdao85/VNHSGE/blob/<full commit>/Dataset/VNHSGE-V/JSON%20format/<split>/<Subject>/<file>.json` and `retrieved_at` is `2026-10-30`

### Requirement: Invalid output stops the conversion
The conversion SHALL exit with an error, before replacing an existing benchmark file, if any output record has an `answer` outside A–D, an empty option or stem, or an id that is not unique.

#### Scenario: Broken output
- **WHEN** a code change causes a record to have an empty `option_c`
- **THEN** the conversion exits with a non-zero status naming the record, and the previous `vnhsge.jsonl` is unchanged

### Requirement: The conversion reports counts
On success the conversion SHALL print the number of kept records per subject and exam type, the number of dropped items per reason, and the split sizes.

#### Scenario: Summary
- **WHEN** the conversion finishes
- **THEN** it prints kept counts per (subject, exam_type), dropped counts per reason, and the calibration and test totals
