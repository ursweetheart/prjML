# Notes

## Hand check (task 4.3), 2026-11-20

30 kept records drawn with `random.Random(7).sample(...)` from `data/processed/vnhsge.jsonl` (16,028 records), each compared with its source item's `Question` and `Choice`.

Result: 30 of 30 match (stem, four options, answer). Subjects covered: Biology 9, English 4, History 4, Geography 3, Physics 3, Chemistry 2, Civic Education 2, Mathematics 2; one illustrative exam, the rest practice.

Observations, not mismatches:

- `Phy_25_40` keeps an embedded figure (`\includegraphics{...}`) in the stem although `Image_Question` is empty. 161 kept records do this (Physics practice 136, Chemistry practice 25). Raised with the user: not covered by the spec's drop rules. Decided 2026-11-21: drop them with reason `image` (a text-only model would guess); spec, design and tasks 5.1–5.2 updated.
- English stems keep the section instruction ("Mark the letter A, B, C, or D ...") and, for reading items, the whole passage. This is the source's format and is left as is.
- Non-breaking spaces after labels (`A.\xa0`) are removed by trimming.

## Question-number prefix (tasks 6.1 and 6.2), 2026-11-22

Found while reading a record: the first stem-number pattern matched only `Câu N:` and `Câu N.`, so 2,998 of 15,867 kept records still began with `CÂU N:` (2,894), `Câu N)` (30) or a bare `Câu N` (74), and 139 English items kept `Question N:`. Fixed with one case-insensitive pattern with an optional separator.

Effect of rerunning: kept 15,867 to **15,827**, `duplicate` 192 to **227** (questions that differed only by the prefix now match), `options` 64 to 69 (stems that were only a number become empty and the item is dropped), `image` 888 unchanged. Calibration 7,909, test 7,918. No kept stem starts with a question number; two runs give the same sha256 (`e9faba66…`); every (subject, exam_type) group has exactly half its records in calibration, rounded down.

Anyone who read the earlier `vnhsge.jsonl` (15,867 records) should rerun the converter: ids and splits of some records changed.
