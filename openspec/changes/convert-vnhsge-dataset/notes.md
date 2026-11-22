# Notes

## Hand check (task 4.3), 2026-11-20

30 kept records drawn with `random.Random(7).sample(...)` from `data/processed/vnhsge.jsonl` (16,028 records), each compared with its source item's `Question` and `Choice`.

Result: 30 of 30 match (stem, four options, answer). Subjects covered: Biology 9, English 4, History 4, Geography 3, Physics 3, Chemistry 2, Civic Education 2, Mathematics 2; one illustrative exam, the rest practice.

Observations, not mismatches:

- `Phy_25_40` keeps an embedded figure (`\includegraphics{...}`) in the stem although `Image_Question` is empty. 161 kept records do this (Physics practice 136, Chemistry practice 25). Raised with the user: not covered by the spec's drop rules. Decided 2026-11-21: drop them with reason `image` (a text-only model would guess); spec, design and tasks 5.1–5.2 updated.
- English stems keep the section instruction ("Mark the letter A, B, C, or D ...") and, for reading items, the whole passage. This is the source's format and is left as is.
- Non-breaking spaces after labels (`A.\xa0`) are removed by trimming.
