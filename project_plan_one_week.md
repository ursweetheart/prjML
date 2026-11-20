# One-Week Execution Plan — Topic 12

Group 4, class QT02 · companion to `machine_learning_project_proposal_topic_12.md`

## Goal

Finish all four required parts (A, B, C, D), the LaTeX report, the repository and the defense rehearsal in 7 working days. The proposal planned 3 weeks, so this plan cuts the slowest work and fixes a strict order.

Dates: Day 1 = Fri Nov 20, Day 7 = Thu Nov 26, submission on Fri Nov 27. If the start date moves, keep the day numbers and shift the dates.

## Four decisions that make one week possible

| Decision | Why it saves time |
| --- | --- |
| 1. Use the open VNHSGE dataset (about 16,000 usable questions, see below) instead of scraping exam PDFs. Scrape only the 2025 THPT exam ourselves, as the post-cutoff set | Removes OCR and parsing, the slowest part of the original plan |
| 2. Use small models (3B parameters or fewer) at FP16. Candidates: Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B. Speculative decoding pair: Qwen2.5-3B (target) and Qwen2.5-0.5B (draft), which share a tokenizer | Fits a free 16 GB GPU; a full benchmark run takes minutes, not hours |
| 3. Run each model once and save the log-probability of every option to disk. Accuracy, confidence intervals, McNemar, conformal sets, APS and ECE are all computed from that cache on CPU | The "at least 5 runs" rule becomes 5 random calibration/test splits on cached numbers, with no extra GPU time |
| 4. Paraphrase only a 300-question sample. Option permutation is free, so apply it to every question | Paraphrasing 3,000 Vietnamese questions well is not realistic in a week |

## Dataset decision

We work as if the instructor has accepted the VNHSGE practice exams as part of the benchmark (see the Data plan in the proposal). Official and illustrative exams are already allowed by D.1.

- VNHSGE is cloned to `data/raw/VNHSGE` and checked: 17,054 questions, 16,085 usable (no image, valid answer key, four options found). Answer keys are present in every split.
- Every VNHSGE question counts as published no later than 2023, so it is on the pre-cutoff side of the contamination check. The 2025 official exam is the post-cutoff side.
- Backup if the instructor later rejects practice exams: keep the 2,938 usable dated official and illustrative questions and also scrape the 2024 official exam.

## Tiny Transformer test bed

We write a minimal decoder-only Transformer in PyTorch without built-in attention modules (Topic 9 A.1–A.2) and train a larger target and a smaller draft version as character-level models on Truyện Kiều. Both share one tokenizer of about 100 characters.

It does not replace the pretrained models. The same speculative decoding and quantization code runs on both. The tiny pair exists because it allows checks that are not feasible with a vocabulary of about 150,000 tokens:

- **A.4 correctness:** checking that speculative decoding keeps the target distribution means sampling many times and counting. The number of samples needed grows with the vocabulary size. With 100 characters, one million samples take about a minute and we can compare them with the exact target distribution. With a 7B model at about 0.1 s per sample, one million samples would take about 28 hours, and a check as precise as the tiny one would need far more.
- **B.3 and C.3:** by training the draft for more or fewer steps we control the acceptance rate a, and test the expected-tokens formula (1 − a^(γ+1))/(1 − a) at several values of a.
- **A.3 and B.5:** we quantize our own weights layer by layer and read our own KV cache before applying the same code to Qwen.

The D.4 speedup measurements are done on the pretrained Qwen pair. On that pair we also add a weaker check of the top 10–20 tokens with a few thousand samples.

## Work flow

```
 Day 1         Day 2-3                Day 4               Day 5          Day 6-7
 ------        -------                -----               -----          -------
 dataset  -->  A.1 scorer --> cache log-probs (3 models) --> stats/conformal --+
 models        A.2 conformal (CPU, uses cache)                                 |
               |                                                               v
 tiny     -->  A.3 quantizer --> INT8/INT4 runs (D.4) ---------------------> report
 Transf.       A.4 spec decode --> tiny check --> Qwen speedup (D.4) ------> defense
               |
 proofs B1-B6 + exercises C1-C4 written in parallel all week ---------------->
```

The critical path is the log-probability cache. Almost all of Part D depends on it, so it must exist by the end of Day 3.

## Day-by-day plan

| Day | Tuấn (leader: data, tiny Transformer) | Bằng (evaluation, statistics) | Tùng (efficiency) |
| --- | --- | --- | --- |
| 1 · Fri Nov 20 | ~~Check VNHSGE format~~ (done). Convert VNHSGE to the record schema: split options out of the question text, read exam type and year from file names, drop image items, empty answers and Literature, remove duplicates, fix or drop the 241 typo items; make the calibration/test split stratified by subject and exam type. Message the instructor about the dataset and the Topic 9 test bed | Set up the Git repository, environment file and Colab notebook; load the three models and check they run | Read up on quantization and speculative decoding; write proofs B.4 and B.5 |
| 2 · Sat Nov 21 | Tiny Transformer: attention, causal mask, sinusoidal positions, decoder block, KV cache; train target and draft on Truyện Kiều | A.1 scorer: log-probability of options A–D with option permutation; compare with `lm-eval` on 100 questions | A.3 quantizer: group-wise INT8/INT4, symmetric and asymmetric; test on the tiny model's weights |
| 3 · Sun Nov 22 | Scrape the 2025 official THPT exam, all subjects except Literature (post-cutoff set); generate 300 paraphrases and hand-check 30 | Run all three models on every question and save the log-probability cache; start A.2 conformal and APS | A.4 speculative decoding; exact distribution check on the tiny pair; test the B.3 formula at three acceptance rates |
| 4 · Mon Nov 23 | D.2 contamination: original vs. paraphrased vs. permuted vs. 2025 questions; write the datasheet | D.3: accuracy with confidence intervals, McNemar for each model pair, conformal sets at α = 0.05 and 0.1, coverage per subject, over 5 splits | Run A.3 on Qwen and compare with `bitsandbytes`; D.4 FP16/INT8/INT4: accuracy, ECE, latency, memory |
| 5 · Tue Nov 24 | Proofs B.1 and B.6; exercises C.1–C.4 | D.5: subjects with the highest uncertainty; abstention rule based on prediction-set size | D.4 speculative decoding speedup with Qwen 0.5B → 3B on math questions; compare with Hugging Face assisted generation |
| 6 · Wed Nov 25 | LaTeX: introduction, data, tiny Transformer section | LaTeX: evaluation and conformal results | LaTeX: proofs B.2–B.5, efficiency results, figures |
| 7 · Thu Nov 26 | Merge the report, model card and README with rerun instructions; defense rehearsal where each member explains another member's part | Same | Same |

Proofs need no GPU, so each member writes their share in the waiting time from Day 1.

## Daily checkpoints

A 10-minute call each evening. If a checkpoint is missed, apply the next item from the cut list.

| End of day | Must be true |
| --- | --- |
| 1 | VNHSGE converted to one JSONL file (about 16,000 questions) with split and exam\_type columns |
| 2 | A.1 matches `lm-eval` within about 1%; the tiny model generates readable Vietnamese text |
| 3 | Log-probability cache exists for all three models; the tiny-pair distribution check passes |
| 4 | All D.3 tables are done |
| 5 | All numbers are final; no new experiments after this day |

## Cut list (apply in this order)

1. Shift test across exam years (already optional).
2. Paraphrases: reduce to 100 questions, keep the permutations.
3. Asymmetric quantization: keep the symmetric mode only.
4. Report length: aim for the 25-page minimum, not 40.

Never cut: the log-probability cache, the tiny-pair check, any of the six proofs, the defense rehearsal.

## Risks

| Risk | Mitigation |
| --- | --- |
| Colab session limits | Save results to Google Drive after every model; Kaggle gives about 30 GPU hours a week per account, and we have three accounts |
| The instructor later rejects the practice exams | Keep the 2,938 dated official and illustrative questions and scrape the 2024 official exam as well |
| Running all models on about 16,000 questions takes longer than planned | Scoring one question is one forward pass, so this should be minutes per model; if not, sample the practice exams down, keeping all official and illustrative questions |
| The instructor does not accept the Topic 9 test bed | Keep the tiny models and describe them as the toy-vocabulary test for A.4, which is already in the proposal |
| The leader is overloaded (data, tiny Transformer, contamination, two proofs) | If Day 2 slips, move the 2025 exam scraping to Bằng |
| The one-week schedule conflicts with the instructor's milestones | The proposal lists three milestones over three weeks; confirm on Day 1 that an early finish is accepted |

## Open items to confirm on Day 1

- Instructor's answer on the VNHSGE practice exams (we proceed as if accepted), and on using a dataset with no license file for evaluation, citing the paper.
- Whether the tiny Transformer test bed (Topic 9 A.1–A.2) is accepted.
- Licenses, Vietnamese quality and training-data cutoffs of the candidate models.
- Who owns `speculative_decode()`: the tiny Transformer and A.4 share this code.
