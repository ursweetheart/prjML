# Project Proposal and Data Plan — Topic 12

Nov 15, 2026 · @tuan

## Project overview

Topic 12 asks us to evaluate open-source language models on Vietnamese multiple-choice exams with statistical guarantees, and to show how quantization and speculative decoding make them cheaper to run without hurting quality.

This is the Machine Learning big assignment for class QT02, Group 4. The topic comes from the instructor's list of 20 topics (TS. Phạm Văn Khánh). Its official title is "Đánh giá có bảo đảm thống kê, lượng tử hóa và suy luận hiệu quả cho mô hình ngôn ngữ" (Statistically guaranteed evaluation, quantization and efficient inference for language models).

The project answers four questions:

| Question | Technique we use |
| --- | --- |
| Is the score real, or did the model memorize the exam? | Contamination control, paraphrased and option-permuted question variants |
| Can we trust the result? | Confidence intervals, McNemar test, split conformal prediction (with the APS variant) |
| Can we shrink the model and keep its quality? | Post-training quantization: FP16, INT8, INT4 |
| Can we generate faster without changing the output? | Speculative decoding with a small draft model |

We do not train a large language model. We implement the evaluation and efficiency tools ourselves (Part A), verify them on a tiny decoder-only Transformer that we write and train from scratch, and apply them to existing pretrained open-source models (Part D).

## Objectives and scope

The core scope is the four required parts of the assignment, graded out of 100 points; Part D carries the most weight.

| Part | Points | What we deliver |
| --- | --- | --- |
| A. Method development and implementation | 20 | From-scratch code for the choice-scoring framework, split conformal prediction and APS, group-wise post-training quantization, and speculative decoding, checked against standard libraries |
| B. Theoretical proofs | 25 | Six proofs: conformal coverage, speculative decoding correctness and acceptance rate, expected tokens per step, quantization error bounds, KV cache size, McNemar test |
| C. Exercises | 15 | Four calculations that check Part B numerically (memory sizing, hand quantization, speedup, conformal threshold) |
| D. Practical application | 30 | Vietnamese exam benchmark of at least 3,000 questions, contamination control, guaranteed evaluation of at least three models, quantization and speedup measurements, uncertainty analysis |
| Report, reproducibility, defense | 10 | LaTeX report, Git repository, datasheet, model card, oral defense |

At least one implementation from Part A must be used directly in the Part D pipeline, for example our own conformal code and our own quantizer.

**Test bed: a tiny Transformer written from scratch.** To verify A.3, A.4 and B.5 under controlled conditions, we implement a minimal decoder-only Transformer in PyTorch without built-in attention modules (scaled dot-product and multi-head attention, causal mask, sinusoidal position encoding, decoder block, KV cache), following Topic 9 A.1–A.2. We train a larger target and a smaller draft version as character-level models on public-domain Vietnamese text (for example Truyện Kiều), so both share one tokenizer by construction. These models do not replace the pretrained models: the same speculative decoding and quantization code runs on both. The tiny pair is used because it allows checks that are not feasible on a vocabulary of about 150,000 tokens:

- A.4 correctness: with a vocabulary of about 100 characters, we draw millions of samples and compare the output distribution with the target distribution exactly, including short multi-token sequences.
- B.3 and C.3: by training the draft for more or fewer steps we control the acceptance rate a, and test the expected-tokens formula (1 − a^(γ+1))/(1 − a) at several values of a instead of one.
- A.3 and B.5: we quantize our own weights layer by layer and read our own KV cache, then apply the same quantizer to the pretrained models.

The speedup measurements required by D.4 are done on the pretrained target/draft pair.

**Out of scope:** training or pretraining a large language model, and collecting any non-public or personal data.

**Optional extensions** are attempted only after the core work is done. They connect Topic 12 to other topics in the instructor's list: LoRA tuning of the draft model to raise its acceptance rate (Topic 9), perplexity under quantization (Topic 9), conformal coverage under distribution shift across exam years (Topic 3), and tokenizer efficiency on Vietnamese text (Topic 10). We will confirm with the instructor that extensions are welcome.

## Team and responsibilities

Group 4 has three members, and each one must be able to explain every proof and every piece of code at the oral defense. The split below is a proposal for the group to adjust.

| Member | Student ID | Proposed lead role |
| --- | --- | --- |
| Trần Xuân Tuấn (leader) | 232631064 | Data pipeline: collection, parsing, cleaning, datasheet, contamination control (Part D.1–D.2); tiny Transformer test bed (target and draft models, KV cache) |
| Trần Xuân Bằng | to be added | Evaluation and statistics: choice-scoring framework, conformal prediction, McNemar and confidence intervals (A.1–A.2, B.1, B.6, D.3) |
| Phạm Việt Tùng | 232631065 | Efficiency: quantization and speculative decoding, latency and memory measurements (A.3–A.4, B.2–B.5, D.4) |

Shared work: Part C exercises, the LaTeX report, and the final uncertainty analysis (D.5). Every proof and every module gets a review from a second member before it is merged, so that all three can defend it.

## Data plan

We will build a benchmark of at least 3,000 Vietnamese multiple-choice questions, starting from the open VNHSGE dataset and adding the newest official exams that we collect ourselves. Every item is tagged by subject, exam year and exam type.

### Sources

- **VNHSGE** (Dao et al., 2023, arXiv 2305.12199; github.com/Xdao85/VNHSGE), Vietnamese version, JSON format. It contains 17,054 questions in 9 subjects, with answer keys for every split. It comes in three kinds:
  - official THPT exams (file names `MET_*_OE_<year>`) and illustrative exams from the Ministry (`MET_*_IE_<year>`), 2019–2023, with the year in the file name;
  - practice exams (`Bio_1`, `Phy_41`, `Math_*`, ...) of unknown source and year.
- **Newest official THPT exams (2025, and 2024 if needed)**, collected by us from public sources. They form the post-cutoff set.

Literature is essay-only and is excluded. Items that depend on a figure we cannot capture as text are dropped.

**Working assumption (pending the instructor's answer):** practice exams are accepted as part of the benchmark. Under this assumption the benchmark has about 16,000 usable questions, and we only collect the 2025 exam for the post-cutoff set. If the instructor accepts only official and illustrative exams, the benchmark is the 2,938 usable dated VNHSGE questions plus the 2024 and 2025 official exams, which brings it above 3,000.

Counts from our check of VNHSGE (Vietnamese JSON):

| | Questions |
| --- | --- |
| All questions | 17,054 |
| Usable: no image, valid answer key, four options found | 16,085 |
| Usable, from dated official or illustrative exams, duplicates removed | 2,938 |

Usable dated questions by subject: Mathematics 867, English 522, History 351, Civic Education 349, Geography 285, Chemistry 196, Biology 188, Physics 180. By year: 2019 456, 2020 557, 2021 583, 2022 984, 2023 358.

### Collection and cleaning

1. Convert VNHSGE to our record schema: split the four options out of the question text, take the answer from `Choice`, and read the exam type and year from the file name.
2. Drop items with an image in the question (726), an empty answer key (42), or the Literature essays (30). Remove duplicate questions (138 across the whole set).
3. Fix or drop the 241 items where four options are not found; many are typos in the source, such as option D labelled "C." a second time.
4. Collect the newest official exams from public pages at a polite request rate, following each site's terms of use and robots.txt. Save the source URL and retrieval date for every file. Parse them into the same schema.
5. Normalize Unicode so Vietnamese diacritics compare correctly, and keep formulas as LaTeX, as VNHSGE already does.
6. Manually check a random sample of items against the source (proposed: 10% of the collected exams, and a sample of converted VNHSGE items).

### Record schema

| Field | Meaning |
| --- | --- |
| id | Unique item id |
| subject, year, exam | Metadata used for per-subject analysis; year is "unknown (≤2023)" for VNHSGE practice exams |
| exam\_type | official, illustrative, or practice |
| level | Difficulty level; VNHSGE has no such field, so it is left empty unless we label it ourselves |
| question, option\_a to option\_d | Normalized text |
| answer | Correct option |
| source\_url, retrieved\_at | Provenance |
| split | calibration, test, or post-cutoff |
| variant\_of | Id of the original item, for paraphrased or option-permuted variants |

### Splits

- **Calibration and test:** a random split stratified by subject and exam type, so that calibration and test items are exchangeable. This is the condition the conformal coverage guarantee needs. We do not use the VNHSGE train/test folders as our split.
- **Post-cutoff subset:** questions published after each model's stated training-data cutoff, evaluated separately for contamination checks.
- **Shift test (optional):** calibrate on older years and test on the newest year to see how coverage changes when exchangeability breaks. This uses only the dated questions.

### Contamination control

The model never sees the year; it is used only to place each question before or after a model's training-data cutoff. VNHSGE was published in May 2023, so every VNHSGE question, including the undated practice exams, counts as published no later than 2023 and therefore before the cutoff of the models we consider (to be checked per model). The 2025 exams are the after-cutoff side.

For each model we record its training-data cutoff. We compare accuracy on pre-cutoff questions, on their paraphrased and option-permuted variants, and on post-cutoff questions. A large drop on variants or on post-cutoff questions suggests memorization.

### Ethics and compliance

We use only public official material and collect no personal data. We respect each source's terms of use and rate limits. If a source's terms do not allow redistribution, we publish our collection scripts and item ids instead of the raw text. VNHSGE has no license file, so we cite the paper, keep the raw data out of our repository, and publish only our conversion scripts and item ids. The dataset ships with a datasheet that describes sources, collection steps, known limits and intended use.

## Models to evaluate

The assignment requires at least three open-source models and does not name them, so we will choose them in week 1 against fixed criteria and record the choice here.

**Selection criteria**

- License allows research use.
- Reasonable Vietnamese support.
- A published training-data cutoff, needed for contamination control.
- Fits a 16 GB GPU (Google Colab or Kaggle) at FP16 or in quantized form. As a rough guide, a 7B-parameter model needs about 14 GB for FP16 weights alone and about 3.5 GB at INT4.
- For speculative decoding, a larger and a smaller model from the same family that share one tokenizer.

| Role | What we need | Used in |
| --- | --- | --- |
| Evaluated models | At least 3 different open-source models | Accuracy, McNemar, conformal sets (D.3) |
| Quantized variants | INT8 and INT4 versions of the same models, built with our own quantizer and checked against a standard library | Accuracy, calibration (ECE), latency, memory (D.4) |
| Target and draft pair | One large and one small model with a shared tokenizer | Speculative decoding speedup (A.4, D.4) |

Open question: which three models and which target/draft pair? To be filled by the end of week 1.

## Timeline

The instructor sets three milestones: proposal and data plan in week 1, progress report in week 2, final submission and defense in week 3. The topic list also says "2 weeks" in its opening section, so we will confirm the exact dates with the instructor.

| Week | Instructor milestone | Our work | Done when |
| --- | --- | --- | --- |
| 1 | Submit proposal and data plan | Fix the source list; start collecting and parsing exams; choose the three models and the target/draft pair; build the choice-scoring framework (A.1); write the tiny Transformer and train its target and draft versions; start the proofs in Part B | This document is submitted; first batch of parsed questions passes the manual check |
| 2 | Progress report: Parts A and B done, Part D dataset ready | Finish A.2–A.4 (conformal, quantizer, speculative decoding); finish all six proofs; reach at least 3,000 questions; run contamination variants; start Part C | All four implementations match standard libraries; dataset and datasheet complete |
| 3 | Submit final product and defend | Run the full Part D experiments (at least 5 runs per setting); finish Part C; write the uncertainty and abstention analysis; write the LaTeX report; prepare the repository and model card; rehearse the defense | Report of 25–40 pages, repository that reruns from scripts, every member able to explain every part |

Optional extensions are scheduled only if Part A, B and the dataset are finished by the end of week 2.

## Deliverables and reproducibility

The final submission has four items, and every result must be reproducible from our scripts.

- A LaTeX report of 25–40 pages with all proofs written in full.
- A Git repository with an environment file and scripts that rerun every experiment.
- The dataset, or our collection scripts, together with a datasheet.
- A model card for the Part D models.

**Reproducibility rules:** fix random seeds; report results as mean ± standard deviation over at least 5 runs or folds; choose hyperparameters on validation data only.

**Academic integrity:** we declare the scope of any generative-AI tool use in an appendix of the report. The committee may question each member on any proof or any piece of code.

## Risks and mitigation

The biggest risks are the short schedule and the 3,000-question target, so data collection starts in week 1.

| Risk | Mitigation |
| --- | --- |
| Fewer than 3,000 clean questions | VNHSGE gives about 16,000 usable questions if practice exams are accepted; if not, 2,938 dated questions plus the 2024 and 2025 official exams |
| Parsing errors from PDFs, formulas or figures | Drop items that depend on figures; manually check a random sample; fix the parser before scaling |
| 16 GB GPU or Colab/Kaggle session limits | Use smaller models or quantized loading; save results to disk after every run |
| Only three weeks for the core work and extensions | Finish the required parts first; start extensions only after week 2 checkpoints are met |
| Unclear training-data cutoffs weaken the contamination check | Record each model's stated cutoff; rely on the paraphrase and permutation test as the main evidence |
| Calibration and test items are not exchangeable, so conformal coverage may fail | Use a random stratified split for the main result; report the temporal split separately as a shift experiment |
| Bugs in speculative decoding that silently change the output distribution | Test on a toy vocabulary first and compare output frequencies with the target distribution |
| A member cannot explain a proof or module at the defense | Cross-review every part; rehearse the defense with questions across all areas |
| Format or deadlines differ from our assumptions | Confirm week-1 format and dates with the instructor |

## Knowledge and techniques we will use

The project combines statistics, Transformer internals and systems measurement, and most of it maps onto one part of the assignment.

| Area | What we need to know | Used in |
| --- | --- | --- |
| Probability and statistics | Exchangeability, quantiles, binomial distribution, confidence intervals, paired tests | B.1, B.6, D.3 |
| Conformal prediction | Split conformal method, nonconformity score 1 − p(y\|x), APS variant, empirical coverage | A.2, B.1, C.4, D.3 |
| Calibration | Expected calibration error (ECE), reliability of predicted probabilities | D.4 |
| Transformer basics | Causal mask, attention, KV cache, grouped-query attention, tokenization, log-probabilities of answer options | A.1, A.4, B.5, C.1 |
| Quantization | Uniform quantization, group-wise scales, symmetric and asymmetric modes, activation outliers, reconstruction error | A.3, B.4, C.2, D.4 |
| Speculative decoding | Accept/reject rule, total variation distance, expected tokens per step, speedup model | A.4, B.2, B.3, C.3, D.4 |
| Data engineering | PDF and web extraction, OCR, Unicode normalization, deduplication, datasheets | D.1, D.2 |
| Experiment practice | Fixed seeds, repeated runs, validation-only tuning, model cards | All parts |

**Knowledge borrowed from Topic 9 for the test bed:** implementing attention, the causal mask, sinusoidal position encoding and a decoder block from scratch, and training a character-level language model.

**Extra knowledge from other topics (optional extensions):** LoRA/QLoRA and perplexity from Topic 9; distribution shift from Topic 3; tokenizer behavior from Topic 10; and, only if time allows, preference optimization (DPO) from Topic 9.

**Key references from the topic list**

- Vovk, Gammerman & Shafer (2005), Algorithmic Learning in a Random World.
- Angelopoulos & Bates (2023), Conformal prediction: A gentle introduction.
- Romano, Sesia & Candès (2020), Classification with valid and adaptive coverage.
- Leviathan, Kalman & Matias (2023), Fast inference from transformers via speculative decoding.
- Chen et al. (2023), Accelerating large language model decoding with speculative sampling.
- Dettmers et al. (2022), LLM.int8().
- Frantar et al. (2023), GPTQ.
- Ainslie et al. (2023), GQA.
- McNemar (1947), Note on the sampling error of the difference between correlated proportions or percentages.

## Planned workflow

The pipeline runs from raw exam papers to a final uncertainty analysis, and each step produces files the next step reads.

1. **Collect and clean the data.** Download public exams, parse them into the record schema, normalize text, remove duplicates, and check a random sample by hand. Output: the benchmark and its datasheet.
2. **Split the data.** Create the stratified calibration and test sets, and mark the post-cutoff subset for each model. Output: fixed split files.
3. **Build the evaluation framework (A.1).** Score each option A–D by its log-probability, with option permutations to measure position bias. Check it on a few models against a standard evaluation library.
4. **Measure contamination (D.2).** Generate paraphrased and permuted variants, then compare accuracy on original and variant questions for every model.
5. **Evaluate at FP16 (D.3).** Run at least three models, report accuracy with confidence intervals, and compare pairs of models with McNemar's test.
6. **Add statistical guarantees (A.2, D.3).** Build conformal prediction sets for α = 0.05 and 0.1 on the calibration set, then check empirical coverage on the test set for each subject.
7. **Quantize (A.3, D.4).** Apply group-wise INT8 and INT4 quantization with our own code, check it against a standard library, and re-run accuracy, ECE, latency and memory.
8. **Speed up generation (A.4, D.4).** Run speculative decoding with the draft and target models. First confirm on the tiny Transformer pair that the output distribution is unchanged and that the measured tokens per step match the formula from B.3, then run the same code on the pretrained pair and measure speedup on questions that need long reasoning.
9. **Analyze and decide (D.5).** Find the subjects and question types with high uncertainty, and propose an abstention rule based on prediction-set size.
10. **Write up.** Complete Part B proofs and Part C exercises, write the LaTeX report, prepare the repository and model card, and rehearse the defense.

Proofs (Part B) and exercises (Part C) run in parallel with steps 3–8, because each implementation in Part A is checked against the matching proof.
