# Topic 12: guaranteed evaluation, quantization and efficient inference on Vietnamese exams

Machine Learning big assignment, class QT02, Group 4. See `machine_learning_project_proposal_topic_12.md` and `project_plan_one_week.md`.

## Data

The benchmark is built from VNHSGE (Dao et al., 2023, arXiv 2305.12199). The raw data is not redistributed here; `data/` is git-ignored.

```bash
git clone --depth 1 https://github.com/Xdao85/VNHSGE.git data/raw/VNHSGE
uv run python -m machinelearningproject.convert_vnhsge
```

This writes `data/processed/vnhsge.jsonl` (one question per line, with a calibration/test split) and `data/processed/vnhsge_dropped.jsonl` (every skipped item with its reason). The output is identical on every run for the same clone.

## Tiny Transformer

A character-level decoder-only Transformer pair (target and draft), written from scratch in `src/machinelearningproject/tiny_transformer.py`. It is the test bed for the quantizer and speculative decoding; it does not answer exam questions.

The training text is Truyện Kiều (public domain), taken from Vietnamese Wikisource; `data/` is git-ignored, so rebuild the file once:

```bash
curl -sG "https://vi.wikisource.org/w/index.php" --data-urlencode "title=Truyện Kiều" --data-urlencode "action=raw" \
  | python3 -c "import re,sys,unicodedata as u;t=sys.stdin.read().split('<poem>')[1].split('</poem>')[0];print('\n'.join(l for l in (u.normalize('NFC',re.sub(r'\{\{số\|\d+\}\}','',x)).strip() for x in t.split('\n')) if l))" \
  > data/raw/truyen_kieu.txt
uv run python -m machinelearningproject.train_tiny
```

This trains both models (about 2 minutes on a GPU, longer on CPU) and writes `target.pt`, `draft.pt` and `vocab.json` to `data/processed/tiny/`. Use `--steps N` to change the number of steps and `--models draft` to train one model. Load a model with `load_model(path)` and the vocabulary with `Tokenizer.load(path)`.

## Scorer check

`src/machinelearningproject/scorer.py` returns, for each benchmark question, the log-probability of the answer letters A–D from a Hugging Face causal language model (renormalised over the four letters), with four cyclic option shifts to measure position bias. This command checks it against `lm-eval` on 100 test questions:

```bash
uv run python -m machinelearningproject.check_scorer
```

It downloads `Qwen/Qwen2.5-0.5B` (about 1 GB), prints both accuracies, the accuracy per option shift and the position bias, and exits 0 when the accuracies differ by at most 0.01. It needs `data/processed/vnhsge.jsonl` (see Data) and runs in about one minute on a 4 GB GPU.

## Quantizer check

`src/machinelearningproject/quantizer.py` quantizes weight matrices group-wise to 8 or 4 bits, symmetric or asymmetric, and `quantize_model` does it for every linear layer of a model. This command applies it to the trained tiny models (see Tiny Transformer above) and prints the per-layer error and the validation loss at FP32, INT8 and INT4:

```bash
uv run python -m machinelearningproject.check_quantizer
```

It needs `data/processed/tiny/` and the Truyện Kiều text, runs in about 15 seconds on a CPU and exits 0 when INT8 changes the loss less than INT4 does.
