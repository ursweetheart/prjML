# Topic 12: guaranteed evaluation, quantization and efficient inference on Vietnamese exams

Machine Learning big assignment, class QT02, Group 4. See `machine_learning_project_proposal_topic_12.md` and `project_plan_one_week.md`.

## Data

The benchmark is built from VNHSGE (Dao et al., 2023, arXiv 2305.12199). The raw data is not redistributed here; `data/` is git-ignored.

```bash
git clone --depth 1 https://github.com/Xdao85/VNHSGE.git data/raw/VNHSGE
uv run python -m machinelearningproject.convert_vnhsge
```

This writes `data/processed/vnhsge.jsonl` (one question per line, with a calibration/test split) and `data/processed/vnhsge_dropped.jsonl` (every skipped item with its reason). The output is identical on every run for the same clone.
