"""A.1 check: our scorer against lm-eval on 100 seeded test-split questions, same prompts, same model.

Run: uv run python -m machinelearningproject.check_scorer [--model NAME]   (downloads the model, about 1 GB)
Spec: openspec/specs/choice-scorer/spec.md (while in flight: openspec/changes/choice-scorer/)
"""

import argparse
import gc
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from machinelearningproject.scorer import (
    LETTERS, format_prompt, letter_logprobs, position_bias, record_options, score_records,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/vnhsge.jsonl"
N, SEED, TOLERANCE = 100, 0, 0.01

TASK_YAML = """task: vnhsge_check
dataset_path: json
dataset_kwargs:
  data_files:
    test: {path}
output_type: multiple_choice
test_split: test
doc_to_text: "{{{{prompt}}}}"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: "{{{{label}}}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
"""  # the default target delimiter is one space, so the continuation is " A".." D", as in our scorer


def sample_records():
    records = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    test = sorted((r for r in records if r["split"] == "test"), key=lambda r: r["id"])
    return random.Random(SEED).sample(test, N)  # sorted first, so the sample does not depend on file order


def lm_eval_predictions(model_name, records, dtype):
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    with tempfile.TemporaryDirectory() as tmp:
        prompts = Path(tmp) / "prompts.jsonl"
        with prompts.open("w", encoding="utf-8") as f:
            for r in records:
                prompt = format_prompt(r["subject"], r["question"], record_options(r))
                f.write(json.dumps({"prompt": prompt, "label": LETTERS.index(r["answer"])}, ensure_ascii=False) + "\n")
        (Path(tmp) / "vnhsge_check.yaml").write_text(TASK_YAML.format(path=prompts), encoding="utf-8")
        res = simple_evaluate(
            model="hf", model_args=f"pretrained={model_name},dtype={dtype}", tasks=["vnhsge_check"],
            task_manager=TaskManager(include_path=tmp), batch_size=1,  # lm-eval keeps full-vocabulary logits per batch; 8 does not fit in 4 GB
            log_samples=True, random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0,
        )
    samples = sorted(res["samples"]["vnhsge_check"], key=lambda s: s["doc_id"])
    pred = [int(np.argmax([float(x[0]) for x in s["filtered_resps"]])) for s in samples]
    return res["results"]["vnhsge_check"]["acc,none"], pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    name = ap.parse_args().model
    records = sample_records()
    gold = np.array([LETTERS.index(r["answer"]) for r in records])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(device).eval()

    scores = score_records(model, tok, records, shifts=(0, 1, 2, 3))  # (N, 4 shifts, 4 letters) in original option order
    ours = scores[:, 0]
    assert np.isfinite(ours).all() and np.allclose(np.exp(ours).sum(-1), 1, atol=1e-5), "scores are not a normalised distribution"
    print("scores are finite and each row sums to 1")
    prompts = [format_prompt(r["subject"], r["question"], record_options(r)) for r in records[:16]]
    drift16 = np.abs(letter_logprobs(model, tok, prompts, batch_size=1) - letter_logprobs(model, tok, prompts, batch_size=8)).max()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # padding is only checkable in float32: in float16 the two batch sizes differ by rounding noise (about 0.01)
    m32 = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    drift32 = np.abs(letter_logprobs(m32, tok, prompts, batch_size=1) - letter_logprobs(m32, tok, prompts, batch_size=8)).max()
    del m32
    print(f"batch size 1 vs 8, max |log-prob| difference: float32 {drift32:.6f}, {dtype} {drift16:.4f}")
    assert drift32 < 1e-4, "batch size changes the scores: the padding is wrong"
    acc_ours = float((ours.argmax(-1) == gold).mean())
    ours_pred = ours.argmax(-1)

    gc.collect()
    torch.cuda.empty_cache()
    acc_lm, lm_pred = lm_eval_predictions(name, records, dtype)

    print(f"\nmodel {name}, {N} test questions, seed {SEED}")
    print(f"accuracy: scorer {acc_ours:.3f}, lm-eval {acc_lm:.3f}, difference {abs(acc_ours - acc_lm):.3f}")
    print(f"questions where the predicted letter differs: {int((ours_pred != np.array(lm_pred)).sum())} of {N}")
    print("\naccuracy per option shift (original order mapped back):")
    for k in range(4):
        print(f"  shift {k}: {(scores[:, k].argmax(-1) == gold).mean():.3f}")
    print(f"  mean of the four shifts: {(np.exp(scores).mean(1).argmax(-1) == gold).mean():.3f}")
    print("\nposition bias, original order (share predicted / share correct):")
    for letter, v in position_bias(ours, [r["answer"] for r in records]).items():
        print(f"  {letter}: {v['predicted']:.2f} / {v['correct']:.2f}")
    ok = abs(acc_ours - acc_lm) <= TOLERANCE + 1e-9
    print(f"\n{'PASS' if ok else 'FAIL'}: tolerance {TOLERANCE}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
