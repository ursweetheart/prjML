"""A.1: score multiple-choice records by the probability a causal LM gives to each answer letter.

Spec: openspec/specs/choice-scorer/spec.md (while in flight: openspec/changes/choice-scorer/)
"""

import numpy as np
import torch

LETTERS = "ABCD"
CUE = "Đáp án:"


def format_prompt(subject, question, options):
    body = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, options))
    return f"Câu hỏi trắc nghiệm môn {subject}. Chọn đáp án đúng.\n\n{question}\n{body}\n{CUE}"


def record_options(record):
    return [record[f"option_{letter.lower()}"] for letter in LETTERS]


def shifted(options, k):
    """Shift k shows original option (i + k) % 4 at position i; k = 0 is the original order."""
    return [options[(i + k) % 4] for i in range(4)]


def unshift(scores, k):
    """Map scores over positions (last axis) back to the original option order."""
    out = np.empty_like(scores)
    for i in range(4):
        out[..., (i + k) % 4] = scores[..., i]
    return out


def letter_ids(tok):
    """Token id of ' A' .. ' D'; each must be one token, or a prefix would be scored instead of the letter."""
    ids = []
    for letter in LETTERS:
        enc = tok(" " + letter, add_special_tokens=False).input_ids
        if len(enc) != 1:
            raise ValueError(f"' {letter}' is {len(enc)} tokens for this tokenizer; cannot score it as one letter")
        ids.append(enc[0])
    return ids


@torch.no_grad()
def letter_logprobs(model, tok, prompts, batch_size=8, max_tokens=4096):
    """(N, 4) log-probabilities of A..D, renormalised over the four letters.

    Prompts are padded on the left, so the last position is a real token for every row; batches are
    built from length-sorted prompts and capped by batch_size and a padded-token budget.
    """
    ids = letter_ids(tok)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    encoded = tok(prompts)["input_ids"]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    out = np.zeros((len(prompts), 4), dtype=np.float32)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and end - start < batch_size and (end - start + 1) * len(encoded[order[end]]) <= max_tokens:
            end += 1
        batch = order[start:end]
        enc = tok.pad({"input_ids": [encoded[i] for i in batch]}, return_tensors="pt").to(model.device)
        logits = model(**enc, logits_to_keep=1).logits[:, -1]  # only the last position; the full vocab at every position would not fit
        out[batch] = torch.log_softmax(logits[:, ids].float(), dim=-1).cpu().numpy()
        start = end
    return out


def score_records(model, tok, records, shifts=(0,), **kwargs):
    """(N, len(shifts), 4) log-probabilities in the original option order, one slice per option shift."""
    result = np.zeros((len(records), len(shifts), 4), dtype=np.float32)
    for s, k in enumerate(shifts):
        prompts = [format_prompt(r["subject"], r["question"], shifted(record_options(r), k)) for r in records]
        result[:, s] = unshift(letter_logprobs(model, tok, prompts, **kwargs), k)
    return result


def position_bias(scores, answers):
    """Per letter: share of questions where it has the top probability, next to the share where it is correct."""
    top = np.bincount(scores.argmax(-1), minlength=4) / len(scores)
    gold = np.bincount([LETTERS.index(a) for a in answers], minlength=4) / len(answers)
    return {letter: {"predicted": float(top[i]), "correct": float(gold[i])} for i, letter in enumerate(LETTERS)}


def _self_test():
    opts = ["một", "hai", "ba", "bốn"]
    p = format_prompt("Geography", "Hỏi gì?", opts)
    assert p == format_prompt("Geography", "Hỏi gì?", opts)
    assert p.startswith("Câu hỏi trắc nghiệm môn Geography.") and p.endswith("\nA. một\nB. hai\nC. ba\nD. bốn\nĐáp án:")
    # every option sits at every position exactly once across the four shifts
    seen = {(opt, pos) for k in range(4) for pos, opt in enumerate(shifted(opts, k))}
    assert len(seen) == 16 and shifted(opts, 0) == opts
    # a position-blind model scores option i the same under every shift once mapped back
    base = np.array([0.1, 0.2, 0.3, 0.4])
    for k in range(4):
        assert np.array_equal(unshift(np.array([base[(i + k) % 4] for i in range(4)]), k), base)
    # a model that always says A: 100% of the top answers are A
    bias = position_bias(np.tile(np.log([0.7, 0.1, 0.1, 0.1]), (5, 1)), list("ABCDA"))
    assert bias["A"]["predicted"] == 1.0 and bias["B"]["predicted"] == 0.0 and bias["A"]["correct"] == 0.4
