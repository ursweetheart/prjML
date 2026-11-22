"""Convert the VNHSGE clone into the benchmark file data/processed/vnhsge.jsonl.

Run: uv run python -m machinelearningproject.convert_vnhsge
Spec: openspec/changes/convert-vnhsge-dataset/specs/benchmark-dataset/spec.md
"""

import collections
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/VNHSGE"
JSON_DIR = RAW / "Dataset/VNHSGE-V/JSON format"
OUT = ROOT / "data/processed/vnhsge.jsonl"
DROPPED = ROOT / "data/processed/vnhsge_dropped.jsonl"
SEED = 0
CALIBRATION_FRACTION = 0.5
TYPE_RANK = {"official": 0, "illustrative": 1, "practice": 2}
UNKNOWN_YEAR = "unknown (≤2023)"

# "A." or "A)" after whitespace or ? . : (not after "+" or "(", so "+C." and "(B)" stay text),
# or a bare "A " at the start of a line.
LABEL = re.compile(r"(?<![^\s?.:])([ABCD])[.)]|^[ \t]*([ABCD])(?=[ \t]+\S)", re.M)
QUESTION_NUMBER = re.compile(r"^\s*Câu\s*\d+\s*[:.]\s*")
YEAR = re.compile(r"(?<!\d)(20[12]\d)(?!\d)")


def parse_options(text):
    """Return (stem, [a, b, c, d]) or None. Uses the last A. that has B., C., D. after it, in order,
    and yields a non-empty stem and options."""
    labels = []
    for m in LABEL.finditer(text):
        g = 1 if m.group(1) else 2
        labels.append((m.group(g), m.start(g), m.end()))
    for i in range(len(labels) - 1, -1, -1):
        if labels[i][0] != "A":
            continue
        chain, want = [labels[i]], "BCD"
        for lab in labels[i + 1:]:
            if want and lab[0] == want[0]:
                chain.append(lab)
                want = want[1:]
        if want:
            continue
        ends = [lab[1] for lab in chain[1:]] + [len(text)]
        options = [text[lab[2]:end].strip() for lab, end in zip(chain, ends)]
        stem = QUESTION_NUMBER.sub("", text[:chain[0][1]]).strip()
        if stem and all(options):
            return stem, options
    return None


def exam_type_and_year(name):
    year = YEAR.search(name)
    if "_IE_" in name:
        kind = "illustrative"
    elif name.startswith("MET_") and year:
        kind = "official"
    else:
        kind = "practice"
    return kind, year.group(1) if year else UNKNOWN_YEAR


def has_figure(q):
    # A text-only model cannot see a figure, whether it is a separate image or embedded LaTeX.
    return bool(str(q.get("Image_Question", "")).strip()) or "\\includegraphics" in str(q.get("Question", ""))


def _self_test():
    assert has_figure({"Image_Question": "a.png", "Question": "x"})
    assert has_figure({"Image_Question": "", "Question": "Hình \\includegraphics{f.png}\nA. x"})
    assert not has_figure({"Image_Question": "", "Question": "Hỏi gì?\nA. x"})
    assert parse_options("Câu 1:  Hỏi gì?\nA. x\nB. y\nC. z\nD. w") == ("Hỏi gì?", ["x", "y", "z", "w"])
    assert parse_options("Câu 17 : Nước nào? \n\tA. Trung Quốc. \tB. Nhật Bản. \tC. Pháp. \tD. Ấn Độ.") == (
        "Nước nào?", ["Trung Quốc.", "Nhật Bản.", "Pháp.", "Ấn Độ."])
    assert parse_options("tăng chủ yếu do A. x\nB. y\nC. z\nD. w") == ("tăng chủ yếu do", ["x", "y", "z", "w"])
    stem, opts = parse_options("Ai vi phạm?\nA. Anh D, ông A và anh\nC. B. Chị B và ông A.\nC. Ông A, anh C.\nD. Ông A.")
    assert opts == ["Anh D, ông A và anh\nC.", "Chị B và ông A.", "Ông A, anh C.", "Ông A."], opts
    assert parse_options("Tim 2 ngăn?\nA. Ếch.\nB. Cá.\nC. Mèo.\nC. Thỏ") is None
    assert parse_options("Câu 3. Hỏi?\nA. 1\nB. 2\nC. 3\nD. 4")[0] == "Hỏi?"
    assert parse_options("A. x\nB. y\nC. z\nD. w") is None  # no stem
    assert parse_options("Ai sai?\n\nA. Ông C và anh A.\nB. Anh V.\nC. Ông C.\nD. Anh Q.")[1] == [
        "Ông C và anh A.", "Anh V.", "Ông C.", "Anh Q."]  # "anh A." inside option A
    assert parse_options("vô sinh?A. Động vật.\nB. Nhiệt độ.\nC. Thực vật.\nD. Vi sinh vật.") == (
        "vô sinh?", ["Động vật.", "Nhiệt độ.", "Thực vật.", "Vi sinh vật."])
    assert parse_options("Question 6: the medicine\nA wonderful\nB. secret\nC attractive\nD. terrible")[1] == [
        "wonderful", "secret", "attractive", "terrible"]
    assert parse_options("Ý nào sai?\nA. x\nC. z\nB. y\nD. w") is None  # labels out of order
    assert parse_options("Nguyên hàm là\nA. x^2+C.\nB. e^x+C.\nC. x+C.\nD. 2x+C.")[1] == [
        "x^2+C.", "e^x+C.", "x+C.", "2x+C."]  # "+C." is the integration constant, not a label
    assert parse_options("Màu?\nA. (A): xanh và (B): tím.\nB. (A): tím.\nC. (B): vàng.\nD. (A): đỏ.")[1] == [
        "(A): xanh và (B): tím.", "(A): tím.", "(B): vàng.", "(A): đỏ."]
    assert exam_type_and_year("MET_Math_OE_2021_101") == ("official", "2021")
    assert exam_type_and_year("MET_Civ_2022_301") == ("official", "2022")
    assert exam_type_and_year("MET_Bio_IE_2020") == ("illustrative", "2020")
    assert exam_type_and_year("Phy_41") == ("practice", UNKNOWN_YEAR)


def _git(*args):
    return subprocess.run(["git", "-C", str(RAW), *args], check=True, capture_output=True, text=True).stdout.strip()


def build():
    commit, date = _git("rev-parse", "HEAD"), _git("log", "-1", "--format=%cs")
    candidates, dropped = [], []
    for path in sorted(JSON_DIR.rglob("*.json")):
        rel = path.relative_to(JSON_DIR).as_posix()
        subject, name = path.parent.name, path.stem
        kind, year = exam_type_and_year(name)
        for n, q in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            def drop(reason):
                dropped.append({"source_id": q.get("ID"), "file": rel, "reason": reason})
            if subject == "Literature":
                drop("literature"); continue
            if has_figure(q):
                drop("image"); continue
            answer = str(q.get("Choice", "")).strip()
            if answer not in ("A", "B", "C", "D"):
                drop("answer"); continue
            parsed = parse_options(unicodedata.normalize("NFC", q["Question"]))
            if parsed is None:
                drop("options"); continue
            stem, opts = parsed
            candidates.append((TYPE_RANK[kind], rel, n, {
                "id": f"vnhsge:{q['ID']}", "subject": subject, "year": year, "exam": name,
                "exam_type": kind, "level": None, "question": stem,
                "option_a": opts[0], "option_b": opts[1], "option_c": opts[2], "option_d": opts[3],
                "answer": answer,
                "source_url": f"https://github.com/Xdao85/VNHSGE/blob/{commit}/Dataset/VNHSGE-V/{quote('JSON format/' + rel)}",
                "retrieved_at": date, "split": None, "variant_of": None,
            }))

    seen, records = set(), []
    for _, rel, _, rec in sorted(candidates, key=lambda c: c[:3]):
        key = " ".join(" ".join([rec["question"], *(rec[f"option_{x}"] for x in "abcd")]).split())
        if key in seen:
            dropped.append({"source_id": rec["id"].removeprefix("vnhsge:"), "file": rel, "reason": "duplicate"})
            continue
        seen.add(key)
        records.append(rec)

    records.sort(key=lambda r: r["id"])
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r["subject"], r["exam_type"])].append(r)
    for group in groups.values():
        random.Random(SEED).shuffle(group)
        cut = int(len(group) * CALIBRATION_FRACTION)
        for i, r in enumerate(group):
            r["split"] = "calibration" if i < cut else "test"
    return records, dropped


def validate(records):
    ids = collections.Counter(r["id"] for r in records)
    for r in records:
        problems = [f for f in ("question", "option_a", "option_b", "option_c", "option_d") if not r[f]]
        if r["answer"] not in ("A", "B", "C", "D"):
            problems.append("answer")
        if ids[r["id"]] > 1:
            problems.append("duplicate id")
        if problems:
            sys.exit(f"invalid record {r['id']}: {', '.join(problems)}; nothing written")


def write_jsonl(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    _self_test()
    records, dropped = build()
    validate(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT, records)
    write_jsonl(DROPPED, dropped)

    kept = collections.Counter((r["subject"], r["exam_type"]) for r in records)
    print(f"kept {len(records)} records -> {OUT.relative_to(ROOT)}")
    for (subject, kind), n in sorted(kept.items()):
        print(f"  {subject:16} {kind:13} {n:6}")
    print(f"dropped {len(dropped)} -> {DROPPED.relative_to(ROOT)}")
    for reason, n in collections.Counter(d["reason"] for d in dropped).most_common():
        print(f"  {reason:16} {n:6}")
    splits = collections.Counter(r["split"] for r in records)
    print(f"split: calibration {splits['calibration']}, test {splits['test']}")


if __name__ == "__main__":
    main()
