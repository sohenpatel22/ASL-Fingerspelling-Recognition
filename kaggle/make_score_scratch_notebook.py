"""Builds kaggle/score_scratch/score_scratch.ipynb and its metadata, then checks that every code cell parses."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).parent / "score_scratch"
OUT.mkdir(exist_ok=True)


def src(text):
    return text.strip("\n").splitlines(keepends=True)


def notebook(cells):
    out = []
    for kind, text in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src(text)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {
        "cells": out,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


CELLS = [
    ("markdown", "# Score the CTC head of the from-scratch model\n\n"
                 "The training log showed the CTC head decoding better than the attention decoder. This scores it with "
                 "the same validation (1000) and test (3000) clips as every other model, greedy CTC decoding."),
    ("code", '''
BRANCH = "phase-5-scratch"
!git clone -q --branch {BRANCH} https://github.com/sohenpatel22/ASL-Fingerspelling-Recognition.git /tmp/repo
%cd /tmp/repo
!pip install -q -e .
'''),
    ("code", '''
import os, zipfile, json
from pathlib import Path

def find_dir(name, root="/kaggle/input"):
    for base, dirs, _ in os.walk(root):
        if name in dirs:
            return Path(base, name)
        if base[len(root):].count(os.sep) >= 5:
            dirs[:] = []
    raise FileNotFoundError(name)

COMP = next(p for p in (Path("/kaggle/input/competitions/asl-fingerspelling"), Path("/kaggle/input/asl-fingerspelling")) if p.exists())
try:
    NPY_TRAIN, NPY_SUPP = find_dir("npy_train"), find_dir("npy_supp")
except FileNotFoundError:
    for archive in [p for p in Path("/kaggle/input").rglob("*.zip") if "competitions" not in str(p)]:
        with zipfile.ZipFile(archive) as z:
            z.extractall("/tmp/npy")
    NPY_TRAIN, NPY_SUPP = find_dir("npy_train", "/tmp/npy"), find_dir("npy_supp", "/tmp/npy")
DATA = f"data.comp_dir={COMP} data.npy_train={NPY_TRAIN} data.npy_supp={NPY_SUPP}"
import torch
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "| cpus:", os.cpu_count())
'''),
    ("code", '''
CKPT = next(Path("/kaggle/input").rglob("best.pth"))
print(CKPT)
!asl-eval --checkpoint {CKPT} --config configs/scratch.yaml --split val --max-samples 1000 --sample-seed 0 --decoder ctc --out /kaggle/working/val_ctc.json --details-out /kaggle/working/val_ctc_details.json {DATA}
!asl-eval --checkpoint {CKPT} --config configs/scratch.yaml --split test --max-samples 3000 --sample-seed 0 --decoder ctc --out /kaggle/working/test_ctc.json --details-out /kaggle/working/test_ctc_details.json {DATA}
val, test = (json.load(open(f"/kaggle/working/{s}_ctc.json")) for s in ("val", "test"))
print("ctc validation CER:", round(val["cer"], 4), "| test CER:", round(test["cer"], 4), "| corpus CER:", round(test["cer_micro"], 4))
'''),
]

(OUT / "score_scratch.ipynb").write_text(json.dumps(notebook(CELLS), indent=1), encoding="utf8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "sohenpatel/asl-fingerspelling-score-scratch", "title": "ASL Fingerspelling Score Scratch", "code_file": "score_scratch.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": True, "dataset_sources": [], "competition_sources": ["asl-fingerspelling"],
    "kernel_sources": ["sohenpatel/asl-fingerspelling-preprocess", "sohenpatel/asl-fingerspelling-scratch"], "model_sources": [],
}, indent=2), encoding="utf8")

problems = 0
for i, cell in enumerate(json.loads((OUT / "score_scratch.ipynb").read_text(encoding="utf8"))["cells"]):
    if cell["cell_type"] != "code":
        continue
    code = "".join(line for line in cell["source"] if not line.lstrip().startswith(("!", "%")))
    try:
        ast.parse(code)
    except SyntaxError as err:
        problems += 1
        print(f"cell {i}: SYNTAX ERROR {err}")
print("syntax problems:", problems)
raise SystemExit(1 if problems else 0)
