"""Builds kaggle/scratch/scratch.ipynb and its metadata, then checks that every code cell parses."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).parent / "scratch"
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
    ("markdown", "# From scratch, with the top solutions' ingredients\n\n"
                 "Trains the model from random weights with heavy augmentation, CutMix, decoder input masking, "
                 "a joint CTC loss and weight averaging, at 192 frames. Scored with the same protocol as every "
                 "other model (3000 random test clips, beam 5, length penalty 0.0) so it can be compared with "
                 "the 0.307 of the tuned v5 checkpoint. SMOKE = True runs a five minute check of every stage first."),
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
SMOKE = False   # True = about 5 minutes of tiny slices of every stage
if SMOKE:
    EXTRA, N_VAL, N_TEST = "train.epochs=2 train.epoch_fraction=0.1 train.eval_samples=200 data.use_supplemental=false", 100, 100
else:
    EXTRA, N_VAL, N_TEST = "", 1000, 3000

!asl-train --config configs/scratch.yaml train.out_dir=/kaggle/working/scratch {DATA} {EXTRA}
assert Path("/kaggle/working/scratch/best.pth").exists(), "training failed: read the output above"
history = [json.loads(l) for l in open("/kaggle/working/scratch/metrics.jsonl")]
for h in history:
    print(int(h["epoch"]), "train", round(h["train_loss"], 3), "val greedy CER", round(h["val_greedy_cer"], 4), "ctc CER", round(h.get("val_ctc_cer", float("nan")), 4), round(h["seconds"]), "s")
'''),
    ("code", '''
!asl-eval --checkpoint /kaggle/working/scratch/best.pth --config configs/scratch.yaml --split val --max-samples {N_VAL} --sample-seed 0 --length-penalty 0.0 --out /kaggle/working/val_scratch.json --details-out /kaggle/working/val_scratch_details.json {DATA}
!asl-eval --checkpoint /kaggle/working/scratch/best.pth --config configs/scratch.yaml --split test --max-samples {N_TEST} --sample-seed 0 --length-penalty 0.0 --out /kaggle/working/test_scratch.json --details-out /kaggle/working/test_scratch_details.json {DATA}
assert Path("/kaggle/working/test_scratch.json").exists(), "scoring failed: read the output above"
val, test = (json.load(open(f"/kaggle/working/{s}_scratch.json")) for s in ("val", "test"))
print("validation CER:", round(val["cer"], 4), "| test CER:", round(test["cer"], 4), "| corpus CER:", round(test["cer_micro"], 4))
print("for comparison the tuned v5 checkpoint scores 0.2698 on validation and 0.3073 on test (when SMOKE is False)")
'''),
]

(OUT / "scratch.ipynb").write_text(json.dumps(notebook(CELLS), indent=1), encoding="utf8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "sohenpatel/asl-fingerspelling-scratch", "title": "ASL Fingerspelling Scratch", "code_file": "scratch.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": True, "dataset_sources": [], "competition_sources": ["asl-fingerspelling"],
    "kernel_sources": ["sohenpatel/asl-fingerspelling-preprocess"], "model_sources": [],
}, indent=2), encoding="utf8")

problems = 0
for i, cell in enumerate(json.loads((OUT / "scratch.ipynb").read_text(encoding="utf8"))["cells"]):
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
