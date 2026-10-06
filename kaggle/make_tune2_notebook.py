"""Builds kaggle/tune2/tune2.ipynb and its metadata, then checks that every code cell parses."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).parent / "tune2"
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
    ("markdown", "# Decoding beyond length penalty 0.0, and the fine-tune that crashed last time\n\n"
                 "The last run's length-penalty sweep was still improving at its edge (0.0), so this tries negative "
                 "values and other beam widths on validation. It also reruns the careful fine-tune (the learning rate "
                 "was parsed as text last time). Test is scored only for choices that win on validation."),
    ("code", '''
BRANCH = "phase-4-accuracy"
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

from asl.checkpoint import fetch_hub_file
LAST = str(fetch_hub_file("hf:SohenP/asl-fingerspelling-conformer/asl_v5_last.pth"))
'''),
    ("markdown", "## 1. More decoding settings, on validation"),
    ("code", '''
settings = {"lp0.0_b5": (0.0, 5), "lp-0.2_b5": (-0.2, 5), "lp-0.4_b5": (-0.4, 5), "lp0.0_b3": (0.0, 3), "lp0.0_b8": (0.0, 8)}
val = {}
for tag, (lp, bw) in settings.items():
    !asl-eval --checkpoint {LAST} --config configs/finetune_v5.yaml --split val --max-samples 1000 --sample-seed 0 --length-penalty {lp} --beam-width {bw} --out /kaggle/working/val_{tag}.json {DATA}
    val[tag] = json.load(open(f"/kaggle/working/val_{tag}.json"))["cer"]
best_tag = min(val, key=val.get)
best_lp, best_bw = settings[best_tag]
print("validation CER by setting:", {k: round(v, 4) for k, v in val.items()})
print("best:", best_tag, "(previous best was lp 0.0, beam 5)")
json.dump({"val": val, "best": best_tag}, open("/kaggle/working/decode_sweep2.json", "w"), indent=1)
'''),
    ("markdown", "## 2. The careful fine-tune, rerun\n\nMain data only, learning rate 1e-5, no label smoothing, weight averaging, evaluation every quarter epoch."),
    ("code", '''
!asl-train --config configs/finetune_v5.yaml train.init_from=hf:SohenP/asl-fingerspelling-conformer/asl_v5_last.pth data.use_supplemental=false train.lr=1e-5 train.label_smoothing=0.0 train.epoch_fraction=0.25 train.epochs=16 train.ema_decay=0.999 train.patience=6 train.warmup_epochs=1 train.eval_samples=1000 train.out_dir=/kaggle/working/ft {DATA}
assert Path("/kaggle/working/ft/best.pth").exists(), "the fine-tune failed: read the output above"
history = [json.loads(l) for l in open("/kaggle/working/ft/metrics.jsonl")]
print("greedy validation CER per quarter epoch:", [round(h["val_greedy_cer"], 4) for h in history])
'''),
    ("code", '''
!asl-eval --checkpoint /kaggle/working/ft/best.pth --config configs/finetune_v5.yaml --split val --max-samples 1000 --sample-seed 0 --length-penalty {best_lp} --beam-width {best_bw} --out /kaggle/working/val_ft.json {DATA}
ft_val = json.load(open("/kaggle/working/val_ft.json"))["cer"]
base_val = val[best_tag]
helps = ft_val < base_val - 0.005
print("validation CER, last checkpoint:", round(base_val, 4), "| fine-tuned (EMA):", round(ft_val, 4), "| helps by more than 0.005:", helps)
'''),
    ("markdown", "## 3. Test, once per decision that won on validation"),
    ("code", '''
results = {}
if best_tag != "lp0.0_b5" and val[best_tag] < val["lp0.0_b5"] - 0.003:
    !asl-eval --checkpoint {LAST} --config configs/finetune_v5.yaml --split test --max-samples 3000 --sample-seed 0 --length-penalty {best_lp} --beam-width {best_bw} --out /kaggle/working/test_decode.json --details-out /kaggle/working/test_decode_details.json {DATA}
    results["decode"] = json.load(open("/kaggle/working/test_decode.json"))
    print("new decoding setting on test:", best_tag, round(results["decode"]["cer"], 4), "(lp 0.0, beam 5 gave 0.3073)")
else:
    print("no new decoding setting beat lp 0.0 / beam 5 on validation by more than 0.003; not scoring test")
if helps:
    !asl-eval --checkpoint /kaggle/working/ft/best.pth --config configs/finetune_v5.yaml --split test --max-samples 3000 --sample-seed 0 --length-penalty {best_lp} --beam-width {best_bw} --out /kaggle/working/test_ft.json --details-out /kaggle/working/test_ft_details.json {DATA}
    results["finetune"] = json.load(open("/kaggle/working/test_ft.json"))
    print("fine-tuned model on test:", round(results["finetune"]["cer"], 4))
else:
    print("fine-tune did not help on validation; not scoring test")
json.dump(results, open("/kaggle/working/summary_tune2.json", "w"), indent=1)
'''),
]

(OUT / "tune2.ipynb").write_text(json.dumps(notebook(CELLS), indent=1), encoding="utf8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "sohenpatel/asl-fingerspelling-tune2", "title": "ASL Fingerspelling Tune 2", "code_file": "tune2.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": True, "dataset_sources": [], "competition_sources": ["asl-fingerspelling"],
    "kernel_sources": ["sohenpatel/asl-fingerspelling-preprocess"], "model_sources": [],
}, indent=2), encoding="utf8")

problems = 0
for i, cell in enumerate(json.loads((OUT / "tune2.ipynb").read_text(encoding="utf8"))["cells"]):
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
