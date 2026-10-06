"""Builds kaggle/tune/tune.ipynb and its metadata, then checks that every code cell parses."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).parent / "tune"
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
    ("markdown", "# Thresholds, decoding and a careful fine-tune\n\n"
                 "Starts from the v5 last-epoch checkpoint on the Hub. Everything is chosen on validation: the "
                 "flag thresholds, the length penalty and whether a fine-tune helps. Test is scored once per "
                 "decision."),
    ("code", '''
BRANCH = "phase-4-accuracy"
!git clone -q --branch {BRANCH} https://github.com/sohenpatel22/ASL-Fingerspelling-Recognition.git /tmp/repo
%cd /tmp/repo
!pip install -q -e .
'''),
    ("code", '''
import os, zipfile
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
print(LAST)
'''),
    ("markdown", "## 1. Flag thresholds, chosen on validation"),
    ("code", '''
import json
from asl.gating import pick_flag_rule

!asl-eval --checkpoint {LAST} --config configs/finetune_v5.yaml --split val --max-samples 1000 --sample-seed 0 --out /kaggle/working/val_last.json --details-out /kaggle/working/val_last_details.json {DATA}
val_records = json.load(open("/kaggle/working/val_last_details.json"))["predictions"]
rule = pick_flag_rule(val_records, max_flagged=0.35)
json.dump(rule, open("/kaggle/working/flag_rule.json", "w"), indent=1)
print(rule)
'''),
    ("markdown", "## 2. Length penalty, chosen on validation"),
    ("code", '''
val_cer = {0.6: json.load(open("/kaggle/working/val_last.json"))["cer"]}
for lp in (0.0, 0.1, 0.2, 0.3, 0.4):
    !asl-eval --checkpoint {LAST} --config configs/finetune_v5.yaml --split val --max-samples 1000 --sample-seed 0 --length-penalty {lp} --out /kaggle/working/val_lp{lp}.json {DATA}
    val_cer[lp] = json.load(open(f"/kaggle/working/val_lp{lp}.json"))["cer"]
best_lp = min(val_cer, key=val_cer.get)
print("validation CER by length penalty:", {k: round(v, 4) for k, v in sorted(val_cer.items())})
print("chosen length penalty:", best_lp)
'''),
    ("markdown", "## 3. Test, once, with the chosen length penalty and the validation-fitted flag rule"),
    ("code", '''
from asl.gating import apply_flag_rule

!asl-eval --checkpoint {LAST} --config configs/finetune_v5.yaml --split test --max-samples 3000 --sample-seed 0 --length-penalty {best_lp} --out /kaggle/working/test_last.json --details-out /kaggle/working/test_last_details.json {DATA}
assert Path("/kaggle/working/test_last.json").exists(), "test scoring failed: read the output above"
test = json.load(open("/kaggle/working/test_last.json"))
records = json.load(open("/kaggle/working/test_last_details.json"))["predictions"]
applied = apply_flag_rule(records, rule)
print("test CER:", round(test["cer"], 4), "corpus CER:", round(test["cer_micro"], 4))
print("flag rule fitted on validation, applied to test:", {k: round(v, 3) for k, v in applied.items()})
json.dump({"test": test, "flag_rule_on_test": applied, "length_penalty": best_lp, "val_cer_by_lp": val_cer}, open("/kaggle/working/summary_tune.json", "w"), indent=1)
'''),
    ("markdown", "## 4. Does a careful fine-tune beat the last checkpoint?\n\n"
                 "Main data only, tiny learning rate, no label smoothing, weight averaging, and evaluation every quarter epoch."),
    ("code", '''
!asl-train --config configs/finetune_v5.yaml train.init_from=hf:SohenP/asl-fingerspelling-conformer/asl_v5_last.pth data.use_supplemental=false train.lr=1e-5 train.label_smoothing=0.0 train.epoch_fraction=0.25 train.epochs=16 train.ema_decay=0.999 train.patience=6 train.warmup_epochs=1 train.eval_samples=1000 train.out_dir=/kaggle/working/ft {DATA}
assert Path("/kaggle/working/ft/best.pth").exists(), "the fine-tune failed: read the output above"
'''),
    ("code", '''
!asl-eval --checkpoint /kaggle/working/ft/best.pth --config configs/finetune_v5.yaml --split val --max-samples 1000 --sample-seed 0 --length-penalty {best_lp} --out /kaggle/working/val_ft.json {DATA}
ft_val = json.load(open("/kaggle/working/val_ft.json"))["cer"]
base_val = val_cer[best_lp]
print("validation CER, last checkpoint:", round(base_val, 4), " fine-tuned (EMA):", round(ft_val, 4))
helps = ft_val < base_val - 0.005
print("fine-tune helps on validation by more than 0.005:", helps)
'''),
    ("code", '''
if helps:
    !asl-eval --checkpoint /kaggle/working/ft/best.pth --config configs/finetune_v5.yaml --split test --max-samples 3000 --sample-seed 0 --length-penalty {best_lp} --out /kaggle/working/test_ft.json --details-out /kaggle/working/test_ft_details.json {DATA}
    print(open("/kaggle/working/test_ft.json").read())
else:
    print("not scoring the fine-tune on test: it did not help on validation")
history = [json.loads(l) for l in open("/kaggle/working/ft/metrics.jsonl")]
print([round(h["val_greedy_cer"], 4) for h in history])
'''),
]

(OUT / "tune.ipynb").write_text(json.dumps(notebook(CELLS), indent=1), encoding="utf8")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": "sohenpatel/asl-fingerspelling-tune", "title": "ASL Fingerspelling Tune", "code_file": "tune.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": True, "dataset_sources": [], "competition_sources": ["asl-fingerspelling"],
    "kernel_sources": ["sohenpatel/asl-fingerspelling-preprocess"], "model_sources": [],
}, indent=2), encoding="utf8")

problems = 0
for i, cell in enumerate(json.loads((OUT / "tune.ipynb").read_text(encoding="utf8"))["cells"]):
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
