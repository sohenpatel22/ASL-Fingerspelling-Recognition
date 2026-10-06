from __future__ import annotations

import argparse
import json
from pathlib import Path

from asl.evaluate import main as evaluate_main
from asl.train import main as train_main

CTC = ["model.ctc=true", "train.ctc_weight=0.3"]
LONGER = ["model.max_seq_len=128"]
VELOCITY = ["model.velocity=true"]
STRONG = ["data.augment=strong"]

RUNS: dict[str, list[str]] = {
    "a0_baseline": [],
    "a1_ctc": CTC,
    "a2_len128": LONGER,
    "a3_velocity": VELOCITY,
    "a4_strong_aug": STRONG,
    "a5_all": [*CTC, *LONGER, *VELOCITY, *STRONG],
}


def markdown_table(results: dict[str, dict]) -> str:
    lines = [
        "| run | val CER (greedy, best epoch) | val CER (beam) | test CER (beam) | test 95% CI | signers |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lo, hi = r["test_ci95"]
        lines.append(
            f"| {name} | {r['best_val_greedy_cer']:.3f} | {r['val_beam_cer']:.3f} | {r['test_cer']:.3f} "
            f"| {lo:.3f} to {hi:.3f} | {r['test_signers']} |"
        )
    return "\n".join(lines)


def run_one(name: str, run_dir: Path, args: argparse.Namespace) -> dict:
    base = ["--config", args.config, *args.overrides]
    train_main([*base, *RUNS[name], f"train.out_dir={run_dir}"])

    def score(split: str, samples: int, *extra: str) -> dict:
        target = run_dir / f"{split}.json"
        evaluate_main([
            "--checkpoint", str(run_dir / "best.pth"), *base, "--split", split,
            "--max-samples", str(samples), "--sample-seed", "0", "--out", str(target), *extra,
        ])
        return json.loads(target.read_text())

    val = score("val", args.val_samples)
    test = score("test", args.test_samples, "--details-out", str(run_dir / "test_details.json"))
    history = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return {
        "overrides": RUNS[name],
        "epochs_run": int(history[-1]["epoch"]),
        "best_val_greedy_cer": min(h["val_greedy_cer"] for h in history),
        "val_beam_cer": val["cer"],
        "test_cer": test["cer"],
        "test_ci95": test["cer_ci95"],
        "test_signers": test.get("n_signers", 0),
        "test_signer_std": test.get("cer_signer_std", 0.0),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="train and score each ablation variant")
    parser.add_argument("--config", default="configs/finetune_v5.yaml")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--only", default=None, help="comma separated run names (default: all)")
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=3000)
    parser.add_argument("overrides", nargs="*", help="applied to every run, e.g. data.comp_dir=...")
    args = parser.parse_args(argv)

    names = args.only.split(",") if args.only else list(RUNS)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for name in names:
        results[name] = run_one(name, out / name, args)
        results_path.write_text(json.dumps(results, indent=2))  # saved after every run
        print(f"{name}: test CER {results[name]['test_cer']:.4f}")

    table = markdown_table(results)
    (out / "results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
