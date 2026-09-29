from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asl.evaluate import main as evaluate_main
from asl.train import main as train_main


def summarize(cers: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(cers)), "std": float(np.std(cers)), "min": min(cers), "max": max(cers)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="k-fold cross-validation over signers")
    parser.add_argument("--config", default=None)
    parser.add_argument("--folds", type=int, default=5, help="number of folds")
    parser.add_argument("--run", default=None, help="comma separated fold ids to run (default: all)")
    parser.add_argument("--out-dir", default="checkpoints/cv")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    which = [int(f) for f in args.run.split(",")] if args.run else list(range(args.folds))
    config = ["--config", args.config] if args.config else []
    results = {}
    for fold in which:
        run_dir = Path(args.out_dir) / f"fold{fold}"
        fold_overrides = [f"data.cv_folds={args.folds}", f"data.cv_fold={fold}", f"train.out_dir={run_dir}"]
        train_main([*config, *args.overrides, *fold_overrides])
        summary_path = run_dir / "test_summary.json"
        evaluate_main([
            "--checkpoint", str(run_dir / "best.pth"), *config, "--split", "test",
            "--out", str(summary_path), *args.overrides, *fold_overrides,
        ])
        results[fold] = json.loads(summary_path.read_text())["cer"]
        print(f"fold {fold}: test CER {results[fold]:.4f}")

    report = {"folds": args.folds, "cer_by_fold": results, **summarize(list(results.values()))}
    out = Path(args.out_dir) / "cv_results.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"test CER over {len(results)} signer folds: {report['mean']:.4f} +/- {report['std']:.4f}")


if __name__ == "__main__":
    main()
