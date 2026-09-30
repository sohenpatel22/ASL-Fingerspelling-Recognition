from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from asl.checkpoint import load_checkpoint
from asl.config import DecodeConfig, load_config
from asl.data import build_datasets
from asl.decode import beam_search
from asl.lm import CharNgramLM
from asl.metrics import bootstrap_ci, cer, corpus_cer, exact_match, group_cer, mean_cer
from asl.model import ASLConformerSeq2Seq
from asl.train import resolve_device
from asl.vocab import Vocab


@torch.no_grad()
def evaluate_dataset(
    model: ASLConformerSeq2Seq,
    dataset: Dataset,
    vocab: Vocab,
    decode_cfg: DecodeConfig,
    max_len: int,
    device: str | torch.device = "cpu",
    max_samples: int | None = None,
    lm: CharNgramLM | None = None,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    model.eval()
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    # a seeded random subset covers many signers; the first n rows usually do not
    if sample_seed is None:
        order = np.arange(n)
    else:
        order = np.random.RandomState(sample_seed).permutation(len(dataset))[:n]
    preds, tgts = [], []
    for i in tqdm(order, desc="evaluating"):
        x, y = dataset[i]
        tokens, _ = beam_search(
            model, x.to(device), vocab, decode_cfg.beam_width, max_len, decode_cfg.length_penalty,
            lm=lm, lm_weight=decode_cfg.lm_weight,
        )
        preds.append(vocab.decode(tokens))
        tgts.append(vocab.decode(y.tolist()[1:]))

    per_sample = [cer(p, t) for p, t in zip(preds, tgts, strict=True) if t]
    lo, hi = bootstrap_ci(per_sample)
    result: dict[str, Any] = {
        "n_samples": n,
        "cer": mean_cer(preds, tgts),
        "cer_micro": corpus_cer(preds, tgts),
        "cer_ci95": [lo, hi],
        "exact_match": exact_match(preds, tgts),
        "beam_width": decode_cfg.beam_width,
        "length_penalty": decode_cfg.length_penalty,
        "lm_weight": decode_cfg.lm_weight if lm is not None else 0.0,
    }
    df = getattr(dataset, "df", None)
    if df is not None and "participant_id" in df.columns:
        by_signer = group_cer(preds, tgts, df["participant_id"].iloc[order].tolist())
        vals = list(by_signer.values())
        result["n_signers"] = len(vals)
        result["cer_signer_mean"] = float(np.mean(vals))
        result["cer_signer_std"] = float(np.std(vals))
        result["cer_by_signer"] = by_signer
    has_signers = df is not None and "participant_id" in df.columns
    signers = df["participant_id"].iloc[order].tolist() if has_signers else [None] * len(order)
    records = [
        {"signer": str(s), "target": t, "prediction": p, "cer": cer(p, t) if t else None}
        for s, p, t in zip(signers, preds, tgts, strict=True)
    ]
    result["predictions"] = records
    result["examples"] = [{"target": r["target"], "prediction": r["prediction"]} for r in records[:20]]
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with beam search")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None, help="pick a random subset with this seed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="write summary metrics json here")
    parser.add_argument("--details-out", default=None, help="write per-signer cer and examples here")
    parser.add_argument("--allow-unsafe", action="store_true", help="trust legacy pickled ckpt")
    parser.add_argument(
        "--sweep-lm-weights", default=None, help="comma separated weights to try (use with --split val)"
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    cfg = load_config(args.config, args.overrides)
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device, allow_unsafe=args.allow_unsafe)
    cfg.model = ckpt["_model_config"]  # data must match the model, whatever the config says
    decode_cfg = DecodeConfig(
        beam_width=int(ckpt.get("beam_width", cfg.decode.beam_width)),
        length_penalty=float(ckpt.get("length_penalty", cfg.decode.length_penalty)),
        lm_path=cfg.decode.lm_path,
        lm_weight=cfg.decode.lm_weight,
    )
    lm = CharNgramLM.load(decode_cfg.lm_path) if decode_cfg.lm_path else None
    dataset = build_datasets(cfg, vocab)[args.split]
    max_len = ckpt["_model_config"].max_phrase_len

    if args.sweep_lm_weights:
        if lm is None:
            raise SystemExit("--sweep-lm-weights needs decode.lm_path")
        table = {}
        for w in [float(v) for v in args.sweep_lm_weights.split(",")]:
            trial = DecodeConfig(decode_cfg.beam_width, decode_cfg.length_penalty, decode_cfg.lm_path, w)
            res = evaluate_dataset(
                model, dataset, vocab, trial, max_len, device, args.max_samples, lm, args.sample_seed
            )
            table[w] = res["cer"]
            print(f"lm_weight {w}: CER {table[w]:.4f}")
        best = min(table, key=table.get)
        print(f"best lm_weight on {args.split}: {best} (CER {table[best]:.4f})")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({"sweep": table, "best_lm_weight": best}, indent=2))
        return

    result = evaluate_dataset(
        model, dataset, vocab, decode_cfg, max_len, device, args.max_samples, lm, args.sample_seed
    )
    summary = {k: v for k, v in result.items() if k not in ("cer_by_signer", "examples", "predictions")}
    print(json.dumps(summary, indent=2))
    for path, payload in ((args.out, summary), (args.details_out, result)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
