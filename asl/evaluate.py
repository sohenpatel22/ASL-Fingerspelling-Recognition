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
from asl.metrics import bootstrap_ci, cer, exact_match, group_cer, mean_cer
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
) -> dict[str, Any]:
    model.eval()
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    preds, tgts = [], []
    for i in tqdm(range(n), desc="evaluating"):
        x, y = dataset[i]
        tokens, _ = beam_search(
            model, x.to(device), vocab, decode_cfg.beam_width, max_len, decode_cfg.length_penalty
        )
        preds.append(vocab.decode(tokens))
        tgts.append(vocab.decode(y.tolist()[1:]))

    per_sample = [cer(p, t) for p, t in zip(preds, tgts, strict=True) if t]
    lo, hi = bootstrap_ci(per_sample)
    result: dict[str, Any] = {
        "n_samples": n,
        "cer": mean_cer(preds, tgts),
        "cer_ci95": [lo, hi],
        "exact_match": exact_match(preds, tgts),
        "beam_width": decode_cfg.beam_width,
        "length_penalty": decode_cfg.length_penalty,
    }
    df = getattr(dataset, "df", None)
    if df is not None and "participant_id" in df.columns:
        by_signer = group_cer(preds, tgts, df["participant_id"].iloc[:n].tolist())
        vals = list(by_signer.values())
        result["n_signers"] = len(vals)
        result["cer_signer_mean"] = float(np.mean(vals))
        result["cer_signer_std"] = float(np.std(vals))
        result["cer_by_signer"] = by_signer
    result["examples"] = [
        {"target": t, "prediction": p} for p, t in list(zip(preds, tgts, strict=True))[:20]
    ]
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with beam search")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="write summary metrics json here")
    parser.add_argument("--details-out", default=None, help="write per-signer cer and examples here")
    parser.add_argument("--allow-unsafe", action="store_true", help="trust legacy pickled ckpt")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    cfg = load_config(args.config, args.overrides)
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device, allow_unsafe=args.allow_unsafe)
    decode_cfg = DecodeConfig(
        beam_width=int(ckpt.get("beam_width", cfg.decode.beam_width)),
        length_penalty=float(ckpt.get("length_penalty", cfg.decode.length_penalty)),
    )
    dataset = build_datasets(cfg, vocab)[args.split]
    result = evaluate_dataset(
        model, dataset, vocab, decode_cfg, ckpt["_model_config"].max_phrase_len, device,
        args.max_samples,
    )
    summary = {k: v for k, v in result.items() if k not in ("cer_by_signer", "examples")}
    print(json.dumps(summary, indent=2))
    for path, payload in ((args.out, summary), (args.details_out, result)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
