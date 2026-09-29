from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from asl.config import DecodeConfig, ModelConfig
from asl.model import ASLConformerSeq2Seq
from asl.vocab import Vocab

log = logging.getLogger(__name__)

DEFAULT_REPO_ID = "SohenP/asl-fingerspelling-conformer"
DEFAULT_FILENAME = "asl_transformer_v6_final.pth"
_ARCH_KEYS = {
    "feature_size": "feature_size",
    "max_seq_len": "max_seq_len",
    "max_phrase_len": "max_phrase_len",
    "d_model": "d_model",
    "enc_layers": "enc_layers",
    "dec_layers": "dec_layers",
    "n_heads": "n_heads",
    "ffn_dim": "ffn_dim",
    "embed_dim": "embed_dim",
}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {(_plain(k) if not isinstance(k, str) else k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_checkpoint(
    path: str | Path,
    model: ASLConformerSeq2Seq,
    model_cfg: ModelConfig,
    vocab: Vocab,
    decode_cfg: DecodeConfig | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decode_cfg = decode_cfg or DecodeConfig()
    payload: dict[str, Any] = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "char_to_idx": vocab.char_to_idx,
        "vocab_size": vocab.vocab_size,
        "n_classes": vocab.n_classes,
        "start_idx": vocab.start_idx,
        "eos_idx": vocab.eos_idx,
        "pad_idx": vocab.pad_idx,
        "model_config": dataclasses.asdict(model_cfg),
        "beam_width": decode_cfg.beam_width,
        "length_penalty": decode_cfg.length_penalty,
        "metrics": _plain(metrics or {}),
    }
    payload.update(_plain(extra or {}))
    torch.save(payload, path)
    return path


def _torch_load(path: str | Path, allow_unsafe: bool) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as err:
        if not allow_unsafe:
            raise RuntimeError(
                f"{path} could not be loaded with weights_only=True ({err}). If you created this "
                "file yourself, run `python -m asl.checkpoint convert SRC DST` to make a safe copy."
            ) from err
        log.warning("Loading %s with weights_only=False (only do this for trusted files)", path)
        return torch.load(path, map_location="cpu", weights_only=False)


def model_config_from_checkpoint(ckpt: dict[str, Any]) -> ModelConfig:
    cfg = dict(ckpt.get("model_config", {}))
    for flat, field in _ARCH_KEYS.items():
        if flat in ckpt and field not in cfg:
            cfg[field] = int(ckpt[flat])
    cfg.setdefault("dropout", 0.0)
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in cfg.items() if k in known})


def _upgrade_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # the v5 notebook called the positional encoding "pos_enc", later versions "posenc"
    return {k.replace(".pos_enc.", ".posenc."): v for k, v in state.items()}


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu", allow_unsafe: bool = False
) -> tuple[ASLConformerSeq2Seq, Vocab, dict[str, Any]]:
    ckpt = _torch_load(path, allow_unsafe)
    state = _upgrade_keys(ckpt.get("model_state_dict", ckpt))
    vocab = Vocab(ckpt["char_to_idx"]) if "char_to_idx" in ckpt else Vocab()
    model_cfg = model_config_from_checkpoint(ckpt) if "model_state_dict" in ckpt else ModelConfig()
    model = ASLConformerSeq2Seq(model_cfg, vocab.vocab_size, vocab.pad_idx)
    model.load_state_dict(state)
    model.to(device).eval()
    ckpt["_model_config"] = model_cfg
    return model, vocab, ckpt


def load_state_dict_only(path: str | Path, allow_unsafe: bool = False) -> dict[str, torch.Tensor]:
    ckpt = _torch_load(path, allow_unsafe)
    return _upgrade_keys(ckpt.get("model_state_dict", ckpt))


def transfer_weights(model: ASLConformerSeq2Seq, state: dict[str, torch.Tensor]) -> dict[str, int]:
    target = model.state_dict()
    stats = {"copied": 0, "resized": 0, "skipped": 0}
    for name, tensor in state.items():
        if name not in target:
            stats["skipped"] += 1
        elif target[name].shape == tensor.shape:
            target[name].copy_(tensor)
            stats["copied"] += 1
        elif "fc_out" in name or "embed" in name:
            slices = tuple(slice(0, min(a, b)) for a, b in zip(tensor.shape, target[name].shape, strict=True))
            target[name][slices].copy_(tensor[slices])
            stats["resized"] += 1
        else:
            stats["skipped"] += 1
    model.load_state_dict(target)
    return stats


def convert_checkpoint(src: str | Path, dst: str | Path) -> Path:
    ckpt = _torch_load(src, allow_unsafe=True)
    if "model_state_dict" not in ckpt:
        ckpt = {"model_state_dict": ckpt}
    ckpt["model_state_dict"] = {k: v.cpu() for k, v in ckpt["model_state_dict"].items()}
    plain = {k: (v if k == "model_state_dict" else _plain(v)) for k, v in ckpt.items()}
    if "idx_to_char" in plain:
        plain.pop("idx_to_char")
    torch.save(plain, dst)
    return Path(dst)


def resolve_checkpoint(
    path: str | Path | None = None,
    repo_id: str | None = None,
    filename: str | None = None,
) -> Path:
    filename = filename or os.environ.get("ASL_MODEL_FILE", DEFAULT_FILENAME)
    for candidate in (path, os.environ.get("ASL_CHECKPOINT"), filename):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    from huggingface_hub import hf_hub_download

    repo_id = repo_id or os.environ.get("ASL_MODEL_REPO", DEFAULT_REPO_ID)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Checkpoint utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    conv = sub.add_parser("convert", help="convert a trusted legacy checkpoint to a safe format")
    conv.add_argument("src")
    conv.add_argument("dst")
    args = parser.parse_args(argv)
    if args.cmd == "convert":
        print(f"wrote {convert_checkpoint(args.src, args.dst)}")


if __name__ == "__main__":
    main()
