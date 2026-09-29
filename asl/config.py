from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    feature_size: int = 84
    max_seq_len: int = 64
    max_phrase_len: int = 34
    d_model: int = 384
    enc_layers: int = 6
    dec_layers: int = 6
    n_heads: int = 6
    ffn_dim: int = 1024
    embed_dim: int = 192
    dropout: float = 0.25
    conv_kernel: int = 31
    ctc: bool = False  # extra CTC head on the encoder for the joint loss
    velocity: bool = False  # append frame-to-frame differences to the input features

    @property
    def input_dim(self) -> int:
        return self.feature_size * (2 if self.velocity else 1)


@dataclass
class DataConfig:
    comp_dir: str = "data/raw/asl-fingerspelling"
    npy_train: str = "data/processed/npy_train"
    npy_supp: str = "data/processed/npy_supp"
    use_supplemental: bool = True
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42
    # generated toy data for smoke tests / CI
    synthetic: bool = False
    synthetic_size: int = 512


@dataclass
class TrainConfig:
    batch_size: int = 96
    grad_accum_steps: int = 1
    epochs: int = 30
    lr: float = 5e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    scheduler: str = "cosine"
    min_lr_ratio: float = 0.0
    label_smoothing: float = 0.20
    ctc_weight: float = 0.0  # 0 = attention loss only; needs model.ctc=true when above 0
    grad_clip: float = 1.0
    num_workers: int = 2
    patience: int = 10
    amp: bool = True
    seed: int = 42
    device: str = "auto"
    eval_samples: int = 500
    eval_batch_size: int = 64
    ss_start_epoch: int | None = None
    ss_eps_start: float = 1.0
    ss_eps_end: float = 0.3
    init_from: str | None = None
    out_dir: str = "checkpoints/run"


@dataclass
class DecodeConfig:
    beam_width: int = 5
    length_penalty: float = 0.6


@dataclass
class TrackingConfig:
    enabled: bool = False
    uri: str = "sqlite:///mlflow.db"
    experiment: str = "asl-fingerspelling"
    run_name: str | None = None
    register_as: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "decode": DecodeConfig,
    "tracking": TrackingConfig,
}


def _build_section(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def _set_nested(raw: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_nested(raw, key.strip(), yaml.safe_load(value))
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise KeyError(f"Unknown config sections: {sorted(unknown)}")
    return Config(**{name: _build_section(cls, raw.get(name) or {}) for name, cls in _SECTIONS.items()})
