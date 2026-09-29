from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from asl.checkpoint import load_checkpoint
from asl.config import DecodeConfig, ModelConfig
from asl.decode import beam_search
from asl.features import to_model_input
from asl.lm import CharNgramLM
from asl.model import ASLConformerSeq2Seq
from asl.vocab import Vocab


@dataclass
class Prediction:
    text: str
    confidence: float
    n_frames: int
    hand_detection_rate: float | None = None


class Predictor:
    def __init__(
        self,
        model: ASLConformerSeq2Seq,
        vocab: Vocab,
        model_cfg: ModelConfig,
        decode_cfg: DecodeConfig | None = None,
        device: str | torch.device = "cpu",
        lm: CharNgramLM | None = None,
    ):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.model_cfg = model_cfg
        self.decode_cfg = decode_cfg or DecodeConfig()
        self.device = torch.device(device)
        self.lm = lm

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        allow_unsafe: bool = False,
        lm_path: str | Path | None = None,
        lm_weight: float | None = None,
    ) -> Predictor:
        model, vocab, ckpt = load_checkpoint(path, device, allow_unsafe=allow_unsafe)
        decode_cfg = DecodeConfig(
            beam_width=int(ckpt.get("beam_width", 5)),
            length_penalty=float(ckpt.get("length_penalty", 0.6)),
            lm_weight=lm_weight or 0.0,
        )
        lm = CharNgramLM.load(lm_path) if lm_path and decode_cfg.lm_weight > 0 else None
        return cls(model, vocab, ckpt["_model_config"], decode_cfg, device, lm)

    @torch.no_grad()
    def predict_landmarks(self, landmarks: np.ndarray) -> Prediction:
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.ndim != 2 or landmarks.shape[1] != self.model_cfg.feature_size or len(landmarks) == 0:
            raise ValueError(
                f"expected a non-empty (T, {self.model_cfg.feature_size}) array, got {landmarks.shape}"
            )
        x = torch.from_numpy(to_model_input(landmarks, self.model_cfg.max_seq_len, self.model_cfg.velocity))
        tokens, score = beam_search(
            self.model,
            x.to(self.device),
            self.vocab,
            beam_width=self.decode_cfg.beam_width,
            max_len=self.model_cfg.max_phrase_len,
            length_penalty=self.decode_cfg.length_penalty,
            lm=self.lm,
            lm_weight=self.decode_cfg.lm_weight,
        )
        n_tokens = max(len(tokens) - 1, 1)
        return Prediction(
            text=self.vocab.decode(tokens),
            confidence=math.exp(score / n_tokens),
            n_frames=len(landmarks),
        )

    def predict_video(self, video_path: str | Path) -> Prediction:
        from asl.video import extract_landmarks

        landmarks, rate = extract_landmarks(video_path)
        pred = self.predict_landmarks(landmarks)
        pred.hand_detection_rate = rate
        return pred
