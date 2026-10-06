from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from tqdm import tqdm

from asl.checkpoint import (
    fetch_hub_file,
    load_checkpoint,
    load_state_dict_only,
    save_checkpoint,
    transfer_weights,
)
from asl.config import Config, TrainConfig, load_config
from asl.data import build_datasets
from asl.decode import ctc_greedy_decode, greedy_decode
from asl.metrics import exact_match, mean_cer
from asl.model import ASLConformerSeq2Seq
from asl.tracking import MLflowTracker
from asl.vocab import Vocab

MetricsCallback = Callable[[int, dict[str, float]], None]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    total = max(cfg.epochs * steps_per_epoch, 1)
    warmup = cfg.warmup_epochs * steps_per_epoch

    def factor(step: int) -> float:
        if cfg.scheduler == "none":
            return 1.0
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def mask_decoder_inputs(y_in: torch.Tensor, prob: float, pad_idx: int) -> torch.Tensor:
    # hide random previous tokens (replaced by PAD, which attention ignores); the first token stays.
    # A decoder that cannot lean on its own previous text has to take the evidence from the encoder.
    if prob <= 0:
        return y_in
    drop = torch.rand(y_in.shape, device=y_in.device) < prob
    drop[:, 0] = False
    return torch.where(drop, torch.full_like(y_in, pad_idx), y_in)


def scheduled_sampling_epsilon(cfg: TrainConfig, epoch: int) -> float:
    if cfg.ss_start_epoch is None or epoch < cfg.ss_start_epoch:
        return 1.0
    span = max(cfg.epochs - cfg.ss_start_epoch, 1)
    frac = min((epoch - cfg.ss_start_epoch) / span, 1.0)
    return cfg.ss_eps_start + (cfg.ss_eps_end - cfg.ss_eps_start) * frac


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: ASLConformerSeq2Seq,
        vocab: Vocab,
        train_ds: Dataset,
        val_ds: Dataset,
        device: torch.device,
        on_metrics: MetricsCallback | None = None,
    ):
        self.cfg, self.model, self.vocab, self.device = cfg, model, vocab, device
        self.val_ds = val_ds
        self.on_metrics = on_metrics
        t = cfg.train
        self.out_dir = Path(t.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        pin = device.type == "cuda"
        sampler = None
        if t.epoch_fraction < 1.0:
            sampler = RandomSampler(train_ds, num_samples=max(1, int(len(train_ds) * t.epoch_fraction)))
        self.train_loader = DataLoader(
            train_ds, batch_size=t.batch_size, shuffle=sampler is None, sampler=sampler,
            num_workers=t.num_workers, pin_memory=pin, persistent_workers=t.num_workers > 0,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=t.eval_batch_size, shuffle=False, num_workers=0, pin_memory=pin
        )
        # same val subset every epoch
        idx = np.random.RandomState(t.seed).permutation(len(val_ds))[: t.eval_samples]
        self.eval_loader = DataLoader(
            Subset(val_ds, idx.tolist()), batch_size=t.eval_batch_size, shuffle=False, num_workers=0
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx, label_smoothing=t.label_smoothing
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
        steps_per_epoch = math.ceil(len(self.train_loader) / t.grad_accum_steps)
        self.scheduler = make_scheduler(self.optimizer, t, steps_per_epoch)
        self.use_amp = t.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)

        self.ema = copy.deepcopy(model).eval() if t.ema_decay > 0 else None
        self.history: list[dict[str, float]] = []
        self.best_cer = float("inf")
        self.bad_epochs = 0
        self.start_epoch = 1

    @property
    def eval_model(self) -> ASLConformerSeq2Seq:
        # validation and checkpoints use the averaged weights when EMA is on
        return self.ema if self.ema is not None else self.model

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = self.cfg.train.ema_decay
        for avg, p in zip(self.ema.parameters(), self.model.parameters(), strict=True):
            avg.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for avg_b, b in zip(self.ema.buffers(), self.model.buffers(), strict=True):
            avg_b.copy_(b)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, enabled=self.use_amp)

    def _decoder_inputs(self, x: torch.Tensor, y_in: torch.Tensor, eps: float) -> torch.Tensor:
        if eps >= 1.0:
            return y_in
        with torch.no_grad(), self._autocast():
            pred = self.model(x, y_in).argmax(-1)
        keep_gold = torch.rand_like(y_in[:, 1:], dtype=torch.float) < eps
        keep_gold |= y_in[:, 1:] == self.vocab.pad_idx
        mixed = y_in.clone()
        mixed[:, 1:] = torch.where(keep_gold, y_in[:, 1:], pred[:, :-1])
        return mixed

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits.reshape(-1, self.vocab.vocab_size).float(), target.reshape(-1))

    def _ctc_loss(self, ctc_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # targets are the plain characters: START, EOS and PAD are all >= n_classes
        log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
        is_char = y < self.vocab.n_classes
        input_lengths = torch.full((y.size(0),), log_probs.size(0), dtype=torch.long)
        return F.ctc_loss(
            log_probs, y[is_char], input_lengths, is_char.sum(1).cpu(),
            blank=self.vocab.pad_idx, zero_infinity=True,
        )

    def _token_acc(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        mask = target != self.vocab.pad_idx
        return (logits.argmax(-1)[mask] == target[mask]).float().mean().item()

    def train_epoch(self, eps: float) -> tuple[float, float]:
        t = self.cfg.train
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = total_acc = 0.0
        n_batches = len(self.train_loader)
        for i, (x, y) in enumerate(tqdm(self.train_loader, desc="train", leave=False)):
            x, y = x.to(self.device), y.to(self.device)
            y_in, target = y[:, :-1], y[:, 1:].contiguous()
            y_in = self._decoder_inputs(x, y_in, eps)
            y_in = mask_decoder_inputs(y_in, t.decoder_mask, self.vocab.pad_idx)
            with self._autocast():
                if t.ctc_weight > 0:
                    logits, ctc_logits = self.model.forward_joint(x, y_in)
                    loss = (1 - t.ctc_weight) * self._loss(logits, target)
                    loss = loss + t.ctc_weight * self._ctc_loss(ctc_logits, y)
                else:
                    logits = self.model(x, y_in)
                    loss = self._loss(logits, target)
            self.scaler.scale(loss / t.grad_accum_steps).backward()
            if (i + 1) % t.grad_accum_steps == 0 or i + 1 == n_batches:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), t.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                if self.ema is not None:
                    self._update_ema()
            total_loss += loss.item()
            total_acc += self._token_acc(logits.detach(), target)
        return total_loss / n_batches, total_acc / n_batches

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        model = self.eval_model
        model.eval()
        loss = acc = 0.0
        n = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            target = y[:, 1:].contiguous()
            with self._autocast():
                logits = model(x, y[:, :-1])
            loss += self._loss(logits, target).item()
            acc += self._token_acc(logits, target)
            n += 1

        preds, ctc_preds, tgts = [], [], []
        has_ctc = model.ctc_head is not None
        for x, y in self.eval_loader:
            x = x.to(self.device)
            decoded = greedy_decode(model, x, self.vocab, self.cfg.model.max_phrase_len)
            ctc_decoded = ctc_greedy_decode(model, x, self.vocab) if has_ctc else decoded
            for p, c, t in zip(decoded, ctc_decoded, y.tolist(), strict=True):
                preds.append(self.vocab.decode(p))
                ctc_preds.append(self.vocab.decode(c))
                tgts.append(self.vocab.decode(t[1:]))
        metrics = {
            "val_loss": loss / n,
            "val_tf_acc": acc / n,
            "val_greedy_cer": mean_cer(preds, tgts),
            "val_greedy_exact": exact_match(preds, tgts),
        }
        if has_ctc:
            metrics["val_ctc_cer"] = mean_cer(ctc_preds, tgts)
        return metrics

    def _save(self, name: str, metrics: dict[str, float], epoch: int) -> None:
        save_checkpoint(
            self.out_dir / name, self.eval_model, self.cfg.model, self.vocab, self.cfg.decode, metrics,
            extra={"epoch": epoch, "best_cer": self.best_cer, "bad_epochs": self.bad_epochs},
        )
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": epoch,
                "best_cer": self.best_cer,
                "bad_epochs": self.bad_epochs,
            },
            self.out_dir / "trainer_state.pt",
        )

    def resume(self) -> None:
        model, _, _ = load_checkpoint(self.out_dir / "last.pth", self.device)
        self.model.load_state_dict(model.state_dict())
        if self.ema is not None:
            self.ema.load_state_dict(model.state_dict())
        state = torch.load(self.out_dir / "trainer_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.best_cer, self.bad_epochs = state["best_cer"], state["bad_epochs"]
        self.start_epoch = state["epoch"] + 1

    def fit(self) -> list[dict[str, float]]:
        t = self.cfg.train
        (self.out_dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        for epoch in range(self.start_epoch, t.epochs + 1):
            t0 = time.time()
            eps = scheduled_sampling_epsilon(t, epoch)
            tr_loss, tr_acc = self.train_epoch(eps)
            metrics = {
                "epoch": epoch, "train_loss": tr_loss, "train_tf_acc": tr_acc, "ss_epsilon": eps,
                "lr": self.optimizer.param_groups[0]["lr"], **self.validate(),
            }
            improved = metrics["val_greedy_cer"] < self.best_cer
            if improved:
                self.best_cer, self.bad_epochs = metrics["val_greedy_cer"], 0
            else:
                self.bad_epochs += 1
            metrics["seconds"] = time.time() - t0
            self.history.append(metrics)
            self._save("last.pth", metrics, epoch)
            if improved:
                self._save("best.pth", metrics, epoch)
            with (self.out_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            if self.on_metrics:
                self.on_metrics(epoch, metrics)
            print(
                f"epoch {epoch:02d}/{t.epochs} | loss {tr_loss:.4f}/{metrics['val_loss']:.4f} | "
                f"greedy CER {metrics['val_greedy_cer']:.4f} | {metrics['seconds']:.0f}s"
                f"{' <- best' if improved else f' (no gain {self.bad_epochs}/{t.patience})'}"
            )
            if self.bad_epochs >= t.patience:
                print("early stopping")
                break
        return self.history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the ASL fingerspelling model")
    parser.add_argument("--config", default=None, help="YAML config (defaults = released v6 setup)")
    parser.add_argument("--resume", action="store_true", help="continue from out_dir/last.pth")
    parser.add_argument("--allow-unsafe", action="store_true", help="trust a legacy pickled init_from")
    parser.add_argument("overrides", nargs="*", help="section.key=value, e.g. train.epochs=5")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if cfg.train.ctc_weight > 0 and not cfg.model.ctc:
        raise SystemExit("train.ctc_weight > 0 needs model.ctc=true")
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    vocab = Vocab()
    model = ASLConformerSeq2Seq(cfg.model, vocab.vocab_size, vocab.pad_idx).to(device)
    print(f"parameters: {model.num_parameters() / 1e6:.1f}M")
    if cfg.train.init_from and not args.resume:
        source = cfg.train.init_from
        if source.startswith("hf:"):
            source = fetch_hub_file(source)
        state = load_state_dict_only(source, allow_unsafe=args.allow_unsafe)
        stats = transfer_weights(model, state)
        print(f"warm-started from {cfg.train.init_from}: {stats}")

    data: dict[str, Any] = build_datasets(cfg, vocab)
    tracker = MLflowTracker(cfg) if cfg.tracking.enabled else nullcontext()
    with tracker as t:
        trainer = Trainer(
            cfg, model, vocab, data["train"], data["val"], device,
            on_metrics=t.log_epoch if t else None,
        )
        if args.resume:
            trainer.resume()
        trainer.fit()
        if t:
            version = t.finish(cfg.train.out_dir, trainer.best_cer)
            if version:
                print(f"registered {cfg.tracking.register_as} v{version} (alias: candidate)")


if __name__ == "__main__":
    main()
