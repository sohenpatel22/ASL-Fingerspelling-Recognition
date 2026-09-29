from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from asl.checkpoint import load_checkpoint
from asl.config import DecodeConfig, ModelConfig, load_config
from asl.data import build_datasets
from asl.decode import beam_search
from asl.infer import Predictor
from asl.metrics import mean_cer
from asl.model import ASLConformerSeq2Seq, TransformerDecoder
from asl.vocab import Vocab

FILES = {"fp32": ("encoder.onnx", "decoder.onnx"), "int8": ("encoder.int8.onnx", "decoder.int8.onnx")}


class ExportDecoder(nn.Module):
    # nn.TransformerDecoder has data dependent branches the exporter can't trace, so this
    # spells out the same pre-norm layer math using the original module's weights
    def __init__(self, decoder: TransformerDecoder):
        super().__init__()
        self.d = decoder

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        d = self.d
        x = d.posenc(d.proj_emb(d.embed(tgt)))
        n = tgt.size(1)
        mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
        for layer in d.decoder.layers:
            h = layer.norm1(x)
            x = x + layer.self_attn(h, h, h, attn_mask=mask, need_weights=False)[0]
            h = layer.norm2(x)
            x = x + layer.multihead_attn(h, memory, memory, need_weights=False)[0]
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        return d.fc_out(x)


def export_onnx(
    model: ASLConformerSeq2Seq, vocab: Vocab, model_cfg: ModelConfig, decode_cfg: DecodeConfig,
    out_dir: str | Path, quantize: bool = True,
) -> dict[str, dict[str, int]]:
    from torch.export import Dim

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    x = torch.randn(2, model_cfg.feature_size, model_cfg.max_seq_len)
    tgt = torch.tensor([[vocab.start_idx, 3, 4, 5], [vocab.start_idx, 7, 8, 9]])
    memory = model.encoder(x)
    batch = Dim("batch", min=1, max=64)
    length = Dim("len", min=2, max=64)
    frames = Dim("frames", min=8, max=512)

    torch.onnx.export(
        model.encoder, (x,), str(out / FILES["fp32"][0]), input_names=["x"], output_names=["memory"],
        dynamic_shapes={"x": {0: batch, 2: frames}}, dynamo=True, external_data=False,
    )
    torch.onnx.export(
        ExportDecoder(model.decoder).eval(), (tgt, memory), str(out / FILES["fp32"][1]),
        input_names=["tgt", "memory"], output_names=["logits"],
        dynamic_shapes={"tgt": {0: batch, 1: length}, "memory": {0: batch, 1: frames}}, dynamo=True,
        external_data=False,
    )
    variants = ["fp32"]
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for src, dst in zip(FILES["fp32"], FILES["int8"], strict=True):
            # matmuls only: int8 convolutions are slow on cpu and missing in older onnxruntime builds
            quantize_dynamic(
                str(out / src), str(out / dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"]
            )
        variants.append("int8")

    import dataclasses

    (out / "meta.json").write_text(json.dumps({
        "model_config": dataclasses.asdict(model_cfg), "char_to_idx": vocab.char_to_idx,
        "beam_width": decode_cfg.beam_width, "length_penalty": decode_cfg.length_penalty,
    }))
    return {v: {f: (out / f).stat().st_size for f in FILES[v]} for v in variants}


class OnnxSeq2Seq:
    # duck-types the parts of the torch model that beam_search / Predictor use
    def __init__(self, encoder_path: str | Path, decoder_path: str | Path, threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(encoder_path), opts, providers=providers)
        self.dec = ort.InferenceSession(str(decoder_path), opts, providers=providers)
        self.encoder = self._encode
        self.decoder = self._decode

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.enc.run(None, {"x": x.numpy()})[0])

    def _decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        feed = {"tgt": tgt.numpy(), "memory": np.ascontiguousarray(memory.numpy())}
        return torch.from_numpy(self.dec.run(None, feed)[0])

    def to(self, _device) -> OnnxSeq2Seq:
        return self

    def eval(self) -> OnnxSeq2Seq:
        return self


def load_onnx_predictor(onnx_dir: str | Path, variant: str = "fp32", threads: int = 0) -> Predictor:
    onnx_dir = Path(onnx_dir)
    meta = json.loads((onnx_dir / "meta.json").read_text())
    enc, dec = FILES[variant]
    cfg = ModelConfig(**meta["model_config"])
    decode_cfg = DecodeConfig(meta["beam_width"], meta["length_penalty"])
    model = OnnxSeq2Seq(onnx_dir / enc, onnx_dir / dec, threads)
    return Predictor(model, Vocab(meta["char_to_idx"]), cfg, decode_cfg)  # type: ignore[arg-type]


def _time_predictor(predictor: Predictor, clips: list[np.ndarray], warmup: int = 3) -> dict[str, float]:
    for clip in clips[:warmup]:
        predictor.predict_landmarks(clip)
    times = []
    for clip in clips:
        t0 = time.perf_counter()
        predictor.predict_landmarks(clip)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "latency_ms_p50": statistics.median(times),
        "latency_ms_p95": times[min(int(len(times) * 0.95), len(times) - 1)],
        "latency_ms_mean": statistics.fmean(times),
    }


def _cer(predictor: Predictor, dataset, vocab: Vocab, n: int) -> float:
    preds, tgts = [], []
    for i in range(min(n, len(dataset))):
        x, y = dataset[i]
        tokens, _ = beam_search(
            predictor.model, x, vocab, predictor.decode_cfg.beam_width,
            predictor.model_cfg.max_phrase_len, predictor.decode_cfg.length_penalty,
        )
        preds.append(vocab.decode(tokens))
        tgts.append(vocab.decode(y.tolist()[1:]))
    return mean_cer(preds, tgts)


def benchmark(
    checkpoint: str | Path, onnx_dir: str | Path, dataset, n: int = 50, allow_unsafe: bool = False
) -> dict[str, dict[str, float]]:
    model, vocab, ckpt = load_checkpoint(checkpoint, "cpu", allow_unsafe=allow_unsafe)
    decode_cfg = DecodeConfig(int(ckpt.get("beam_width", 5)), float(ckpt.get("length_penalty", 0.6)))
    predictors = {"torch": Predictor(model, vocab, ckpt["_model_config"], decode_cfg)}
    for variant in ("fp32", "int8"):
        if (Path(onnx_dir) / FILES[variant][0]).exists():
            predictors[f"onnx_{variant}"] = load_onnx_predictor(onnx_dir, variant)

    # latency is measured on raw landmark clips, the same input the service receives
    clips = [dataset[i][0].T.numpy() for i in range(min(n, len(dataset)))]
    results = {}
    for name, predictor in predictors.items():
        row = _time_predictor(predictor, clips)
        row["cer"] = _cer(predictor, dataset, vocab, n)
        if name == "torch":
            row["size_mb"] = Path(checkpoint).stat().st_size / 1e6
        else:
            variant = name.split("_")[1]
            row["size_mb"] = sum((Path(onnx_dir) / f).stat().st_size for f in FILES[variant]) / 1e6
        results[name] = row
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="export a checkpoint to ONNX (+int8) and benchmark it")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="onnx")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N", help="time and score N test clips")
    parser.add_argument("--report", default=None, help="write the benchmark json here")
    parser.add_argument("--allow-unsafe", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # torch's exporter prints unicode ticks
        sys.stdout.reconfigure(encoding="utf-8")
    model, vocab, ckpt = load_checkpoint(args.checkpoint, "cpu", allow_unsafe=args.allow_unsafe)
    decode_cfg = DecodeConfig(int(ckpt.get("beam_width", 5)), float(ckpt.get("length_penalty", 0.6)))
    sizes = export_onnx(model, vocab, ckpt["_model_config"], decode_cfg, args.out_dir, not args.no_quantize)
    print(json.dumps(sizes, indent=2))
    if args.benchmark:
        cfg = load_config(args.config, args.overrides)
        dataset = build_datasets(cfg, vocab)["test"]
        report = benchmark(args.checkpoint, args.out_dir, dataset, args.benchmark, args.allow_unsafe)
        print(json.dumps(report, indent=2))
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
