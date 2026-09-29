from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from asl.config import Config

CANDIDATE = "candidate"
PRODUCTION = "production"


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class MLflowTracker:
    def __init__(self, cfg: Config):
        import mlflow

        self.mlflow = mlflow
        self.cfg = cfg
        mlflow.set_tracking_uri(cfg.tracking.uri)
        mlflow.set_experiment(cfg.tracking.experiment)
        self.run = None

    def __enter__(self) -> MLflowTracker:
        self.run = self.mlflow.start_run(run_name=self.cfg.tracking.run_name)
        params = {k: v for k, v in _flatten(self.cfg.to_dict()).items() if not k.startswith("tracking.")}
        self.mlflow.log_params(params)
        self.mlflow.set_tag("git_sha", _git_sha())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.mlflow.end_run("FAILED" if exc_type else "FINISHED")

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        numeric = {k: v for k, v in metrics.items() if k != "epoch" and isinstance(v, (int, float))}
        self.mlflow.log_metrics(numeric, step=epoch)

    def finish(self, out_dir: str | Path, best_cer: float) -> str | None:
        out_dir = Path(out_dir)
        self.mlflow.log_metric("best_val_greedy_cer", best_cer)
        for name in ("best.pth", "config.json", "metrics.jsonl"):
            if (out_dir / name).exists():
                self.mlflow.log_artifact(str(out_dir / name), "run")
        name = self.cfg.tracking.register_as
        if not name:
            return None
        client = self.mlflow.tracking.MlflowClient()
        try:
            client.create_registered_model(name)
        except self.mlflow.exceptions.MlflowException:
            pass  # already exists
        source = f"{self.run.info.artifact_uri}/run/best.pth"
        version = client.create_model_version(name, source, run_id=self.run.info.run_id)
        client.set_registered_model_alias(name, CANDIDATE, version.version)
        return version.version


def promote(uri: str, name: str, metrics_file: str, max_cer: float) -> bool:
    import mlflow

    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()
    cer = json.loads(Path(metrics_file).read_text())["cer"]
    if cer > max_cer:
        print(f"not promoting: cer {cer:.4f} > {max_cer}")
        return False
    version = client.get_model_version_by_alias(name, CANDIDATE)
    client.set_registered_model_alias(name, PRODUCTION, version.version)
    print(f"{name} v{version.version} -> {PRODUCTION} (cer {cer:.4f})")
    return True


def download_model(uri: str, model_uri: str) -> Path:
    import mlflow

    mlflow.set_tracking_uri(uri)
    local = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    path = Path(local)
    return path if path.is_file() else next(path.glob("*.pth"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="promote a registered model if it passes the metric gate")
    parser.add_argument("--uri", default="sqlite:///mlflow.db")
    parser.add_argument("--name", default="asl-fingerspelling")
    parser.add_argument("--metrics", required=True, help="json with a 'cer' field, from asl-eval")
    parser.add_argument("--max-cer", type=float, required=True)
    args = parser.parse_args(argv)
    raise SystemExit(0 if promote(args.uri, args.name, args.metrics, args.max_cer) else 1)


if __name__ == "__main__":
    main()
