import json

import pytest

mlflow = pytest.importorskip("mlflow")

from asl.tracking import CANDIDATE, PRODUCTION, download_model, promote  # noqa: E402
from asl.train import main as train_main  # noqa: E402


def _train(tmp_path, register=True):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    train_main([
        "--config", "configs/smoke.yaml",
        "train.epochs=2", f"train.out_dir={(tmp_path / 'run').as_posix()}",
        "tracking.enabled=true", f"tracking.uri={uri}", "tracking.run_name=test",
        "tracking.register_as=asl-test" if register else "tracking.register_as=null",
    ])
    return uri


def test_training_logs_params_metrics_and_artifacts(tmp_path):
    uri = _train(tmp_path, register=False)
    mlflow.set_tracking_uri(uri)
    run = mlflow.search_runs(experiment_names=["asl-fingerspelling"]).iloc[0]
    assert run["params.train.epochs"] == "2" and run["params.model.d_model"] == "64"
    assert run["metrics.best_val_greedy_cer"] >= 0
    client = mlflow.tracking.MlflowClient()
    steps = client.get_metric_history(run["run_id"], "val_greedy_cer")
    assert [m.step for m in steps] == [1, 2]
    logged = {a.path for a in client.list_artifacts(run["run_id"], "run")}
    assert {"run/best.pth", "run/config.json"} <= logged


def test_register_promote_and_download(tmp_path):
    uri = _train(tmp_path)
    client = mlflow.tracking.MlflowClient()
    assert str(client.get_model_version_by_alias("asl-test", CANDIDATE).version) == "1"

    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps({"cer": 0.30}))
    bad.write_text(json.dumps({"cer": 0.90}))
    assert not promote(uri, "asl-test", str(bad), max_cer=0.5)
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias("asl-test", PRODUCTION)

    assert promote(uri, "asl-test", str(good), max_cer=0.5)
    assert str(client.get_model_version_by_alias("asl-test", PRODUCTION).version) == "1"
    path = download_model(uri, f"models:/asl-test@{PRODUCTION}")
    assert path.suffix == ".pth"
