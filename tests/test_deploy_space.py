import os
import subprocess
import sys

import pytest

sys.path.insert(0, "scripts")
from deploy_space import build_staging  # noqa: E402

from asl.checkpoint import DEFAULT_FILENAME, resolve_checkpoint  # noqa: E402


def test_staged_space_is_self_contained(tmp_path):
    stage = build_staging(tmp_path / "space")
    for name in ("app.py", "README.md", "requirements.txt", "packages.txt", "asl/model.py", "asl/infer.py"):
        assert (stage / name).exists(), name
    readme = (stage / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---") and "sdk: gradio" in readme and "app_file: app.py" in readme
    requirements = (stage / "requirements.txt").read_text()
    assert "torch==2.11.0" in requirements and "whl/cpu" not in requirements  # zerogpu needs cuda torch
    assert 'python_version: "3.12.12"' in readme
    assert not list(stage.rglob("__pycache__"))


def test_staged_app_imports_its_own_asl_package(tmp_path):
    pytest.importorskip("gradio")
    stage = build_staging(tmp_path / "space")
    code = "import app, asl, pathlib; print(pathlib.Path(asl.__file__).parent.parent)"
    out = subprocess.run([sys.executable, "-c", code], cwd=stage, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert str(stage) in out.stdout


def test_inference_function_is_wrapped_by_spaces_gpu(tmp_path):
    pytest.importorskip("gradio")
    stage = build_staging(tmp_path / "space")
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "spaces.py").write_text(
        "import pathlib\n"
        "def GPU(duration=None):\n"
        "    def wrap(fn):\n"
        "        pathlib.Path('wrapped.txt').write_text(fn.__name__)\n"
        "        return fn\n"
        "    return wrap\n"
    )
    env = {**os.environ, "PYTHONPATH": str(stub)}
    cmd = [sys.executable, "-c", "import app"]
    out = subprocess.run(cmd, cwd=stage, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert (stage / "wrapped.txt").read_text() == "predict"


def test_staged_app_imports_without_the_packages_the_space_does_not_install(tmp_path):
    pytest.importorskip("gradio")
    stage = build_staging(tmp_path / "space")
    blocker = """
import sys

# gradio and huggingface_hub already bring pandas, tqdm and fastapi; these four are not installed there
NOT_ON_THE_SPACE = {"editdistance", "mlflow", "onnxruntime", "prometheus_client"}


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in NOT_ON_THE_SPACE:
            raise ImportError("not installed on the Space: " + name)


sys.meta_path.insert(0, Block())
import app
"""
    out = subprocess.run([sys.executable, "-c", blocker], cwd=stage, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-600:]


def test_resolve_checkpoint_order(tmp_path, monkeypatch):
    local = tmp_path / "local.pth"
    local.write_bytes(b"x")
    assert resolve_checkpoint(local) == local

    env_file = tmp_path / "env.pth"
    env_file.write_bytes(b"x")
    monkeypatch.setenv("ASL_CHECKPOINT", str(env_file))
    assert resolve_checkpoint(tmp_path / "missing.pth") == env_file

    monkeypatch.delenv("ASL_CHECKPOINT")
    calls = {}

    def fake_download(repo_id, filename):
        calls.update(repo_id=repo_id, filename=filename)
        return str(tmp_path / "downloaded.pth")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setenv("ASL_MODEL_REPO", "someone/model")
    monkeypatch.chdir(tmp_path)  # no local default file here
    resolve_checkpoint()
    assert calls == {"repo_id": "someone/model", "filename": DEFAULT_FILENAME}
