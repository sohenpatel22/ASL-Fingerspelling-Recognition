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
    assert "download.pytorch.org/whl/cpu" in (stage / "requirements.txt").read_text()
    assert not list(stage.rglob("__pycache__"))


def test_staged_app_imports_its_own_asl_package(tmp_path):
    pytest.importorskip("gradio")
    stage = build_staging(tmp_path / "space")
    code = "import app, asl, pathlib; print(pathlib.Path(asl.__file__).parent.parent)"
    out = subprocess.run([sys.executable, "-c", code], cwd=stage, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    assert str(stage) in out.stdout


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
