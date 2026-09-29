import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FRONT_MATTER = """---
title: ASL Fingerspelling
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.10.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

Upload or record a short clip of ASL fingerspelling and get the text back.
MediaPipe hand landmarks go into a Conformer encoder + Transformer decoder.
Code: https://github.com/sohenpatel22/ASL-Fingerspelling-Recognition
"""

REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0
numpy==1.26.4
pyyaml
huggingface_hub
mediapipe==0.10.21
opencv-python-headless
gTTS
groq
"""

# mediapipe pulls in a non-headless opencv that needs these system libraries
PACKAGES = "libgl1\nlibglib2.0-0\n"


def build_staging(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "asl", dest / "asl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "app" / "app.py", dest / "app.py")
    (dest / "README.md").write_text(FRONT_MATTER, encoding="utf-8")
    (dest / "requirements.txt").write_text(REQUIREMENTS)
    (dest / "packages.txt").write_text(PACKAGES)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="stage (and optionally push) the Hugging Face Space")
    parser.add_argument("--space", help="user/name of the Space, needed with --push")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--out", default=None, help="stage into this folder instead of a temp dir")
    args = parser.parse_args()

    dest = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="asl-space-"))
    build_staging(dest)
    print(f"staged Space files in {dest}")
    if not args.push:
        return

    from huggingface_hub import HfApi

    if not args.space:
        raise SystemExit("--space user/name is required with --push")
    api = HfApi(token=os.environ.get("HF_TOKEN"))  # falls back to the token from `hf auth login`
    api.create_repo(
        args.space, repo_type="space", space_sdk="gradio", space_hardware="cpu-basic", exist_ok=True
    )
    api.upload_folder(folder_path=str(dest), repo_id=args.space, repo_type="space")
    print(f"pushed to https://huggingface.co/spaces/{args.space}")

    requested = api.get_space_runtime(args.space).raw.get("hardware", {}).get("requested")
    if requested not in (None, "cpu-basic"):
        print(
            f"warning: this Space asks for {requested} hardware but the app runs on CPU. "
            "Recreate it with 'CPU basic' selected (a ZeroGPU Space can't be switched back for free)."
        )


if __name__ == "__main__":
    main()
