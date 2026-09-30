# ASL Fingerspelling Recognition

[![CI](https://github.com/sohenpatel22/ASL-Fingerspelling-Recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/sohenpatel22/ASL-Fingerspelling-Recognition/actions/workflows/ci.yml)
[![Hugging Face Space](https://img.shields.io/badge/demo-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/SohenP/asl-fingerspelling)

**Live demo:** https://huggingface.co/spaces/SohenP/asl-fingerspelling (record or upload a short
fingerspelling clip; it runs the v5 model, test CER about 0.33, on a free ZeroGPU Space)

Turns a short video of ASL fingerspelling into text. MediaPipe pulls hand landmarks out of the
video, a Conformer encoder + Transformer decoder reads them, and beam search produces the
characters. There's a Gradio demo, a FastAPI service with monitoring, and the training side is
set up with experiment tracking, data versioning and CI.

I trained the model on the Kaggle
[Google - American Sign Language Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling)
data. It started as a course project (MIE1517, UofT). Since then I've reworked it into a package
with tests and configs and built the MLOps side around it.

```
video -> MediaPipe -> (T, 84) landmarks -> wrist-center + scale -> pad/resample to 64 frames
      -> Conformer encoder -> Transformer decoder -> beam search -> text
```

Each frame is 84 numbers: 21 landmarks x (x, y) x 2 hands, zeros when a hand isn't found.
Centering on the wrist and scaling makes the model care about hand shape instead of where the
hand sits in the frame. The model has about 27.8M parameters.

## Results

Split is 70/15/15 by signer, so val and test signers never show up in training. The number I trust
is the one from `reports/phase4/`, scored with beam search on a seeded random sample of 3000 test
clips from the 14 held-out signers.

| model | test CER (mean per clip) | test CER (total edits / total characters) |
|---|---|---|
| **v5 weights, as deployed** | **0.334** | **0.322** |
| v5 fine-tuned again with this repo's trainer | 0.399 | |

The 0.43 and 0.44 in the original notebooks came from a different evaluation and shouldn't be
compared with these. For scale: the winning Kaggle solutions scored about 0.81 to 0.82 on the
competition's metric (roughly 0.18 to 0.19 in these units, same idea as the second column).

What the errors look like (v5, same 3000 clips):

- They are lumpy, not spread out. 36% of clips are decoded perfectly, and the 12% of clips with
  CER of 0.9 or more account for 72% of all the error.
- Most of those failures are not near misses. In two thirds of them the model outputs a different
  *kind* of phrase than the one signed (a phone number comes out as an address) and the output is
  longer than the target, as if the decoder made up something fluent when it couldn't see the hands.
- Which signer it is matters more than anything else. Per-signer CER runs from 0.03 to 0.78 and
  correlates at -0.96 with how often MediaPipe found a hand in that signer's frames (signers with a
  hand detected in under a third of frames are at 0.6 to 0.8 CER).

## Setup

```bash
git clone https://github.com/sohenpatel22/ASL-Fingerspelling-Recognition.git
cd ASL-Fingerspelling-Recognition
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev,serve,mlops,export]"
pytest                                   # about 110 tests, ~2 minutes on CPU
asl-train --config configs/smoke.yaml
```

Extras: `app` (Gradio demo), `serve` (API), `mlops` (MLflow, DVC), `export` (ONNX), `preprocess`
(parquet reading). Install only what you need. The smoke config trains a tiny model on generated
data, which is handy for checking that everything still works without downloading anything.

## Training on the real data

Get the competition data from Kaggle, then turn the parquet files into normalized `.npy`
sequences once:

```bash
asl-preprocess --comp-dir data/raw/asl-fingerspelling
asl-train --config configs/default.yaml
asl-train --config configs/local_4gb.yaml        # 4GB GPU: batch 24 x 4 accumulation steps
asl-train --config configs/default.yaml train.epochs=5 train.lr=1e-4
asl-eval --checkpoint checkpoints/v6/best.pth --config configs/default.yaml --split test
```

Any config value can be overridden on the command line like that. `--resume` continues from
`last.pth`, and `train.init_from=<ckpt>` warm-starts from an older model (it handles the 61 -> 62
token change from adding EOS).

### Pipeline with DVC

`dvc.yaml` chains preprocess -> train -> evaluate, so `dvc repro` only reruns the stages whose
inputs changed, and `dvc metrics show` prints the test CER. I checked it end to end on fake
competition-format data (`tests/fake_competition.py`). To share data and checkpoints, add a
remote first, e.g. `dvc remote add -d storage gdrive://<folder-id>`.

### Experiment tracking with MLflow

```bash
asl-train --config configs/default.yaml tracking.enabled=true tracking.register_as=asl-fingerspelling
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

That logs params, per-epoch metrics, the git commit and the checkpoint, and registers the model
with the `candidate` alias. `asl-promote` moves the `production` alias to it only if the test CER
passes a threshold:

```bash
asl-eval --checkpoint checkpoints/v6/best.pth --out metrics/test.json
asl-promote --name asl-fingerspelling --metrics metrics/test.json --max-cer 0.45
```

## Improving accuracy

I tried five ideas to bring the error down, plus a language model, all switchable in config and unit
tested. Then I ran an ablation on Kaggle (T4, about 5 hours): every variant warm-started from the v5
weights and trained on the same signer-independent split, and scored with the protocol above.

| run | what changed | val CER (beam) | test CER (beam) |
|---|---|---|---|
| v5 raw | the weights as deployed, no training | | **0.334** |
| a0_baseline | fine-tune v5 with this repo's trainer and the supplemental data | 0.350 | 0.399 |
| a1_ctc | + CTC loss next to the attention loss | 0.383 | 0.433 |
| a2_len128 | 128 frames instead of 64 | 0.407 | 0.450 |
| a3_velocity | + frame-to-frame velocity features | 0.365 | 0.408 |
| a4_strong_aug | + rotation, aspect jitter, missed-hand spans | 0.355 | 0.399 |
| a5_all | all four together | 0.392 | 0.436 |

Numbers are in `reports/phase4/`. What I take from them:

- **Nothing beat the v5 weights, and fine-tuning made them worse.** The plain fine-tune (a0) is worse
  than raw v5 for all 14 signers (0.065 higher on average, paired CI 0.047 to 0.089). I first read
  a0 as a fair baseline and only caught this once I scored raw v5 with the same protocol.
- **The ablation was not a fair test of the ideas.** Every variant started from v5 and only trained
  for a handful of epochs at a low learning rate, so anything that changes the input (128 frames,
  velocity) or adds a new head (CTC) had to be adapted onto weights trained for something else. A
  variant that loses here might still win when trained from scratch. What it does show is that
  bolting these on to v5 doesn't help.
- **More training on the same signers overfits.** Validation was best after the first epoch and
  then drifted worse while training accuracy kept climbing.
- **The character language model didn't help.** Tuned on validation, the best weight was 0.
- **The remaining error is mostly a data problem.** See the last bullet under Results: it tracks how
  often the hand is detected, and the model hallucinates when it can't see one.

Choices were made on validation only, and test was scored once at the end. Cross-validation
(`asl-cv`) has to train from scratch because v5 has already seen most signers, so I haven't run it.

## Serving

```bash
export ASL_CHECKPOINT=/path/to/asl_transformer_v6_final.pth     # or ASL_MODEL_URI=models:/asl-fingerspelling@production
asl-serve
```

`POST /predict` takes a video upload, `POST /predict/landmarks` takes raw `(T, 84)` landmarks as
JSON, `GET /health` reports the model version and `GET /metrics` is Prometheus. Each request
also writes a JSON log line. The metrics cover request rate and latency, beam search time,
prediction confidence, how many frames had a hand in them, and videos with no hands at all.

For drift monitoring, build a reference from the training data and point the service at it:

```bash
asl-reference --npy-dir data/processed/npy_train --out monitoring/reference.json
ASL_REFERENCE=monitoring/reference.json asl-serve
```

It then exports a PSI score per input feature (clip length, share of frames with a left or right
hand, amount of motion) over a rolling window. `deploy/alerts.yml` has Prometheus alerts for
drift, a high no-hands rate, slow inference and low confidence.

### ONNX export

```bash
asl-export --checkpoint checkpoints/v5/asl_v5_best.pth --out-dir onnx --config configs/default.yaml --benchmark 200
ASL_ONNX_DIR=onnx ASL_ONNX_VARIANT=fp32 asl-serve
```

This exports the encoder and decoder separately (dynamic batch, frame count and sequence
length), also makes an int8 copy, and benchmarks torch vs onnx fp32 vs onnx int8 on latency,
size, CER (when there are labels) and how often the output text matches the torch model. The
ONNX decoder matches PyTorch to about 1e-6 in tests.

Numbers on the real v5 weights, 40 noise clips (I don't have labelled test data locally, so this
measures speed and agreement, not accuracy), 4 CPU threads on my laptop:

| | median latency per clip | size | same text as torch |
|---|---|---|---|
| torch | ~660 ms | 113 MB | - |
| onnx fp32 | ~410 ms | 114 MB | 100% |
| onnx int8 (matmuls only) | ~370 ms | 47 MB | 80% |

So fp32 ONNX is about 1.6x faster than PyTorch with identical output. int8 is a bit faster again
and 2.4x smaller, but it changes the output on one clip in five, so I wouldn't ship it without
checking CER on real test data. The service defaults to fp32. (Quantizing the convolutions too
made things worse: slower, less faithful, and older onnxruntime builds can't run them.)

Separately, beam search runs all live beams in one decoder call instead of one call per beam.
On the real weights that took a clip from about 1.4 s to 0.64 s, with identical output on every
clip tried (there's also a test against the original implementation).

### Docker

```bash
docker build -t asl-fingerspelling .
docker run -p 8000:8000 -v "$PWD/onnx:/models/onnx:ro" -e ASL_ONNX_DIR=/models/onnx asl-fingerspelling
docker compose up        # api + mlflow + prometheus + grafana
```

The image runs as a non-root user and has a healthcheck. It's about 2.6 GB, mostly PyTorch and
MediaPipe's dependencies. I built it and ran it against a toy ONNX model, including a video
upload, but I haven't tried the compose stack yet (mlflow, prometheus and grafana containers).
Grafana comes up on port 3000 with a dashboard already provisioned.

## CI

`ci.yml` runs on every push: lint, the test suite, a short training run on generated data, an
evaluation with a metric gate (fails if CER or exact match regress), an ONNX export, and then
builds the Docker image and hits the running container. `release.yml` pushes the image to GHCR
when a `v*` tag is pushed. I haven't seen either run on GitHub yet.

## Free deployment on Hugging Face Spaces

The demo runs as a Gradio Space on ZeroGPU, which is what a free Hugging Face account can host
(up to 2 Spaces; a plain CPU Gradio Space needs a PRO plan). The model call is wrapped in
`@spaces.GPU` and everything else, MediaPipe included, runs on the CPU. Free visitors get a few
minutes of GPU time a day, and one clip takes about a second. The weights live in a Hugging Face
model repo and the Space downloads them at startup, so nothing big goes in git.

1. Upload the weights to a model repo (once): `hf upload SohenP/asl-fingerspelling-conformer checkpoints/v5/asl_v5_best.pth asl_v5_best.pth`
   and set the Space variable `ASL_MODEL_FILE=asl_v5_best.pth` (the repo name defaults to
   `SohenP/asl-fingerspelling-conformer`).
2. Create a Gradio Space with **ZeroGPU** hardware, or let the script do it:
   `python scripts/deploy_space.py --push --space SohenP/asl-fingerspelling`
   (uses `HF_TOKEN` if set, otherwise your `hf auth login` session).
3. For automatic deploys, add a repo variable `HF_SPACE` (`SohenP/asl-fingerspelling`) and a repo secret
   `HF_TOKEN` (a token with write access). `deploy-space.yml` then redeploys whenever `asl/` or
   `app/` changes on `main`, and does nothing if `HF_SPACE` isn't set.

Without `--push` the script only stages the files, which is a quick way to check what gets uploaded.
The ZeroGPU path can only be tested on Hugging Face itself; locally the `@spaces.GPU` decorator
is a no-op and the app runs on CPU.

## Demo

```bash
export ASL_CHECKPOINT=/path/to/asl_transformer_v6_final.pth
python app/app.py
```

Weights aren't in the repo. The app looks for `ASL_CHECKPOINT`, a local file, or the Hugging Face
Hub (`SohenP/asl-fingerspelling-conformer`), in that order. Checkpoints saved by the old
notebooks have numpy scalars in them, so convert a trusted one first with
`python -m asl.checkpoint convert old.pth new.pth`.

The LLM option needs `GROQ_API_KEY`. It only knows a fixed list of words, so it's a demo
feature and none of the numbers above use it.

## Layout

```
asl/          the package (model, data, training, decoding, eval, inference, serving, export,
              language model, cross-validation, ablations)
app/          gradio demo
configs/      default, local_4gb, smoke
deploy/       prometheus, alert rules, grafana dashboard
tests/        pytest
scripts/      webcam demo, metric gate
kaggle/       preprocess and GPU training notebooks for Kaggle
notebooks/    v6 fine-tune notebook from Kaggle
reports/      original course report
dvc.yaml      data + training pipeline
Dockerfile, docker-compose.yml
```

## Still to do

- a fair from-scratch test of the ideas above, plus inputs that still carry signal when the hands are not detected (pose and lips landmarks)
- have the app warn when the hand is rarely detected, since that is when the model makes things up
- z coordinates as extra features (needs re-preprocessing)
- put the FastAPI service somewhere public too (the Space only runs the Gradio demo)

## Limitations

Only isolated fingerspelling clips, no full sentences. It struggles with bad lighting, odd
camera angles and signers unlike the training set, and short words tend to get hallucinated.
Also, the model was trained on the competition's landmarks but the demo runs MediaPipe on your
own video, so there's a bit of a domain gap.

MIT license.
