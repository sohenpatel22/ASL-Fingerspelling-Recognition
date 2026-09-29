# ASL Fingerspelling Recognition

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

Split is 70/15/15 by signer, so val and test signers never show up in training. These numbers
come from the original notebooks.

| | |
|---|---|
| Test CER (beam 5, length penalty 0.6) | 0.43 |
| v5 baseline test CER | 0.44 |
| Val CER with beam search | about 0.33 (first 200 val samples) |

A few caveats. The "CER" printed each epoch in the v6 notebook was computed from teacher-forced
predictions, so it looks better than the model really is at inference. The trainer in this repo
early-stops on greedy-decoded CER instead. The test set is only ~15% of the signers, so it's a
noisy number; `asl-eval` also prints per-signer CER and a bootstrap interval. And adding the 50K
supplemental sequences only moved test CER by about 0.01, so more data alone isn't going to fix
this.

## Setup

```bash
git clone https://github.com/sohenpatel22/ASL-Fingerspelling-Recognition.git
cd ASL-Fingerspelling-Recognition
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev,serve,mlops,export]"
pytest
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

### ONNX and int8

```bash
asl-export --checkpoint checkpoints/v6/best.pth --out-dir onnx --config configs/default.yaml --benchmark 200
ASL_ONNX_DIR=onnx ASL_ONNX_VARIANT=int8 asl-serve
```

This exports the encoder and decoder separately (dynamic batch, frame count and sequence
length), quantizes both to int8, and benchmarks torch vs onnx fp32 vs onnx int8 on latency, size
and CER. The ONNX decoder matches PyTorch to about 1e-6 in tests.

I haven't run this on the trained weights yet, since I don't have them locally. What I did
measure, on a full-size model with random weights (so decoding always runs to the max length,
which makes these worst-case latencies), 4 CPU threads on my laptop, rough numbers:

| | median latency per clip | size |
|---|---|---|
| torch | ~1040 ms | 113 MB |
| onnx fp32 | ~820 ms | 114 MB |
| onnx int8 | ~230 ms | 39 MB |

Beam search itself now runs all live beams in one decoder call instead of one call per beam,
which took a clip from about 2.3 s to 0.94 s on that same model and gives identical output
(there's a test against the original implementation). Whether int8 keeps CER on the real model is
something I still need to check. On the toy model it did.

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

## Demo

```bash
export ASL_CHECKPOINT=/path/to/asl_transformer_v6_final.pth
python app/app.py
```

Weights aren't in the repo. The app looks for `ASL_CHECKPOINT`, a local file, or the Hugging Face
Hub (`sohenpatel22/asl-fingerspelling-conformer`), in that order. Checkpoints saved by the old
notebooks have numpy scalars in them, so convert a trusted one first with
`python -m asl.checkpoint convert old.pth new.pth`.

The LLM option needs `GROQ_API_KEY`. It only knows a fixed list of words, so it's a demo
feature and none of the numbers above use it.

## Layout

```
asl/          the package (model, data, training, decoding, eval, inference, serving, export)
app/          gradio demo
configs/      default, local_4gb, smoke
deploy/       prometheus, alert rules, grafana dashboard
tests/        pytest
scripts/      webcam demo, metric gate
notebooks/    original kaggle notebook
reports/      original course report
dvc.yaml      data + training pipeline
Dockerfile, docker-compose.yml
```

## Still to do

- run the export benchmark on the real weights
- better accuracy: CTC loss alongside attention, sequences longer than 64 frames, z and velocity
  features, an n-gram LM for rescoring
- deploy the API somewhere public

## Limitations

Only isolated fingerspelling clips, no full sentences. It struggles with bad lighting, odd
camera angles and signers unlike the training set, and short words tend to get hallucinated.
Also, the model was trained on the competition's landmarks but the demo runs MediaPipe on your
own video, so there's a bit of a domain gap.

MIT license.
