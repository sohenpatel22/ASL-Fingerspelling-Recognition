# ASL Fingerspelling Recognition

Turns a short video of ASL fingerspelling into text. MediaPipe pulls hand landmarks out of the
video, a Conformer encoder + Transformer decoder reads them, and beam search produces the
characters. There's a Gradio demo on top, with optional LLM cleanup and text-to-speech.

I trained it on the Kaggle
[Google - American Sign Language Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling)
data. Originally this was a course project (MIE1517, UofT). I've been reworking it into a proper
package with tests, configs and CI, and I'm adding the MLOps side (tracking, data versioning,
serving) on top.

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
pip install -e ".[dev,app]"
pytest
asl-train --config configs/smoke.yaml
```

The smoke config trains a tiny model on generated data, which is handy for checking that
training, checkpointing and evaluation all still work without downloading anything.

## Training on the real data

Get the competition data from Kaggle, then turn the parquet files into normalized `.npy`
sequences once (needs `pip install -e ".[preprocess]"`):

```bash
asl-preprocess --comp-dir data/raw/asl-fingerspelling
```

Train and evaluate:

```bash
asl-train --config configs/default.yaml
asl-train --config configs/local_4gb.yaml        # for a 4GB GPU
asl-train --config configs/default.yaml train.epochs=5 train.lr=1e-4
asl-eval --checkpoint checkpoints/v6/best.pth --config configs/default.yaml --split test
```

Any config value can be overridden on the command line like that. `--resume` picks up from
`last.pth`, and `train.init_from=<ckpt>` warm-starts from an older model (it handles the 61 -> 62
token change from adding EOS).

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
asl/          the package (model, data, training, decoding, eval, inference)
app/          gradio demo
configs/      default, local_4gb, smoke
tests/        pytest
scripts/      webcam demo
notebooks/    original kaggle notebook
reports/      original course report
```

## Still to do

- DVC for data, MLflow for experiments
- FastAPI service, Docker, monitoring
- ONNX export and quantization
- better accuracy: CTC loss alongside attention, sequences longer than 64 frames, z and velocity
  features, an n-gram LM for rescoring

## Limitations

Only isolated fingerspelling clips, no full sentences. It struggles with bad lighting, odd
camera angles and signers unlike the training set, and short words tend to get hallucinated.
Also, the model was trained on the competition's landmarks but the demo runs MediaPipe on your
own video, so there's a bit of a domain gap.

MIT license.
