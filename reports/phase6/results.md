# Phase 6: continuing the scratch run

Restarted from the phase 5 weights (`configs/continue.yaml`): same recipe, a fresh cosine schedule
from lr 3e-4, up to 40 more epochs. Early stopping (patience 12, on the attention decoder's greedy
validation CER) ended it at epoch 26 after about 3h40m, and the checkpoint it kept is epoch 14.
Scored with the usual protocol: seeded random clips (1000 val, 3000 test), beam 5 with length
penalty 0.0 for the attention decoder, greedy decoding for the CTC head.

| model | val CER | test CER (mean per clip) | test CER (corpus) | exact match |
|---|---|---|---|---|
| tuned v5 (deployed) | 0.270 | **0.307** | 0.299 | 0.392 |
| continued, CTC head | 0.284 | 0.316 | 0.311 | 0.200 |
| continued, attention decoder | 0.365 | 0.414 | 0.382 | 0.257 |
| phase 5 (16 epochs), CTC head | 0.300 | 0.332 | 0.328 | 0.174 |

Paired per-signer differences on the 14 test signers (negative means the first is better):

| first | second | mean diff | 95% CI | signers where first is better |
|---|---|---|---|---|
| continued CTC | tuned v5 | +0.005 | -0.011 to 0.020 | 6 of 14 |
| continued CTC | phase 5 CTC | -0.017 | -0.021 to -0.014 | 14 of 14 |
| continued attention | tuned v5 | +0.109 | 0.074 to 0.147 | 0 of 14 |

What I take from it:

- **More training helped, but not enough to replace v5.** The CTC head is 0.017 better than after 16
  epochs, for every signer. Against the tuned v5 it is 0.008 behind on test, and the paired interval
  contains zero, so I can't tell the two apart. I only deploy a model that is clearly better, so
  nothing changed in the app.
- **The attention decoder is still the problem.** Its teacher-forced accuracy looks fine (about 0.80) but it
  decodes at 0.36 validation CER and barely moved in 26 epochs, while the CTC head kept improving
  (0.305 to 0.288). I don't know why; decoder input masking and CutMix may make its job
  harder than it is for CTC.
- **I stopped on the wrong metric.** Early stopping and checkpoint choice used the attention decoder's
  CER, which is noisy and flat, so the kept checkpoint is epoch 14. The CTC head's validation CER was
  0.292 there and 0.288 at epochs 20 to 26. That is a gap of about 0.004, so it would not have
  changed the conclusion, but it is the same lesson as with v5: pick the checkpoint with the metric
  you will actually serve.
- The remaining error is probably the data. Everything here sees only hand x,y landmarks, and the
  earlier analysis put most of v5's error on signers whose hands are missed.

Not tried: rescoring beam search hypotheses with the CTC head, averaging this model with v5, and
pose and lip landmarks.

Files: `raw/` (metrics per epoch and all four scoring runs), `paired_*.json` (made with
`scripts/paired.py`).
