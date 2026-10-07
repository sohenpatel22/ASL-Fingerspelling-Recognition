# Phase 5: training from scratch

One run on a Kaggle T4, 16 epochs, about 2h45m. Recipe in `configs/scratch.yaml`: random initial
weights, 192 frames, heavy augmentation, CutMix on half the clips, decoder input masking (0.2), a joint
CTC loss (weight 0.25), weight averaging (EMA 0.999), main + supplemental data, same signer split as
everything else. Scored with the same protocol as the earlier phases: seeded random clips (1000 val,
3000 test), beam 5, length penalty 0.0.

| model | val CER | test CER (mean per clip) | test CER (corpus) | exact match |
|---|---|---|---|---|
| tuned v5 (deployed) | 0.270 | **0.307** | 0.299 | 0.392 |
| scratch, CTC head (greedy) | 0.300 | 0.332 | 0.328 | 0.174 |
| scratch, attention decoder (beam) | 0.373 | 0.429 | 0.396 | |

Paired per-signer differences on the 14 test signers (negative means the first is better):

| first | second | mean diff | 95% CI | signers where first is better |
|---|---|---|---|---|
| scratch CTC | tuned v5 | +0.022 | 0.009 to 0.035 | 4 of 14 |
| scratch CTC | scratch attention | -0.101 | -0.153 to -0.058 | 14 of 14 |
| scratch attention | tuned v5 | +0.124 | 0.089 to 0.166 | 0 of 14 |

What I take from it:

- **It did not beat v5, so nothing was deployed.** The best of the two decoders is 0.025 behind the tuned
  v5 on test and the interval excludes zero.
- **The training curve was still going down when the schedule ended.** Validation greedy CER went
  1.39, 0.74, 0.52, 0.48, 0.45 over the first five epochs and then crept to 0.373 by epoch 16
  (`raw/scratch/metrics.jsonl`). The winning solutions trained for about 300 epochs, so this
  recipe is under-trained rather than broken.
- **The CTC head is the better decoder here.** It beats the attention decoder for every one of the 14
  signers. With 16 epochs the attention decoder has not learned to read the encoder yet, while the
  CTC loss forces a monotonic alignment from the start. Its 0.332 is level with the first v5
  checkpoint (0.334), though it still loses to the tuned one.
- The CTC head has no sequence confidence, so the flag rule (which needs one) only applies to the
  attention decoder.

Not tried: a longer run (60 or more epochs, roughly 8 hours of T4 time), pose and lip landmarks for
clips where the hands are missed, and using the CTC output to rescore beam search hypotheses.

Files: `raw/` (attention decoder results and training log), `ctc/` (CTC head results),
`paired_*.json` (made with `scripts/paired.py`).
