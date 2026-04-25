# ASL Fingerspelling Recognition

**End-to-End ASL Fingerspelling Recognition using MediaPipe, Conformer-Transformer Modeling, and Gradio Deployment**


---

## Overview

This project implements an end-to-end American Sign Language (ASL) fingerspelling recognition pipeline that converts hand gesture video input into predicted English text.

The system combines computer vision, sequence modeling, and deployment:

- MediaPipe-based hand landmark extraction
- Landmark normalization and temporal preprocessing
- Conformer-Transformer sequence-to-sequence modeling
- Beam search decoding for character-level prediction
- Gradio application for interactive inference

The goal of this project was to build a full deep learning pipeline that connects raw video input to text output, rather than only training a standalone model.

---

## Project Pipeline

```text
Raw ASL Video
      |
MediaPipe Hand Landmark Extraction
      |
Wrist-Based Landmark Normalization
      |
Temporal Padding / Sequence Processing
      |
Conformer Encoder
      |
Transformer Decoder
      |
Beam Search Decoding
      |
Predicted Text Output

```

---

## Problem Statement

ASL fingerspelling involves spelling words letter by letter using hand gestures. Recognizing fingerspelling from video is challenging because:

* finger movements are fast and subtle
* different signers have different hand shapes and signing speeds
* video quality, lighting, and camera angle affect hand detection
* the model must learn both spatial hand structure and temporal movement patterns

This project focuses on isolated ASL fingerspelling recognition, where the model predicts a character sequence from a short input video.

---

## Key Features

* End-to-end video-to-text recognition pipeline
* MediaPipe hand landmark extraction
* 84-dimensional per-frame feature representation
* Wrist-centered landmark normalization
* Conformer encoder for local and global temporal representation
* Transformer decoder for sequence prediction
* Beam search decoding
* Gradio-based inference application
* Organized training, inference, video processing, and deployment modules

---

## Model Architecture

The core model is a Conformer-Transformer sequence model.

### MediaPipe Feature Extraction

MediaPipe is used to detect hand landmarks from each video frame.

For each frame:

* 21 landmarks are extracted per hand
* each landmark contains x and y coordinates
* two hands produce an 84-dimensional feature vector

```text
21 landmarks * 2 coordinates * 2 hands = 84 features per frame
```

The extracted landmarks are normalized using a wrist-centered strategy to reduce sensitivity to camera distance and hand position.

---

### Conformer Encoder

The Conformer encoder is used to learn temporal hand movement representations.

It combines:

* self-attention for global sequence context
* convolutional layers for local temporal motion patterns

This is useful for fingerspelling because many letters differ by small hand and finger movements across time.

---

### Transformer Decoder

The Transformer decoder predicts the output character sequence autoregressively.

It attends to the encoded video representation and generates one character at a time until the sequence is complete.

---

### Beam Search Decoding

Instead of selecting only the highest-probability character at each step, beam search keeps multiple candidate sequences during decoding.

This improves sequence-level prediction quality by reducing the chance of early decoding mistakes.

The project uses:

```text
Beam width: 5
Length penalty: 0.6
```

---

## Repository Structure

```text
asl-fingerspelling-recognition/
│
├── .github/                  # GitHub configuration files
│
├── app/                      # Gradio inference application
│
├── src/                      # Core machine learning pipeline
│   ├── __init__.py
│   ├── inference/            # Inference and decoding logic
│   ├── models/               # Model architecture implementation
│   ├── train/                # Training pipeline modules
│   └── video_pipeline/       # Video processing and feature pipeline
│
├── scripts/                  # Helper scripts
│   └── __init__.py
│
├── notebooks/                # Experiment and analysis notebooks
│   └── Models-asl-v6-supplemental-final.ipynb
│
├── mediapipe/                # MediaPipe landmark extraction module
│   └── asl_mediapipe/
│
├── setup/                    # Setup and environment-related files
│
├── reports/                  # Final project report
│   └── Team12_Final_Report
│
├── docs/                     # Supporting documentation
│   ├── GITHUB_SETUP
│   └── project_plan
│
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sohenpatel22/asl-fingerspelling-recognition.git
cd asl-fingerspelling-recognition
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Gradio inference app:

```bash
python app/app.py
```

Then open the local Gradio URL shown in the terminal.

Usually this will be:

```text
http://localhost:7860
```

The app allows the user to upload or record an ASL fingerspelling video and returns the predicted text output.

---

## Model Weights

The trained model checkpoint is included as:

```text
asl_transformer_v6_best.pth
```

If the model file is too large for normal GitHub storage, it should be tracked using Git LFS.

To install and use Git LFS:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

Then add and commit the model file normally.

---

## Dataset

This project is based on ASL fingerspelling recognition data involving hand landmark sequences and character-level labels.

The broader task uses:

* hand landmark sequences
* signer-level variation
* character-level transcription targets
* special tokens such as START, EOS, and PAD

The model learns to map temporal hand landmark sequences to text predictions.

---

## Training and Experiments

Training and experimentation are organized across:

```text
src/train/
notebooks/
scripts/
```

The supplemental notebook contains experimental work and model development details:

```text
notebooks/Models-asl-v6-supplemental-final.ipynb
```

The training pipeline includes:

* sequence preparation
* landmark normalization
* model training
* checkpoint saving
* validation monitoring

---

## Inference Flow

During inference:

1. A video is provided through the app.
2. MediaPipe extracts hand landmarks frame by frame.
3. Landmark sequences are normalized.
4. The sequence is passed into the trained Conformer-Transformer model.
5. Beam search decoding generates the predicted character sequence.
6. The final text prediction is displayed in the app.

---

## Project Report

The full team project report is available in:

```text
reports/Final_Report.ipynb
```

The report includes:

* project motivation
* methodology
* architecture details
* experiments
* observations
* limitations
* future work

---

## Limitations

This project currently focuses on isolated ASL fingerspelling recognition.

Current limitations include:

* sensitivity to lighting conditions
* sensitivity to camera angle and hand visibility
* difficulty with very fast or unclear fingerspelling
* limited generalization to unseen signing styles
* no full ASL sentence-level understanding

---

## Future Work

Possible improvements include:

* real-time webcam streaming inference
* stronger signer-independent generalization
* improved data augmentation for hand landmarks
* attention visualization for interpretability
* larger-scale training on more diverse signers
* extending from fingerspelling to full ASL phrase recognition
* adding text-to-speech output after prediction

---

## Skills Demonstrated

This project demonstrates practical experience with:

* computer vision preprocessing
* MediaPipe landmark extraction
* PyTorch deep learning
* sequence-to-sequence modeling
* Conformer and Transformer architectures
* beam search decoding
* model inference pipelines
* Gradio deployment
* ML project organization

---