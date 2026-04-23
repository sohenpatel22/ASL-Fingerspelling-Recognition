---
title: MIE1517 ASL Project
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
python_version: "3.10"
---

# 🤟 ASL Fingerspelling Recognition: End-to-End Deployment
**University of Toronto | MIE1517 Final Project | Group 12**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00A89D.svg?style=flat&logo=Google&logoColor=white)](https://developers.google.com/mediapipe)

## 📌 Project Overview
This repository contains the deployment architecture for an end-to-end American Sign Language (ASL) fingerspelling recognition system. It bridges the gap between raw video input and natural language text prediction using a custom **Conformer-Transformer** sequence-to-sequence model.

The application is containerized and deployed via a Gradio web interface, allowing for real-time video upload, feature extraction, and character-level translation.

## 🧠 Model Architecture & Pipeline
Our pipeline transforms raw `.mp4` video into transcribed text through three distinct stages:

### 1. Spatio-Temporal Feature Extraction (MediaPipe)
* **Input:** Raw RGB video frames.
* **Extraction:** We utilize Google MediaPipe (`solutions.hands`) to extract 21 2D landmarks (x, y coordinates) per hand per frame, resulting in an 84-dimensional feature vector.
* **Normalization:** Coordinates are passed through a custom `wrist_normalize` function. This centers the spatial data around the wrist landmark and scales it dynamically to make the model invariant to the signer's distance from the camera.
* **Temporal Padding:** Sequences are linearly interpolated or zero-padded to a fixed length of `MAX_SEQ_LEN = 64` to maintain tensor uniformity.

### 2. The Conformer-Transformer Network
The core deep learning model is built in PyTorch and consists of two main components:
* **Conformer Encoder:** Replaces standard Transformer self-attention with Conformer blocks (Convolution-augmented Transformer). The 1D Depthwise Convolutions allow the network to capture localized, high-frequency temporal dependencies (the micro-movements of fingers changing shapes) before passing the features to global attention layers.
* **Transformer Decoder:** An autoregressive decoder that attends to the Conformer's memory bank and predicts character tokens sequentially.

### 3. Beam Search Decoding
Instead of greedy decoding (taking the highest probability character at each step), the inference script utilizes **Beam Search** with a defined `BEAM_WIDTH = 5` and `LENGTH_PENALTY = 0.6`. This allows the model to explore multiple prediction paths simultaneously, significantly reducing the Character Error Rate (CER) on complex words.

---

## 📂 Repository Structure

```text
/
├── app.py                            # Main Gradio application, vocabulary mapping, and PyTorch architecture
├── requirements.txt                  # Deployment dependencies (MediaPipe pinned to 0.10.21)
├── asl_transformer_v6_best.pth       # Model weights
├── .gitattributes                    # Git LFS tracking for the large .pth file
└── README.md                         # Project documentation and HF routing
