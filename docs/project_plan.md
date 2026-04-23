# MIE1517 Final Sprint: ASL Fingerspelling Translator

## 1. The 7-Day Sprint Timeline

* **Days 1–2 (Now):** Build and test the MediaPipe + Gradio inference pipeline locally using the v5 weights.
* **Day 3:** Analyze the Scheduled Sampling run. If the CER drops to ~0.38, freeze the model. If it continues to hallucinate, implement the CTC Auxiliary Loss and launch the final training run.
* **Day 4:** Push the Gradio app to Hugging Face Spaces. Iron out any dependency issues.
* **Day 5:** Model freeze. Take whichever `.pth` weights perform best and plug them into the Gradio app.
* **Day 6:** Pre-record 3 to 5 clean demo videos for the presentation. Ensure they are signed deliberately and clearly.
* **Day 7:** Finalize the project report, focusing on the analysis of exposure bias, domain shifts, and engineering tradeoffs.

---

## 2. Team Handoff: Gradio Deployment Pipeline

**The Goal:** Build the live demonstration pipeline for our final presentation. 
**The Problem:** Our PyTorch model (`ASLConformerSeq2Seq`) only understands floating-point coordinate arrays. It cannot read `.mp4` files or webcam feeds directly.
**Your Mission:** Use Google MediaPipe to extract hand skeletons from a video, normalize the data, feed it into the PyTorch model, and display the translated text via a Gradio web interface.

*Note: Build and test this pipeline right now using our older `v5` checkpoint. When the final `v6` model is ready, we will just swap the `.pth` file.*

### Step 1: Set Up Local Environment
Run this in your terminal to install dependencies:
`pip install torch torchvision mediapipe opencv-python gradio numpy editdistance`

### Step 2: Project Structure
Create a new folder with these exactly four files:
1.  `model.py` (Holds the PyTorch architecture)
2.  `app.py` (Holds the MediaPipe logic and Gradio UI)
3.  `requirements.txt` (For Hugging Face deployment)
4.  `asl_transformer_v5_final.pth` (Download from Kaggle and place here)

### Step 3: Populate `model.py`
Copy the following classes/functions from our Kaggle training notebook into `model.py`:
* `Config` class
* `wrist_normalize` function (Ensure it has the `.any(axis=1)` fix)
* `PositionalEncoding`
* `ConformerBlock`, `ConformerEncoder`, `TransformerDecoder`
* `ASLConformerSeq2Seq`

### Step 4: Populate `app.py`
This script extracts the skeleton, formats the tensors, runs the beam search, and launches the UI. Paste this into `app.py`:

```python
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
import gradio as gr
from model import ASLConformerSeq2Seq, Config, wrist_normalize

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Model
CKPT_PATH = 'asl_transformer_v5_final.pth'
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
char_to_idx = ckpt.get('char_to_idx')
idx_to_char = ckpt.get('idx_to_char', {int(v): k for k, v in char_to_idx.items()})
START_IDX, EOS_IDX, PAD_IDX = ckpt.get('start_idx', 59), ckpt.get('eos_idx', 60), ckpt.get('pad_idx', 61)

model = ASLConformerSeq2Seq(Config).to(DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 2. MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        frame_landmarks = np.zeros(84, dtype=np.float32)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[hand_idx].classification[0].label
                coords_x = [lm.x for lm in hand_landmarks.landmark]
                coords_y = [lm.y for lm in hand_landmarks.landmark]
                
                if label == 'Left': frame_landmarks[0:21], frame_landmarks[21:42] = coords_x, coords_y
                elif label == 'Right': frame_landmarks[42:63], frame_landmarks[63:84] = coords_x, coords_y
                    
        frames_data.append(frame_landmarks)
        
    cap.release()
    if not frames_data: return None
        
    seq = wrist_normalize(np.vstack(frames_data))
    T = len(seq)
    if T > Config.MAX_SEQ_LEN: seq = seq[np.linspace(0, T-1, Config.MAX_SEQ_LEN, dtype=int)]
    elif T < Config.MAX_SEQ_LEN: seq = np.vstack([seq, np.zeros((Config.MAX_SEQ_LEN - T, 84), dtype=np.float32)])
        
    return torch.from_numpy(seq.T).float().unsqueeze(0).to(DEVICE)

@torch.no_grad()
def process_video(video_file):
    if video_file is None: return "Please upload a video."
    x = extract_landmarks(video_file)
    if x is None: return "Error: No hands detected by MediaPipe."
        
    memory = model.encoder(x)
    beams = [([START_IDX], 0.0)]
    
    for _ in range(Config.MAX_PHRASE_LEN - 1):
        candidates = []
        for seq, score in beams:
            if seq[-1] == EOS_IDX:
                candidates.append((seq, score))
                continue
            tgt = torch.tensor([seq], dtype=torch.long, device=DEVICE)
            logits = model.decoder(tgt, memory)
            logp = F.log_softmax(logits[0, -1], dim=-1)
            topk = logp.topk(Config.BEAM_WIDTH)
            for lp, idx in zip(topk.values, topk.indices): candidates.append((seq + [idx.item()], score + lp.item()))
                
        beams = sorted(candidates, key=lambda b: b[1] / max(len(b[0]) - 1, 1) ** Config.LENGTH_PENALTY, reverse=True)[:Config.BEAM_WIDTH]
        if all(b[0][-1] == EOS_IDX for b in beams): break
            
    best_seq = beams[0][0]
    return ''.join([idx_to_char[i] for i in best_seq if i not in (START_IDX, EOS_IDX, PAD_IDX) and i in idx_to_char])

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload ASL Fingerspelling Video"),
    outputs=gr.Textbox(label="Predicted Text", text_align="center"),
    title="ASL Fingerspelling Translator (Group 12)"
)

if __name__ == "__main__":
    iface.launch()