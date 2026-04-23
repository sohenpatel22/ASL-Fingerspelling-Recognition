import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import math
import os
from gtts import gTTS

# ==========================================
# 1. Setup, Hardcoded Config & Vocab
# ==========================================
DEVICE = torch.device('cpu') 
CKPT_PATH = 'asl_transformer_v6_best.pth'  # <-- Updated file name

class Config:
    FEATURE_SIZE   = 84
    MAX_SEQ_LEN    = 64
    MAX_PHRASE_LEN = 34
    D_MODEL        = 384
    ENC_LAYERS     = 6
    DEC_LAYERS     = 6
    N_HEADS        = 6
    FFN_DIM        = 1024
    EMBED_DIM      = 192
    DROPOUT        = 0.0 
    BEAM_WIDTH     = 5
    LENGTH_PENALTY = 0.6

# Official Kaggle Competition Character Map
char_to_idx = {' ': 0, '!': 1, '#': 2, '$': 3, '%': 4, '&': 5, "'": 6, '(': 7, ')': 8, '*': 9, '+': 10, ',': 11, '-': 12, '.': 13, '/': 14, '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20, '6': 21, '7': 22, '8': 23, '9': 24, ':': 25, ';': 26, '=': 27, '?': 28, '@': 29, '[': 30, '_': 31, 'a': 32, 'b': 33, 'c': 34, 'd': 35, 'e': 36, 'f': 37, 'g': 38, 'h': 39, 'i': 40, 'j': 41, 'k': 42, 'l': 43, 'm': 44, 'n': 45, 'o': 46, 'p': 47, 'q': 48, 'r': 49, 's': 50, 't': 51, 'u': 52, 'v': 53, 'w': 54, 'x': 55, 'y': 56, 'z': 57, '~': 58}
idx_to_char = {int(v): k for k, v in char_to_idx.items()}

N_CLASSES   = 59
START_IDX   = N_CLASSES
EOS_IDX     = N_CLASSES + 1
PAD_IDX     = N_CLASSES + 2
VOCAB_SIZE  = N_CLASSES + 3

# ==========================================
# 2. Model Architecture
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, maxlen=512, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(maxlen, d_model)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, kernel_size=31, dropout=0.0):
        super().__init__()
        self.ff1       = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv      = nn.Sequential(nn.Conv1d(d_model, 2*d_model, 1), nn.GLU(dim=1), nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model), nn.BatchNorm1d(d_model), nn.SiLU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout))
        self.ff2      = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
        self.norm_out = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        r, _ = self.attn(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x))
        x = x + self.drop_attn(r)
        x = x + self.conv(self.norm_conv(x).transpose(1, 2)).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)

class ConformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj   = nn.Sequential(nn.Conv1d(cfg.FEATURE_SIZE, cfg.D_MODEL, kernel_size=3, padding=1), nn.BatchNorm1d(cfg.D_MODEL), nn.ReLU())
        self.posenc = PositionalEncoding(cfg.D_MODEL, dropout=cfg.DROPOUT)
        self.layers = nn.ModuleList([ConformerBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.FFN_DIM, dropout=cfg.DROPOUT) for _ in range(cfg.ENC_LAYERS)])
    def forward(self, x):
        x = self.posenc(self.proj(x).permute(0, 2, 1))
        for layer in self.layers: x = layer(x)
        return x 

class TransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed    = nn.Embedding(VOCAB_SIZE, cfg.EMBED_DIM, padding_idx=PAD_IDX)
        self.proj_emb = nn.Linear(cfg.EMBED_DIM, cfg.D_MODEL)
        self.posenc   = PositionalEncoding(cfg.D_MODEL, dropout=cfg.DROPOUT)
        dec_layer     = nn.TransformerDecoderLayer(d_model=cfg.D_MODEL, nhead=cfg.N_HEADS, dim_feedforward=cfg.FFN_DIM, dropout=cfg.DROPOUT, batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=cfg.DEC_LAYERS)
        self.fc_out   = nn.Linear(cfg.D_MODEL, VOCAB_SIZE)
    def forward(self, tgt, memory):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        return self.fc_out(self.decoder(self.posenc(self.proj_emb(self.embed(tgt))), memory, tgt_mask=tgt_mask, tgt_key_padding_mask=(tgt == PAD_IDX)))

class ASLConformerSeq2Seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ConformerEncoder(cfg)
        self.decoder = TransformerDecoder(cfg)
    def forward(self, x, tgt): return self.decoder(tgt, self.encoder(x))

# Load Model Safely 
print("Loading model checkpoint...")
model = ASLConformerSeq2Seq(Config).to(DEVICE)
try:
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    # Handle if the checkpoint is just the state_dict directly or wrapped in a dict
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")

model.eval()

# ==========================================
# 3. Pipeline Processing Functions
# ==========================================
def wrist_normalize(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    for offset in [0, 42]:
        lx, ly = out[:, offset:offset+21], out[:, offset+21:offset+42]
        wx, wy = lx[:, 0:1], ly[:, 0:1]
        visible = (lx != 0).any(axis=1, keepdims=True)  
        lx, ly = np.where(visible, lx - wx, 0.0), np.where(visible, ly - wy, 0.0)
        span = max(float(np.abs(lx).max()), float(np.abs(ly).max()), 1e-6)
        out[:, offset:offset+21], out[:, offset+21:offset+42] = lx / span, ly / span
    return out.astype(np.float32)

def process_video_to_text(video_path):
    if not video_path: return "Please upload or record a video."
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_landmarks = np.zeros(84, dtype=np.float32)
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[hand_idx].classification[0].label
                idx_offset = 0 if label == 'Left' else 42
                frame_landmarks[idx_offset:idx_offset+21] = [lm.x for lm in hand_landmarks.landmark]
                frame_landmarks[idx_offset+21:idx_offset+42] = [lm.y for lm in hand_landmarks.landmark]
        frames_data.append(frame_landmarks)
    cap.release()
    
    if not frames_data: return "Error: MediaPipe could not detect any hands in this video. Please ensure good lighting."
    
    seq = wrist_normalize(np.vstack(frames_data))
    T = len(seq)
    if T > Config.MAX_SEQ_LEN: seq = seq[np.linspace(0, T-1, Config.MAX_SEQ_LEN, dtype=int)]
    elif T < Config.MAX_SEQ_LEN: seq = np.vstack([seq, np.zeros((Config.MAX_SEQ_LEN - T, 84), dtype=np.float32)])
    
    tensor_input = torch.from_numpy(seq.T).float().unsqueeze(0).to(DEVICE)
    
    # Beam Search
    with torch.no_grad():
        memory = model.encoder(tensor_input)
        beams = [([START_IDX], 0.0)]
        for _ in range(Config.MAX_PHRASE_LEN - 1):
            candidates = []
            for seq, score in beams:
                if seq[-1] == EOS_IDX:
                    candidates.append((seq, score))
                    continue
                logits = model.decoder(torch.tensor([seq], dtype=torch.long, device=DEVICE), memory)
                topk = F.log_softmax(logits[0, -1], dim=-1).topk(Config.BEAM_WIDTH)
                for lp, idx in zip(topk.values, topk.indices):
                    candidates.append((seq + [idx.item()], score + lp.item()))
            beams = sorted(candidates, key=lambda b: b[1] / max(len(b[0]) - 1, 1) ** Config.LENGTH_PENALTY, reverse=True)[:Config.BEAM_WIDTH]
            if all(b[0][-1] == EOS_IDX for b in beams): break
                
    return ''.join([idx_to_char[i] for i in beams[0][0] if i not in (START_IDX, EOS_IDX, PAD_IDX) and i in idx_to_char])


# 4. Text to Speech Component (TTS) 
def process_video_and_speak(video_file):
    # 1. Run your existing working model prediction
    predicted_text = process_video_to_text(video_file) 
    
    # 2. Convert that text to an audio file
    tts = gTTS(text=predicted_text, lang='en')
    audio_path = "output_audio.mp3"
    tts.save(audio_path)
    
    # 3. Return BOTH the text and the audio file to the UI
    return predicted_text, audio_path
    
# ==========================================
# 4. Gradio Interface (Blocks Version)
# ==========================================
with gr.Blocks(theme="ocean") as demo:
    gr.Markdown("# 🤟 ASL Fingerspelling Translator")
    gr.Markdown("### University of Toronto | MIE1517 Group 12")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Capture ASL", height=400)
            submit_btn = gr.Button("Translate Signs", variant="primary")
        with gr.Column():
            text_output = gr.Textbox(label="Model Prediction", lines=4)
            # --- ADDED AUDIO PLAYER ---
            audio_output = gr.Audio(label="Audio Output", autoplay=True) 
    
    # --- UPDATED BUTTON CLICK ---
    submit_btn.click(
        fn=process_video_and_speak,          # Points to the new wrapper function
        inputs=video_input, 
        outputs=[text_output, audio_output]  # Now outputs to both text and audio
    )

if __name__ == "__main__":
    demo.launch()
