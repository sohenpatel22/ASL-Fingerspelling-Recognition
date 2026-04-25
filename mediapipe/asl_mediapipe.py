import cv2
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from pathlib import Path
from collections import deque

# 1. CONFIG
CHAR_MAP_PATH  = Path('character_to_prediction_index.json')
MODEL_CKPT     = Path('asl_transformer_v5_best.pth')

FEATURE_SIZE   = 84
MAX_SEQ_LEN    = 64 
MAX_PHRASE_LEN = 34
D_MODEL        = 384
ENC_LAYERS     = 6
DEC_LAYERS     = 6
N_HEADS        = 6
FFN_DIM        = 1024
EMBED_DIM      = 192
DROPOUT        = 0.15
BEAM_WIDTH     = 5
LENGTH_PENALTY = 0.6

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# 2. VOCAB
with open(CHAR_MAP_PATH) as f:
    char_to_idx = json.load(f)

idx_to_char = {v: k for k, v in char_to_idx.items()}
N_CLASSES   = len(char_to_idx)   # 59

START_IDX  = N_CLASSES           # 59
EOS_IDX    = N_CLASSES + 1       # 60
PAD_IDX    = N_CLASSES + 2       # 61
VOCAB_SIZE = N_CLASSES + 3       # 62

print(f'START={START_IDX}  EOS={EOS_IDX}  PAD={PAD_IDX}  VOCAB={VOCAB_SIZE}')

# 3. MODEL
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size,
                      padding=kernel_size // 2, groups=d_model),
            nn.BatchNorm1d(d_model), nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        r = self.norm_attn(x); r, _ = self.attn(r, r, r)
        x = x + self.drop_attn(r)
        r = self.norm_conv(x).transpose(1, 2)
        x = x + self.conv(r).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(FEATURE_SIZE, D_MODEL, kernel_size=3, padding=1),
            nn.BatchNorm1d(D_MODEL), nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(D_MODEL, dropout=DROPOUT)
        self.layers  = nn.ModuleList([
            ConformerBlock(D_MODEL, N_HEADS, FFN_DIM, dropout=DROPOUT)
            for _ in range(ENC_LAYERS)
        ])

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 1)
        x = self.pos_enc(x)
        for layer in self.layers: x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed    = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_IDX)
        self.proj_emb = nn.Linear(EMBED_DIM, D_MODEL)
        self.pos_enc  = PositionalEncoding(D_MODEL, dropout=DROPOUT)
        dec_layer     = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=DEC_LAYERS)
        self.fc_out   = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, tgt, memory):
        L    = tgt.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad_mask = (tgt == PAD_IDX)
        emb  = self.pos_enc(self.proj_emb(self.embed(tgt)))
        out  = self.decoder(emb, memory, tgt_mask=mask,
                            tgt_key_padding_mask=tgt_pad_mask)
        return self.fc_out(out)


class ASLTransformerSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConformerEncoder()
        self.decoder = TransformerDecoder()

    def forward(self, x, tgt):
        return self.decoder(tgt, self.encoder(x))

# 4. LOAD WEIGHTS
model = ASLTransformerSeq2Seq().to(DEVICE)
ckpt  = torch.load(MODEL_CKPT, map_location=DEVICE)
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.eval()
print('**Model loaded**')

# 5. PREPROCESSING
def wrist_normalize(seq):
    """Matches training notebook exactly"""
    out = seq.copy()
    for offset in [0, 42]:
        lx, ly = out[:, offset:offset+21], out[:, offset+21:offset+42]
        wx, wy = lx[:, 0:1], ly[:, 0:1]
        visible = (lx != 0).any(axis=1, keepdims=True)
        lx = np.where(visible, lx - wx, 0.0)
        ly = np.where(visible, ly - wy, 0.0)
        span = max(np.abs(lx).max(), np.abs(ly).max(), 1e-6)
        out[:, offset:offset+21]    = lx / span
        out[:, offset+21:offset+42] = ly / span
    return out


def prepare_sequence(frame_buffer):
    """Convert buffer → (1, 84, MAX_SEQ_LEN) tensor for model"""
    seq = np.array(frame_buffer, dtype=np.float32)  # (T, 84)
    seq = wrist_normalize(seq)

    T = len(seq)
    if T >= MAX_SEQ_LEN:
        seq = seq[np.linspace(0, T - 1, MAX_SEQ_LEN, dtype=int)]
    else:
        seq = np.concatenate([
            seq,
            np.zeros((MAX_SEQ_LEN - T, FEATURE_SIZE), dtype=np.float32)
        ])

    seq = seq.T  # (84, MAX_SEQ_LEN) ← matches training
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def landmarks_to_array(left_hand, right_hand):
    """MediaPipe → (84,) array. Layout matches training."""
    row = np.zeros(84, dtype=np.float32)
    if left_hand:
        for i, lm in enumerate(left_hand.landmark):
            row[i]      = lm.x
            row[21 + i] = lm.y
    if right_hand:
        for i, lm in enumerate(right_hand.landmark):
            row[42 + i] = lm.x
            row[63 + i] = lm.y
    return row

# 6. BEAM SEARCH
@torch.no_grad()
def beam_search(frame_buffer):
    x      = prepare_sequence(frame_buffer)   # (1, 84, MAX_SEQ_LEN)
    memory = model.encoder(x)

    beams = [{'score': 0.0, 'tokens': [START_IDX], 'done': False}]

    for _ in range(MAX_PHRASE_LEN):
        active = [b for b in beams if not b['done']]
        if not active:
            break

        next_beams = [b for b in beams if b['done']]

        for beam in active:
            t      = torch.tensor([beam['tokens']], device=DEVICE)
            logit  = model.decoder(t, memory)[:, -1, :]
            logp   = F.log_softmax(logit, dim=-1).squeeze(0)
            topk_lp, topk_ids = logp.topk(BEAM_WIDTH)

            for lp, idx in zip(topk_lp.tolist(), topk_ids.tolist()):
                next_beams.append({
                    'score' : beam['score'] + lp,
                    'tokens': beam['tokens'] + [idx],
                    'done'  : (idx == EOS_IDX),
                })

        def norm(b):
            l = max(len(b['tokens']) - 1, 1)
            return b['score'] / (l ** LENGTH_PENALTY)

        next_beams.sort(key=norm, reverse=True)
        beams = next_beams[:BEAM_WIDTH]

    best = max(beams, key=lambda b: b['score'] / max(len(b['tokens']) - 1, 1) ** LENGTH_PENALTY)
    result = []
    for t in best['tokens'][1:]:
        if t == EOS_IDX:
            break
        result.append(idx_to_char.get(t, ''))
    return ''.join(result)

# 7. MEDIAPIPE SETUP
import urllib.request

# Download the hand landmarker model if not present
MODEL_PATH = 'hand_landmarker.task'
if not Path(MODEL_PATH).exists():
    print('Downloading hand landmarker model...')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        MODEL_PATH
    )
    print('**Model downloaded**')

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult  = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5)

detector = HandLandmarker.create_from_options(options)

def get_hands_from_result(result):
    """Extract left/right hand landmarks from new API result"""
    left_hand  = None
    right_hand = None
    if result.hand_landmarks and result.handedness:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name  # 'Left' or 'Right'
            if label == 'Left':
                left_hand  = landmarks
            else:
                right_hand = landmarks
    return left_hand, right_hand

def landmarks_to_array_new(left_hand, right_hand):
    """New API uses NormalizedLandmark objects"""
    row = np.zeros(84, dtype=np.float32)
    if left_hand:
        for i, lm in enumerate(left_hand):
            row[i]      = lm.x
            row[21 + i] = lm.y
    if right_hand:
        for i, lm in enumerate(right_hand):
            row[42 + i] = lm.x
            row[63 + i] = lm.y
    return row

# 8. MAIN LOOP
cap          = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=MAX_SEQ_LEN)
prediction   = ''
recording    = False

print('\n=== Controls ===')
print('SPACE : start/stop recording')
print('Q     : quit')

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)
    left_hand, right_hand = get_hands_from_result(result)

    # Draw landmarks
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy  = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Collect frames if recording
    if recording:
        row = landmarks_to_array_new(left_hand, right_hand)
        frame_buffer.append(row)
        frame_count += 1

        # Predict every 30 frames while recording (rolling prediction)
        if len(frame_buffer) >= 10 and len(frame_buffer) % 30 == 0:
            print('Running prediction...')
            prediction = beam_search(frame_buffer)
            print(f'Prediction: {prediction}')

    frame = cv2.flip(frame, 1)

    # UI overlay
    status = 'RECORDING' if recording else 'SPACE to record'
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if recording else (200, 200, 200), 2)
    cv2.putText(frame, f'Pred: {prediction}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f'Frames: {frame_count}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow('ASL Fingerspelling', frame)

# Key handling — AFTER waitKey
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
            recording = not recording
            if not recording:
                if len(frame_buffer) >= 10:
                    print('Running final prediction...')
                    prediction = beam_search(frame_buffer)
                    print(f'Final Prediction: {prediction}')
                frame_buffer.clear()
                frame_count = 0 
cap.release()
cv2.destroyAllWindows()
detector.close()