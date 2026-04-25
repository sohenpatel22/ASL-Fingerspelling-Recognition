"""
Inference and deployment utilities for ASL Fingerspelling Recognition.

This module handles:
- Loading trained models
- Running inference on parquet data and video
- Batch prediction
- Result formatting and export

"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
from tqdm import tqdm

from config import (
    DEVICE, CHECKPOINT_DIR, IDX_TO_CHAR, START_IDX, EOS_IDX, PAD_IDX,
    BEAM_CONFIG, MAX_FRAMES, NORMALIZE_LANDMARKS,
)
from models import CNNLSTMFingerSpeller
from utils import compute_cer


class InferenceEngine:
    """
    Inference engine for ASL fingerspelling recognition.
    
    Loads model checkpoint and runs inference on landmark sequences.
    """
    
    def __init__(self, checkpoint_path: Path = None):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_DIR / "best_model.pth"
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint."""
        model = CNNLSTMFingerSpeller()
        
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
            
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            else:
                model.load_state_dict(checkpoint)
            
            print(f"✓ Loaded model from {self.checkpoint_path}")
        else:
            print(f"⚠ Checkpoint not found at {self.checkpoint_path}")
        
        model = model.to(DEVICE)
        return model
    
    @torch.no_grad()
    def predict(
        self,
        landmarks: np.ndarray,
        beam_width: int = BEAM_CONFIG["width"],
    ) -> str:
        """
        Predict fingerspelling text from landmarks.
        
        Args:
            landmarks: (T, D) numpy array of landmark coordinates
            beam_width: number of hypotheses in beam search
        
        Returns:
            predicted_text: string of predicted characters
        """
        # Preprocess
        landmarks = self._preprocess_landmarks(landmarks)  # (D, T)
        
        # Add batch dimension
        landmarks = torch.from_numpy(landmarks).float().unsqueeze(0)  # (1, D, T)
        landmarks = landmarks.to(DEVICE)
        
        # Encode
        encoder_out = self.model.cnn_encoder(landmarks)  # (1, T, hidden)
        lstm_out, _ = self.model.lstm_decoder(encoder_out)  # (1, T, hidden*2)
        
        # CTC beam search
        ctc_logits = self.model.char_decoder.ctc_projection(lstm_out)  # (1, T, vocab)
        
        predictions = self._beam_search(ctc_logits, beam_width)
        predicted_indices = predictions[0]
        
        # Convert indices to text
        predicted_text = self._decode_indices(predicted_indices)
        
        return predicted_text
    
    def _preprocess_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Preprocess landmark sequence."""
        # landmarks: (T, D)
        
        # Normalize
        if NORMALIZE_LANDMARKS:
            landmarks = np.clip(landmarks, -1.0, 1.0)
        
        # Pad/truncate
        T = landmarks.shape[0]
        if T < MAX_FRAMES:
            padding = np.zeros((MAX_FRAMES - T, landmarks.shape[1]))
            landmarks = np.vstack([landmarks, padding])
        elif T > MAX_FRAMES:
            landmarks = landmarks[:MAX_FRAMES]
        
        # Transpose to (D, T)
        landmarks = landmarks.T
        
        return landmarks
    
    def _beam_search(
        self,
        ctc_logits: torch.Tensor,
        beam_width: int = 5,
    ) -> List[List[int]]:
        """
        Simple beam search decoding.
        
        Args:
            ctc_logits: (B, T, vocab)
        
        Returns:
            predictions: list of predicted sequences
        """
        batch_size = ctc_logits.size(0)
        predictions = []
        
        for b in range(batch_size):
            logits = ctc_logits[b]  # (T, vocab)
            
            # Greedy: take argmax at each timestep
            chars = torch.argmax(logits, dim=1)  # (T,)
            
            # Collapse repeated characters (CTC rule)
            collapsed = []
            for i, c in enumerate(chars):
                if i == 0 or c.item() != chars[i-1].item():
                    collapsed.append(c.item())
            
            # Remove blanks and pads
            collapsed = [c for c in collapsed if c != PAD_IDX and c != 46]
            
            predictions.append(collapsed)
        
        return predictions
    
    def _decode_indices(self, indices: List[int]) -> str:
        """Convert character indices to text."""
        text = ""
        for idx in indices:
            if idx == START_IDX:
                continue
            elif idx == EOS_IDX:
                break
            elif idx == PAD_IDX:
                continue
            elif idx in IDX_TO_CHAR:
                text += IDX_TO_CHAR[idx]
        
        return text
    
    def predict_batch(
        self,
        landmarks_batch: List[np.ndarray],
    ) -> List[str]:
        """
        Predict on batch of sequences.
        
        Args:
            landmarks_batch: list of (T, D) arrays
        
        Returns:
            predictions: list of predicted texts
        """
        predictions = []
        
        for landmarks in tqdm(landmarks_batch, desc="Inferring"):
            pred = self.predict(landmarks)
            predictions.append(pred)
        
        return predictions


def evaluate_on_parquet(
    engine: InferenceEngine,
    parquet_dir: Path,
    metadata_csv: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate model on parquet landmark files.
    
    Args:
        engine: InferenceEngine instance
        parquet_dir: directory with .parquet files
        metadata_csv: CSV with sequence_id and phrase columns
        output_path: where to save predictions
    
    Returns:
        metrics: dict with CER, accuracy, etc.
    """
    metadata = pd.read_csv(metadata_csv)
    
    predictions = []
    ground_truths = []
    sequence_ids = []
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        seq_id = row['sequence_id']
        phrase = row['phrase']
        
        parquet_path = parquet_dir / f"{seq_id}.parquet"
        
        if not parquet_path.exists():
            continue
        
        try:
            df = pd.read_parquet(parquet_path)
            landmarks = df.values.astype(np.float32)
            
            pred = engine.predict(landmarks)
            
            predictions.append(pred)
            ground_truths.append(phrase)
            sequence_ids.append(seq_id)
        except Exception as e:
            print(f"Error processing {seq_id}: {e}")
            continue
    
    # Compute metrics
    cer = compute_cer(predictions, ground_truths)
    accuracy = sum(1 for p, g in zip(predictions, ground_truths) if p == g) / len(predictions)
    
    metrics = {
        "cer": cer,
        "accuracy": accuracy,
        "num_samples": len(predictions),
    }
    
    # Save predictions
    if output_path:
        results_df = pd.DataFrame({
            "sequence_id": sequence_ids,
            "ground_truth": ground_truths,
            "prediction": predictions,
            "correct": [p == g for p, g in zip(predictions, ground_truths)],
        })
        results_df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")
    
    return metrics


def export_for_deployment(
    checkpoint_path: Path = None,
    export_format: str = "torchscript",
) -> Path:
    """
    Export model for deployment (TorchScript, ONNX, etc.)
    
    Args:
        checkpoint_path: path to model checkpoint
        export_format: "torchscript" or "onnx"
    
    Returns:
        export_path: path to exported model
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    
    engine = InferenceEngine(checkpoint_path)
    model = engine.model
    model.eval()
    
    if export_format == "torchscript":
        # Create dummy input
        dummy_input = torch.randn(1, 130, 64).to(DEVICE)
        
        # Trace or script the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        export_path = checkpoint_path.parent / "model_torchscript.pt"
        torch.jit.save(traced_model, export_path)
        
        print(f"✓ Exported TorchScript model to {export_path}")
    
    elif export_format == "onnx":
        try:
            import onnx
            import onnxruntime as ort
            
            dummy_input = torch.randn(1, 130, 64).to(DEVICE)
            
            export_path = checkpoint_path.parent / "model.onnx"
            
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                input_names=["landmarks"],
                output_names=["char_logits", "ctc_logits"],
                opset_version=12,
            )
            
            print(f"✓ Exported ONNX model to {export_path}")
        except ImportError:
            print("⚠ ONNX export requires: pip install onnx onnxruntime")
            return None
    
    return export_path
