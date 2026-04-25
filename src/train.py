"""
Training utilities and loops for ASL Fingerspelling Recognition.

This module handles:
- Training loops with scheduled sampling
- Validation and evaluation
- Checkpoint management
- Learning rate scheduling
- Curriculum learning coordination

"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import json

from config import (
    DEVICE, LEARNING_RATE, WARMUP_EPOCHS, WEIGHT_DECAY,
    MAX_GRAD_NORM, OPTIMIZER, BATCH_SIZE, TOTAL_EPOCHS,
    USE_AMP, CHECKPOINT_DIR, SCHEDULED_SAMPLING,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    SAVE_CHECKPOINT_EVERY_N_EPOCHS, SAVE_BEST_ONLY,
)
from models import CNNLSTMFingerSpeller, HybridCTCCELoss
from utils import CharacterErrorRate, compute_cer


class ScheduledSampler:
    """
    Scheduled sampling: gradually transition from teacher forcing to model predictions.
    
    Probability of using teacher forcing: decay from start_prob to end_prob over training.
    """
    
    def __init__(self, start_prob: float = 1.0, end_prob: float = 0.5, total_steps: int = 1000):
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_prob(self) -> float:
        """Get current teacher forcing probability."""
        progress = self.current_step / self.total_steps
        prob = self.start_prob - (self.start_prob - self.end_prob) * progress
        return np.clip(prob, self.end_prob, self.start_prob)
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1


class Trainer:
    """
    Unified trainer for ASL fingerspelling recognition.
    
    Handles training, validation, checkpointing, and curriculum learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        learning_rate: float = LEARNING_RATE,
        checkpoint_dir: Path = CHECKPOINT_DIR,
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Loss function
        self.criterion = HybridCTCCELoss()
        
        # Optimizer
        if OPTIMIZER == "adamw":
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            self.optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # LR Scheduler
        self.lr_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=WARMUP_EPOCHS * len(train_loader),
        )
        
        # AMP scaler
        self.scaler = GradScaler() if USE_AMP else None
        
        # Scheduled sampling
        if SCHEDULED_SAMPLING["enabled"]:
            self.scheduled_sampler = ScheduledSampler(
                start_prob=SCHEDULED_SAMPLING["start_prob"],
                end_prob=SCHEDULED_SAMPLING["end_prob"],
                total_steps=TOTAL_EPOCHS * len(train_loader),
            )
        else:
            self.scheduled_sampler = None
        
        # Metrics
        self.cer_metric = CharacterErrorRate()
        
        # Best model tracking
        self.best_val_cer = float("inf")
        self.patience_counter = 0
        
        # History
        self.history = {
            "train_loss": [],
            "train_cer": [],
            "val_loss": [],
            "val_cer": [],
            "learning_rate": [],
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            metrics: dict with loss and CER
        """
        self.model.train()
        total_loss = 0.0
        total_cer = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} - Train")
        
        for batch_idx, (landmarks, target_tokens) in enumerate(pbar):
            landmarks = landmarks.to(DEVICE)  # (B, D, T)
            target_tokens = target_tokens.to(DEVICE)  # (B, L)
            
            # Scheduled sampling: decide whether to use teacher forcing
            use_teacher_forcing = True
            if self.scheduled_sampler:
                prob = self.scheduled_sampler.get_prob()
                use_teacher_forcing = np.random.random() < prob
                self.scheduled_sampler.step()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if USE_AMP and self.scaler:
                with autocast():
                    char_logits, ctc_logits = self.model(
                        landmarks,
                        target_tokens=target_tokens if use_teacher_forcing else None,
                        use_teacher_forcing=use_teacher_forcing,
                    )
                    
                    # Compute loss
                    input_lengths = torch.full((landmarks.size(0),), landmarks.size(2), dtype=torch.long)
                    target_lengths = torch.sum(target_tokens != 46, dim=1)  # 46 = PAD_IDX
                    
                    loss, ctc_loss, ce_loss = self.criterion(
                        char_logits,
                        ctc_logits,
                        target_tokens,
                        input_lengths,
                        target_lengths,
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                char_logits, ctc_logits = self.model(
                    landmarks,
                    target_tokens=target_tokens if use_teacher_forcing else None,
                    use_teacher_forcing=use_teacher_forcing,
                )
                
                input_lengths = torch.full((landmarks.size(0),), landmarks.size(2), dtype=torch.long)
                target_lengths = torch.sum(target_tokens != 46, dim=1)
                
                loss, ctc_loss, ce_loss = self.criterion(
                    char_logits,
                    ctc_logits,
                    target_tokens,
                    input_lengths,
                    target_lengths,
                )
                
                loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.lr_scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": total_loss / num_batches})
        
        avg_loss = total_loss / num_batches
        
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            metrics: dict with loss and CER
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            
            for landmarks, target_tokens in pbar:
                landmarks = landmarks.to(DEVICE)
                target_tokens = target_tokens.to(DEVICE)
                
                char_logits, ctc_logits = self.model(
                    landmarks,
                    target_tokens=target_tokens,
                    use_teacher_forcing=True,
                )
                
                input_lengths = torch.full((landmarks.size(0),), landmarks.size(2), dtype=torch.long)
                target_lengths = torch.sum(target_tokens != 46, dim=1)
                
                loss, _, _ = self.criterion(
                    char_logits,
                    ctc_logits,
                    target_tokens,
                    input_lengths,
                    target_lengths,
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"val_loss": total_loss / num_batches})
        
        avg_loss = total_loss / num_batches
        
        return {"val_loss": avg_loss}
    
    def train(
        self,
        num_epochs: int = TOTAL_EPOCHS,
        curriculum_phases: Optional[Dict] = None,
    ):
        """
        Train model for multiple epochs with optional curriculum learning.
        
        Args:
            num_epochs: total number of epochs
            curriculum_phases: dict with phase configurations
        """
        print(f"\n{'='*60}")
        print(f"Starting training: {num_epochs} epochs on {DEVICE}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            # Print
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % SAVE_CHECKPOINT_EVERY_N_EPOCHS == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "metrics": {k: v[-1] for k, v in self.history.items()},
                    },
                    checkpoint_path,
                )
                print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            if val_metrics["val_loss"] < self.best_val_cer - EARLY_STOPPING_MIN_DELTA:
                self.best_val_cer = val_metrics["val_loss"]
                self.patience_counter = 0
                
                # Save best model
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "metrics": {k: v[-1] for k, v in self.history.items()},
                    },
                    best_path,
                )
                print(f"  ✓ New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\n✓ Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best model: {self.checkpoint_dir / 'best_model.pth'}")
        print(f"{'='*60}\n")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_history(self, output_path: Path = None):
        """Save training history."""
        if output_path is None:
            output_path = self.checkpoint_dir / "training_history.json"
        
        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Saved training history to {output_path}")
