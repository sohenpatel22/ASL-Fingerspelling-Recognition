"""
Model architectures for ASL Fingerspelling Recognition.

This module contains:
- CNN-LSTM with Attention (primary model)
- Isolated Signs Classifier (for pre-training)
- Transformer-based alternative (fallback)
- Loss functions and decoding utilities

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np

from config import (
    CNN_LSTM_MODEL, TRANSFORMER_MODEL, ISOLATED_SIGNS_MODEL,
    DEVICE, CTC_CONFIG, LOSS_CONFIG, BEAM_CONFIG,
    START_IDX, EOS_IDX, PAD_IDX, VOCAB_SIZE=62
)

# ISOLATED SIGNS CLASSIFIER (Pre-training

class IsolatedSignsClassifier(nn.Module):
    """
    CNN-based classifier for isolated sign language (250 signs).
    
    Used as pre-training before transfer to fingerspelling recognition.
    Learns a hand pose feature encoder that generalizes across tasks.
    """
    
    def __init__(self, config: dict = ISOLATED_SIGNS_MODEL):
        super().__init__()
        
        self.input_dim = config["input_dim"]
        self.num_classes = config["num_classes"]
        self.conv_channels = config["conv_channels"]
        self.kernel_sizes = config["kernel_sizes"]
        self.dropout = config["dropout"]
        
        # 1D Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels, kernel_size in zip(self.conv_channels, self.kernel_sizes):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.MaxPool1d(2),
                )
            )
            in_channels = out_channels
        
        # Compute flattened dimension after conv layers
        # Input: (B, 1, input_dim) -> after 3 conv layers with maxpool: input_dim / 8
        self.flat_dim = self.conv_channels[-1] * (self.input_dim // 8)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, config["fc_dim"]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(config["fc_dim"], self.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) landmark features
        
        Returns:
            logits: (B, num_classes)
        """
        # Reshape to (B, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        logits = self.fc(x)
        
        return logits

# CNN-LSTM WITH ATTENTION (Primary Model

class AttentionLayer(nn.Module):
    """Attention mechanism for temporal sequences."""
    
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_dim, attention_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) sequence of frames
        
        Returns:
            attended: (B, T, D) attended features
            weights: (B, T) attention weights for interpretation
        """
        attended, _ = self.attention(x, x, x)
        return attended, _


class CNNEncoder(nn.Module):
    """Spatial CNN encoder for landmark features."""
    
    def __init__(self, config: dict = CNN_LSTM_MODEL):
        super().__init__()
        
        self.input_dim = config["input_dim"]
        cnn_cfg = config["cnn"]
        
        # 1D convolutions (apply across time dimension)
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_ch in cnn_cfg["out_channels"]:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_ch,
                        kernel_size=cnn_cfg["kernel_size"],
                        stride=cnn_cfg["stride"],
                        padding=cnn_cfg["padding"],
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(cnn_cfg["dropout"]),
                )
            )
            in_channels = out_ch
        
        # Output projection
        self.output_projection = nn.Linear(cnn_cfg["out_channels"][-1], cnn_cfg["output_dim"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T) landmark features
        
        Returns:
            encoded: (B, T, output_dim)
        """
        # x: (B, D, T)
        for conv in self.conv_layers:
            x = conv(x)
        
        # x: (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        
        x = self.output_projection(x)  # (B, T, output_dim)
        
        return x


class BiLSTMDecoder(nn.Module):
    """Bidirectional LSTM for temporal sequence modeling."""
    
    def __init__(self, config: dict = CNN_LSTM_MODEL):
        super().__init__()
        
        lstm_cfg = config["lstm"]
        
        self.lstm = nn.LSTM(
            input_size=config["cnn"]["output_dim"],
            hidden_size=lstm_cfg["hidden_dim"],
            num_layers=lstm_cfg["num_layers"],
            bidirectional=lstm_cfg["bidirectional"],
            dropout=lstm_cfg["dropout"],
            batch_first=True,
        )
        
        # Attention mechanism
        if config["attention"]["enabled"]:
            self.attention = AttentionLayer(
                lstm_cfg["hidden_dim"] * (2 if lstm_cfg["bidirectional"] else 1),
                config["attention"]["hidden_dim"],
            )
        else:
            self.attention = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            x: (B, T, input_dim)
        
        Returns:
            output: (B, T, hidden_dim*2 or hidden_dim)
            (h_n, c_n): final LSTM states
        """
        lstm_out, (h_n, c_n) = self.lstm(x)  # (B, T, hidden*2)
        
        if self.attention:
            lstm_out, _ = self.attention(lstm_out)
        
        return lstm_out, (h_n, c_n)


class CharacterDecoder(nn.Module):
    """Autoregressive character decoder with CTC output."""
    
    def __init__(self, config: dict = CNN_LSTM_MODEL):
        super().__init__()
        
        lstm_cfg = config["lstm"]
        dec_cfg = config["decoder"]
        
        hidden_dim = lstm_cfg["hidden_dim"] * (2 if lstm_cfg["bidirectional"] else 1)
        
        # Character embedding
        self.embedding = nn.Embedding(dec_cfg["vocab_size"], dec_cfg["embedding_dim"])
        
        # Decoder LSTM (for autoregressive generation)
        self.decoder_lstm = nn.LSTM(
            input_size=dec_cfg["embedding_dim"],
            hidden_size=dec_cfg["hidden_dim"],
            num_layers=dec_cfg["num_layers"],
            dropout=dec_cfg["dropout"],
            batch_first=True,
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(dec_cfg["hidden_dim"], dec_cfg["vocab_size"])
        
        # CTC head (frame-level character prediction)
        self.ctc_projection = nn.Linear(hidden_dim, dec_cfg["vocab_size"])
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_output: (B, T, hidden_dim*2) from BiLSTM
            target_tokens: (B, L) ground truth characters (for training)
            use_teacher_forcing: whether to use target_tokens for next input
        
        Returns:
            logits: (B*L, vocab_size) decoded character logits
            ctc_logits: (B, T, vocab_size) for CTC loss
        """
        B, T, hidden_dim = encoder_output.shape
        
        # CTC logits (frame-level predictions)
        ctc_logits = self.ctc_projection(encoder_output)  # (B, T, vocab)
        
        # Autoregressive decoding
        if target_tokens is None:
            # Inference mode (not implemented here, use beam search separately)
            return logits, ctc_logits
        
        # Training mode: teacher forcing
        batch_size, target_len = target_tokens.shape
        
        # Embed target tokens
        token_embeddings = self.embedding(target_tokens)  # (B, L, embed_dim)
        
        # Decode
        decoder_out, _ = self.decoder_lstm(token_embeddings)  # (B, L, hidden)
        logits = self.output_projection(decoder_out)  # (B, L, vocab)
        
        return logits, ctc_logits


class CNNLSTMFingerSpeller(nn.Module):
    """
    Complete CNN-LSTM model for ASL fingerspelling recognition.
    
    Architecture:
    1. CNN Encoder: Extract spatial features from landmarks
    2. BiLSTM Decoder: Model temporal dependencies
    3. Attention: Weight important frames
    4. Character Decoder: Generate character sequence
    
    Loss function: CTC + CrossEntropy hybrid
    """
    
    def __init__(self, config: dict = CNN_LSTM_MODEL):
        super().__init__()
        
        self.config = config
        
        # Encoder
        self.cnn_encoder = CNNEncoder(config)
        
        # Temporal decoder
        self.lstm_decoder = BiLSTMDecoder(config)
        
        # Character decoder
        self.char_decoder = CharacterDecoder(config)
    
    def forward(
        self,
        landmarks: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            landmarks: (B, D, T) normalized hand landmarks
            target_tokens: (B, L) ground truth character indices
            use_teacher_forcing: whether to use ground truth during training
        
        Returns:
            char_logits: (B, L, vocab_size) character predictions
            ctc_logits: (B, T, vocab_size) frame-level predictions
        """
        # CNN encoding
        encoder_out = self.cnn_encoder(landmarks)  # (B, T, output_dim)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm_decoder(encoder_out)  # (B, T, hidden*2)
        
        # Character decoder
        char_logits, ctc_logits = self.char_decoder(
            lstm_out,
            target_tokens=target_tokens,
            use_teacher_forcing=use_teacher_forcing,
        )
        
        return char_logits, ctc_logits
    
    def beam_search_decode(
        self,
        landmarks: torch.Tensor,
        beam_width: int = BEAM_CONFIG["width"],
        length_penalty: float = BEAM_CONFIG["length_penalty"],
        max_len: int = 34,
    ) -> List[List[int]]:
        """
        Beam search decoding for inference.
        
        Args:
            landmarks: (B, D, T)
            beam_width: number of hypotheses to keep
            length_penalty: penalty for long sequences
            max_len: maximum output length
        
        Returns:
            predictions: list of predicted character sequences
        """
        # Encode
        encoder_out = self.cnn_encoder(landmarks)
        lstm_out, _ = self.lstm_decoder(encoder_out)
        
        # CTC-based beam search (simplified)
        ctc_logits = self.char_decoder.ctc_projection(lstm_out)  # (B, T, vocab)
        
        batch_size = ctc_logits.size(0)
        predictions = []
        
        for b in range(batch_size):
            # Get best path per sequence
            logits = ctc_logits[b]  # (T, vocab)
            chars = torch.argmax(logits, dim=1)  # (T,)
            
            # Collapse repeated characters (CTC rule)
            collapsed = [chars[0].item()]
            for c in chars[1:]:
                if c.item() != collapsed[-1]:
                    collapsed.append(c.item())
            
            # Remove PAD tokens
            collapsed = [c for c in collapsed if c != PAD_IDX]
            predictions.append(collapsed)
        
        return predictions

# TRANSFORMER ALTERNATIVE (Fallback

class TransformerFingerSpeller(nn.Module):
    """
    Transformer-based model for fingerspelling (alternative to CNN-LSTM).
    
    Use if CNN-LSTM plateaus and more capacity is needed.
    """
    
    def __init__(self, config: dict = TRANSFORMER_MODEL):
        super().__init__()
        
        self.config = config
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config["max_seq_length"], config["d_model"])
        )
        
        # Input projection
        self.input_projection = nn.Linear(config["input_dim"], config["d_model"])
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config["num_encoder_layers"])
        
        # Character embedding for decoder
        self.char_embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config["num_decoder_layers"])
        
        # Output projection
        self.output_projection = nn.Linear(config["d_model"], config["vocab_size"])
    
    def forward(
        self,
        landmarks: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            landmarks: (B, D, T)
            target_tokens: (B, L) for training
        
        Returns:
            logits: (B, L, vocab_size)
        """
        # Reshape landmarks for transformer
        landmarks = landmarks.transpose(1, 2)  # (B, T, D)
        
        # Project to d_model
        x = self.input_projection(landmarks)  # (B, T, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Encode
        encoded = self.encoder(x)  # (B, T, d_model)
        
        if target_tokens is not None:
            # Decode with teacher forcing
            tgt_embed = self.char_embedding(target_tokens)  # (B, L, embed_dim)
            tgt_projected = F.linear(tgt_embed, self.output_projection.weight[:, :tgt_embed.size(-1)])
            
            decoded = self.decoder(tgt_projected, encoded)  # (B, L, d_model)
        else:
            # Greedy decoding (simplified)
            decoded = encoded
        
        logits = self.output_projection(decoded)  # (B, L/T, vocab)
        
        return logits

# LOSS FUNCTION

class HybridCTCCELoss(nn.Module):
    """
    Hybrid loss combining CTC and Cross-Entropy.
    
    CTC: Alignment-free, better for variable-length outputs
    CE: Supervised character-level loss
    
    Total: α * CTC_loss + (1-α) * CE_loss
    """
    
    def __init__(self, config: dict = LOSS_CONFIG, ctc_config: dict = CTC_CONFIG):
        super().__init__()
        
        self.config = config
        self.ctc_weight = config["ctc_weight"]
        self.ce_weight = config["ce_weight"]
        
        self.ctc_loss = nn.CTCLoss(
            blank=ctc_config["blank_idx"],
            reduction=ctc_config["reduction"],
            zero_infinity=ctc_config["zero_infinity"],
        )
        
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=config["label_smoothing"],
            ignore_index=PAD_IDX,
        )
    
    def forward(
        self,
        char_logits: torch.Tensor,
        ctc_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            char_logits: (B, L, vocab_size) from character decoder
            ctc_logits: (B, T, vocab_size) from CTC head
            target_tokens: (B, L) ground truth
            input_lengths: (B,) actual lengths of inputs
            target_lengths: (B,) actual lengths of targets
        
        Returns:
            total_loss: weighted combination
            ctc_loss_val: CTC loss component
            ce_loss_val: CE loss component
        """
        # CTC loss
        if self.config["use_ctc"]:
            # Prepare for CTC: (T, B, vocab)
            ctc_log_probs = F.log_softmax(ctc_logits, dim=2).transpose(0, 1)
            ctc_loss_val = self.ctc_loss(ctc_log_probs, target_tokens, input_lengths, target_lengths)
        else:
            ctc_loss_val = torch.tensor(0.0, device=char_logits.device)
        
        # Cross-entropy loss
        if self.config["use_ce"]:
            ce_loss_val = self.ce_loss(
                char_logits.view(-1, char_logits.size(-1)),
                target_tokens.view(-1),
            )
        else:
            ce_loss_val = torch.tensor(0.0, device=char_logits.device)
        
        # Combine
        total_loss = self.ctc_weight * ctc_loss_val + self.ce_weight * ce_loss_val
        
        return total_loss, ctc_loss_val, ce_loss_val
