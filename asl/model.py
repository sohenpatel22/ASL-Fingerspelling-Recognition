from __future__ import annotations

import math

import torch
import torch.nn as nn

from asl.config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, maxlen: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(maxlen, d_model)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


def _feed_forward(d_model: int, ffn_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, ffn_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(ffn_dim, d_model),
        nn.Dropout(dropout),
    )


class ConformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.ff1 = _feed_forward(d_model, ffn_dim, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )
        self.ff2 = _feed_forward(d_model, ffn_dim, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.norm_attn(x)
        r, _ = self.attn(h, h, h)
        x = x + self.drop_attn(r)
        x = x + self.conv(self.norm_conv(x).transpose(1, 2)).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(cfg.input_dim, cfg.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.d_model),
            nn.ReLU(),
        )
        self.posenc = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            ConformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.conv_kernel, cfg.dropout)
            for _ in range(cfg.enc_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.posenc(self.proj(x).permute(0, 2, 1))
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=pad_idx)
        self.proj_emb = nn.Linear(cfg.embed_dim, cfg.d_model)
        self.posenc = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.dec_layers)
        self.fc_out = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        emb = self.posenc(self.proj_emb(self.embed(tgt)))
        out = self.decoder(
            emb, memory, tgt_mask=mask, tgt_key_padding_mask=(tgt == self.pad_idx)
        )
        return self.fc_out(out)


class ASLConformerSeq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, pad_idx: int):
        super().__init__()
        self.encoder = ConformerEncoder(cfg)
        self.decoder = TransformerDecoder(cfg, vocab_size, pad_idx)
        self.ctc_head = nn.Linear(cfg.d_model, vocab_size) if cfg.ctc else None

    def forward(self, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.decoder(tgt, self.encoder(x))

    def forward_joint(self, x: torch.Tensor, tgt: torch.Tensor):
        # one encoder pass shared by the attention decoder and the CTC head
        memory = self.encoder(x)
        ctc_logits = self.ctc_head(memory) if self.ctc_head is not None else None
        return self.decoder(tgt, memory), ctc_logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
