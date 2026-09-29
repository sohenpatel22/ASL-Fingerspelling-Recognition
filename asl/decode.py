from __future__ import annotations

import torch
import torch.nn.functional as F

from asl.model import ASLConformerSeq2Seq
from asl.vocab import Vocab


@torch.no_grad()
def greedy_decode(
    model: ASLConformerSeq2Seq, x: torch.Tensor, vocab: Vocab, max_len: int
) -> list[list[int]]:
    memory = model.encoder(x)
    batch = x.size(0)
    tgt = torch.full((batch, 1), vocab.start_idx, dtype=torch.long, device=x.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=x.device)
    for _ in range(max_len - 1):
        nxt = model.decoder(tgt, memory)[:, -1].argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, vocab.pad_idx), nxt)
        tgt = torch.cat([tgt, nxt.unsqueeze(1)], dim=1)
        finished |= nxt == vocab.eos_idx
        if bool(finished.all()):
            break
    return tgt[:, 1:].tolist()


@torch.no_grad()
def beam_search(
    model: ASLConformerSeq2Seq,
    x: torch.Tensor,
    vocab: Vocab,
    beam_width: int = 5,
    max_len: int = 34,
    length_penalty: float = 0.6,
) -> tuple[list[int], float]:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    device = x.device
    memory = model.encoder(x)
    beams: list[tuple[list[int], float]] = [([vocab.start_idx], 0.0)]

    def rank(b: tuple[list[int], float]) -> float:
        return b[1] / max(len(b[0]) - 1, 1) ** length_penalty

    for _ in range(max_len - 1):
        candidates: list[tuple[list[int], float]] = []
        for seq, score in beams:
            if seq[-1] == vocab.eos_idx:
                candidates.append((seq, score))
                continue
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            logp = F.log_softmax(model.decoder(tgt, memory)[0, -1], dim=-1)
            top = logp.topk(beam_width)
            for lp, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                candidates.append((seq + [idx], score + lp))
        beams = sorted(candidates, key=rank, reverse=True)[:beam_width]
        if all(b[0][-1] == vocab.eos_idx for b in beams):
            break
    return beams[0]
