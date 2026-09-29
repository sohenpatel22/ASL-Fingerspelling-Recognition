from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

_CHARS = " !#$%&'()*+,-./0123456789:;=?@[_abcdefghijklmnopqrstuvwxyz~"


def default_char_to_idx() -> dict[str, int]:
    return {c: i for i, c in enumerate(_CHARS)}


class Vocab:
    def __init__(self, char_to_idx: dict[str, int] | None = None):
        self.char_to_idx = dict(char_to_idx) if char_to_idx else default_char_to_idx()
        self.idx_to_char = {int(i): c for c, i in self.char_to_idx.items()}
        self.n_classes = len(self.char_to_idx)
        self.start_idx = self.n_classes
        self.eos_idx = self.n_classes + 1
        self.pad_idx = self.n_classes + 2
        self.vocab_size = self.n_classes + 3

    @classmethod
    def from_json(cls, path: str | Path) -> Vocab:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def encode(self, phrase: str, max_len: int) -> list[int]:
        chars = [self.char_to_idx.get(c, self.pad_idx) for c in phrase[: max_len - 2]]
        seq = [self.start_idx, *chars, self.eos_idx]
        seq += [self.pad_idx] * (max_len - len(seq))
        return seq[:max_len]

    def decode(self, indices: Iterable[int]) -> str:
        out = []
        for i in indices:
            i = int(i)
            if i == self.eos_idx:
                break
            if i in (self.pad_idx, self.start_idx):
                continue
            if i in self.idx_to_char:
                out.append(self.idx_to_char[i])
        return "".join(out)
