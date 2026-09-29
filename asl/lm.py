from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

BOS = "\x02"
EOS = "\x03"


class CharNgramLM:
    # interpolated Witten-Bell smoothing over characters, with an end-of-phrase symbol
    def __init__(self, order: int = 5, vocab_size: int = 60):
        self.order = order
        self.vocab_size = vocab_size  # only used for the uniform base distribution
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def fit(self, phrases: Iterable[str]) -> CharNgramLM:
        for phrase in phrases:
            text = BOS * (self.order - 1) + phrase + EOS
            for i in range(self.order - 1, len(text)):
                nxt = text[i]
                for n in range(self.order):  # contexts of length 0 .. order-1
                    ctx = text[i - n : i]
                    self.counts[ctx][nxt] += 1
        return self

    def logprob(self, context: str, ch: str) -> float:
        history = (BOS * (self.order - 1) + context)[-(self.order - 1) :] if self.order > 1 else ""
        return math.log(self._prob(history, ch))

    def _prob(self, history: str, ch: str) -> float:
        if not history:
            table = self.counts.get("")
            base = 1.0 / self.vocab_size
        else:
            table = self.counts.get(history)
            base = self._prob(history[1:], ch)
        if not table:
            return base
        total, distinct = sum(table.values()), len(table)
        return (table.get(ch, 0) + distinct * base) / (total + distinct)

    def score(self, text: str) -> float:
        # total log probability of a phrase including the end symbol
        return sum(self.logprob(text[:i], c) for i, c in enumerate(text)) + self.logprob(text, EOS)

    def save(self, path: str | Path) -> None:
        payload = {"order": self.order, "vocab_size": self.vocab_size, "counts": self.counts}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> CharNgramLM:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        lm = cls(payload["order"], payload["vocab_size"])
        for ctx, table in payload["counts"].items():
            lm.counts[ctx] = defaultdict(int, table)
        return lm


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    from asl.config import load_config
    from asl.data import participant_split

    parser = argparse.ArgumentParser(description="fit a character n-gram LM on the training split phrases")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="monitoring/char_lm.json.gz")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    d = cfg.data
    comp = Path(d.comp_dir)
    meta = pd.read_csv(comp / "train.csv")
    train, _, _ = participant_split(meta, d.val_fraction, d.test_fraction, d.split_seed)
    phrases = list(train["phrase"].astype(str))
    if d.use_supplemental:
        phrases += list(pd.read_csv(comp / "supplemental_metadata.csv")["phrase"].astype(str))
    lm = CharNgramLM(args.order).fit(phrases)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lm.save(args.out)
    print(f"fitted order-{args.order} LM on {len(phrases)} phrases -> {args.out}")


if __name__ == "__main__":
    main()
