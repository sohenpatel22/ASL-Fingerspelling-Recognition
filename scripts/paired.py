import argparse
import json

import numpy as np


def by_signer(path):
    return json.load(open(path))["cer_by_signer"]


def main():
    parser = argparse.ArgumentParser(description="paired per-signer comparison of two details files")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    a, b = by_signer(args.first), by_signer(args.second)
    signers = sorted(set(a) & set(b))
    diff = np.array([a[s] - b[s] for s in signers])
    rng = np.random.default_rng(0)
    means = rng.choice(diff, size=(2000, diff.size), replace=True).mean(axis=1)
    result = {
        "mean_diff": round(float(diff.mean()), 4),
        "ci95": [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)],
        "signers_better": int((diff < 0).sum()),
        "signers": len(signers),
    }
    print("negative = first is better")
    print(json.dumps(result))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
