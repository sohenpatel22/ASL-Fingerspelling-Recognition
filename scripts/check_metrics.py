import json
import operator
import re
import sys

OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}


def main() -> int:
    # usage: check_metrics.py metrics.json "cer<=0.35" "exact_match>=0.5"
    path, *rules = sys.argv[1:]
    metrics = json.load(open(path))
    failed = False
    for rule in rules:
        key, op, limit = re.match(r"(\w+)(<=|>=|<|>)([\d.]+)", rule).groups()
        value = metrics[key]
        ok = OPS[op](value, float(limit))
        print(f"{'ok  ' if ok else 'FAIL'} {key}={value:.4f} (want {op} {limit})")
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
