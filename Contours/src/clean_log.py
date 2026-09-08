"""
Turn a messy copy-pasted terminal log into a clean, sorted CSV.

Usage:
    python clean_log.py train.log train_metrics.csv
"""

import re
import sys

PATTERN = re.compile(
    r"Epoch (\d+) Complete \| Average Loss: ([\d.]+) \| Mean edge probability: ([\d.]+)"
)


def clean(log_path, csv_path):
    with open(log_path, "r", errors="ignore") as f:
        text = f.read()

    # A dict keyed by epoch number does the deduplicating for us: writing the same
    # key twice just overwrites it, so the last copy of each epoch is what survives.
    metrics = {}
    for match in PATTERN.finditer(text):
        epoch = int(match.group(1))
        metrics[epoch] = (float(match.group(2)), float(match.group(3)))

    if not metrics:
        print("No epoch lines matched. Check that the log format is what you expect.")
        return

    with open(csv_path, "w") as f:
        f.write("epoch,loss,edge_prob\n")
        for epoch in sorted(metrics):
            loss, edge_prob = metrics[epoch]
            f.write(f"{epoch},{loss},{edge_prob}\n")

    epochs = sorted(metrics)
    missing = [e for e in range(1, epochs[-1] + 1) if e not in metrics]

    print(f"Wrote {len(metrics)} unique epochs to {csv_path} "
          f"(range {epochs[0]} to {epochs[-1]})")
    if missing:
        print(f"WARNING: no lines found for these epochs: {missing}")


if __name__ == "__main__":
    clean(sys.argv[1], sys.argv[2])