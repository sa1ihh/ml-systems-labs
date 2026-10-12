"""Lab 1, step 2: train the two baselines and record test accuracy.

Usage (from repo root):
    python lab01/src/train_baselines.py
"""
import csv

import numpy as np
from sklearn.metrics import accuracy_score

from common import RESULTS_DIR, load_split, make_models, set_seeds


def main():
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples, "
          f"Features: {X_train.shape[1]}")
    print(f"Malignant/benign ratio  train: {np.bincount(y_train)}  test: {np.bincount(y_test)}")

    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})
        print(f"{name:20s} test accuracy = {acc:.4f}")

    out = RESULTS_DIR / "baseline_accuracy.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out.relative_to(RESULTS_DIR.parents[1])}")


if __name__ == "__main__":
    main()
