"""Memory of a minimal inference process (what a real deployment would run).

Started by measure_cost.py as a fresh Python process. It imports only what is
needed to serve the model (no torch, no training code), loads the saved model,
runs single-sample predictions and prints its memory use as JSON.

Usage:
    python deploy_probe.py <model.joblib> <comma-separated feature values>
"""
import json
import sys

import joblib
import numpy as np
import psutil

INFER_REPEATS = 100


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2**20


def peak_rss_mb() -> float:
    """Peak RSS of this process since it started."""
    if sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB


def main():
    model_path, values = sys.argv[1], sys.argv[2]
    x_one = np.array([float(v) for v in values.split(",")]).reshape(1, -1)

    before_load = rss_mb()
    model = joblib.load(model_path)
    after_load = rss_mb()
    for _ in range(INFER_REPEATS):
        model.predict(x_one)

    print(json.dumps({
        "rss_before_load_mb": before_load,
        "rss_after_load_mb": after_load,
        "peak_rss_mb": peak_rss_mb(),
    }))


if __name__ == "__main__":
    main()
