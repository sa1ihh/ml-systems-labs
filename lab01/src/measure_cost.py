"""Lab 1, step 3: measure the system cost of both baselines.

For each model:
  - training time: 1 warm-up fit, then median of 5 timed fits
  - single-sample inference latency: 1 warm-up, then median of 100 timed predictions
  - model size: joblib.dump to results/, size in bytes and KB
  - peak memory (RSS) during training and during inference, via memory_profiler

Usage (from repo root):
    python lab01/src/measure_cost.py
"""
import csv
import json
import statistics
import time

import joblib
import psutil
from memory_profiler import memory_usage

from common import RESULTS_DIR, load_split, make_models, set_seeds

TRAIN_REPEATS = 5
INFER_REPEATS = 100
MEM_INTERVAL_S = 0.001  # memory_profiler sampling interval


def rss_mb() -> float:
    """Current resident set size of this process in MB (MiB)."""
    return psutil.Process().memory_info().rss / 2**20


def time_training(name, X_train, y_train):
    make_models()[name].fit(X_train, y_train)  # warm-up, not recorded
    times = []
    for _ in range(TRAIN_REPEATS):
        model = make_models()[name]
        start = time.perf_counter()
        model.fit(X_train, y_train)
        times.append(time.perf_counter() - start)
    return model, times


def time_inference(model, x_one):
    model.predict(x_one)  # warm-up, not recorded
    times = []
    for _ in range(INFER_REPEATS):
        start = time.perf_counter()
        model.predict(x_one)
        times.append(time.perf_counter() - start)
    return times


def peak_memory(func, *args):
    """Run func(*args) while sampling RSS; return (baseline MB, peak MB)."""
    baseline = rss_mb()
    peak = memory_usage((func, args), interval=MEM_INTERVAL_S, max_usage=True)
    return baseline, peak


def predict_many(model, x_one):
    for _ in range(INFER_REPEATS):
        model.predict(x_one)


def main():
    set_seeds()
    X_train, X_test, y_train, _ = load_split()
    x_one = X_test[:1]  # a single sample, shape (1, 30)

    rows, raw = [], {}
    for name in make_models():
        set_seeds()
        model, train_times = time_training(name, X_train, y_train)
        infer_times = time_inference(model, x_one)

        model_path = RESULTS_DIR / f"model_{name}.joblib"
        joblib.dump(model, model_path)
        size_bytes = model_path.stat().st_size

        train_base, train_peak = peak_memory(make_models()[name].fit, X_train, y_train)
        infer_base, infer_peak = peak_memory(predict_many, model, x_one)

        row = {
            "model": name,
            "train_time_median_ms": round(statistics.median(train_times) * 1e3, 3),
            "infer_latency_median_ms": round(statistics.median(infer_times) * 1e3, 4),
            "model_size_bytes": size_bytes,
            "model_size_kb": round(size_bytes / 1024, 2),
            "process_rss_baseline_mb": round(train_base, 1),
            "peak_rss_train_mb": round(train_peak, 1),
            "peak_rss_infer_mb": round(infer_peak, 1),
            "train_mem_increase_mb": round(train_peak - train_base, 2),
            "infer_mem_increase_mb": round(infer_peak - infer_base, 2),
        }
        rows.append(row)
        raw[name] = {"train_times_s": train_times, "infer_times_s": infer_times}
        for key, value in row.items():
            print(f"  {key:26s} {value}")
        print()

    with (RESULTS_DIR / "system_cost.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "timings_raw.json").write_text(json.dumps(raw, indent=2))
    print("Saved lab01/results/system_cost.csv and lab01/results/timings_raw.json")


if __name__ == "__main__":
    main()
