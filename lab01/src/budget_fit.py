"""Lab 1, step 4: compare the measured costs with the four deployment budgets.

Memory rule (conservative): a target's memory range gives the RAM a device of
that class can offer. If the requirement fits the LOWER bound, every device in
the class can run it ("yes"); if it only fits the UPPER bound, only the larger
devices can ("conditional"); above the upper bound it does not fit ("no").

Memory is judged twice:
  - process: peak RSS of the minimal Python inference process (deploy_peak_rss_mb)
  - model only: the model's own footprint (serialized size), i.e. what remains
    if the model were ported out of Python (e.g. to C on a microcontroller)

Latency was measured on a desktop CPU, so it is a lower bound for slower devices.

Usage (from repo root):
    python lab01/src/budget_fit.py
"""
import csv

from common import RESULTS_DIR

KB_PER_MB = 1024

# Budgets from the lab manual. Memory in MB as (lower, upper); None = unbounded.
BUDGETS = {
    "Cloud":  {"mem_mb": (1024, None),        "latency_ms": 100, "size_kb": 500 * KB_PER_MB},
    "Edge":   {"mem_mb": (256, 1024),         "latency_ms": 50,  "size_kb": 50 * KB_PER_MB},
    "Mobile": {"mem_mb": (64, 256),           "latency_ms": 20,  "size_kb": 10 * KB_PER_MB},
    "TinyML": {"mem_mb": (None, 256 / KB_PER_MB), "latency_ms": 10, "size_kb": 100},
}


def memory_verdict(need_mb, bounds):
    lower, upper = bounds
    if lower is not None and need_mb <= lower:
        return "yes"
    if upper is None or need_mb <= upper:
        return "yes" if lower is None else "conditional"
    return "no"


def combine(*verdicts):
    if "no" in verdicts:
        return "no"
    return "conditional" if "conditional" in verdicts else "yes"


def main():
    with (RESULTS_DIR / "system_cost.csv").open() as f:
        costs = list(csv.DictReader(f))

    rows = []
    for c in costs:
        size_kb = float(c["model_size_kb"])
        latency = float(c["infer_latency_median_ms"])
        process_mb = float(c["deploy_peak_rss_mb"])
        model_mb = size_kb / KB_PER_MB
        for target, b in BUDGETS.items():
            size_ok = "yes" if size_kb <= b["size_kb"] else "no"
            latency_ok = "yes" if latency <= b["latency_ms"] else "no"
            process_fit = memory_verdict(process_mb, b["mem_mb"])
            model_fit = memory_verdict(model_mb, b["mem_mb"])
            rows.append({
                "model": c["model"],
                "target": target,
                "size_kb": size_kb,
                "size_limit_kb": b["size_kb"],
                "size_ok": size_ok,
                "latency_ms": latency,
                "latency_limit_ms": b["latency_ms"],
                "latency_ok": latency_ok,
                "process_mem_mb": process_mb,
                "process_mem_fit": process_fit,
                "model_mem_kb": size_kb,
                "model_mem_fit": model_fit,
                "fit_as_python_process": combine(size_ok, latency_ok, process_fit),
                "fit_if_ported": combine(size_ok, latency_ok, model_fit),
            })

    with (RESULTS_DIR / "deployment_fit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Compact matrix for the report.
    targets = list(BUDGETS)
    lines = ["| Model | " + " | ".join(targets) + " |",
             "|---|" + "---|" * len(targets)]
    for c in costs:
        cells = []
        for t in targets:
            r = next(r for r in rows if r["model"] == c["model"] and r["target"] == t)
            cells.append(f'{r["fit_as_python_process"]} / {r["fit_if_ported"]}')
        lines.append(f'| {c["model"]} | ' + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cell format: fit as a Python process / fit if the model is ported out of Python.")
    table = "\n".join(lines)
    (RESULTS_DIR / "deployment_fit.md").write_text(table + "\n")

    for r in rows:
        print(f'{r["model"]:20s} {r["target"]:7s} size={r["size_ok"]:3s} '
              f'latency={r["latency_ok"]:3s} process_mem={r["process_mem_fit"]:11s} '
              f'model_mem={r["model_mem_fit"]:11s} -> {r["fit_as_python_process"]} / {r["fit_if_ported"]}')
    print()
    print(table)
    print("\nSaved lab01/results/deployment_fit.csv and lab01/results/deployment_fit.md")


if __name__ == "__main__":
    main()
