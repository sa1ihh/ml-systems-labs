# Lab 1: Environment and first system measurements

## 1. Goal

Set up a reproducible Python environment, train two baseline classifiers on the
Breast Cancer Wisconsin (Diagnostic) dataset, and measure what each model costs as
a system: training time, single-sample inference latency, model size and memory.
Use these measurements to decide on which deployment targets (Cloud, Edge, Mobile,
TinyML) each model could run.

## 2. Method

**Environment.** Python 3.11.8 in an isolated virtual environment with all
packages pinned as in the lab manual. The exact versions are recorded in
`results/versions.txt` by `src/print_versions.py`. The pinned `mlflow==2.14.1`
requires `pyarrow<16`, which conflicts with the pinned `pyarrow==16.1.0`; mlflow was
therefore installed without dependency resolution so that every pin stays exact
(details in the repository README). Hardware: Intel 64-bit desktop CPU
(Family 6 Model 186), Windows 11. No GPU was used.

**Data and models.** 569 samples, 30 numeric features, binary target
(malignant / benign). Stratified 70/30 split with `random_state=42`
(398 train / 171 test; class ratio 37% / 63% in both parts). All seeds
(`random`, `numpy`, `torch`) set to 42.
- Linear model: `LogisticRegression(max_iter=1000, random_state=42)`
- Tree ensemble: `RandomForestClassifier(n_estimators=100, random_state=42)`

The features were used unscaled, exactly as specified.

**Measurements** (`src/measure_cost.py`):
- Training time: one warm-up fit, then the median of 5 timed fits.
- Inference latency: one sample, one warm-up call, then the median of 100 timed calls.
- Model size: `joblib.dump` to `results/model.joblib`; bytes and KB recorded.
  Because both models are dumped to the same file in turn, a per-model copy is
  kept as `results/model_<name>.joblib`.
- Peak memory, two views:
  1. Peak RSS of the measurement process during training and inference, sampled
     with `memory_profiler` (1 ms interval). This process also imports torch for
     seeding, so it overstates what serving the model needs.
  2. Peak RSS of a fresh, inference-only Python process (`src/deploy_probe.py`)
     that imports only numpy, joblib and psutil, loads the saved model and runs
     100 predictions. Median of 5 processes. This is the memory figure used for
     the deployment decision.

**Deployment fit** (`src/budget_fit.py`). Each measurement is compared with the
budgets from the manual. For memory, the lower bound of a target's range is the
RAM that every device in that class offers: fitting it means "yes", fitting only
the upper bound means "conditional" (larger devices only), exceeding the upper
bound means "no". Memory is judged twice: as a Python process, and for the model
alone (serialized size), which is what remains if the model is ported out of
Python, for example to C on a microcontroller.

## 3. Results

### Accuracy (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| Logistic Regression | 0.9474 |
| Random Forest | 0.9357 |

### System cost (`results/system_cost.csv`)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Training time, median of 5 (ms) | 234.2 | 149.4 |
| Inference latency, 1 sample, median of 100 (ms) | 0.031 | 2.700 |
| Model size (bytes) | 1,055 | 290,889 |
| Model size (KB) | 1.03 | 284.07 |
| Peak RSS, measurement process (MB) | 229.2 | 230.8 |
| Peak RSS, inference-only process (MB) | 92.4 | 99.5 |
| of which: loading the model and scikit-learn (MB) | 52.0 | 58.9 |

Imports alone explain most of the memory: an empty Python process uses about
19 MB, adding numpy and scikit-learn raises it to about 98 MB, and adding torch
to about 225 MB.

### Deployment fit (`results/deployment_fit.md`)

Budgets: Cloud (memory ≥ 1 GB, latency ≤ 100 ms, size ≤ 500 MB),
Edge (256–1024 MB, ≤ 50 ms, ≤ 50 MB), Mobile (64–256 MB, ≤ 20 ms, ≤ 10 MB),
TinyML (≤ 256 KB, ≤ 10 ms, ≤ 100 KB).

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | yes / yes | yes / yes | conditional / yes | no / yes |
| Random Forest | yes / yes | yes / yes | conditional / yes | no / no |

Cell format: fit as a Python process / fit if the model is ported out of Python.

- **Latency** is never the limiting factor: the slowest result (Random Forest,
  2.7 ms) is 3.7 times below the tightest budget (TinyML, 10 ms).
- **Mobile** is conditional for both models because the inference process
  (92–100 MB) exceeds the 64 MB lower bound but fits the 256 MB upper bound.
- **TinyML**: no Python process fits in 256 KB. Logistic Regression itself is only
  1 KB (30 weights and a bias), so it fits if ported. Random Forest does not fit
  even when ported, because its 284 KB model exceeds the 100 KB size budget.

### Notes and limitations

- Logistic Regression raised a `ConvergenceWarning`: with unscaled features, the
  solver used all 1000 iterations without fully converging. This also explains
  why its training time is higher than the Random Forest's.
- Timings vary between runs on the same machine. Across three runs, the Random
  Forest median training time ranged from 89 to 149 ms and its median latency
  from 1.25 to 2.70 ms. The tables report the final run. All deployment
  decisions hold for every run.
- Latency was measured on a desktop CPU. A microcontroller is much slower, so
  the TinyML latency check is optimistic.
- The memory increase caused by a single fit was too small to measure reliably
  (under 1.1 MB), because the Python allocator reuses memory from earlier fits.
- Re-running the measurements reproduced byte-identical model files, which
  confirms that the seeds make training deterministic.

## 4. Conclusions

1. **A more complex model did not buy better accuracy.** The Random Forest reached
   0.9357 test accuracy against 0.9474 for Logistic Regression, while being about
   90 times slower per prediction (2.70 ms vs 0.031 ms) and about 275 times larger
   (284 KB vs 1 KB). On a small, well-structured dataset like this one, the extra
   cost of an ensemble is not justified, so a simple baseline should always be
   measured first before choosing a heavier model.

2. **The bottleneck for TinyML is the Python environment, not the model.** The
   Logistic Regression model is only 1 KB (30 weights and a bias), but the smallest
   Python process that can serve it needs 92 MB, because an empty interpreter already
   takes about 19 MB and loading numpy and scikit-learn adds most of the rest. To run
   on a microcontroller with 256 KB of memory, the model would have to be exported
   out of Python, for example as a few lines of C code, rather than made smaller.

3. **For a small device such as a smartwatch or a sensor, Logistic Regression is
   the right choice.** It is the only one of the two models that can fit the TinyML
   budget once ported (1 KB against the 100 KB limit), while the Random Forest
   (284 KB) cannot fit even then. It is also the more accurate model here, and
   because it is just a weighted sum of the inputs, it is simple to implement and
   predictable to run on constrained hardware. Choosing it costs nothing in accuracy.
