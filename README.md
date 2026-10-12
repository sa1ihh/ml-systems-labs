# ML Systems Labs

Lab work for the Machine Learning Systems course.
Textbook: Vijay Janapa Reddi, *Introduction to Machine Learning Systems*.

## Labs

| Lab | Topic | Report |
|---|---|---|
| 01 | Environment and first system measurements | [lab01/report.md](lab01/report.md) |

## Repository layout

```
README.md
requirements.txt
.gitignore
lab01/ ... lab11/
    report.md     goal, method, results, conclusions
    src/          code
    results/      measured outputs (CSV, plots, logs)
project/
    README.md
    src/
    results/
    data_card.md
    model_card.md
```

Datasets are downloaded by the scripts into `data/` (git-ignored).

## Environment setup

Python 3.11.8 (also pinned in `.python-version`) in an isolated virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Note on the pinned versions

The pinned `mlflow==2.14.1` declares `pyarrow<16`, which conflicts with the pinned
`pyarrow==16.1.0`, so `pip install -r requirements.txt` fails with `ResolutionImpossible`.
To keep every pin exactly as specified, install in two steps:

```bash
grep -v '^mlflow' requirements.txt > req_no_mlflow.txt   # Windows: findstr /v "^mlflow" requirements.txt > req_no_mlflow.txt
pip install -r req_no_mlflow.txt
pip install --no-deps mlflow==2.14.1
pip install alembic "cachetools>=5,<6" cloudpickle docker entrypoints gitpython graphene \
    "importlib-metadata>=3.7,<8" markdown opentelemetry-api opentelemetry-sdk \
    querystring-parser sqlalchemy sqlparse waitress "packaging<25" "protobuf<5,>=3.20.2" \
    "pytz<2025" pyarrow==16.1.0
```

After this, `pip check` reports only the expected `pyarrow` warning, and
`import mlflow` works with pyarrow 16.1.0.

## Reproducibility

- Seeds: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
- Timings: one warm-up run, then at least 5 repeats; the median is reported.
- Each lab records Python and library versions in `labXX/results/versions.txt`.
