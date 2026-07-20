# MLOps Pipeline — Modular ML Lifecycle with CI

A machine learning pipeline built as a set of independent, reusable stages, with
model persistence and continuous integration on every push.

The focus of this project is **pipeline engineering and automation**, not the
modelling task itself: the dataset is deliberately small so that the structure,
reproducibility and CI remain the subject.

---

## Pipeline stages

Each stage is a standalone function in `function.py` and can be run independently:

| Stage | Function | Description |
|---|---|---|
| 1. Prepare | `prepare_data()` | Load dataset, stratified train/test split, standardize features |
| 2. Train | `train_model(name, X, y)` | Fit a classifier — `rf`, `ada` or `xgb` |
| 3. Evaluate | `evaluate_model(model, X, y)` | Accuracy, confusion matrix, classification report |
| 4. Save | `save_model(model, scaler)` | Persist model and scaler with Joblib |
| 5. Load | `load_model()` | Restore a saved model and scaler |

Separating the stages means the scaler fitted at preparation time is saved and
reused at inference time, which avoids train/serve skew.

## Models

Three classifiers are interchangeable behind the same interface:

- **Random Forest** (`rf`)
- **AdaBoost** (`ada`)
- **XGBoost** (`xgb`)

## Dataset

Iris (built into scikit-learn) — 150 samples, 4 features, 3 balanced classes.
It requires no download and keeps CI runs fast, which is why it was chosen for a
project about pipeline mechanics.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Interactive step-by-step CLI:

```bash
python main.py
```

It prompts for the stage to run (1–5), and for the model to train at stage 2.

Run the whole pipeline non-interactively:

```bash
python smoke_test.py
```

---

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. Set up Python 3.11 with pip caching
2. Install dependencies
3. Execute `smoke_test.py`

The smoke test runs the full chain — prepare, train all three models, evaluate,
save, reload — and asserts that the reloaded model reproduces the predictions of
the original. CI therefore fails if any stage or the persistence round-trip breaks.

`main.py` is not used in CI because it is interactive and would block waiting on
standard input.

---

## Project structure

```
.
├── .github/workflows/ci.yml   # CI pipeline
├── function.py                # Pipeline stages
├── main.py                    # Interactive CLI
├── smoke_test.py              # Non-interactive end-to-end run (used by CI)
└── requirements.txt
```
