# ImBa — AutoML for Imbalanced Binary Classification

> **ImBa** is a lightweight AutoML framework for **binary classification with imbalanced classes**.
> It searches over **end-to-end pipelines** (preprocessing → imbalance handling → model → optional ensembling) and optimizes **F1-score** (optionally with **threshold tuning**).

![ImBa logo](logo.png)

---

## Why ImBa

Imbalanced datasets break a lot of “default” ML habits. ImBa is built around that reality:

- **Imbalance-aware search**: combine **oversampling**, **undersampling**, and **class weights** (use each separately or together).
- **Optuna optimization**: Bayesian/TPE-style search over pipeline choices and model parameters.
- **Threshold tuning for F1**: optionally selects the best decision threshold in CV to maximize F1.
- **Ensembling at the end**: blend top-N candidates (soft voting) and optionally **stack** models.
- **Reproducible experiments**: saves all trials/metrics to CSV and plots (PNG), plus the best model and a final report.

---

## Installation

Create an environment and install requirements:

```bash
pip install -U pip
pip install -r requirements.txt
```

Core dependencies:
- `scikit-learn`, `pandas`, `numpy`, `joblib`
- `optuna`
- `imblearn` (imbalanced-learn)

Optional (auto-detected):
- `xgboost`, `lightgbm`, `catboost`
- `category_encoders` (for target encoding)
- `shap` (for SHAP explanations)
- `mlflow` (for experiment tracking)
- `graphviz` + Python package `graphviz` (DOT → PNG/SVG)
- `networkx`, `matplotlib` (for lightweight PNG visualizations)

---

## Quickstart (Python API)

```python
import pandas as pd
from imba.core import AutoMLBinary, AutoMLConfig
from imba.evaluate import EvalConfig

df = pd.read_csv("train.csv")
y = df.pop("target")

automl = AutoMLBinary(
    AutoMLConfig(
        budget=60,
        out_dir="automl_outputs",
        random_state=42,
        # ensemble_top_n=3,
        # use_stacking=False,
    ),
    eval_cfg=EvalConfig(
        n_splits=5,
        scoring_primary="f1",
        # tune_threshold=True,
        # threshold_min=0.05, threshold_max=0.95, threshold_steps=19,
        # stratified=True,
    ),
)

automl.fit(df, y)

pred = automl.predict(df)
proba = automl.predict_proba(df)
```

---

## CLI usage

Run AutoML on a CSV file:

```bash
python run_imba.py --input path/to/data.csv --target target_column --budget 60 --cv 5 --out automl_outputs
```

(Your `run_imba.py` may also include visualization options; see `--help`.)

---

## What ImBa searches over

### 1) Preprocessing
- **Categorical encoding**: `onehot`, `ordinal`, `target` (needs `category_encoders`)
- **Numeric imputation**: `median`, `mean`, `min` (min-imputer)
- **Scaling**: `none`, `standard`, `robust`, `minmax`, `power`
- Optional skew correction: `skew_auto=True` (PowerTransformer on numeric columns)

### 2) Dimensionality reduction
- `none` or `pca` (with `dr_ratio`)

### 3) Imbalance handling (combinable)
- `oversample`: RandomOverSampler
- `undersample`: RandomUnderSampler
- `class_weight` / `pos_weight` where supported

### 4) Models (auto-detected)
- `logreg`, `rf`, `gb`
- optional: `xgb`, `lgbm`, `cat`

### 5) Optional ensembling
- **Blending**: soft voting of top-N pipelines
- **Stacking**: meta-model on top of top-N pipelines (optional)

---

## Outputs (artifacts)

In `out_dir` ImBa saves key artifacts:

- `best_model.joblib` — the best fitted pipeline (and threshold wrapper if enabled)
- `label_mapping.json` — mapping of original labels to (0, 1) and back
- `best_summary.json` — best config + score
- `final_report.json` — aggregated CV report (precision/recall/F1/ROC-AUC/AP/etc.)
- `metrics_vs_iteration.csv` — metrics per trial/iteration
- `metrics_vs_iteration.png` — learning curve of trial scores
- `decision_graph.dot` — best pipeline decision graph in DOT format
- `decision_graph*.png/svg` — optional rendered images
- `shap.joblib` — optional SHAP outputs (if available)
- `feature_importances.csv` — optional feature importance (if supported by the best model)

---

## DOT → PNG (decision graph)

If Graphviz is installed:

```bash
dot -Tpng automl_outputs/decision_graph.dot -o automl_outputs/decision_graph.png
```

Python alternative (requires system Graphviz + `pip install graphviz`):

```python
from graphviz import Source
Source.from_file("automl_outputs/decision_graph.dot").render("automl_outputs/decision_graph", format="png", cleanup=True)
```

---

## Benchmarking on many datasets (OpenML buckets)

If you collected bucket files like `datasets/bucket_le_1000.csv`, `datasets/bucket_75_25.csv`, etc.,
you can run batch evaluation:

```bash
python run_bucket_benchmark.py --datasets-dir datasets --out-root bucket_runs --budget 50 --cv 5 --tune-threshold
```

This will:
- download each dataset by `did` and use `target` from the CSV
- run ImBa per dataset and save artifacts to `bucket_runs/<bucket>/<dataset>/`
- compute per-bucket summaries (mean/std) and save boxplots (PNG)

**OpenML IDs**
- `did` = dataset id (download the dataset)
- `tid` = task id (dataset + task definition). For simple download you typically only need `did` + `target`.

---

## Tips & troubleshooting

- **Windows Tkinter error / main thread issue** when plotting:
  use a non-GUI backend (`matplotlib.use("Agg")`) — this is already done in benchmark scripts.
- **Target encoding requires** `category_encoders`.
- If you see `xgboost` warning about `use_label_encoder`:
  remove that parameter (newer XGBoost ignores it).
- If you get “unexpected keyword argument” errors (e.g. `lr`):
  check parameter names match the model (`learning_rate` for GradientBoostingClassifier).

---

## Project structure (typical)

- `imba/core.py` — high-level AutoML loop (Optuna, saving artifacts, best model)
- `imba/components.py` — preprocessors, DR, samplers, models, full pipeline factory
- `imba/evaluate.py` — CV evaluation and metrics (F1/ROC-AUC/…)
- `imba/visualize.py` — decision graph utilities
- `imba/shap_utils.py` — optional SHAP export
- `run_imba.py` — single-dataset runner
- `run_bucket_benchmark.py` — multi-dataset benchmark runner

---

## License
MIT
