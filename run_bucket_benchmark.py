#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run AutoMLBinary on datasets listed in ./datasets/bucket_*.csv,
download from OpenML by did, use target from meta CSV,
aggregate metrics per bucket, save results and plots.

Outputs:
  bucket_runs/<bucket>/<dataset_name>/*   # arifacts for each dataset
  bucket_runs/<bucket>/summary_metrics.csv
  bucket_runs/<bucket>/summary_agg.csv
  bucket_runs/<bucket>/metrics_boxplot.png
"""

import os
import glob
import json
import warnings
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# --- safe non-GUI plotting ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openml

from imba.core import AutoMLBinary, AutoMLConfig
from imba.evaluate import EvalConfig
from imba.components import FullPipelineFactory


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_bucket_files(datasets_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(datasets_dir, "bucket_*.csv")))
    if not files:
        raise RuntimeError(f"No bucket_*.csv found in {datasets_dir}")
    return files


def download_openml_dataset(did: int, target: str):
    """
    Returns (X_df, y_series, dataset_name)
    """
    ds = openml.datasets.get_dataset(int(did))
    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=target)
    return X, pd.Series(y), ds.name


def compute_scores_for_best_cfg(
    automl: AutoMLBinary,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """
    Recompute CV metrics for best configuration.
    """
    y_bin, _ = AutoMLBinary._encode_binary_labels(y)
    best_cfg = automl.best_["config"]
    pipe = FullPipelineFactory.build(best_cfg, n_features_hint=X.shape[1], n_classes=2)
    scores = automl.eval.cv_score(pipe, X, y_bin)
    return scores


def maybe_save_feature_importances(pipe, X, y_bin, out_dir: str, top_k: int = 50):
    """
    Try to save top-k feature importances for the final best pipe.
    """
    try:
        pipe.fit(X, y_bin)
    except Exception:
        return

    pre = getattr(pipe, "named_steps", {}).get("pre", None)
    clf = getattr(pipe, "named_steps", {}).get("clf", None)
    if clf is None:
        return

    # get feature names if possible
    feature_names = None
    if pre is not None and hasattr(pre, "ct_"):
        ct = pre.ct_
        if hasattr(ct, "get_feature_names_out"):
            try:
                feature_names = list(ct.get_feature_names_out())
            except Exception:
                feature_names = None

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)

    if importances is None:
        return

    n_feats = len(importances)
    if feature_names is None or len(feature_names) != n_feats:
        feature_names = [f"feature_{i}" for i in range(n_feats)]

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df.sort_values("importance", ascending=False, inplace=True)
    if top_k and top_k > 0:
        imp_df = imp_df.head(top_k)

    imp_df.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)


def save_bucket_plots(bucket_dir: str, res_df: pd.DataFrame):
    """
    Save boxplot for f1 and roc for this bucket.
    """
    if res_df.empty:
        return
    plt.figure(figsize=(6, 4))
    res_df[["f1", "roc"]].boxplot()
    plt.title("Distribution of F1 and ROC AUC")
    plt.ylabel("Score")
    plt.tight_layout()
    out_path = os.path.join(bucket_dir, "metrics_boxplot.png")
    plt.savefig(out_path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------
# Core per-bucket run
# ---------------------------------------------------------------------
def run_bucket(bucket_csv: str, args):
    bucket_name = os.path.splitext(os.path.basename(bucket_csv))[0].replace("bucket_", "")
    bucket_dir = os.path.join(args.out_root, bucket_name)
    ensure_dir(bucket_dir)

    print(f"\n==============================")
    print(f"BUCKET: {bucket_name}")
    print(f"Meta file: {bucket_csv}")
    print(f"Outputs: {bucket_dir}")
    print(f"==============================")

    meta = pd.read_csv(bucket_csv)
    if meta.empty:
        print("Bucket is empty, skipping.")
        return

    per_dataset_scores: List[Dict[str, Any]] = []

    for i, row in meta.iterrows():
        did = int(row["did"])
        target = str(row["target"]).strip()
        name_hint = str(row.get("name", f"did_{did}")).strip()

        ds_out_dir = os.path.join(bucket_dir, name_hint)
        ensure_dir(ds_out_dir)

        print(f"\n[{i+1}/{len(meta)}] Download did={did} name={name_hint} target={target}")

        try:
            X, y, ds_name = download_openml_dataset(did, target)
        except Exception as e:
            print(f"[download] failed for did={did}: {e}")
            continue

        # AutoML
        automl = AutoMLBinary(
            AutoMLConfig(
                budget=args.budget,
                out_dir=ds_out_dir,
                random_state=args.seed,
                ensemble_top_n=args.top_n,
                use_stacking=args.stacking,
            ),
            eval_cfg=EvalConfig(
                n_splits=args.cv,
                scoring_primary="f1",
                stratified=not args.no_stratified,
                f1_average="binary",
                tune_threshold=args.tune_threshold,
                threshold_min=args.thr_min,
                threshold_max=args.thr_max,
                threshold_steps=args.thr_steps,
            ),
        )

        try:
            automl.fit(X, y)
        except Exception as e:
            print(f"[automl] failed on {ds_name}: {e}")
            continue

        # recompute CV metrics for best config
        try:
            scores = compute_scores_for_best_cfg(automl, X, y)
        except Exception as e:
            print(f"[score] recompute failed: {e}")
            continue

        # feature importances (optional)
        try:
            y_bin, _ = AutoMLBinary._encode_binary_labels(y)
            best_cfg = automl.best_["config"]
            pipe = FullPipelineFactory.build(best_cfg, n_features_hint=X.shape[1], n_classes=2)
            maybe_save_feature_importances(pipe, X, y_bin, ds_out_dir, top_k=args.feat_top_k)
        except Exception:
            pass

        record = {
            "dataset": ds_name,
            "did": did,
            "target": target,
            **scores
        }
        per_dataset_scores.append(record)

        print(f"[done] {ds_name} | F1={scores['f1']:.4f} ROC={scores['roc']:.4f} "
              f"thr={scores.get('best_threshold', 0.5):.3f}")

    if not per_dataset_scores:
        print("No successful datasets in this bucket.")
        return

    res_df = pd.DataFrame(per_dataset_scores)

    # save per-dataset metrics
    summary_metrics_path = os.path.join(bucket_dir, "summary_metrics.csv")
    res_df.to_csv(summary_metrics_path, index=False)

    # aggregate mean/std for f1, roc
    agg = res_df[["f1", "roc"]].agg(["mean", "std"]).T
    summary_agg_path = os.path.join(bucket_dir, "summary_agg.csv")
    agg.to_csv(summary_agg_path)

    # save plot
    save_bucket_plots(bucket_dir, res_df)

    print(f"\nSaved per-dataset metrics: {summary_metrics_path}")
    print(f"Saved agg metrics:        {summary_agg_path}")
    print(f"Saved boxplot:            {os.path.join(bucket_dir, 'metrics_boxplot.png')}")
    print("\nBucket summary:")
    print(agg)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Run AutoMLBinary on OpenML datasets listed in ./datasets bucket CSVs"
    )
    p.add_argument("--datasets-dir", default="datasets",
                   help="Folder with bucket_*.csv (default: ./datasets)")
    p.add_argument("--out-root", default="bucket_runs",
                   help="Root folder to store runs (default: ./bucket_runs)")

    p.add_argument("--budget", type=int, default=50,
                   help="Optuna trials per dataset")
    p.add_argument("--cv", type=int, default=5,
                   help="CV folds")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--top-n", type=int, default=3,
                   help="Top-N for blending/stacking")
    p.add_argument("--stacking", action="store_true",
                   help="Use stacking instead of soft voting")
    p.add_argument("--no-stratified", action="store_true",
                   help="Use KFold instead of StratifiedKFold")

    # threshold tuning
    p.add_argument("--tune-threshold", action="store_true",
                   help="Enable threshold tuning for F1 in CV")
    p.add_argument("--thr-min", type=float, default=0.05)
    p.add_argument("--thr-max", type=float, default=0.95)
    p.add_argument("--thr-steps", type=int, default=19)

    # feature importance
    p.add_argument("--feat-top-k", type=int, default=50,
                   help="Top-K features to store per dataset")

    args = p.parse_args()

    ensure_dir(args.out_root)

    bucket_files = load_bucket_files(args.datasets_dir)
    print(f"Found {len(bucket_files)} bucket files in {args.datasets_dir}")

    for bf in bucket_files:
        run_bucket(bf, args)

    print("\nALL BUCKETS DONE.")


if __name__ == "__main__":
    main()
