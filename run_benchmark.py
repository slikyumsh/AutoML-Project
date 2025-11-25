#!/usr/bin/env python3
import argparse
import os

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imba.core import AutoMLBinary, AutoMLConfig
from imba.evaluate import EvalConfig
from imba.components import FullPipelineFactory


# -----------------------------
# helpers
# -----------------------------
def find_meta_csv(meta_dir: str) -> Optional[str]:
    """
    Ищет CSV в meta_dir, который похож на файл-описание датасетов:
    должен содержать колонки name и target.
    """
    if not os.path.isdir(meta_dir):
        return None

    for f in os.listdir(meta_dir):
        if not f.lower().endswith(".csv"):
            continue
        path = os.path.join(meta_dir, f)
        try:
            df = pd.read_csv(path, nrows=5)
            cols = set(df.columns.str.lower())
            if "name" in cols and "target" in cols:
                return path
        except Exception:
            pass
    return None


def load_target_map(meta_path: str) -> Dict[str, str]:
    """
    Грузим таблицу meta и строим mapping:
      dataset_name (из колонки name) -> target column
    """
    meta = pd.read_csv(meta_path)
    # нормализуем имена
    meta.columns = [c.strip() for c in meta.columns]
    if "name" not in meta.columns or "target" not in meta.columns:
        raise ValueError(f"Meta file must contain columns: name, target. Got: {list(meta.columns)}")

    name2target = {}
    for _, row in meta.iterrows():
        nm = str(row["name"]).strip()
        tg = str(row["target"]).strip()
        if nm and tg:
            name2target[nm] = tg
    return name2target


def guess_target_column(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    """
    Эвристика для target, если он не найден в meta.
    """
    if explicit is not None:
        if explicit in df.columns:
            return explicit
        else:
            raise ValueError(f"Explicit target '{explicit}' not found in columns: {list(df.columns)}")

    candidates = ["target", "Target", "TARGET", "class", "Class", "label", "Label", "y", "Y"]
    for c in candidates:
        if c in df.columns:
            return c

    return df.columns[-1]


def compute_feature_importances(
    pipe,
    X: pd.DataFrame,
    y_bin: np.ndarray,
    out_dir: str,
    top_k: int = 50,
) -> Optional[str]:
    """
    Фитим pipeline и пытаемся вытащить важности признаков.
    """
    try:
        pipe.fit(X, y_bin)
    except Exception as e:
        print(f"[featimp] Fit failed, skip: {e}")
        return None

    # достаём препроцессор и модель
    try:
        pre = pipe.named_steps.get("pre", None)
    except Exception:
        pre = None

    try:
        clf = pipe.named_steps.get("clf", None)
    except Exception:
        clf = None

    if clf is None:
        print("[featimp] No 'clf' step found, skip feature_importances")
        return None

    # feature names
    feature_names = None
    if pre is not None and hasattr(pre, "ct_"):
        ct = pre.ct_
        if hasattr(ct, "get_feature_names_out"):
            try:
                feature_names = list(ct.get_feature_names_out())
            except Exception:
                feature_names = None

    # importances
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)

    if importances is None:
        print("[featimp] Estimator has no feature_importances_ or coef_, skip")
        return None

    n_feats = len(importances)
    if feature_names is None or len(feature_names) != n_feats:
        feature_names = [f"feature_{i}" for i in range(n_feats)]

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df.sort_values("importance", ascending=False, inplace=True)
    if top_k and top_k > 0:
        imp_df = imp_df.head(top_k)

    out_path = os.path.join(out_dir, "feature_importances.csv")
    imp_df.to_csv(out_path, index=False)
    print(f"[featimp] Saved feature importances to {out_path}")
    return out_path


def evaluate_dataset(
    csv_path: str,
    args,
    name2target: Dict[str, str],
) -> Dict[str, Any]:
    """
    Один датасет:
      - читаем
      - target либо из meta, либо эвристикой
      - запускаем AutoMLBinary
      - пересчитываем CV-метрики для лучшей конфигурации
      - сохраняем важности признаков
    """
    print(f"\n=== Dataset: {os.path.basename(csv_path)} ===")
    df = pd.read_csv(csv_path)

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    # target priority:
    # 1) args.target if set
    # 2) meta name->target
    # 3) heuristic
    if args.target is not None:
        target_col = guess_target_column(df, args.target)
        source = "explicit --target"
    elif dataset_name in name2target and name2target[dataset_name] in df.columns:
        target_col = name2target[dataset_name]
        source = "meta (data/datasets)"
    else:
        target_col = guess_target_column(df, None)
        source = "heuristic"

    print(f"Target column: {target_col}  [source: {source}]")

    y = df.pop(target_col)
    X = df

    out_dir = os.path.join(args.out_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    automl = AutoMLBinary(
        AutoMLConfig(
            budget=args.budget,
            out_dir=out_dir,
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

    automl.fit(X, y)

    y_bin, _ = AutoMLBinary._encode_binary_labels(y)

    best_cfg = automl.best_["config"]
    pipe = FullPipelineFactory.build(best_cfg, n_features_hint=X.shape[1], n_classes=2)
    scores = automl.eval.cv_score(pipe, X, y_bin)

    try:
        compute_feature_importances(pipe, X, y_bin, out_dir, top_k=args.feat_top_k)
    except Exception as e:
        print(f"[featimp] Failed: {e}")

    scores["dataset"] = dataset_name
    scores["target"] = target_col
    scores["target_source"] = source
    return scores


# -----------------------------
# main
# -----------------------------
def main():
    p = argparse.ArgumentParser(
        description="Benchmark AutoMLBinary on multiple datasets and aggregate F1 / ROC AUC"
    )
    p.add_argument(
        "--data-dir",
        default="downloaded datasets",
        help="Directory with CSV datasets (default: 'downloaded datasets')",
    )
    p.add_argument(
        "--meta-dir",
        default="data/datasets",
        help="Directory with meta CSV containing name/target (default: data/datasets)",
    )
    p.add_argument(
        "--meta-path",
        default=None,
        help="Explicit meta CSV path. If None, auto-detect in meta-dir.",
    )
    p.add_argument(
        "--target",
        default=None,
        help="Force same target name for all datasets (overrides meta).",
    )
    p.add_argument(
        "--out-root",
        default="benchmark_outputs",
        help="Root directory to store all experiment outputs",
    )
    p.add_argument("--budget", type=int, default=50, help="Optuna trials per dataset")
    p.add_argument("--cv", type=int, default=5, help="CV folds")
    p.add_argument("--seed", type=int, default=42, help="Random seed for Optuna / CV")
    p.add_argument("--top-n", type=int, default=3, help="Top-N models for blending/stacking")
    p.add_argument("--stacking", action="store_true", help="Use stacking instead of soft voting")
    p.add_argument("--no-stratified", action="store_true", help="Use KFold instead StratifiedKFold")

    # threshold tuning
    p.add_argument("--tune-threshold", action="store_true", help="Enable threshold tuning for F1 in CV")
    p.add_argument("--thr-min", type=float, default=0.05, help="Min threshold")
    p.add_argument("--thr-max", type=float, default=0.95, help="Max threshold")
    p.add_argument("--thr-steps", type=int, default=19, help="Number of thresholds")

    # feature importance
    p.add_argument("--feat-top-k", type=int, default=50, help="Top-K features to save per dataset")

    args = p.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # ---- load meta mapping ----
    meta_path = args.meta_path or find_meta_csv(args.meta_dir)
    if meta_path is None:
        print(f"[meta] No suitable meta CSV found in '{args.meta_dir}'. "
              f"Targets will be guessed heuristically.")
        name2target = {}
    else:
        print(f"[meta] Using meta file: {meta_path}")
        name2target = load_target_map(meta_path)

    # ---- collect CSV datasets ----
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    csv_files = [
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.lower().endswith(".csv")
    ]
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {args.data_dir}")

    print(f"Found {len(csv_files)} CSV datasets in '{args.data_dir}'")

    all_scores: List[Dict[str, Any]] = []

    for path in sorted(csv_files):
        try:
            scores = evaluate_dataset(path, args, name2target)
            all_scores.append(scores)
            print(
                f"Done: {os.path.basename(path)} | "
                f"F1={scores['f1']:.4f}, ROC_AUC={scores['roc']:.4f}, "
                f"thr={scores.get('best_threshold', 0.5):.3f}"
            )
        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")

    if not all_scores:
        raise RuntimeError("No successful runs, nothing to aggregate.")

    # ---- aggregate ----
    res_df = pd.DataFrame(all_scores)
    summary_path = os.path.join(args.out_root, "summary_metrics.csv")
    res_df.to_csv(summary_path, index=False)

    agg = res_df[["f1", "roc"]].agg(["mean", "std"]).T
    agg_path = os.path.join(args.out_root, "summary_agg.csv")
    agg.to_csv(agg_path)

    print("\n=== Aggregated metrics across datasets ===")
    print(agg)

    # ---- boxplot ----
    plt.figure(figsize=(6, 4))
    res_df[["f1", "roc"]].boxplot()
    plt.title("Distribution of F1 and ROC AUC across datasets")
    plt.ylabel("Score")
    plt.tight_layout()
    boxplot_path = os.path.join(args.out_root, "metrics_boxplot.png")
    plt.savefig(boxplot_path, dpi=160)
    plt.close()

    print(f"\nSaved per-dataset metrics to: {summary_path}")
    print(f"Saved summary (mean/std) to: {agg_path}")
    print(f"Saved boxplot to: {boxplot_path}")


if __name__ == "__main__":
    main()
