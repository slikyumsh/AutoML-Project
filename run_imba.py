#!/usr/bin/env python3
import argparse
import os
import pandas as pd

from imba import AutoMLBinary
from imba.core import AutoMLConfig
from imba.evaluate import EvalConfig

def main():
    p = argparse.ArgumentParser(description="Custom AutoML for imbalanced binary classification (Optuna)")
    p.add_argument("--input", required=True, help="Path to CSV with data")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--budget", type=int, default=60, help="Optuna trials")
    p.add_argument("--out", default="automl_outputs", help="Output directory")
    p.add_argument("--cv", type=int, default=5, help="CV folds")
    p.add_argument("--no-stratified", action="store_true", help="Use plain KFold instead StratifiedKFold")
    p.add_argument("--top-n", type=int, default=3, help="Top-N models for blending/stacking")
    p.add_argument("--stacking", action="store_true", help="Use stacking instead of blending")

    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.input)
    y = df.pop(args.target)

    automl = AutoMLBinary(
        AutoMLConfig(
            budget=args.budget,
            out_dir=args.out,
            ensemble_top_n=args.top_n,
            use_stacking=True
        ),
        eval_cfg=EvalConfig(
            n_splits=args.cv,
            scoring_primary="f1",
             tune_threshold=True,
            stratified=not args.no_stratified
        )
    )
    automl.fit(df, y)

    best_path = os.path.join(args.out, "best_model.joblib")
    print(f"Best model saved to: {best_path}")
    print(f"History CSV: {os.path.join(args.out, 'metrics_vs_iteration.csv')}")
    print(f"History PNG: {os.path.join(args.out, 'metrics_vs_iteration.png')}")

if __name__ == "__main__":
    main()
