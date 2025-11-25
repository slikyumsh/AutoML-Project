import os
import json
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import openml

warnings.filterwarnings("ignore")

SIZE_BUCKETS = [
    ("le_1000",    1,     1000),
    ("le_10000",   1001,  10000),
    ("le_50000",   10001, 50000),
]

IMBALANCE_BUCKETS = [
    ("90_10", 0.10),
    ("75_25", 0.25),
]

TARGET_PER_BUCKET = 10
MAX_FETCH_TASKS = 20000
PAUSE_SEC = 0.0

MIN_ROWS = 1
MAX_COLS = 500
MAX_MISSING_RATIO = 0.3

ARTIFACTS_DIR = "datasets"
CSV_ALL = os.path.join(ARTIFACTS_DIR, "selected_binary_tasks.csv")
JSON_BUCKETS = os.path.join(ARTIFACTS_DIR, "selected_binary_tasks_by_bucket.json")

PRINT_EVERY = 50
SEED = 42

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def pick_size_bucket(n_rows: int):
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= n_rows <= hi:
            return name
    return None

def pick_imbalance_bucket_exclusive(y: pd.Series):
    """
    Взаимоисключающий выбор корзины дисбаланса:
      minor_ratio <= 0.10  -> '90_10'
      else minor_ratio <= 0.25 -> '75_25'
      else None
    """
    vc = y.value_counts(dropna=True)
    if len(vc) != 2:
        return None

    minor_ratio = vc.min() / float(vc.sum())

    if minor_ratio <= 0.10:
        return "90_10"
    if minor_ratio <= 0.25:
        return "75_25"
    return None

def dataset_missing_ratio(X: pd.DataFrame) -> float:
    try:
        return float(X.isna().mean().mean())
    except Exception:
        return 1.0

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return repr(x)

def print_progress(checked, size_collected, imb_collected):
    parts = [f"checked={checked}"]
    for sb, _, _ in SIZE_BUCKETS:
        parts.append(f"{sb}={len(size_collected[sb])}/{TARGET_PER_BUCKET}")
    for ib, _ in IMBALANCE_BUCKETS:
        parts.append(f"{ib}={len(imb_collected[ib])}/{TARGET_PER_BUCKET}")
    print(" | ".join(parts))

def all_buckets_full(size_collected, imb_collected) -> bool:
    for sb, _, _ in SIZE_BUCKETS:
        if len(size_collected[sb]) < TARGET_PER_BUCKET:
            return False
    for ib, _ in IMBALANCE_BUCKETS:
        if len(imb_collected[ib]) < TARGET_PER_BUCKET:
            return False
    return True


def main():
    np.random.seed(SEED)
    ensure_dir(ARTIFACTS_DIR)

    print("Загружаю список задач с OpenML…")
    tasks_df = openml.tasks.list_tasks(output_format='dataframe')
    tasks_df = tasks_df[tasks_df['task_type'] == 'Supervised Classification'].copy()

    tasks_df = tasks_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    if len(tasks_df) > MAX_FETCH_TASKS:
        tasks_df = tasks_df.head(MAX_FETCH_TASKS)

    size_collected = {sb: [] for sb, _, _ in SIZE_BUCKETS}
    imb_collected = {ib: [] for ib, _ in IMBALANCE_BUCKETS}

    all_rows = [] 
    checked = 0

    for _, row in tasks_df.iterrows():
        checked += 1
        try:
            tid = int(row['tid'])
            did = int(row['did'])
            target_name = row.get('target_feature', None)

            ds = openml.datasets.get_dataset(did)

            if not target_name or target_name == '' or pd.isna(target_name):
                try:
                    qualities = ds.qualities
                    if isinstance(qualities, dict) and 'class_attribute' in qualities:
                        target_name = qualities['class_attribute']
                except Exception:
                    pass

            if not target_name or target_name == '' or pd.isna(target_name):
                continue

            X, y, _, _ = ds.get_data(dataset_format='dataframe', target=target_name)

            n_rows, n_cols = X.shape
            if n_rows < MIN_ROWS or n_cols > MAX_COLS:
                continue

            miss_ratio = dataset_missing_ratio(X)
            if miss_ratio > MAX_MISSING_RATIO:
                continue

            y_series = pd.Series(y).dropna()
            if y_series.nunique() != 2:
                continue

            size_bucket = pick_size_bucket(n_rows)
            if size_bucket is None:
                continue

            imbalance_bucket = pick_imbalance_bucket_exclusive(y_series)
            if imbalance_bucket is None:
                continue

            vc = y_series.value_counts()
            maj_class = safe_str(vc.idxmax())
            min_class = safe_str(vc.idxmin())
            maj_count = int(vc.max())
            min_count = int(vc.min())
            total = maj_count + min_count
            minor_ratio = min_count / float(total)
            major_ratio = maj_count / float(total)

            meta_base = dict(
                tid=tid,
                did=did,
                name=ds.name,
                target=target_name,
                n_rows=n_rows,
                n_features=n_cols,
                majority_class=maj_class,
                minority_class=min_class,
                majority_ratio=round(major_ratio, 6),
                minority_ratio=round(minor_ratio, 6),
                missing_ratio=round(miss_ratio, 6),
            )

            placed_any = False

            if len(size_collected[size_bucket]) < TARGET_PER_BUCKET:
                meta = dict(**meta_base, size_bucket=size_bucket, imbalance_bucket=None)
                size_collected[size_bucket].append(meta)
                all_rows.append({"group": size_bucket, "group_type": "size", **meta})
                placed_any = True

            if len(imb_collected[imbalance_bucket]) < TARGET_PER_BUCKET:
                meta = dict(**meta_base, size_bucket=None, imbalance_bucket=imbalance_bucket)
                imb_collected[imbalance_bucket].append(meta)
                all_rows.append({"group": imbalance_bucket, "group_type": "imbalance", **meta})
                placed_any = True

            if placed_any and PAUSE_SEC > 0:
                time.sleep(PAUSE_SEC)

            if checked % PRINT_EVERY == 0:
                print_progress(checked, size_collected, imb_collected)

            if all_buckets_full(size_collected, imb_collected):
                break

        except KeyboardInterrupt:
            raise
        except Exception:
            continue

    print_progress(checked, size_collected, imb_collected)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(CSV_ALL, index=False, encoding="utf-8")
    print(f"Saved: {CSV_ALL} ({len(df_all)} rows)")

    by_bucket = {}

    for sb_name, _, _ in SIZE_BUCKETS:
        items = size_collected.get(sb_name, [])
        by_bucket[sb_name] = items
        out_path = os.path.join(ARTIFACTS_DIR, f"bucket_{sb_name}.csv")
        if items:
            pd.DataFrame(items).sort_values(
                by=["n_rows", "minority_ratio", "tid"]
            ).to_csv(out_path, index=False, encoding="utf-8")
            print(f"Saved: {out_path} ({len(items)} rows)")
        else:
            print(f"{sb_name}: empty")

    for ib_name, _ in IMBALANCE_BUCKETS:
        items = imb_collected.get(ib_name, [])
        by_bucket[ib_name] = items
        out_path = os.path.join(ARTIFACTS_DIR, f"bucket_{ib_name}.csv")
        if items:
            pd.DataFrame(items).sort_values(
                by=["n_rows", "minority_ratio", "tid"]
            ).to_csv(out_path, index=False, encoding="utf-8")
            print(f"Saved: {out_path} ({len(items)} rows)")
        else:
            print(f"{ib_name}: empty")

    with open(JSON_BUCKETS, "w", encoding="utf-8") as f:
        json.dump(by_bucket, f, ensure_ascii=False, indent=2)
    print(f"Saved: {JSON_BUCKETS}")

    print("\nSUMMARY:")
    for sb_name, _, _ in SIZE_BUCKETS:
        print(f"{sb_name}: {len(size_collected[sb_name])}/{TARGET_PER_BUCKET}")
    for ib_name, _ in IMBALANCE_BUCKETS:
        print(f"{ib_name}: {len(imb_collected[ib_name])}/{TARGET_PER_BUCKET}")

    print(f"\nTotal checked: {checked}")
    print("Done.")


if __name__ == "__main__":
    main()
