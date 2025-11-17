# -*- coding: utf-8 -*-
# Подбираем по 10 бинарных задач на корзину размер×дисбаланс.
# Пересечения по дисбалансу разрешены (одна задача может попасть в обе).
# Все артефакты сохраняются в папку ./datasets
import os
import json
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import openml

warnings.filterwarnings("ignore")

# ----------------------------
# ПАРАМЕТРЫ
# ----------------------------
SIZE_BUCKETS = [
    ("le_1000",    1,    1000),
    ("le_10000",   1001, 10000),
    ("le_50000",   10001, 50000),
]

# Минорный класс <= threshold
IMBALANCE_BUCKETS = [
    ("75_25", 0.25),
    ("90_10", 0.10),
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

# ----------------------------
# ВСПОМОГАТЕЛЬНЫЕ
# ----------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def pick_size_bucket(n_rows: int):
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= n_rows <= hi:
            return name
    return None

def pick_imbalance_buckets(y: pd.Series):
    vc = y.value_counts(dropna=True)
    if len(vc) != 2:
        return set()
    minor_ratio = vc.min() / float(vc.sum())
    return {name for name, thr in IMBALANCE_BUCKETS if minor_ratio <= thr}

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

def summarize_counts(buckets_collected):
    # Возвращает словарь: {(size, imb): count}
    return { (sb, ib): len(items) for (sb, ib), items in buckets_collected.items() }

def print_progress(checked, buckets_collected):
    counts = summarize_counts(buckets_collected)
    parts = [f"checked={checked}"]
    for sb, _, _ in SIZE_BUCKETS:
        for ib, _ in IMBALANCE_BUCKETS:
            c = counts.get((sb, ib), 0)
            parts.append(f"{sb}/{ib}={c}/{TARGET_PER_BUCKET}")
    print(" | ".join(parts))

# ----------------------------
# ОСНОВА
# ----------------------------
def main():
    np.random.seed(SEED)
    ensure_dir(ARTIFACTS_DIR)

    print("Загружаю список задач с OpenML…")
    tasks_df = openml.tasks.list_tasks(output_format='dataframe')
    tasks_df = tasks_df[tasks_df['task_type'] == 'Supervised Classification'].copy()

    # Перемешаем, чтобы быстрее набрать корзины (часто первые — крупные репозитории)
    tasks_df = tasks_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    if len(tasks_df) > MAX_FETCH_TASKS:
        tasks_df = tasks_df.head(MAX_FETCH_TASKS)

    buckets_collected = defaultdict(list)  # {(size, imb): [meta,...]}
    results_rows = []
    checked = 0
    total_target = TARGET_PER_BUCKET * len(SIZE_BUCKETS) * len(IMBALANCE_BUCKETS)

    def bucket_full(sb, ib):
        return len(buckets_collected[(sb, ib)]) >= TARGET_PER_BUCKET

    for _, row in tasks_df.iterrows():
        checked += 1
        try:
            tid = int(row['tid'])
            did = int(row['did'])
            target_name = row.get('target_feature', None)

            ds = openml.datasets.get_dataset(did)

            # Попытка определить целевую, если её нет в задаче
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

            imbalance_buckets = pick_imbalance_buckets(y_series)
            if not imbalance_buckets:
                continue

            vc = y_series.value_counts()
            maj_class = safe_str(vc.idxmax())
            min_class = safe_str(vc.idxmin())
            maj_count = int(vc.max())
            min_count = int(vc.min())
            total = maj_count + min_count
            minor_ratio = min_count / float(total)
            major_ratio = maj_count / float(total)

            placed = False
           
            for ib_name, _thr in IMBALANCE_BUCKETS:
                if ib_name in imbalance_buckets and not bucket_full(size_bucket, ib_name):
                    meta = dict(
                        size_bucket=size_bucket,
                        imbalance_bucket=ib_name,
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
                    buckets_collected[(size_bucket, ib_name)].append(meta)
                    results_rows.append(meta)
                    placed = True

            if placed and PAUSE_SEC > 0:
                time.sleep(PAUSE_SEC)

            if checked % PRINT_EVERY == 0:
                print_progress(checked, buckets_collected)

            have = sum(len(v) for v in buckets_collected.values())
            if have >= total_target:
                break

        except KeyboardInterrupt:
            raise
        except Exception:
            continue

    print_progress(checked, buckets_collected)

    # ----------------------------
    # СОХРАНЕНИЕ
    # ----------------------------
    df_all = pd.DataFrame(results_rows)
    df_all_sorted = df_all.sort_values(
        by=["size_bucket", "imbalance_bucket", "n_rows", "minority_ratio", "tid"]
    )
    df_all_sorted.to_csv(CSV_ALL, index=False, encoding="utf-8")
    print(f"Saved: {CSV_ALL} ({len(df_all_sorted)} rows)")

    by_bucket = {}
    for sb_name, _, _ in SIZE_BUCKETS:
        for ib_name, _ in IMBALANCE_BUCKETS:
            key = (sb_name, ib_name)
            items = buckets_collected.get(key, [])
            by_bucket[f"{sb_name}__{ib_name}"] = items
            out_path = os.path.join(ARTIFACTS_DIR, f"bucket_{sb_name}_{ib_name}.csv")
            if items:
                pd.DataFrame(items).sort_values(
                    by=["n_rows", "minority_ratio", "tid"]
                ).to_csv(out_path, index=False, encoding="utf-8")
                print(f"Saved: {out_path} ({len(items)} rows)")
            else:
                print(f"{sb_name}/{ib_name}: empty")

    with open(JSON_BUCKETS, "w", encoding="utf-8") as f:
        json.dump(by_bucket, f, ensure_ascii=False, indent=2)
    print(f"Saved: {JSON_BUCKETS}")

    print("\nSUMMARY:")
    for sb_name, _, _ in SIZE_BUCKETS:
        for ib_name, _ in IMBALANCE_BUCKETS:
            key = (sb_name, ib_name)
            print(f"{sb_name}/{ib_name}: {len(buckets_collected.get(key, []))}/{TARGET_PER_BUCKET}")
    print(f"\nTotal checked: {checked}")
    print("Done.")

if __name__ == "__main__":
    main()
