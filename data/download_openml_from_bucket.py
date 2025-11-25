import os
import re
import time
import argparse
import pandas as pd
import openml

def sanitize(name: str, maxlen: int = 80) -> str:
    name = re.sub(r"[^\w\-.]+", "_", str(name))
    return name[:maxlen].strip("._")

def resolve_target(ds, fallback: str | None) -> str | None:
    if fallback and isinstance(fallback, str) and fallback.strip():
        return fallback
    if getattr(ds, "default_target_attribute", None):
        return ds.default_target_attribute
    try:
        q = ds.qualities or {}
        if isinstance(q, dict) and q.get("class_attribute"):
            return q["class_attribute"]
    except Exception:
        pass
    return None

def download_one(did: int, name: str | None, target_hint: str | None, outdir: str, overwrite: bool = False) -> dict:
    rec = {"did": int(did), "name": name, "target": target_hint, "status": "error", "path": "", "rows": None, "cols": None, "error": ""}
    try:
        ds = openml.datasets.get_dataset(int(did))
        target_name = resolve_target(ds, target_hint)
        if not target_name:
            raise RuntimeError("target not found")
        X, y, _, _ = ds.get_data(dataset_format="dataframe", target=target_name)
        base = f"{did}_{sanitize(name or ds.name)}"
        fpath = os.path.join(outdir, f"{base}.csv")
        if os.path.exists(fpath) and not overwrite:
            rec.update({"status": "exists", "path": fpath})
            return rec
        df = X.copy()
        if y is not None:
            if getattr(y, "name", None) != target_name:
                df[target_name] = pd.Series(y).values
            else:
                df[target_name] = y
        df.to_csv(fpath, index=False, encoding="utf-8")
        rec.update({"status": "ok", "path": fpath, "rows": df.shape[0], "cols": df.shape[1]})
        return rec
    except Exception as e:
        rec["error"] = str(e)
        return rec

def main():
    ap = argparse.ArgumentParser(description="Download OpenML datasets listed in a bucket CSV (by 'did').")
    ap.add_argument("--index", required=True, help=r"Путь к bucket CSV (например: data\datasets\bucket_le_1000_75_25.csv)")
    ap.add_argument("--outdir", required=True, help="Папка для сохранения датасетов (будет создана)")
    ap.add_argument("--limit", type=int, default=None, help="Максимум датасетов к скачиванию")
    ap.add_argument("--sleep", type=float, default=0.0, help="Пауза между загрузками (сек)")
    ap.add_argument("--overwrite", action="store_true", help="Перезаписывать существующие файлы")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    idx = pd.read_csv(args.index)
    if "did" not in idx.columns:
        raise SystemExit("В CSV не найдена колонка 'did'")

    idx = idx.drop_duplicates(subset=["did"]).reset_index(drop=True)
    if args.limit is not None:
        idx = idx.head(args.limit)

    report = []
    for i, row in idx.iterrows():
        did = int(row["did"])
        name = row["name"] if "name" in idx.columns else None
        target = row["target"] if "target" in idx.columns else None
        rec = download_one(did, name, target, args.outdir, overwrite=args.overwrite)
        report.append(rec)
        print(f"[{i+1}/{len(idx)}] did={did} -> {rec['status']} {rec['path'] or rec['error']}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    rep_df = pd.DataFrame(report)
    rep_path = os.path.join(args.outdir, "download_report.csv")
    rep_df.to_csv(rep_path, index=False, encoding="utf-8")
    print(f"\nОтчёт: {rep_path}")
    ok = (rep_df["status"] == "ok").sum()
    exists = (rep_df["status"] == "exists").sum()
    err = (rep_df["status"] == "error").sum()
    print(f"Итого: ok={ok}, exists={exists}, error={err}")

if __name__ == "__main__":
    main()
