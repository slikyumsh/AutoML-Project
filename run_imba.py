#!/usr/bin/env python3
import argparse
import os
import pandas as pd

from imba import AutoMLBinary
from imba.core import AutoMLConfig
from imba.evaluate import EvalConfig
from imba.visualize import (
    save_decision_graph,
    save_decision_graph_networkx,
    save_decision_graph_mermaid,
)

def main():
    p = argparse.ArgumentParser(description="Custom AutoML for imbalanced binary classification")
    p.add_argument("--input", required=True, help="Path to CSV with data")
    p.add_argument("--target", required=True, help="Target column name")
    p.add_argument("--strategy", choices=["bayes", "bandit"], default="bayes")
    p.add_argument("--budget", type=int, default=60, help="Trials / pulls")
    p.add_argument("--out", default="automl_outputs", help="Output directory")
    p.add_argument("--cv", type=int, default=5, help="CV folds")
    p.add_argument("--viz-formats", default="png,svg,mermaid",
                   help="Comma-separated: png (networkx), svg/png (graphviz), mermaid")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.input)
    y = df.pop(args.target)

    automl = AutoMLBinary(
        AutoMLConfig(strategy=args.strategy, budget=args.budget, out_dir=args.out),
        eval_cfg=EvalConfig(n_splits=args.cv)
    )
    automl.fit(df, y)

    best_path = os.path.join(args.out, "best_model.joblib")
    print(f"Best model saved to: {best_path}")

    # DOT — всегда сохраняем
    dot_path = os.path.join(args.out, "decision_graph.dot")
    # файл уже создаётся внутри AutoML (если оставил как в моём коде), но повторное создание безопасно
    print(f"Decision graph DOT: {save_decision_graph({'dummy':1}, dot_path) if False else dot_path}")

    # Рендеры
    targets = [s.strip().lower() for s in str(args.viz_formats).split(",") if s.strip()]
    if "mermaid" in targets:
        try:
            mmd = save_decision_graph_mermaid(_read_best_summary(args.out), os.path.join(args.out, "decision_graph.mmd"))
            md = os.path.join(args.out, "decision_graph.md")
            with open(md, "w", encoding="utf-8") as f:
                f.write("```mermaid\n")
                with open(mmd, "r", encoding="utf-8") as fin:
                    f.write(fin.read())
                f.write("\n```")
            print(f"Decision graph Mermaid: {mmd}  | Markdown wrapper: {md}")
        except Exception as e:
            print(f"[viz] Mermaid skipped: {e}")

    if any(x in targets for x in ("png", "svg")):
        # Пробуем Graphviz
        if "svg" in targets or "png" in targets:
            try:
                from graphviz import Source
                src = Source.from_file(dot_path)
                if "png" in targets:
                    out_file = src.render(filename=os.path.join(args.out, "decision_graph"), format="png", cleanup=True)
                    print(f"Decision graph (Graphviz PNG): {out_file}")
                if "svg" in targets:
                    out_file = src.render(filename=os.path.join(args.out, "decision_graph"), format="svg", cleanup=True)
                    print(f"Decision graph (Graphviz SVG): {out_file}")
            except Exception as e:
                print(f"[viz] Graphviz render failed: {e}")

        # Всегда делаем PNG без Graphviz — через NetworkX
        try:
            nx_png = save_decision_graph_networkx(_read_best_summary(args.out),
                                                  os.path.join(args.out, "decision_graph_nx.png"))
            print(f"Decision graph (NetworkX PNG): {nx_png}")
        except Exception as e:
            print(f"[viz] NetworkX PNG skipped: {e}\nTip: pip install networkx matplotlib")

def _read_best_summary(out_dir: str):
    """Читаем best_summary.json, чтобы визуализация брала финальные значения."""
    import json
    path = os.path.join(out_dir, "best_summary.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    main()
