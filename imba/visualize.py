from typing import Dict

def _graph_spec(best_trial: Dict):
    metric = best_trial.get("metric_name", "AP")
    score = best_trial.get("score")
    nodes = [
        ("start", "start"),
        ("enc", f"encoding: {best_trial.get('encoding')}"),
        ("sc", f"scaler: {best_trial.get('scaler')}"),
        ("imp", f"num_impute: {best_trial.get('num_impute')}"),
        ("skw", f"skew_auto: {best_trial.get('skew_auto')}"),
        ("imb", f"imbalance: {best_trial.get('imbalance_mode')}"),
        ("dr", f"DR: {best_trial.get('dr_kind')}"),
        ("mdl", f"model: {best_trial.get('model_name')}"),
        ("score", f"CV {metric}: {score:.4f}" if isinstance(score, (int, float)) else f"CV {metric}: {score}"),
    ]
    edges = [("start","enc"),("enc","sc"),("sc","imp"),("imp","skw"),("skw","imb"),("imb","dr"),("dr","mdl"),("mdl","score")]
    return nodes, edges

def save_decision_graph(best_trial: Dict, path: str = "decision_graph.dot") -> str:
    nodes, _ = _graph_spec(best_trial)
    parts = [
        "digraph AutoML {",
        "  rankdir=LR; node [shape=box, style=rounded];",
        "  start [label=\"start\"];",
    ]
    for key, label in nodes:
        if key == "start": continue
        parts.append(f'  {key} [label="{label}"];')
    parts += ["  start -> enc -> sc -> imp -> skw -> imb -> dr -> mdl -> score;", "}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path

def save_decision_graph_mermaid(best_trial: Dict, path: str = "decision_graph.mmd") -> str:
    nodes, edges = _graph_spec(best_trial)
    id2label = {i: lbl for i, lbl in nodes}
    lines = ["flowchart LR"]
    for a, b in edges:
        lines.append(f'  {a}["{id2label[a]}"] --> {b}["{id2label[b]}"]')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def save_decision_graph_networkx(best_trial: Dict, path: str = "decision_graph_nx.png") -> str:
    import networkx as nx
    import matplotlib.pyplot as plt
    nodes, edges = _graph_spec(best_trial)
    G = nx.DiGraph()
    G.add_nodes_from([n for n, _ in nodes])
    G.add_edges_from(edges)
    order = ["start","enc","sc","imp","skw","imb","dr","mdl","score"]
    pos = {name: (i, 0) for i, name in enumerate(order)}
    labels = {n: lbl for n, lbl in nodes}
    fig = plt.figure(figsize=(13, 2.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    nx.draw(G, pos, with_labels=False, node_shape="s", node_size=9000, arrows=True, width=1.0, ax=ax)
    for n, (x, y) in pos.items():
        ax.text(x, y, labels[n], ha="center", va="center", fontsize=9)
    ax.axis("off")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
