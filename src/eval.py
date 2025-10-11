import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def evaluate_model(model, X_test, y_test, config: dict):
    pred = model.predict(X_test).data[:, 0]
    
    metrics = {}
    if "auc" in config["metrics"]:
        metrics["AUC"] = roc_auc_score(y_test, pred)
    if "accuracy" in config["metrics"]:
        metrics["Accuracy"] = accuracy_score(y_test, (pred > 0.5).astype(int))
    if "f1" in config["metrics"]:
        metrics["F1"] = f1_score(y_test, (pred > 0.5).astype(int))
    
    save_metrics(metrics, config["metrics_path"])
    return metrics


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("LightAutoML Evaluation Metrics\n")
        f.write("==============================\n\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
