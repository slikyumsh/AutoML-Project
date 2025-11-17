from typing import Optional, List
import numpy as np
import pandas as pd
import joblib

try:
    import shap
except Exception:
    shap = None

def compute_and_save_shap(pipeline, X: pd.DataFrame, out_path: str, max_samples: int = 1000) -> Optional[str]:
    if shap is None:
        return None
    # Transform features up to the final estimator
    try:
        pre = pipeline.named_steps.get("pre") or pipeline.named_steps.get("pre__pre")
        clf = pipeline.named_steps.get("clf")
    except Exception:
        return None
    try:
        X_t = pre.transform(X)
    except Exception:
        # Fit if needed (e.g., best pipeline may not be refit on full data yet)
        pre.fit(X, getattr(pipeline, "y_train_", None))
        X_t = pre.transform(X)
    # Choose explainer
    try:
        expl = shap.Explainer(clf)
    except Exception:
        try:
            expl = shap.TreeExplainer(clf)
        except Exception:
            return None
    Xt_sample = X_t[: min(len(X_t), max_samples)]
    sv = expl(Xt_sample)
    # Save as joblib for later interactive analysis
    joblib.dump({"shap_values": sv, "data": Xt_sample}, out_path)
    return out_path