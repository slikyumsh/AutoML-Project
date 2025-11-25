from typing import Optional
import numpy as np

def _extract_parts(pipeline):
    if hasattr(pipeline, "named_steps"):
        pre = []
        est = None
        for name, step in pipeline.named_steps.items():
            if name == "clf":
                est = step
                break
            pre.append(step)
        return pre, est or pipeline
    return [], pipeline

def _transform_seq(steps, X):
    Xc = X
    for st in steps:
        if hasattr(st, "transform"):
            Xc = st.transform(Xc)
    return Xc

def compute_and_save_shap(pipeline, X_df, out_path: str, max_samples: int = 200) -> Optional[str]:
    try:
        import shap
    except Exception:
        return None
    pre_steps, est = _extract_parts(pipeline)
    Xp = _transform_seq(pre_steps, X_df)
    Xp = np.asarray(Xp)
    if Xp.shape[0] > max_samples:
        Xb = Xp[:max_samples]
    else:
        Xb = Xp
    try:
        expl = None
        base = getattr(est, "base", est)
        if "xgboost" in base.__class__.__module__ or "lightgbm" in base.__class__.__module__:
            expl = shap.TreeExplainer(base)
        if base.__class__.__name__.lower().startswith("catboost"):
            expl = shap.TreeExplainer(base)
        if expl is None:
            if hasattr(base, "coef_"):
                expl = shap.LinearExplainer(base, Xb, feature_dependence="independent")
            else:
                expl = shap.Explainer(base, Xb)
        sv = expl(Xb)
        import joblib
        joblib.dump({"shap_values": sv, "sample_X": Xb}, out_path)
        return out_path
    except Exception:
        return None
