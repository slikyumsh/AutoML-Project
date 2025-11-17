from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

@dataclass
class EvalConfig:
    n_splits: int = 5
    random_state: int = 42
    scoring_primary: str = "average_precision"  # robust for imbalance

DEFAULT_SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}

class Evaluator:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def cv_score(self, estimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        cv = StratifiedKFold(n_splits=self.cfg.n_splits, shuffle=True, random_state=self.cfg.random_state)
        res = cross_validate(
            estimator,
            X,
            y,
            scoring=DEFAULT_SCORING,
            cv=cv,
            return_estimator=False,
            n_jobs=-1,
        )
        out = {f"mean_{k}": float(np.nanmean(v)) for k, v in res.items() if k.startswith("test_") for v in [res[k]]}
        # Alias primary
        out["primary"] = out.get(f"mean_test_{self.cfg.scoring_primary}", np.nan)
        return out