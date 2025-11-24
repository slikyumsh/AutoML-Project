from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, recall_score,
    precision_score, accuracy_score, balanced_accuracy_score
)

@dataclass
class EvalConfig:
    n_splits: int = 5
    scoring_primary: str = "average_precision"  # ap как главная метрика


class Evaluator:
    """Собственная реализация CV без sklearn scorers — чтобы избежать ошибок с response_method"""
    def __init__(self, cfg: EvalConfig | None = None):
        self.cfg = cfg or EvalConfig()

    def _proba_or_score(self, est, X):
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X)
            if isinstance(p, list):  # на всякий
                p = p[0]
            # бинарная задача -> колонка класса 1
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1]
            # некоторые модели возвращают 1D
            return p.ravel()
        if hasattr(est, "decision_function"):
            s = est.decision_function(X)
            return s.ravel()
        return None

    def cv_score(self, estimator, X, y) -> Dict[str, float]:
        cv = StratifiedKFold(n_splits=self.cfg.n_splits, shuffle=True, random_state=42)

        ap_list, roc_list = [], []
        f1_list, rec_list, prec_list = [], [], []
        acc_list, bacc_list = [], []

        for tr, vl in cv.split(X, y):
            est = clone(estimator)
            Xtr, Xvl = X.iloc[tr] if hasattr(X, "iloc") else X[tr], X.iloc[vl] if hasattr(X, "iloc") else X[vl]
            ytr, yvl = y[tr], y[vl]

            est.fit(Xtr, ytr)

            prob = self._proba_or_score(est, Xvl)
            if prob is not None and prob.ndim == 1:
                try:
                    ap_list.append(average_precision_score(yvl, prob))
                except Exception:
                    ap_list.append(np.nan)
                try:
                    roc_list.append(roc_auc_score(yvl, prob))
                except Exception:
                    roc_list.append(np.nan)
            else:
                ap_list.append(np.nan)
                roc_list.append(np.nan)

            y_pred = est.predict(Xvl)
            f1_list.append(f1_score(yvl, y_pred, zero_division=0))
            rec_list.append(recall_score(yvl, y_pred, zero_division=0))
            prec_list.append(precision_score(yvl, y_pred, zero_division=0))
            acc_list.append(accuracy_score(yvl, y_pred))
            bacc_list.append(balanced_accuracy_score(yvl, y_pred))

        means = {
            "ap": float(np.nanmean(ap_list)),
            "roc": float(np.nanmean(roc_list)),
            "f1": float(np.mean(f1_list)),
            "rec": float(np.mean(rec_list)),
            "prec": float(np.mean(prec_list)),
            "acc": float(np.mean(acc_list)),
            "bacc": float(np.mean(bacc_list)),
        }
        means["primary"] = means["ap"] if self.cfg.scoring_primary == "average_precision" else means["roc"]
        return means
