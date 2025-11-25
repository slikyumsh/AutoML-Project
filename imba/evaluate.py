from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, recall_score,
    precision_score, accuracy_score, balanced_accuracy_score
)

@dataclass
class EvalConfig:
    n_splits: int = 5
    scoring_primary: str = "f1"      
    stratified: bool = True
    f1_average: str = "binary"      # binary|macro|weighted

    # --- threshold tuning ---
    tune_threshold: bool = False
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 19        
    threshold_grid: Optional[List[float]] = None


class Evaluator:
    def __init__(self, cfg: EvalConfig | None = None):
        self.cfg = cfg or EvalConfig()

    def _proba_or_score(self, est, X):
        """Return scores aligned to class=1 if possible."""
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X)
            if isinstance(p, list):
                p = p[0]
            if p.ndim == 2 and p.shape[1] >= 2:
                # align to class 1 if classes_ exists
                if hasattr(est, "classes_") and 1 in est.classes_:
                    idx = int(np.where(est.classes_ == 1)[0][0])
                    return p[:, idx]
                return p[:, 1]
            return p.ravel()
        if hasattr(est, "decision_function"):
            s = est.decision_function(X)
            return s.ravel()
        return None

    def _thresholds(self) -> np.ndarray:
        if self.cfg.threshold_grid is not None:
            return np.array(self.cfg.threshold_grid, dtype=float)
        return np.linspace(self.cfg.threshold_min, self.cfg.threshold_max, self.cfg.threshold_steps)

    def _best_threshold_f1(self, y_true, scores) -> tuple[float, float]:
        """Grid search best threshold for F1."""
        thr_grid = self._thresholds()
        best_thr = 0.5
        best_f1 = -1.0
        for thr in thr_grid:
            y_pred = (scores >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, average=self.cfg.f1_average, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        return best_thr, float(best_f1)

    def cv_score(self, estimator, X, y) -> Dict[str, float]:
        if self.cfg.stratified:
            cv = StratifiedKFold(n_splits=self.cfg.n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.cfg.n_splits, shuffle=True, random_state=42)

        ap_list, roc_list = [], []
        f1_list, rec_list, prec_list = [], [], []
        acc_list, bacc_list = [], []
        thr_list = []

        for tr, vl in cv.split(X, y):
            est = clone(estimator)
            Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
            Xvl = X.iloc[vl] if hasattr(X, "iloc") else X[vl]
            ytr, yvl = y[tr], y[vl]

            est.fit(Xtr, ytr)

            scores = self._proba_or_score(est, Xvl)

            # --- threshold tuning ---
            if self.cfg.tune_threshold and scores is not None and np.isfinite(scores).any():
                best_thr, best_f1 = self._best_threshold_f1(yvl, scores)
                thr_list.append(best_thr)
                y_pred = (scores >= best_thr).astype(int)
                f1_list.append(best_f1)
            else:
                y_pred = est.predict(Xvl)
                f1_list.append(f1_score(yvl, y_pred,
                                        average=self.cfg.f1_average,
                                        zero_division=0))

            # proba-metrics
            if scores is not None and scores.ndim == 1:
                try:
                    ap_list.append(average_precision_score(yvl, scores))
                except Exception:
                    ap_list.append(np.nan)
                try:
                    roc_list.append(roc_auc_score(yvl, scores))
                except Exception:
                    roc_list.append(np.nan)
            else:
                ap_list.append(np.nan)
                roc_list.append(np.nan)

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

        if self.cfg.scoring_primary == "f1":
            means["primary"] = means["f1"]
        elif self.cfg.scoring_primary == "roc_auc":
            means["primary"] = means["roc"]
        else:
            means["primary"] = means["ap"]

        means["best_threshold"] = float(np.mean(thr_list)) if thr_list else 0.5
        return means
