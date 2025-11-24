from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import os, json, joblib
import numpy as np
import pandas as pd

from .components import (
    PreprocessConfig, DRConfig, ImbalanceConfig, ModelConfig, PipelineConfig,
    FullPipelineFactory
)
from .evaluate import Evaluator, EvalConfig
from .logging import RunLogger
from .strategies import BayesianSearch, BanditUCB1, SearchResult
from .visualize import save_decision_graph
from .shap_utils import compute_and_save_shap


@dataclass
class AutoMLConfig:
    strategy: str = "bayes"
    budget: int = 60
    out_dir: str = "automl_outputs"
    random_state: int = 42


class AutoMLBinary:
    def __init__(self, config: AutoMLConfig, eval_cfg: Optional[EvalConfig] = None):
        self.cfg = config
        self.eval = Evaluator(eval_cfg or EvalConfig())
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        self.logger = RunLogger(self.cfg.out_dir)
        self.best_: Optional[Dict[str, Any]] = None
        self.fitted_pipeline_ = None
        self._inv_label_mapping: Optional[Dict[int, Any]] = None
        self._trial_idx: int = 0

    @staticmethod
    def _has(lib: str) -> bool:
        try:
            __import__(lib)
            return True
        except Exception:
            return False

    @staticmethod
    def _compute_pos_weight(y: pd.Series) -> float:
        vc = pd.Series(y).value_counts(dropna=False)
        if len(vc) != 2:
            return 1.0
        n_pos = float(vc.min()) or 1.0
        n_neg = float(vc.max())
        return n_neg / n_pos

    @staticmethod
    def _encode_binary_labels(y: pd.Series) -> Tuple[np.ndarray, Dict[int, Any]]:
        ys = pd.Series(y)
        vc = ys.value_counts(dropna=False)
        if len(vc) != 2:
            raise ValueError(f"Target must be binary, got {len(vc)} unique values.")
        major_label = vc.idxmax()
        minor_label = vc.idxmin()
        to_bin = {major_label: 0, minor_label: 1}
        inv = {0: major_label, 1: minor_label}
        y_bin = ys.map(to_bin).astype(int).values
        if np.isnan(y_bin.astype(float)).any():
            raise ValueError("Found unmapped labels during encoding.")
        return y_bin, inv

    # --------- arms and spaces (LDA убран) ---------
    def _arms(self, pos_weight: float) -> List[Dict[str, Any]]:
        encs = ["onehot", "ordinal", "target"]
        imps = ["median", "mean", "min"]
        scalers = ["none", "standard", "robust", "minmax", "power"]
        skews = [False, True]
        drs = [("none", 1.0), ("pca", 0.5), ("pca", 0.8)]
        imbs = ["over", "under", "weight", "none"]

        models = [
            ("logreg", {"C": [0.1, 1.0, 10.0]}),
            ("rf", {"n_estimators": [200, 500], "max_depth": [None, 8, 16]}),
            ("gb", {"learning_rate": [0.05, 0.1], "n_estimators": [200, 400]}),
            ("vote_soft", {}), ("vote_hard", {}), ("stack_lr", {}),
        ]
        if self._has("xgboost"):
            models.append(("xgb", {"max_depth": [3, 5], "n_estimators": [200, 400], "learning_rate": [0.05, 0.1]}))
        if self._has("lightgbm"):
            models.append(("lgbm", {"num_leaves": [31, 63], "n_estimators": [200, 400], "learning_rate": [0.05, 0.1]}))
        if self._has("catboost"):
            models.append(("cat", {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "iterations": [300, 600]}))

        arms: List[Dict[str, Any]] = []
        for e in encs:
            for im in imps:
                for sc in scalers:
                    for sk in skews:
                        for imb in imbs:
                            for (drk, drr) in drs:
                                for mname, grid in models:
                                    arms.append({
                                        "encoding": e, "num_impute": im, "scaler": sc,
                                        "skew_auto": sk, "skew_thr": 1.0,
                                        "imbalance_mode": imb,
                                        "dr_kind": drk, "dr_ratio": drr,
                                        "model_name": mname, "model_grid": grid,
                                        "use_class_weight": (imb in ["weight", "none"]),
                                        "pos_weight": pos_weight,
                                    })
        return arms

    def _hyperopt_space(self, pos_weight: float):
        from hyperopt import hp
        names = ["logreg", "rf", "gb", "vote_soft", "vote_hard", "stack_lr"]
        if self._has("xgboost"): names.append("xgb")
        if self._has("lightgbm"): names.append("lgbm")
        if self._has("catboost"): names.append("cat")

        space = {
            "encoding": hp.choice("encoding", ["onehot", "ordinal", "target"]),
            "num_impute": hp.choice("num_impute", ["median", "mean", "min"]),
            "scaler": hp.choice("scaler", ["none", "standard", "robust", "minmax", "power"]),
            "skew_auto": hp.choice("skew_auto", [False, True]),
            "skew_thr": hp.choice("skew_thr", [0.8, 1.0, 1.5]),
            "imbalance_mode": hp.choice("imbalance_mode", ["over", "under", "weight", "none"]),
            "dr_kind": hp.choice("dr_kind", ["none", "pca"]),          # LDA убран
            "dr_ratio": hp.uniform("dr_ratio", 0.4, 0.95),
            "model_name": hp.choice("model_name", names),
            "use_class_weight": hp.choice("use_class_weight", [True, False]),
            "pos_weight": pos_weight,
            # model-specific
            "C": hp.loguniform("C", np.log(0.05), np.log(20.0)),
            "rf_n_estimators": hp.choice("rf_n_estimators", [200, 500]),
            "rf_max_depth": hp.choice("rf_max_depth", [None, 8, 16]),
            "gb_lr": hp.choice("gb_lr", [0.05, 0.1]),
            "gb_ne": hp.choice("gb_ne", [200, 400]),
            "xgb_depth": hp.choice("xgb_depth", [3, 5]),
            "xgb_ne": hp.choice("xgb_ne", [200, 400]),
            "xgb_lr": hp.choice("xgb_lr", [0.05, 0.1]),
            "lgbm_leaves": hp.choice("lgbm_leaves", [31, 63]),
            "lgbm_ne": hp.choice("lgbm_ne", [200, 400]),
            "lgbm_lr": hp.choice("lgbm_lr", [0.05, 0.1]),
            "cat_depth": hp.choice("cat_depth", [4, 6, 8]),
            "cat_lr": hp.choice("cat_lr", [0.03, 0.1]),
            "cat_iter": hp.choice("cat_iter", [300, 600]),
        }
        return space

    # --------- objective ---------
    def _make_objective(self, X: pd.DataFrame, y_bin: np.ndarray):
        y_series = pd.Series(y_bin)
        pos_weight = self._compute_pos_weight(y_series)
        eval_metric = self.eval.cfg.scoring_primary

        def build_and_score(params: Dict[str, Any]) -> float:
            self._trial_idx += 1
            trial = self._trial_idx
            model_name = params.get("model_name")
            encoding = params.get("encoding"); num_impute = params.get("num_impute")
            scaler = params.get("scaler"); imb = params.get("imbalance_mode")
            skew_auto = bool(params.get("skew_auto", False)); skew_thr = float(params.get("skew_thr", 1.0))
            dr_kind = params.get("dr_kind"); dr_ratio = float(params.get("dr_ratio", 0.8))

            # per-model
            mparams: Dict[str, Any] = {}
            if model_name == "logreg":
                mparams = {"C": float(params.get("C", 1.0)), "solver": "lbfgs", "max_iter": 5000}
            elif model_name == "rf":
                mparams = {"n_estimators": int(params.get("rf_n_estimators", 200)),
                           "max_depth": params.get("rf_max_depth", None), "n_jobs": -1, "random_state": 42}
            elif model_name == "gb":
                mparams = {"n_estimators": int(params.get("gb_ne", 200)),
                           "learning_rate": float(params.get("gb_lr", 0.1)), "random_state": 42}
            elif model_name == "xgb":
                mparams = {"n_estimators": int(params.get("xgb_ne", 200)),
                           "learning_rate": float(params.get("xgb_lr", 0.1)),
                           "max_depth": int(params.get("xgb_depth", 3)), "n_jobs": -1, "eval_metric": "logloss", "random_state": 42}
            elif model_name == "lgbm":
                mparams = {"n_estimators": int(params.get("lgbm_ne", 200)),
                           "learning_rate": float(params.get("lgbm_lr", 0.1)),
                           "num_leaves": int(params.get("lgbm_leaves", 31)), "random_state": 42}
            elif model_name == "cat":
                mparams = {"depth": int(params.get("cat_depth", 6)),
                           "learning_rate": float(params.get("cat_lr", 0.1)),
                           "iterations": int(params.get("cat_iter", 300)), "verbose": 0, "random_state": 42}

            pre_cfg = PreprocessConfig(encoding=encoding, num_impute=num_impute, scaler=scaler,
                                       skew_auto=skew_auto, skew_thr=skew_thr)
            dr_cfg = DRConfig(kind=dr_kind, ratio=dr_ratio)
            pipe_cfg = PipelineConfig(
                preprocess=pre_cfg,
                dr=dr_cfg,
                imbalance=ImbalanceConfig(mode=imb),
                model=ModelConfig(name=model_name, params=mparams,
                                  use_class_weight=bool(params.get("use_class_weight", True)),
                                  pos_weight=pos_weight)
            )

            try:
                pipe = FullPipelineFactory.build(pipe_cfg, n_features_hint=X.shape[1], n_classes=2)
                scores = self.eval.cv_score(pipe, X, y_series)
                score = float(scores.get("primary", np.nan))
            except Exception:
                scores = {"ap": np.nan, "roc": np.nan, "f1": np.nan,
                          "rec": np.nan, "prec": np.nan, "acc": np.nan, "bacc": np.nan, "primary": np.nan}
                score = 0.0

            # логируем (параметры могут повторяться между трейлами — добавляем step)
            try:
                self.logger.log_metrics({f"cv_{k}": v for k, v in scores.items()}, step=trial)
                self.logger.log_params({
                    "encoding": encoding, "num_impute": num_impute, "scaler": scaler,
                    "skew_auto": skew_auto, "skew_thr": skew_thr,
                    "imbalance": imb, "dr_kind": dr_kind, "dr_ratio": dr_ratio,
                    "model": model_name, **mparams
                }, step=trial)
            except Exception:
                pass

            if (self.best_ is None) or (score > self.best_["score"]):
                self.best_ = {
                    "score": score, "metric_name": eval_metric, "config": pipe_cfg,
                    "encoding": encoding, "num_impute": num_impute, "scaler": scaler,
                    "skew_auto": skew_auto, "imbalance_mode": imb, "dr_kind": dr_kind,
                    "model_name": model_name, "model_params": mparams
                }
            return score

        return build_and_score

    # --------- final report (без cross_val_predict) ---------
    def _final_report(self, cfg: PipelineConfig, X: pd.DataFrame, y_bin: np.ndarray):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, accuracy_score,
            roc_auc_score, average_precision_score, confusion_matrix, classification_report
        )
        cv = StratifiedKFold(n_splits=self.eval.cfg.n_splits, shuffle=True, random_state=self.cfg.random_state)

        y_true_all, y_pred_all, prob_all = [], [], []

        def proba_or_score(est, X_):
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X_)
                p = p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else np.ravel(p)
                return p
            if hasattr(est, "decision_function"):
                return np.ravel(est.decision_function(X_))
            return None

        for tr, vl in cv.split(X, y_bin):
            est = clone(FullPipelineFactory.build(cfg, n_features_hint=X.shape[1], n_classes=2))
            Xtr, Xvl = X.iloc[tr] if hasattr(X, "iloc") else X[tr], X.iloc[vl] if hasattr(X, "iloc") else X[vl]
            ytr, yvl = y_bin[tr], y_bin[vl]
            est.fit(Xtr, ytr)
            y_pred = est.predict(Xvl)
            y_prob = proba_or_score(est, Xvl)

            y_true_all.append(yvl)
            y_pred_all.append(y_pred)
            prob_all.append(y_prob if y_prob is not None else np.full_like(yvl, np.nan, dtype=float))

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        prob = np.concatenate(prob_all)

        rep = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, prob)) if np.isfinite(prob).any() else float("nan"),
            "average_precision": float(average_precision_score(y_true, prob)) if np.isfinite(prob).any() else float("nan"),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        }
        with open(os.path.join(self.cfg.out_dir, "final_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        return rep

    # --------- public API ---------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_bin, inv_map = self._encode_binary_labels(pd.Series(y))
        self._inv_label_mapping = inv_map
        with open(os.path.join(self.cfg.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(inv_map, f, ensure_ascii=False, indent=2, default=str)

        objective = self._make_objective(X, y_bin)
        if self.cfg.strategy == "bayes":
            space = self._hyperopt_space(self._compute_pos_weight(pd.Series(y_bin)))
            search = BayesianSearch(space=space, objective=objective, max_evals=self.cfg.budget, seed=self.cfg.random_state)
        elif self.cfg.strategy == "bandit":
            arms = self._arms(self._compute_pos_weight(pd.Series(y_bin)))
            search = BanditUCB1(arms=arms, objective_arm=objective, budget=self.cfg.budget)
        else:
            raise ValueError("strategy must be 'bayes' or 'bandit'")
        _result: SearchResult = search.run()

        if self.best_ is None:
            raise RuntimeError("No successful trials")

        best_cfg: PipelineConfig = self.best_["config"]
        best_pipe = FullPipelineFactory.build(best_cfg, n_features_hint=X.shape[1], n_classes=2)
        best_pipe.fit(X, y_bin)
        best_path = os.path.join(self.cfg.out_dir, "best_model.joblib")
        joblib.dump(best_pipe, best_path)
        self.logger.log_artifact(best_path)

        # граф решений
        dot_path = os.path.join(self.cfg.out_dir, "decision_graph.dot")
        save_decision_graph({**self.best_, "dr_kind": best_cfg.dr.kind}, dot_path)
        self.logger.log_artifact(dot_path)

        # SHAP
        shap_path = os.path.join(self.cfg.out_dir, "shap.joblib")
        try:
            saved = compute_and_save_shap(best_pipe, X, shap_path)
            if saved: self.logger.log_artifact(saved)
        except Exception:
            pass

        # Final metrics report
        try:
            _ = self._final_report(best_cfg, X, y_bin)
            self.logger.log_artifact(os.path.join(self.cfg.out_dir, "final_report.json"))
        except Exception:
            pass

        with open(os.path.join(self.cfg.out_dir, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump({k: (str(v) if k == "config" else v) for k, v in self.best_.items()}, f, ensure_ascii=False, indent=2)

        self.logger.finalize()
        self.fitted_pipeline_ = best_pipe
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.fitted_pipeline_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.fitted_pipeline_.predict(X)
        if self._inv_label_mapping is not None:
            inv = self._inv_label_mapping
            return np.array([inv.get(int(v), v) for v in y_pred])
        return y_pred
