from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import os, json, joblib, time

import numpy as np
import pandas as pd
import optuna

from .components import (
    PreprocessConfig, DRConfig, ImbalanceConfig, ModelConfig, PipelineConfig,
    FullPipelineFactory
)
from .evaluate import Evaluator, EvalConfig
from .logging import RunLogger
from .visualize import save_decision_graph
from .shap_utils import compute_and_save_shap


# ---- small wrapper to apply tuned threshold ----
class ThresholdWrapper:
    def __init__(self, base_estimator, threshold: float = 0.5):
        self.base_estimator = base_estimator
        self.threshold = float(threshold)

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        if hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X)
        raise AttributeError("Base estimator has no predict_proba")

    def decision_function(self, X):
        if hasattr(self.base_estimator, "decision_function"):
            return self.base_estimator.decision_function(X)
        raise AttributeError("Base estimator has no decision_function")

    def predict(self, X):
        # Prefer proba, fallback to decision_function
        if hasattr(self.base_estimator, "predict_proba"):
            p = self.base_estimator.predict_proba(X)
            if hasattr(self.base_estimator, "classes_") and 1 in self.base_estimator.classes_:
                idx = int(np.where(self.base_estimator.classes_ == 1)[0][0])
                s = p[:, idx]
            else:
                s = p[:, 1]
        elif hasattr(self.base_estimator, "decision_function"):
            s = np.ravel(self.base_estimator.decision_function(X))
        else:
            # last resort
            return self.base_estimator.predict(X)

        return (s >= self.threshold).astype(int)


@dataclass
class AutoMLConfig:
    budget: int = 60
    out_dir: str = "automl_outputs"
    random_state: int = 42

    ensemble_top_n: int = 3
    use_stacking: bool = False


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
        self.study_: Optional[optuna.Study] = None
        self.history_: List[Dict[str, Any]] = []
        self.best_threshold_: float = 0.5

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

    # -------------------------------------------------------------------------
    # Expanded Optuna search space (valid param names)
    # -------------------------------------------------------------------------
    def _suggest_params(self, trial: optuna.Trial, pos_weight: float) -> PipelineConfig:
        encoding = trial.suggest_categorical("encoding", ["onehot", "ordinal", "target"])
        num_impute = trial.suggest_categorical("num_impute", ["median", "mean", "min"])
        scaler = trial.suggest_categorical("scaler", ["none", "standard", "robust", "minmax", "power"])
        skew_auto = trial.suggest_categorical("skew_auto", [False, True])

        pre_cfg = PreprocessConfig(
            encoding=encoding,
            num_impute=num_impute,
            scaler=scaler,
            skew_auto=skew_auto,
            skew_thr=1.0
        )

        dr_kind = trial.suggest_categorical("dr_kind", ["none", "pca"])
        dr_ratio = 0.8
        if dr_kind == "pca":
            dr_ratio = trial.suggest_float("dr_ratio", 0.3, 0.99)
        dr_cfg = DRConfig(kind=dr_kind, ratio=float(dr_ratio))

        oversample = trial.suggest_categorical("oversample", [False, True])
        undersample = trial.suggest_categorical("undersample", [False, True])
        use_class_weight = trial.suggest_categorical("use_class_weight", [False, True])

        imb_cfg = ImbalanceConfig(
            oversample=oversample,
            undersample=undersample,
            use_class_weight=use_class_weight
        )

        model_names = ["logreg", "rf", "gb"]
        if self._has("xgboost"): model_names.append("xgb")
        if self._has("lightgbm"): model_names.append("lgbm")
        if self._has("catboost"): model_names.append("cat")

        model_name = trial.suggest_categorical("model_name", model_names)
        mparams: Dict[str, Any] = {}

        if model_name == "logreg":
            mparams["C"] = trial.suggest_float("logreg_C", 1e-3, 100.0, log=True)

        elif model_name == "rf":
            mparams["n_estimators"] = trial.suggest_int("rf_n_estimators", 200, 2000)
            mparams["max_depth"] = trial.suggest_categorical(
                "rf_max_depth", [None, 4, 6, 8, 12, 16, 24, 32, 48]
            )
            mparams["min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 20)
            mparams["min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 10)
            mparams["max_features"] = trial.suggest_categorical(
                "rf_max_features", ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0]
            )
            mparams["bootstrap"] = trial.suggest_categorical("rf_bootstrap", [True, False])

        elif model_name == "gb":
            mparams["n_estimators"] = trial.suggest_int("gb_n_estimators", 50, 1000)
            mparams["learning_rate"] = trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True)
            mparams["max_depth"] = trial.suggest_int("gb_max_depth", 2, 5)
            mparams["subsample"] = trial.suggest_float("gb_subsample", 0.5, 1.0)
            mparams["max_features"] = trial.suggest_categorical("gb_max_features", [None, "sqrt", "log2"])

        elif model_name == "xgb":
            mparams["n_estimators"] = trial.suggest_int("xgb_n_estimators", 100, 2000)
            mparams["learning_rate"] = trial.suggest_float("xgb_learning_rate", 0.005, 0.3, log=True)
            mparams["max_depth"] = trial.suggest_int("xgb_max_depth", 2, 12)
            mparams["min_child_weight"] = trial.suggest_float("xgb_min_child_weight", 1.0, 20.0)
            mparams["subsample"] = trial.suggest_float("xgb_subsample", 0.5, 1.0)
            mparams["colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0)
            mparams["gamma"] = trial.suggest_float("xgb_gamma", 0.0, 5.0)
            mparams["reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 1e-6, 5.0, log=True)
            mparams["reg_lambda"] = trial.suggest_float("xgb_reg_lambda", 1e-3, 20.0, log=True)
            mparams["tree_method"] = "hist"

        elif model_name == "lgbm":
            mparams["n_estimators"] = trial.suggest_int("lgbm_n_estimators", 100, 2000)
            mparams["learning_rate"] = trial.suggest_float("lgbm_learning_rate", 0.005, 0.3, log=True)
            mparams["num_leaves"] = trial.suggest_int("lgbm_num_leaves", 15, 255)
            mparams["max_depth"] = trial.suggest_categorical("lgbm_max_depth", [-1, 4, 6, 8, 12, 16])
            mparams["min_child_samples"] = trial.suggest_int("lgbm_min_child_samples", 5, 100)
            mparams["subsample"] = trial.suggest_float("lgbm_subsample", 0.5, 1.0)
            mparams["colsample_bytree"] = trial.suggest_float("lgbm_colsample_bytree", 0.5, 1.0)
            mparams["reg_alpha"] = trial.suggest_float("lgbm_reg_alpha", 1e-6, 5.0, log=True)
            mparams["reg_lambda"] = trial.suggest_float("lgbm_reg_lambda", 1e-3, 20.0, log=True)

        elif model_name == "cat":
            mparams["iterations"] = trial.suggest_int("cat_iterations", 300, 2000)
            mparams["depth"] = trial.suggest_int("cat_depth", 3, 10)
            mparams["learning_rate"] = trial.suggest_float("cat_learning_rate", 0.005, 0.3, log=True)
            mparams["l2_leaf_reg"] = trial.suggest_float("cat_l2_leaf_reg", 1.0, 10.0, log=True)
            mparams["bagging_temperature"] = trial.suggest_float("cat_bagging_temperature", 0.0, 1.0)
            mparams["border_count"] = trial.suggest_int("cat_border_count", 32, 255)

        model_cfg = ModelConfig(
            name=model_name,
            params=mparams,
            use_class_weight=use_class_weight,
            pos_weight=pos_weight
        )

        return PipelineConfig(preprocess=pre_cfg, dr=dr_cfg, imbalance=imb_cfg, model=model_cfg)

    # -------------------------------------------------------------------------
    def _objective(self, X: pd.DataFrame, y_bin: np.ndarray):
        y_series = pd.Series(y_bin)
        pos_weight = self._compute_pos_weight(y_series)
        eval_metric = self.eval.cfg.scoring_primary

        def _obj(trial: optuna.Trial) -> float:
            self._trial_idx += 1
            tnum = self._trial_idx
            cfg = self._suggest_params(trial, pos_weight)

            start = time.time()
            try:
                pipe = FullPipelineFactory.build(cfg, n_features_hint=X.shape[1], n_classes=2)
                scores = self.eval.cv_score(pipe, X, y_series)
                score = float(scores["primary"])
                best_thr = float(scores.get("best_threshold", 0.5))
            except Exception:
                scores = {"ap": np.nan, "roc": np.nan, "f1": np.nan,
                          "rec": np.nan, "prec": np.nan, "acc": np.nan, "bacc": np.nan,
                          "primary": 0.0, "best_threshold": 0.5}
                score = 0.0
                best_thr = 0.5

            elapsed = time.time() - start

            try:
                self.logger.log_metrics({f"cv_{k}": v for k, v in scores.items()}, step=tnum)
                flat_params = {
                    "encoding": cfg.preprocess.encoding,
                    "num_impute": cfg.preprocess.num_impute,
                    "scaler": cfg.preprocess.scaler,
                    "skew_auto": cfg.preprocess.skew_auto,
                    "dr_kind": cfg.dr.kind,
                    "dr_ratio": cfg.dr.ratio,
                    "oversample": cfg.imbalance.oversample,
                    "undersample": cfg.imbalance.undersample,
                    "use_class_weight": cfg.imbalance.use_class_weight,
                    "model_name": cfg.model.name,
                    **cfg.model.params
                }
                self.logger.log_params(flat_params, step=tnum)
            except Exception:
                pass

            self.history_.append({
                "iteration": tnum,
                "score": score,
                "best_threshold": best_thr,
                "metric": eval_metric,
                "elapsed_sec": elapsed,
                "params": trial.params
            })

            if (self.best_ is None) or (score > self.best_["score"]):
                self.best_ = {
                    "score": score,
                    "metric_name": eval_metric,
                    "config": cfg,
                    "trial_params": trial.params,
                    "best_threshold": best_thr
                }

            return score

        return _obj

    # -------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_bin, inv_map = self._encode_binary_labels(pd.Series(y))
        self._inv_label_mapping = inv_map

        with open(os.path.join(self.cfg.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(inv_map, f, ensure_ascii=False, indent=2, default=str)

        objective = self._objective(X, y_bin)

        sampler = optuna.samplers.TPESampler(seed=self.cfg.random_state)
        self.study_ = optuna.create_study(direction="maximize", sampler=sampler)
        self.study_.optimize(objective, n_trials=self.cfg.budget, n_jobs=1)

        if self.best_ is None:
            raise RuntimeError("No successful trials")

        self.best_threshold_ = float(self.best_.get("best_threshold", 0.5))

        trials_sorted = sorted(
            [t for t in self.study_.trials if t.value is not None],
            key=lambda t: t.value, reverse=True
        )
        top_trials = trials_sorted[: max(1, self.cfg.ensemble_top_n)]

        base_estimators = []
        pos_weight = self._compute_pos_weight(pd.Series(y_bin))

        for t in top_trials:
            frozen = t.params

            pre_cfg = PreprocessConfig(
                encoding=frozen["encoding"],
                num_impute=frozen["num_impute"],
                scaler=frozen["scaler"],
                skew_auto=bool(frozen["skew_auto"]),
                skew_thr=1.0
            )
            dr_cfg = DRConfig(kind=frozen["dr_kind"], ratio=float(frozen.get("dr_ratio", 0.8)))

            imb_cfg = ImbalanceConfig(
                oversample=bool(frozen["oversample"]),
                undersample=bool(frozen["undersample"]),
                use_class_weight=bool(frozen["use_class_weight"])
            )

            model_name = frozen["model_name"]
            prefix = model_name + "_"
            mparams = {k[len(prefix):]: v for k, v in frozen.items() if k.startswith(prefix)}

            model_cfg = ModelConfig(
                name=model_name,
                params=mparams,
                use_class_weight=imb_cfg.use_class_weight,
                pos_weight=pos_weight
            )

            cfg_i = PipelineConfig(preprocess=pre_cfg, dr=dr_cfg, imbalance=imb_cfg, model=model_cfg)
            base_estimators.append(
                (f"{model_name}_trial{t.number}",
                 FullPipelineFactory.build(cfg_i, n_features_hint=X.shape[1], n_classes=2))
            )

        from sklearn.ensemble import VotingClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression

        if len(base_estimators) == 1:
            ensemble = base_estimators[0][1]
        else:
            if self.cfg.use_stacking:
                meta = LogisticRegression(max_iter=2000, class_weight="balanced")
                ensemble = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=meta,
                    cv=self.eval.cfg.n_splits
                )
            else:
                ensemble = VotingClassifier(estimators=base_estimators, voting="soft")

        ensemble.fit(X, y_bin)

        # ---- wrap with threshold
        ensemble_thr = ThresholdWrapper(ensemble, threshold=self.best_threshold_)
        best_path = os.path.join(self.cfg.out_dir, "best_model.joblib")
        joblib.dump(ensemble_thr, best_path)
        self.logger.log_artifact(best_path)

        # ---- save history csv+png
        hist_df = pd.DataFrame([{
            "iteration": h["iteration"],
            "score": h["score"],
            "best_threshold": h.get("best_threshold", 0.5),
            "elapsed_sec": h["elapsed_sec"]
        } for h in self.history_])

        csv_path = os.path.join(self.cfg.out_dir, "metrics_vs_iteration.csv")
        hist_df.to_csv(csv_path, index=False)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(hist_df["iteration"], hist_df["score"], marker="o")
        plt.title("F1-score vs Optimization Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("F1-score")
        plt.grid(True)
        plt.tight_layout()
        png_path = os.path.join(self.cfg.out_dir, "metrics_vs_iteration.png")
        plt.savefig(png_path, dpi=160)
        plt.close()

        self.logger.log_artifact(csv_path)
        self.logger.log_artifact(png_path)

        # ---- decision graph
        dot_path = os.path.join(self.cfg.out_dir, "decision_graph.dot")
        save_decision_graph({
            **(self.best_ or {}),
            "dr_kind": self.best_["config"].dr.kind,
            "encoding": self.best_["config"].preprocess.encoding,
            "scaler": self.best_["config"].preprocess.scaler,
            "num_impute": self.best_["config"].preprocess.num_impute,
            "skew_auto": self.best_["config"].preprocess.skew_auto,
            "imbalance_mode": {
                "over": self.best_["config"].imbalance.oversample,
                "under": self.best_["config"].imbalance.undersample,
                "weight": self.best_["config"].imbalance.use_class_weight
            },
            "model_name": self.best_["config"].model.name,
            "best_threshold": self.best_threshold_
        }, dot_path)
        self.logger.log_artifact(dot_path)

        # ---- SHAP
        shap_path = os.path.join(self.cfg.out_dir, "shap.joblib")
        try:
            saved = compute_and_save_shap(ensemble, X, shap_path)  # for shap use raw ensemble
            if saved:
                self.logger.log_artifact(saved)
        except Exception:
            pass

        # ---- final report
        try:
            _ = self._final_report(self.best_["config"], X, y_bin)
            self.logger.log_artifact(os.path.join(self.cfg.out_dir, "final_report.json"))
        except Exception:
            pass

        # ---- save best summary
        best_summary = {
            "best_score": float(self.best_["score"]),
            "metric": self.best_["metric_name"],
            "best_trial_params": self.best_["trial_params"],
            "best_threshold": self.best_threshold_,
            "ensemble_top_n": self.cfg.ensemble_top_n,
            "use_stacking": self.cfg.use_stacking,
        }
        with open(os.path.join(self.cfg.out_dir, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump(best_summary, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.cfg.out_dir, "best_model_params.json"), "w", encoding="utf-8") as f:
            json.dump({
                "best_trial": best_summary,
                "top_trials": [
                    {"number": t.number, "score": float(t.value), "params": t.params}
                    for t in top_trials
                ]
            }, f, ensure_ascii=False, indent=2)

        self.logger.finalize()
        self.fitted_pipeline_ = ensemble_thr
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.fitted_pipeline_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = self.fitted_pipeline_.predict(X)
        if self._inv_label_mapping is not None:
            inv = self._inv_label_mapping
            return np.array([inv.get(int(v), v) for v in y_pred])
        return y_pred
