from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import os
import json
import joblib
import numpy as np
import pandas as pd

from .components import (
    PreprocessConfig,
    ImbalanceConfig,
    ModelConfig,
    PipelineConfig,
    FullPipelineFactory,
)
from .evaluate import Evaluator, EvalConfig
from .logging import RunLogger
from .strategies import BayesianSearch, BanditUCB1, SearchResult
from .visualize import save_decision_graph
from .shap_utils import compute_and_save_shap


@dataclass
class AutoMLConfig:
    strategy: str = "bayes"  # 'bayes' | 'bandit'
    budget: int = 60          # trials or pulls
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
        self._inv_label_mapping: Optional[Dict[int, Any]] = None  # {0: major_label, 1: minor_label}
        self._trial_idx: int = 0  # счётчик трейлов для логирования

    # -------- helpers --------
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
        """
        Возвращает y_bin (0/1) и инвертированный маппинг {0: major_label, 1: minor_label}.
        Минорный класс -> 1 (позитив), мажорный -> 0.
        """
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

    # -------- search spaces (for bandit discrete arms) --------
    def _arms(self, pos_weight: float) -> List[Dict[str, Any]]:
        encs = ["onehot", "ordinal", "target"]
        imps = ["median", "mean", "min"]
        scalers = ["none", "standard", "robust", "minmax", "power"]
        imbs = ["over", "under", "weight", "none"]

        models = [
            ("logreg", {"C": [0.1, 1.0, 10.0]}),
            ("rf", {"n_estimators": [200, 500], "max_depth": [None, 8, 16]}),
            ("gb", {"learning_rate": [0.05, 0.1], "n_estimators": [200, 400]}),
            ("linsvc", {"C": [0.5, 1.0, 2.0]}),
        ]
        try:
            import xgboost  # noqa: F401
            models.append(("xgb", {"max_depth": [3, 5], "n_estimators": [200, 400], "learning_rate": [0.05, 0.1]}))
        except Exception:
            pass
        try:
            import lightgbm  # noqa: F401
            models.append(("lgbm", {"num_leaves": [31, 63], "n_estimators": [200, 400], "learning_rate": [0.05, 0.1]}))
        except Exception:
            pass

        arms: List[Dict[str, Any]] = []
        for e in encs:
            for im in imps:
                for sc in scalers:
                    for imb in imbs:
                        for mname, grid in models:
                            arms.append(
                                {
                                    "encoding": e,
                                    "num_impute": im,
                                    "scaler": sc,
                                    "imbalance_mode": imb,
                                    "model_name": mname,
                                    "model_grid": grid,
                                    "use_class_weight": (imb in ["weight", "none"]),
                                    "pos_weight": pos_weight,
                                }
                            )
        return arms

    def _hyperopt_space(self, pos_weight: float):
        from hyperopt import hp
        names = ["logreg", "rf", "gb"]
        if self._has("xgboost"):
            names.append("xgb")
        if self._has("lightgbm"):
            names.append("lgbm")

        space = {
            "encoding": hp.choice("encoding", ["onehot", "ordinal", "target"]),
            "num_impute": hp.choice("num_impute", ["median", "mean", "min"]),
            "scaler": hp.choice("scaler", ["none", "standard", "robust", "minmax", "power"]),
            "imbalance_mode": hp.choice("imbalance_mode", ["over", "under", "weight", "none"]),
            "model_name": hp.choice("model_name", names),
            "use_class_weight": hp.choice("use_class_weight", [True, False]),
            "pos_weight": pos_weight,
            # model-specific (значения)
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
        }
        return space

    # -------- objective builder --------
    def _make_objective(self, X: pd.DataFrame, y_bin: np.ndarray):
        y_series = pd.Series(y_bin)
        pos_weight = self._compute_pos_weight(y_series)
        eval_metric = self.eval.cfg.scoring_primary

        def build_and_score(params: Dict[str, Any]) -> float:
            # номер текущего трейла
            self._trial_idx += 1
            trial = self._trial_idx

            model_name = params.get("model_name")
            encoding = params.get("encoding")
            num_impute = params.get("num_impute")
            scaler = params.get("scaler")
            imb = params.get("imbalance_mode")

            # per-model params (значения напрямую)
            mparams: Dict[str, Any] = {}
            if model_name == "logreg":
                mparams = {"C": float(params.get("C", 1.0)), "solver": "lbfgs", "max_iter": 5000}
            elif model_name == "rf":
                mparams = {
                    "n_estimators": int(params.get("rf_n_estimators", 200)),
                    "max_depth": params.get("rf_max_depth", None),
                    "n_jobs": -1,
                }
            elif model_name == "gb":
                mparams = {
                    "n_estimators": int(params.get("gb_ne", 200)),
                    "learning_rate": float(params.get("gb_lr", 0.1)),
                }
            elif model_name == "xgb":
                mparams = {
                    "n_estimators": int(params.get("xgb_ne", 200)),
                    "learning_rate": float(params.get("xgb_lr", 0.1)),
                    "max_depth": int(params.get("xgb_depth", 3)),
                    "n_jobs": -1,
                }
            elif model_name == "lgbm":
                mparams = {
                    "n_estimators": int(params.get("lgbm_ne", 200)),
                    "learning_rate": float(params.get("lgbm_lr", 0.1)),
                    "num_leaves": int(params.get("lgbm_leaves", 31)),
                }

            pipe_cfg = PipelineConfig(
                preprocess=PreprocessConfig(encoding=encoding, num_impute=num_impute, scaler=scaler),
                imbalance=ImbalanceConfig(mode=imb),
                model=ModelConfig(
                    name=model_name,
                    params=mparams,
                    use_class_weight=bool(params.get("use_class_weight", True)),
                    pos_weight=pos_weight,
                ),
            )
            pipe = FullPipelineFactory.build(pipe_cfg)
            scores = self.eval.cv_score(pipe, X, y_series)  # используем закодированный y
            score = float(scores.get("primary", np.nan))

            # лог текущего трейла (в MLflow метрики с step, параметры с префиксом t{step}__)
            self.logger.log_metrics({f"cv_{k}": v for k, v in scores.items()}, step=trial)
            self.logger.log_params(
                {
                    "encoding": encoding,
                    "num_impute": num_impute,
                    "scaler": scaler,
                    "imbalance": imb,
                    "model": model_name,
                    **mparams,
                },
                step=trial,
            )

            # сохранить лучшее
            if (self.best_ is None) or (score > self.best_["score"]):
                self.best_ = {
                    "score": score,
                    "metric_name": eval_metric,
                    "config": pipe_cfg,
                    "encoding": encoding,
                    "num_impute": num_impute,
                    "scaler": scaler,
                    "imbalance_mode": imb,
                    "model_name": model_name,
                    "model_params": mparams,
                }
            return score

        return build_and_score

    # -------- public API --------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # 1) Кодируем таргет в 0/1 (минорный -> 1)
        y_bin, inv_map = self._encode_binary_labels(pd.Series(y))
        self._inv_label_mapping = inv_map
        with open(os.path.join(self.cfg.out_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(inv_map, f, ensure_ascii=False, indent=2, default=str)

        # 2) Запускаем поиск
        objective = self._make_objective(X, y_bin)
        if self.cfg.strategy == "bayes":
            space = self._hyperopt_space(self._compute_pos_weight(pd.Series(y_bin)))
            search = BayesianSearch(
                space=space, objective=objective, max_evals=self.cfg.budget, seed=self.cfg.random_state
            )
        elif self.cfg.strategy == "bandit":
            arms = self._arms(self._compute_pos_weight(pd.Series(y_bin)))
            search = BanditUCB1(arms=arms, objective_arm=objective, budget=self.cfg.budget)
        else:
            raise ValueError("strategy must be 'bayes' or 'bandit'")

        _result: SearchResult = search.run()  # история уже логируется в objective

        if self.best_ is None:
            raise RuntimeError("No successful trials")

        # 3) Обучаем лучшую на всех данных с закодированным y
        best_cfg: PipelineConfig = self.best_["config"]
        best_pipe = FullPipelineFactory.build(best_cfg)
        best_pipe.fit(X, y_bin)

        best_path = os.path.join(self.cfg.out_dir, "best_model.joblib")
        joblib.dump(best_pipe, best_path)
        self.logger.log_artifact(best_path)

        dot_path = os.path.join(self.cfg.out_dir, "decision_graph.dot")
        save_decision_graph(
            {**self.best_, "score": self.best_["score"]},
            dot_path
        )
        self.logger.log_artifact(dot_path)

        shap_path = os.path.join(self.cfg.out_dir, "shap.joblib")
        try:
            saved = compute_and_save_shap(best_pipe, X, shap_path)
            if saved:
                self.logger.log_artifact(saved)
        except Exception:
            pass

        with open(os.path.join(self.cfg.out_dir, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: (str(v) if k == "config" else v) for k, v in self.best_.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )

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
