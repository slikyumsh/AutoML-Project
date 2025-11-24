from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# --- Configs ------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    encoding: str = "onehot"       # onehot | ordinal | target
    num_impute: str = "median"     # median | mean | min
    scaler: str = "none"           # none | standard | robust | minmax | power
    skew_auto: bool = False        # если True, применяем PowerTransformer к числовым фичам
    skew_thr: float = 1.0          # (не используется в этой версии, интерфейс сохранён)


@dataclass
class DRConfig:
    kind: str = "none"             # none | pca
    ratio: float = 0.8             # для pca — доля объяснённой дисперсии


@dataclass
class ImbalanceConfig:
    mode: str = "none"             # none | over | under | weight


@dataclass
class ModelConfig:
    name: str = "logreg"
    params: Dict[str, Any] = None
    use_class_weight: bool = True
    pos_weight: float = 1.0


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig
    dr: DRConfig
    imbalance: ImbalanceConfig
    model: ModelConfig


# --- Helpers ------------------------------------------------------------------

def _num_cat_columns(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


class ColumnMinImputer(BaseEstimator, TransformerMixin):
    """Импутация числовых колонок их минимумами (без NaN)."""
    def __init__(self):
        self.mins_: Optional[pd.Series] = None
        self.columns_: Optional[list[str]] = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = X.columns.tolist()
        self.mins_ = X.min(axis=0, skipna=True)
        # если колонка целиком NaN — подставим 0
        self.mins_ = self.mins_.fillna(0.0)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.fillna(self.mins_)
        # сохранить порядок столбцов
        return X[self.columns_].values


def _build_numeric_imputer(strategy: str):
    if strategy == "median":
        return SimpleImputer(strategy="median")
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    if strategy == "min":
        return ColumnMinImputer()
    # fallback
    return SimpleImputer(strategy="median")


def _build_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "power":
        # Yeo-Johnson; корректно работает на произвольных значениях
        return PowerTransformer(standardize=True)
    return "passthrough"


def _build_encoder(kind: str, cat_cols: list[str]):
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if kind == "target":
        try:
            import category_encoders as ce
        except Exception:
            # запасной вариант — ordinal
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        # TargetEncoder работает именно на категориальных колонках
        return ce.TargetEncoder(cols=cat_cols)
    # default
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


# --- Preprocessor -------------------------------------------------------------

class Preprocessor(BaseEstimator, TransformerMixin):
    """Колонно-ориентированный препроцессор: импутация, энкодинг, скейлинг, опционально коррекция скоса."""
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.num_cols_: Optional[list[str]] = None
        self.cat_cols_: Optional[list[str]] = None
        self.ct_: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        self.num_cols_, self.cat_cols_ = _num_cat_columns(X)

        # числовой блок
        num_steps = []
        if self.cfg.skew_auto:
            num_steps.append(("skew", PowerTransformer(standardize=True)))
        num_steps.append(("impute", _build_numeric_imputer(self.cfg.num_impute)))
        scaler = _build_scaler(self.cfg.scaler)
        if scaler != "passthrough":
            num_steps.append(("scale", scaler))
        from sklearn.pipeline import Pipeline as SkPipeline
        num_pipe = SkPipeline(num_steps) if len(num_steps) > 1 else (num_steps[0][1] if num_steps else "passthrough")

        # категориальный блок
        cat_imputer = SimpleImputer(strategy="most_frequent")
        encoder = _build_encoder(self.cfg.encoding, self.cat_cols_)

        # ColumnTransformer
        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.num_cols_),
                ("cat", (encoder if self.cfg.encoding != "target" else encoder), self.cat_cols_)
            ],
            remainder="drop",
            sparse_threshold=0.0,
            n_jobs=None,
        )

        # важно: если target encoding — он учит по y
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return self.ct_.transform(X)


# --- DR (только PCA) ----------------------------------------------------------

def _build_dr(dr_cfg: DRConfig):
    if dr_cfg.kind == "pca":
        return PCA(n_components=dr_cfg.ratio, svd_solver="full", random_state=42)
    return None


# --- Model factory ------------------------------------------------------------

def _build_model(mcfg: ModelConfig, n_classes: int = 2):
    name = mcfg.name
    params = dict(mcfg.params or {})
    use_w = bool(mcfg.use_class_weight)
    pos_w = float(mcfg.pos_weight)

    def _maybe_add_class_weight(p: Dict[str, Any]):
        if not use_w:
            return p
        # sklearn модели
        if name in ("logreg", "rf", "gb"):
            p = dict(p)
            p["class_weight"] = "balanced"
        return p

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        p = {"solver": "lbfgs", "max_iter": 5000}
        p.update(params)
        p = _maybe_add_class_weight(p)
        return LogisticRegression(**p)

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        p = {"n_estimators": 200, "n_jobs": -1, "random_state": 42}
        p.update(params)
        p = _maybe_add_class_weight(p)
        return RandomForestClassifier(**p)

    if name == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        p = {"n_estimators": 200, "learning_rate": 0.1, "random_state": 42}
        p.update(params)
        # у GBC нет class_weight
        return GradientBoostingClassifier(**p)

    if name == "xgb":
        from xgboost import XGBClassifier
        p = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3, "n_jobs": -1, "eval_metric": "logloss", "random_state": 42}
        p.update(params)
        if use_w:
            p["scale_pos_weight"] = pos_w
        return XGBClassifier(**p)

    if name == "lgbm":
        from lightgbm import LGBMClassifier
        p = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31, "random_state": 42, "verbose": -1}
        p.update(params)
        if use_w:
            p["class_weight"] = {0: 1.0, 1: pos_w}
        return LGBMClassifier(**p)

    if name == "cat":
        from catboost import CatBoostClassifier
        p = {"depth": 6, "learning_rate": 0.1, "iterations": 300, "verbose": 0, "random_state": 42}
        p.update(params)
        if use_w:
            # веса классов в виде списка [w0, w1]
            p["class_weights"] = [1.0, pos_w]
        return CatBoostClassifier(**p)

    if name in ("vote_soft", "vote_hard", "stack_lr"):
        # базовые слабые модели (без перевеса классов внутри ансамбля — перевес делаем общим режимом)
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
        base_lr = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
        base_rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        base_gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.08, random_state=42)

        if name.startswith("vote"):
            voting = "soft" if name == "vote_soft" else "hard"
            return VotingClassifier(
                estimators=[("lr", base_lr), ("rf", base_rf), ("gb", base_gb)],
                voting=voting,
                n_jobs=None
            )
        else:
            final_lr = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
            return StackingClassifier(
                estimators=[("lr", base_lr), ("rf", base_rf), ("gb", base_gb)],
                final_estimator=final_lr,
                passthrough=False,
                n_jobs=None
            )

    raise ValueError(f"Unknown model name: {name}")


# --- Full pipeline ------------------------------------------------------------

class FullPipelineFactory:
    @staticmethod
    def build(cfg: PipelineConfig, n_features_hint: int = 0, n_classes: int = 2):
        steps = []

        # preprocess
        pre = Preprocessor(cfg.preprocess)
        steps.append(("pre", pre))

        # dimensionality reduction (только PCA)
        dr_est = _build_dr(cfg.dr)
        if dr_est is not None:
            steps.append(("dr", dr_est))

        # imbalance
        imb = cfg.imbalance.mode
        if imb == "over":
            steps.append(("imb", RandomOverSampler(random_state=42)))
        elif imb == "under":
            steps.append(("imb", RandomUnderSampler(random_state=42)))
        # 'weight' и 'none' — ничего не вставляем, это на уровне модели

        # model
        model = _build_model(cfg.model, n_classes=n_classes)
        steps.append(("model", model))

        return ImbPipeline(steps, memory=None, verbose=False)
