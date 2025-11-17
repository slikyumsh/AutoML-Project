# Lightweight, composable building blocks for preprocessing, sampling and models
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from inspect import signature

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
)
from sklearn.impute import SimpleImputer

try:
    import category_encoders as ce  # for TargetEncoder
except Exception:
    ce = None

# Imbalance samplers (optional)
try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    RandomOverSampler = None
    RandomUnderSampler = None
    ImbPipeline = Pipeline  # graceful fallback


# ---------------------------
# Smart imputers / encoders / scalers
# ---------------------------
class MinImputer(BaseEstimator, TransformerMixin):
    """Fill numeric NaNs with per-column minimum; pass-through non-numerics."""
    def __init__(self):
        self.mins_: Optional[pd.Series] = None
        self.numeric_cols_: Optional[List[str]] = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        if self.numeric_cols_:
            self.mins_ = X[self.numeric_cols_].min(axis=0)
        else:
            self.mins_ = pd.Series(dtype=float)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.numeric_cols_:
            for c in self.numeric_cols_:
                X[c] = X[c].fillna(self.mins_.get(c, np.nan))
        return X


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Impute numerics and categoricals in a DataFrame, preserving column names & dtypes."""
    def __init__(self, num_strategy: str = "median", cat_strategy: str = "most_frequent"):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.num_imputer_: Optional[TransformerMixin] = None
        self.cat_imputer_: TransformerMixin = SimpleImputer(strategy=cat_strategy)
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.num_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        self.cat_cols_ = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
        # numeric imputer: поддержка 'min'
        if self.num_cols_:
            if self.num_strategy in ("median", "mean"):
                self.num_imputer_ = SimpleImputer(strategy=self.num_strategy)
                self.num_imputer_.fit(X[self.num_cols_])
            elif self.num_strategy == "min":
                self.num_imputer_ = MinImputer()
                self.num_imputer_.fit(X[self.num_cols_])
            else:
                raise ValueError(f"Unknown num_strategy: {self.num_strategy}")
        if self.cat_cols_:
            self.cat_imputer_.fit(X[self.cat_cols_])
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.num_cols_ and self.num_imputer_ is not None:
            X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])
        if self.cat_cols_:
            X[self.cat_cols_] = self.cat_imputer_.transform(X[self.cat_cols_])
        return X


class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    """Target encoding for categorical columns (requires category_encoders)."""
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.cat_cols_: List[str] = []
        self.encoder_ = None

    def fit(self, X, y):
        if ce is None:
            raise ImportError("category_encoders is required for target encoding")
        X = pd.DataFrame(X)
        self.cat_cols_ = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
        if self.cat_cols_:
            self.encoder_ = ce.TargetEncoder(cols=self.cat_cols_, smoothing=self.smoothing)
            self.encoder_.fit(X[self.cat_cols_], y)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.cat_cols_ and self.encoder_ is not None:
            X[self.cat_cols_] = self.encoder_.transform(X[self.cat_cols_])
        return X


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Apply selected scaler to numeric columns in a DataFrame, preserve columns."""
    def __init__(self, kind: str = "none"):
        self.kind = kind
        self.num_cols_: List[str] = []
        self.scaler_: Optional[TransformerMixin] = None

    @staticmethod
    def _make(kind: str):
        if kind == "none" or kind is None:
            return None
        if kind == "standard":
            return StandardScaler()
        if kind == "robust":
            return RobustScaler()
        if kind == "minmax":
            return MinMaxScaler()
        if kind == "power":
            return PowerTransformer(method="yeo-johnson", standardize=True)
        raise ValueError(f"Unknown scaler: {kind}")

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.num_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        self.scaler_ = self._make(self.kind)
        if self.scaler_ is not None and self.num_cols_:
            self.scaler_.fit(X[self.num_cols_])
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.scaler_ is not None and self.num_cols_:
            X[self.num_cols_] = self.scaler_.transform(X[self.num_cols_])
        return X


class DFImputeThenTargetAndScale(BaseEstimator, TransformerMixin):
    """Single transformer: DataFrameImputer -> TargetEncoderWrapper -> DataFrameScaler (no nested Pipelines)."""
    def __init__(self, num_strategy: str = "median", smoothing: float = 1.0, scaler: str = "none"):
        self.num_strategy = num_strategy
        self.smoothing = smoothing
        self.scaler = scaler
        self._imp = DataFrameImputer(num_strategy=num_strategy, cat_strategy="most_frequent")
        self._te = TargetEncoderWrapper(smoothing=smoothing)
        self._sc = DataFrameScaler(kind=scaler)

    def fit(self, X, y):
        X_imp = self._imp.fit_transform(X, y)
        self._te.fit(X_imp, y)
        X_te = self._te.transform(X_imp)
        self._sc.fit(X_te)
        return self

    def transform(self, X):
        X_imp = self._imp.transform(X)
        X_te = self._te.transform(X_imp)
        X_sc = self._sc.transform(X_te)
        return X_sc


# ---------------------------
# Preprocessor factory
# ---------------------------
@dataclass
class PreprocessConfig:
    encoding: str  # 'onehot' | 'ordinal' | 'target'
    num_impute: str  # 'median' | 'mean' | 'min'
    scaler: str = "none"  # 'none' | 'standard' | 'robust' | 'minmax' | 'power'


def _make_onehot():
    """Create OneHotEncoder that is compatible across sklearn versions."""
    params = {"handle_unknown": "ignore"}
    sig = signature(OneHotEncoder.__init__)
    if "sparse_output" in sig.parameters:         # sklearn >= 1.2
        params["sparse_output"] = False
    else:                                         # sklearn <= 1.1
        params["sparse"] = False
    return OneHotEncoder(**params)


def _make_num_imputer(kind: str):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "mean":
        return SimpleImputer(strategy="mean")
    if kind == "min":
        return MinImputer()
    raise ValueError(f"Unknown num_impute: {kind}")


def _make_scaler(kind: str):
    if not kind or kind == "none":
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "power":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    raise ValueError(f"Unknown scaler: {kind}")


class PreprocessorFactory:
    @staticmethod
    def build(cfg: PreprocessConfig):
        # 'target' — единый трансформер без вложенных Pipeline, со скейлингом по выбору
        if cfg.encoding == "target":
            return DFImputeThenTargetAndScale(num_strategy=cfg.num_impute, smoothing=1.0, scaler=cfg.scaler)

        # onehot / ordinal с ColumnTransformer (числа: иммутация -> нативный скейлер)
        num_imputer = _make_num_imputer(cfg.num_impute)
        cat_imputer = SimpleImputer(strategy="most_frequent")

        if cfg.encoding == "onehot":
            encoder = _make_onehot()
        elif cfg.encoding == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        else:
            raise ValueError(f"Unknown encoding: {cfg.encoding}")

        num_steps = [("imp", num_imputer)]
        scaler_obj = _make_scaler(cfg.scaler)
        if scaler_obj is not None:
            num_steps.append(("scale", scaler_obj))
        num_pipe = Pipeline(num_steps)

        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, make_column_selector(dtype_include=[np.number])),
                ("cat", Pipeline([("imp", cat_imputer), ("enc", encoder)]),
                 make_column_selector(dtype_include=["object", "category", "bool"])),
            ],
            remainder="drop",
        )
        return pre  # важно: единый трансформер, без внешнего Pipeline


# ---------------------------
# Sampler factory
# ---------------------------
@dataclass
class ImbalanceConfig:
    mode: str  # 'none' | 'over' | 'under' | 'weight'


class SamplerFactory:
    @staticmethod
    def build(cfg: ImbalanceConfig):
        if cfg.mode == "over":
            if RandomOverSampler is None:
                raise ImportError("imblearn is required for oversampling")
            return RandomOverSampler()
        if cfg.mode == "under":
            if RandomUnderSampler is None:
                raise ImportError("imblearn is required for undersampling")
            return RandomUnderSampler()
        return None  # 'none' or 'weight' handled inside models


# ---------------------------
# Model factory
# ---------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None


@dataclass
class ModelConfig:
    name: str  # 'logreg' | 'rf' | 'gb' | 'linsvc' | 'xgb' | 'lgbm'
    params: Dict[str, Any]
    use_class_weight: bool = True
    pos_weight: Optional[float] = None  # for xgb/lgbm


class ModelFactory:
    @staticmethod
    def build(cfg: ModelConfig):
        n = cfg.name
        p = dict(cfg.params)

        if n == "logreg":
            if cfg.use_class_weight:
                p.setdefault("class_weight", "balanced")
            p.setdefault("solver", "lbfgs")
            p.setdefault("max_iter", 5000)
            return LogisticRegression(**p)

        if n == "rf":
            if cfg.use_class_weight:
                p.setdefault("class_weight", "balanced")
            return RandomForestClassifier(**p)

        if n == "gb":
            return GradientBoostingClassifier(**p)

        if n == "linsvc":
            if cfg.use_class_weight:
                p.setdefault("class_weight", "balanced")
            return LinearSVC(**p)

        if n == "xgb":
            if xgb is None:
                raise ImportError("xgboost not installed")
            if cfg.pos_weight is not None:
                p.setdefault("scale_pos_weight", cfg.pos_weight)
            p.setdefault("eval_metric", "logloss")
            p.setdefault("n_jobs", -1)
            return xgb.XGBClassifier(**p)

        if n == "lgbm":
            if lgb is None:
                raise ImportError("lightgbm not installed")
            if cfg.pos_weight is not None:
                p.setdefault("scale_pos_weight", cfg.pos_weight)
            # тихие и мягкие дефолты под маленькие датасеты
            p.setdefault("verbose", -1)
            p.setdefault("min_child_samples", 5)
            p.setdefault("min_data_in_leaf", 5)
            p.setdefault("min_data_in_bin", 3)
            p.setdefault("feature_fraction", 1.0)
            p.setdefault("bagging_fraction", 1.0)
            p.setdefault("bagging_freq", 0)
            p.setdefault("num_leaves", p.get("num_leaves", 31))
            return lgb.LGBMClassifier(**p)

        raise ValueError(f"Unknown model: {n}")


# ---------------------------
# End-to-end pipeline factory (preproc + sampler + model)
# ---------------------------
@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig
    imbalance: ImbalanceConfig
    model: ModelConfig


class FullPipelineFactory:
    @staticmethod
    def build(cfg: PipelineConfig):
        pre = PreprocessorFactory.build(cfg.preprocess)
        sampler = SamplerFactory.build(cfg.imbalance)
        est = ModelFactory.build(cfg.model)
        if sampler is not None and ImbPipeline is not Pipeline:
            return ImbPipeline([
                ("pre", pre),
                ("sampler", sampler),
                ("clf", est),
            ])
        return Pipeline([
            ("pre", pre),
            ("clf", est),
        ])
