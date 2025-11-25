from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
from inspect import signature

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler,
    RobustScaler, MinMaxScaler, PowerTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# optional libs
try:
    import category_encoders as ce
except Exception:
    ce = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    encoding: str = "onehot"       # onehot | ordinal | target
    num_impute: str = "median"     # median | mean | min
    scaler: str = "none"           # none | standard | robust | minmax | power
    skew_auto: bool = False        # if True, apply PowerTransformer to numeric cols
    skew_thr: float = 1.0          # interface kept, not used directly


@dataclass
class DRConfig:
    kind: str = "none"             # none | pca
    ratio: float = 0.8             # for pca: explained variance ratio


@dataclass
class ImbalanceConfig:
    oversample: bool = False
    undersample: bool = False
    use_class_weight: bool = False
    over_kind: str = "random"      # random | smoteenn (if both over+under)
    under_kind: str = "random"     # reserved for future


@dataclass
class ModelConfig:
    name: str = "logreg"           # logreg|rf|gb|linsvc|xgb|lgbm|cat
    params: Dict[str, Any] = field(default_factory=dict)
    use_class_weight: bool = True
    pos_weight: Optional[float] = None   # for xgb/lgbm


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig
    dr: DRConfig
    imbalance: ImbalanceConfig
    model: ModelConfig


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _num_cat_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


class ColumnMinImputer(BaseEstimator, TransformerMixin):
    """Impute numerical columns with their (non-NaN) minimum."""
    def __init__(self):
        self.mins_: Optional[pd.Series] = None
        self.columns_: Optional[List[str]] = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = X.columns.tolist()
        mins = X.min(axis=0, skipna=True)
        self.mins_ = mins.fillna(0.0)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.fillna(self.mins_)


def _make_onehot():
    """Create OneHotEncoder compatible with sklearn <=1.1 and >=1.2."""
    params = {"handle_unknown": "ignore"}
    sig = signature(OneHotEncoder.__init__)
    if "sparse_output" in sig.parameters:  # sklearn >= 1.2
        params["sparse_output"] = False
    else:                                  # sklearn <= 1.1
        params["sparse"] = False
    return OneHotEncoder(**params)


def _build_num_imputer(kind: str):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "mean":
        return SimpleImputer(strategy="mean")
    if kind == "min":
        return ColumnMinImputer()
    return SimpleImputer(strategy="median")


def _build_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "power":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    return "passthrough"


def _build_encoder(kind: str, cat_cols: List[str]):
    if kind == "onehot":
        return _make_onehot()
    if kind == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if kind == "target":
        if ce is None:
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        return ce.TargetEncoder(cols=cat_cols, smoothing=1.0)
    return _make_onehot()


# -----------------------------------------------------------------------------
# Preprocessor
# -----------------------------------------------------------------------------

class Preprocessor(BaseEstimator, TransformerMixin):
    """Column-wise preprocessing: impute, encode, scale, optional skew fix."""
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.num_cols_: Optional[List[str]] = None
        self.cat_cols_: Optional[List[str]] = None
        self.ct_: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        self.num_cols_, self.cat_cols_ = _num_cat_columns(X)

        num_steps = []
        if self.cfg.skew_auto:
            num_steps.append(("skew", PowerTransformer(method="yeo-johnson", standardize=True)))
        num_steps.append(("impute", _build_num_imputer(self.cfg.num_impute)))
        scaler = _build_scaler(self.cfg.scaler)
        if scaler != "passthrough":
            num_steps.append(("scale", scaler))
        num_pipe = SkPipeline(num_steps)

        cat_steps = [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _build_encoder(self.cfg.encoding, self.cat_cols_)),
        ]
        cat_pipe = SkPipeline(cat_steps)

        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.num_cols_),
                ("cat", cat_pipe, self.cat_cols_),
            ],
            remainder="drop",
        )
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        if self.ct_ is None:
            raise RuntimeError("Preprocessor is not fitted.")
        X = pd.DataFrame(X).copy()
        return self.ct_.transform(X)


# -----------------------------------------------------------------------------
# Dimensionality Reduction factory
# -----------------------------------------------------------------------------

class DimReductionFactory:
    @staticmethod
    def build(cfg: DRConfig):
        if cfg.kind == "pca":
            return PCA(n_components=cfg.ratio, svd_solver="full", random_state=42)
        return None


# -----------------------------------------------------------------------------
# Sampler factory (supports combinations)
# -----------------------------------------------------------------------------

class SamplerFactory:
    @staticmethod
    def build(cfg: ImbalanceConfig):
        if cfg.oversample and cfg.undersample:
            return SMOTEENN(random_state=42)
        if cfg.oversample:
            return RandomOverSampler(random_state=42)
        if cfg.undersample:
            return RandomUnderSampler(random_state=42)
        return None


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

class ModelFactory:
    @staticmethod
    def build(cfg: ModelConfig):
        n = cfg.name
        p = dict(cfg.params or {})

        if n == "logreg":
            if cfg.use_class_weight:
                p.setdefault("class_weight", "balanced")
            p.setdefault("solver", "lbfgs")
            p.setdefault("max_iter", 5000)
            return LogisticRegression(**p)

        if n == "rf":
            if cfg.use_class_weight:
                p.setdefault("class_weight", "balanced")
            p.setdefault("n_estimators", 200)
            p.setdefault("n_jobs", -1)
            p.setdefault("random_state", 42)
            return RandomForestClassifier(**p)

        if n == "gb":
            p.setdefault("n_estimators", 200)
            p.setdefault("learning_rate", 0.1)
            p.setdefault("random_state", 42)
            return GradientBoostingClassifier(**p)

        if n == "linsvc":
            if cfg.use_class_weight:
                p.setdefault("class_weight", "balanced")
            p.setdefault("random_state", 42)
            return LinearSVC(**p)

        if n == "xgb":
            if xgb is None:
                raise ImportError("xgboost not installed")
            if cfg.pos_weight is not None:
                p.setdefault("scale_pos_weight", float(cfg.pos_weight))
            p.setdefault("eval_metric", "logloss")
            p.setdefault("n_jobs", -1)
            p.setdefault("random_state", 42)
            return xgb.XGBClassifier(**p)

        if n == "lgbm":
            if lgb is None:
                raise ImportError("lightgbm not installed")
            if cfg.pos_weight is not None:
                p.setdefault("scale_pos_weight", float(cfg.pos_weight))
            if cfg.use_class_weight:
                p.setdefault("class_weight", "balanced")
            p.setdefault("random_state", 42)
            p.setdefault("verbose", -1)
            return lgb.LGBMClassifier(**p)

        if n == "cat":
            if CatBoostClassifier is None:
                raise ImportError("catboost not installed")
            if cfg.use_class_weight:
                p.setdefault("auto_class_weights", "Balanced")
            p.setdefault("verbose", 0)
            p.setdefault("random_state", 42)
            return CatBoostClassifier(**p)

        raise ValueError(f"Unknown model: {n}")


# -----------------------------------------------------------------------------
# End-to-end pipeline factory (preproc + dr + sampler + model)
# -----------------------------------------------------------------------------

class FullPipelineFactory:
    @staticmethod
    def build(cfg: PipelineConfig, **kwargs):
        pre = Preprocessor(cfg.preprocess)
        dr = DimReductionFactory.build(cfg.dr)
        sampler = SamplerFactory.build(cfg.imbalance)
        est = ModelFactory.build(cfg.model)

        steps = [("pre", pre)]
        if dr is not None:
            steps.append(("dr", dr))

        if sampler is not None:
            steps.append(("sampler", sampler))
            steps.append(("clf", est))
            return ImbPipeline(steps)

        steps.append(("clf", est))
        return SkPipeline(steps)
