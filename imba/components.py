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
# Dim factory
# ---------------------------

@dataclass
class DimensionalityReductionConfig:
    method: str  # 'none' | 'pca' | 'lda' | 'umap' |
    n_components: Optional[int] = None
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}




class DimReductionFactory:
    @staticmethod
    def build(cfg: DimensionalityReductionConfig):
            
        if cfg.method == "pca":
            return PCATransformer(n_components=cfg.n_components, **cfg.params)
            
        elif cfg.method == "lda":
            return LDATransformer(n_components=cfg.n_components, **cfg.params)
            
        elif cfg.method == "umap":
            return UMAPTransformer(n_components=cfg.n_components, **cfg.params)
            
        else:
            return None # Либо можно IdentityTransformer()
            





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
