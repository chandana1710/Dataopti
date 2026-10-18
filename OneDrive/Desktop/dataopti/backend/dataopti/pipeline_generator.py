"""Build reproducible sklearn pipelines with preprocessing inside each split."""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

from .feature_selection import make_selector


class TextColumnToStrings(BaseEstimator, TransformerMixin):
    """Flatten one imputed text column for TfidfVectorizer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=object).reshape(-1)
        return np.asarray(["" if value is None else str(value) for value in values], dtype=object)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else ["text"], dtype=object)


class IQRCapper(BaseEstimator, TransformerMixin):
    """Learn IQR bounds from training rows and cap later rows at those bounds."""

    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        columns = self.columns if self.columns is not None else [column for column in frame.columns if frame[column].dtype.kind in "biufc"]
        self.bounds_ = {}
        for column in columns:
            series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            self.bounds_[column] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        for column, (lower, upper) in self.bounds_.items():
            frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(lower, upper)
        return frame

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)


def make_preprocessor(X, robust_numeric=False):
    numeric = [column for column in X.columns if X[column].dtype.kind in "biufc"]
    text_columns = []
    for column in X.columns:
        if column in numeric:
            continue
        sample = X[column].dropna().astype(str).head(1000)
        if not sample.empty and sample.str.len().mean() >= 40:
            text_columns.append(column)
    categorical = [column for column in X.columns if column not in numeric and column not in text_columns]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", RobustScaler() if robust_numeric else StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2, max_categories=50)),
        ]), categorical))
    for index, column in enumerate(text_columns):
        transformers.append((f"text_{index}", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("flatten", TextColumnToStrings()),
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)),
        ]), [column]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def build_pipeline_specs(task, X_train, random_state=42):
    preprocessor = make_preprocessor(X_train)
    if task == "classification":
        baseline = RandomForestClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced_subsample", random_state=random_state, n_jobs=1)
        embedded_model = ExtraTreesClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=1)
        linear_model = LogisticRegression(max_iter=400, class_weight="balanced", random_state=random_state)
    else:
        baseline = RandomForestRegressor(n_estimators=60, min_samples_leaf=2, random_state=random_state, n_jobs=1)
        embedded_model = ExtraTreesRegressor(n_estimators=60, min_samples_leaf=2, random_state=random_state, n_jobs=1)
        linear_model = Ridge(alpha=1.0)

    selector_template = make_selector(task, "embedded", max(1, X_train.shape[1]), random_state)
    kbest_template = make_selector(task, "filter", max(1, X_train.shape[1]), random_state)
    return [
        {
            "name": "Baseline: Random Forest",
            "steps": [("preprocessor", preprocessor), ("model", baseline)],
            "feature_selection": False,
            "outlier_strategy": "keep",
        },
        {
            "name": "Remove outliers + selection: Extra Trees",
            "steps": [("preprocessor", make_preprocessor(X_train)), ("selection", selector_template), ("model", embedded_model)],
            "feature_selection": True,
            "outlier_strategy": "remove training rows outside IQR bounds",
        },
        {
            "name": "Cap outliers + compact features: Linear model",
            "steps": [("outlier_capper", IQRCapper([column for column in X_train.columns if X_train[column].dtype.kind in "biufc"])), ("preprocessor", make_preprocessor(X_train, robust_numeric=True)), ("selection", kbest_template), ("model", linear_model)],
            "feature_selection": True,
            "outlier_strategy": "cap at training IQR bounds",
        },
    ]


def build_pipeline(spec):
    return Pipeline(spec["steps"])
