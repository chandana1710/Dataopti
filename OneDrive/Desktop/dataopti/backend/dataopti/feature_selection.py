"""Leakage-safe feature selectors that are fitted only inside ML pipelines."""

from functools import partial

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, mutual_info_regression


def make_selector(task, mode, feature_count, random_state=42):
    if mode == "embedded":
        estimator = ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor
        params = {"n_estimators": 80, "random_state": random_state, "n_jobs": 1}
        if task == "classification":
            params["class_weight"] = "balanced"
        return SelectFromModel(estimator(**params), threshold="median")
    k = max(1, min(10, int(feature_count)))
    scorer = mutual_info_classif if task == "classification" else mutual_info_regression
    score_function = partial(scorer, random_state=random_state)
    return SelectKBest(score_func=score_function, k=k)


def selected_feature_names(selector, feature_names):
    names = list(feature_names)
    support = selector.get_support()
    return [name for name, selected in zip(names, support) if selected]

