"""Holdout evaluation, feature selection, comparison, and recommendations."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from .feature_engineering import engineer_features
from .feature_selection import selected_feature_names
from .pipeline_generator import build_pipeline, build_pipeline_specs
from .cleaning import duplicate_mask, MAX_CATEGORIES_PER_COLUMN, MAX_ENCODED_CELLS
from .cleaning import clean_dataframe
from .value_utils import stable_value


MAX_TRAINING_ROWS = 20_000
MAX_CV_ROWS = 2_000


def infer_task(target):
    distinct = int(target.nunique(dropna=True))
    return "regression" if pd.api.types.is_numeric_dtype(target) and distinct > 20 else "classification"


def evaluate_pipelines(df, target_name, random_state=42, include_pipeline=False, include_cross_validation=True):
    """Train comparable pipelines with holdout-only evaluation and no leakage."""
    target_column = next((column for column in df.columns if str(column) == str(target_name)), None)
    if target_column is None:
        raise ValueError("The automatically detected target is missing from the dataset.")

    usable = df.loc[df[target_column].notna()].copy()
    duplicate_count = int(duplicate_mask(usable).sum())
    usable = usable.loc[~duplicate_mask(usable)].copy()
    original_usable_rows = len(usable)
    if len(usable) > MAX_TRAINING_ROWS:
        usable = usable.sample(n=MAX_TRAINING_ROWS, random_state=random_state).sort_index().copy()
    if len(usable) < 12:
        raise ValueError("At least 12 rows with a target value are needed to compare ML pipelines.")
    y = usable[target_column]
    distinct = int(y.nunique(dropna=True))
    if distinct < 2:
        raise ValueError("The detected target contains only one class or value; ML evaluation is not possible.")
    task = infer_task(y)
    if task == "classification" and distinct > 100:
        raise ValueError("The detected classification target has more than 100 classes and is probably an identifier.")

    X = usable.drop(columns=[target_column]).copy()
    if X.empty:
        raise ValueError("No feature columns remain after removing the detected target.")
    X, engineering = engineer_features(X)
    if X.empty:
        raise ValueError("No usable input features remain after feature engineering.")

    for column in X.columns:
        if X[column].dtype == "object":
            X[column] = X[column].map(stable_value)

    test_size = max(0.25, 2 / len(X))
    stratify = None
    if task == "classification":
        counts = y.value_counts(dropna=False)
        test_rows = max(2, int(np.ceil(len(y) * test_size)))
        if counts.min() >= 2 and len(counts) <= test_rows:
            stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    if task == "classification" and y_train.nunique() < 2:
        raise ValueError("The training split contains only one target class. More labeled rows are needed.")

    specs = build_pipeline_specs(task, X_train, random_state)
    results = []
    fitted = {}
    for spec in specs:
        pipeline = build_pipeline(spec)
        candidate_X_train, candidate_y_train = X_train, y_train
        outlier_rows_removed = 0
        if spec["outlier_strategy"].startswith("remove"):
            numeric = candidate_X_train.select_dtypes(include=[np.number])
            if not numeric.empty:
                q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
                iqr = q3 - q1
                row_outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).any(axis=1)
                outlier_rows_removed = int(row_outliers.sum())
                candidate_X_train = candidate_X_train.loc[~row_outliers]
                candidate_y_train = candidate_y_train.loc[~row_outliers]
            if len(candidate_X_train) < 5 or (task == "classification" and candidate_y_train.nunique() < 2):
                results.append({"name": spec["name"], "status": "failed", "feature_selection": spec["feature_selection"], "outlier_strategy": spec["outlier_strategy"], "error": "Removing training outliers left too few rows or target classes."})
                continue
        start = time.perf_counter()
        try:
            pipeline.fit(candidate_X_train, candidate_y_train)
            predictions = pipeline.predict(X_test)
            duration = round(time.perf_counter() - start, 4)
            cv_metric = "accuracy" if task == "classification" else "r2"
            cross_validation = {
                "status": "not_run",
                "metric": cv_metric,
                "reason": "Cross-validation is calculated for the recommended pipeline only.",
            }
            if task == "classification":
                metrics = {
                    "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
                    "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 6),
                    "precision_weighted": round(float(precision_score(y_test, predictions, average="weighted", zero_division=0)), 6),
                    "recall_weighted": round(float(recall_score(y_test, predictions, average="weighted", zero_division=0)), 6),
                    "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 6),
                }
                ranking_score = metrics["f1_weighted"]
                primary_metric = "f1_weighted"
            else:
                mse = float(np.mean((np.asarray(y_test, dtype=float) - np.asarray(predictions, dtype=float)) ** 2))
                metrics = {
                    "r2": round(float(r2_score(y_test, predictions)), 6),
                    "rmse": round(float(np.sqrt(mse)), 6),
                    "mae": round(float(mean_absolute_error(y_test, predictions)), 6),
                }
                ranking_score = metrics["r2"]
                primary_metric = "r2"

            fitted[spec["name"]] = pipeline
            results.append({
                "name": spec["name"],
                "status": "complete",
                "feature_selection": spec["feature_selection"],
                "outlier_strategy": spec["outlier_strategy"],
                "metrics": metrics,
                "ranking_metric": primary_metric,
                "ranking_score": ranking_score,
                "training_seconds": duration,
                "cross_validation": cross_validation,
                "train_rows": int(len(candidate_X_train)),
                "test_rows": int(len(X_test)),
                "training_outlier_rows_removed": outlier_rows_removed,
            })
        except Exception as error:
            results.append({"name": spec["name"], "status": "failed", "feature_selection": spec["feature_selection"], "error": str(error)})

    successful = [result for result in results if result["status"] == "complete"]
    if not successful:
        errors = "; ".join(result.get("error", "unknown error") for result in results)
        raise ValueError(f"No ML pipeline could be trained: {errors}")

    baseline = next((item for item in successful if item["name"].startswith("Baseline:")), successful[0])
    best = max(successful, key=lambda item: item["ranking_score"])
    if include_cross_validation:
        best_spec = next(spec for spec in specs if spec["name"] == best["name"])
        cv_X, cv_y = X_train, y_train
        if best_spec["outlier_strategy"].startswith("remove"):
            numeric = cv_X.select_dtypes(include=[np.number])
            if not numeric.empty:
                q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
                iqr = q3 - q1
                row_outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).any(axis=1)
                candidate_X, candidate_y = cv_X.loc[~row_outliers], cv_y.loc[~row_outliers]
                if len(candidate_X) >= 5 and (task != "classification" or candidate_y.nunique() >= 2):
                    cv_X, cv_y = candidate_X, candidate_y
        if len(cv_X) > MAX_CV_ROWS:
            cv_X = cv_X.sample(n=MAX_CV_ROWS, random_state=random_state)
            cv_y = cv_y.loc[cv_X.index]
        cv_class_counts = cv_y.value_counts() if task == "classification" else None
        cv_folds = 3 if task != "classification" or cv_class_counts.min() >= 3 else 0
        if cv_folds:
            try:
                cv_start = time.perf_counter()
                cv_scores = cross_val_score(fitted[best["name"]], cv_X, cv_y, cv=cv_folds, scoring=best["ranking_metric"] if task == "classification" else "r2", n_jobs=1)
                best["cross_validation"] = {
                    "status": "complete",
                    "metric": best["ranking_metric"] if task == "classification" else "r2",
                    "folds": cv_folds,
                    "rows": int(len(cv_X)),
                    "mean": round(float(np.mean(cv_scores)), 6),
                    "standard_deviation": round(float(np.std(cv_scores)), 6),
                    "scores": [round(float(score), 6) for score in cv_scores],
                    "seconds": round(time.perf_counter() - cv_start, 4),
                }
            except Exception as cv_error:
                best["cross_validation"] = {"status": "unavailable", "metric": cv_metric, "reason": str(cv_error)}
        else:
            best["cross_validation"] = {
                "status": "unavailable",
                "metric": cv_metric,
                "reason": "At least three examples per target class are required for 3-fold cross-validation.",
            }
    selection_result = next((item for item in successful if item["feature_selection"]), None)
    selected_names = []
    if selection_result:
        selection_pipeline = fitted[selection_result["name"]]
        transformed_names = selection_pipeline.named_steps["preprocessor"].get_feature_names_out()
        selector = selection_pipeline.named_steps["selection"]
        selected_names = selected_feature_names(selector, transformed_names)
    if not selected_names:
        selected_names = [str(column) for column in X.columns]

    best_pipeline = fitted[best["name"]]
    transformed_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    if "selection" in best_pipeline.named_steps:
        transformed_names = selected_feature_names(best_pipeline.named_steps["selection"], transformed_names)
    estimator = best_pipeline.named_steps["model"]
    importance_values = getattr(estimator, "feature_importances_", None)
    if importance_values is None and hasattr(estimator, "coef_"):
        importance_values = np.abs(np.asarray(estimator.coef_)).mean(axis=0) if np.asarray(estimator.coef_).ndim > 1 else np.abs(estimator.coef_)
    feature_importance = []
    if importance_values is not None and len(importance_values) == len(transformed_names):
        ranked = sorted(zip(transformed_names, importance_values), key=lambda pair: float(pair[1]), reverse=True)[:15]
        feature_importance = [{"feature": str(name), "importance": round(float(value), 6)} for name, value in ranked]

    improvement = best["ranking_score"] - baseline["ranking_score"]
    metric_label = "weighted F1" if task == "classification" else "R²"
    reason = (
        f"{best['name']} scored best on the held-out test split using {metric_label} "
        f"({best['ranking_score']:.3f}); the baseline scored {baseline['ranking_score']:.3f}."
    )
    evaluation = {
        "status": "complete",
        "target_column": str(target_name),
        "task_type": task,
        "rows_used": int(len(usable)),
        "rows_subsampled_from": int(original_usable_rows) if original_usable_rows > len(usable) else None,
        "duplicate_rows_excluded_before_split": duplicate_count,
        "feature_count_before_engineering": int(df.shape[1] - 1),
        "feature_count_after_engineering": int(X.shape[1]),
        "feature_engineering": engineering,
        "random_state": random_state,
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "method": "single stratified holdout" if stratify is not None else "single random holdout"},
        "pipelines": results,
        "baseline": {"name": baseline["name"], "metrics": baseline["metrics"]},
        "optimized": {"name": best["name"], "metrics": best["metrics"]},
        "comparison": {"metric": primary_metric, "baseline_score": baseline["ranking_score"], "optimized_score": best["ranking_score"], "improvement": round(float(improvement), 6)},
        "feature_selection": {
            "method": selection_result["name"] if selection_result else "all available features",
            "input_features": int(len(transformed_names)) if selection_result else int(X.shape[1]),
            "selected_features": selected_names,
            "selected_count": len(selected_names),
            "removed_count": max(0, int(len(transformed_names) - len(selected_names))) if selection_result else 0,
            "importance": feature_importance,
        },
        "recommended_pipeline": {"name": best["name"], "reason": reason, "metric": primary_metric, "score": best["ranking_score"]},
        "training_seconds": round(sum(item.get("training_seconds", 0) for item in successful), 4),
        "cross_validation": next((item["cross_validation"] for item in successful if item.get("cross_validation", {}).get("status") == "complete"), None),
    }
    if include_pipeline:
        # Refit the selected workflow on every usable labeled row before export.
        # The holdout-trained estimator is appropriate for scoring, but would
        # unnecessarily discard the held-out rows in a downloadable artifact.
        final_X, final_y = X, y
        if best["outlier_strategy"].startswith("remove"):
            numeric = final_X.select_dtypes(include=[np.number])
            if not numeric.empty:
                q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
                iqr = q3 - q1
                row_outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).any(axis=1)
                candidate_X = final_X.loc[~row_outliers]
                candidate_y = final_y.loc[~row_outliers]
                if len(candidate_X) >= 5 and (task != "classification" or candidate_y.nunique() >= 2):
                    final_X, final_y = candidate_X, candidate_y
        best_spec = next(spec for spec in specs if spec["name"] == best["name"])
        exported_pipeline = build_pipeline(best_spec)
        refit_start = time.perf_counter()
        exported_pipeline.fit(final_X, final_y)
        evaluation["artifact_fit_seconds"] = round(time.perf_counter() - refit_start, 4)
        evaluation["_fitted_pipeline"] = exported_pipeline
    return evaluation


def build_feature_selected_dataset(df, target_name, selector_name=None, random_state=42):
    """Refit the chosen selector on all labeled rows and return its CSV-ready data."""
    cleaned, cleaning_report = clean_dataframe(df, target_name=target_name, make_ml_ready=False)
    target_column = next((column for column in cleaned.columns if str(column) == str(target_name)), None)
    if target_column is None:
        raise ValueError("The detected target was not retained during cleaning.")
    X = cleaned.drop(columns=[target_column])
    y = cleaned[target_column]
    X, engineering = engineer_features(X)
    for column in X.columns:
        if X[column].dtype == "object":
            X[column] = X[column].map(stable_value)
    numeric_columns = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    estimated_cells = len(X) * (
        len(numeric_columns)
        + sum(min(int(X[column].nunique(dropna=False)), MAX_CATEGORIES_PER_COLUMN) for column in categorical_columns)
    )
    if estimated_cells > MAX_ENCODED_CELLS:
        raise ValueError("The feature-selected export would be too large to materialize safely; the cleaned dataset is still available.")
    task = infer_task(y)
    specs = build_pipeline_specs(task, X, random_state)
    selected_spec = next((spec for spec in specs if spec["feature_selection"] and (selector_name is None or spec["name"] == selector_name)), None)
    if selected_spec is None:
        raise ValueError("No feature-selection pipeline is available for this dataset.")
    if selected_spec["outlier_strategy"].startswith("remove"):
        numeric = X.select_dtypes(include=[np.number])
        if not numeric.empty:
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr = q3 - q1
            row_outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).any(axis=1)
            X, y = X.loc[~row_outliers], y.loc[~row_outliers]
    selector_index = next(index for index, (name, _) in enumerate(selected_spec["steps"]) if name == "selection")
    prefix = Pipeline(selected_spec["steps"][:selector_index])
    selector = selected_spec["steps"][selector_index][1]
    transformed = prefix.fit_transform(X, y)
    selected_matrix = selector.fit_transform(transformed, y)
    feature_names = selected_feature_names(selector, prefix.get_feature_names_out())
    if selected_matrix.shape[0] * selected_matrix.shape[1] > 10_000_000:
        raise ValueError("The selected-feature export would be too large to materialize safely.")
    if hasattr(selected_matrix, "toarray"):
        selected_matrix = selected_matrix.toarray()
    result = pd.DataFrame(selected_matrix, columns=feature_names, index=X.index)
    result[str(target_name)] = y.to_numpy()
    return result, {
        "filename_suffix": "_feature_selected.csv",
        "rows": int(len(result)),
        "feature_count": int(len(feature_names)),
        "features": feature_names,
        "target_column": str(target_name),
        "cleaning": cleaning_report,
        "feature_engineering": engineering,
    }
