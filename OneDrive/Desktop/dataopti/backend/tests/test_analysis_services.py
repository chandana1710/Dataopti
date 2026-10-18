"""Smoke tests for the current profiling and preparation workflow."""

import base64
import io
import json
import unittest
import zipfile

import numpy as np
import pandas as pd

from app import app
from dataopti import analyze_dataframe, clean_dataframe, detect_dataset_type, infer_target_column
from dataopti.drift import compare_datasets
from dataopti.anomaly import detect_anomalies
from dataopti.feature_engineering import engineer_features
from dataopti.modeling import evaluate_pipelines
from dataopti.validation import inspect_invalid_values


class AnalysisServiceTests(unittest.TestCase):
    def setUp(self):
        count = 72
        age = np.arange(count) % 55 + 18
        target = np.where(age < 36, "low", np.where(age < 55, "medium", "high"))
        self.data = pd.DataFrame({
            "customer_id": np.arange(count),
            "age": age.astype(float),
            "income": np.arange(count) * 125.0,
            "city": ["A", "B", "C"] * (count // 3),
            "primary_category": target,
        })
        self.data.loc[2, "age"] = np.nan
        self.data.loc[5, "city"] = None
        self.data = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        self.client = app.test_client()

    def test_profile_returns_measured_quality_and_target(self):
        profile = analyze_dataframe(self.data, "customers.csv")
        self.assertEqual(profile["target_inference"]["column"], "primary_category")
        self.assertEqual(profile["column_profiles"][1]["missing_count"], 1)
        self.assertLess(profile["data_quality"]["score"], 100)
        self.assertEqual(profile["dataset_type"], "tabular")

    def test_sensor_dataset_is_detected_from_axes(self):
        kind = detect_dataset_type(
            "sensor.csv",
            ["timestamp", "accelerometer_x", "accelerometer_y", "accelerometer_z"],
        )
        self.assertEqual(kind, "sensor")

    def test_sensor_example_gets_time_and_magnitude_features(self):
        frame = pd.read_csv("examples/sensor_activity.csv")
        profile = analyze_dataframe(frame, "sensor_activity.csv")
        self.assertEqual(profile["dataset_type"], "sensor")
        self.assertEqual(profile["target_inference"]["column"], "activity")
        self.assertEqual(profile["sensor_profile"]["sampling_interval_seconds_median"], 1.0)
        features, report = engineer_features(frame.drop(columns=["activity"]))
        self.assertIn("accelerometer_magnitude", features.columns)
        self.assertIn("timestamp_year", features.columns)
        self.assertIn("accelerometer_magnitude", report["generated_features"])

    def test_obvious_invalid_age_values_are_reported(self):
        frame = pd.DataFrame({"age": [-4, 25, 500], "salary": [-10, 100, 200]})
        findings = inspect_invalid_values(frame)
        self.assertEqual({item["column"]: item["count"] for item in findings}, {"age": 2, "salary": 1})

    def test_cleaning_trims_obvious_categorical_whitespace(self):
        frame = pd.DataFrame({"city": ["Delhi", " Mumbai ", "Delhi"], "value": [1, 2, 3]})
        cleaned, report = clean_dataframe(frame)
        self.assertIn("Mumbai", cleaned["city"].to_list())
        self.assertEqual(report["invalid_values_treated"], 1)

    def test_cleaner_removes_duplicates_and_prepares_detected_target(self):
        target = infer_target_column(self.data)["column"]
        cleaned, report = clean_dataframe(self.data, target, make_ml_ready=True)
        self.assertEqual(report["duplicate_rows_removed"], 1)
        self.assertEqual(report["missing_values_after"], 0)
        self.assertIn("primary_category", cleaned.columns)
        self.assertEqual(report["outliers_removed"], 0)

    def test_models_are_trained_compared_and_feature_selection_is_measured(self):
        target = infer_target_column(self.data)["column"]
        result = evaluate_pipelines(self.data, target)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len([item for item in result["pipelines"] if item["status"] == "complete"]), 3)
        self.assertGreater(result["feature_selection"]["selected_count"], 0)
        self.assertEqual(result["comparison"]["metric"], "f1_weighted")

    def test_numeric_target_runs_regression_pipelines(self):
        rows = 72
        age = np.arange(rows) % 50 + 18
        frame = pd.DataFrame({
            "age": age,
            "experience": np.arange(rows) % 15,
            "revenue": age * 125.0 + (np.arange(rows) % 7) * 31.0,
        })
        result = evaluate_pipelines(frame, "revenue")
        self.assertEqual(result["task_type"], "regression")
        self.assertEqual(result["comparison"]["metric"], "r2")
        self.assertTrue(all(item["status"] == "complete" for item in result["pipelines"]))

    def test_upload_endpoint_returns_profile_and_quality(self):
        response = self.client.post(
            "/api/upload",
            data={"files": (io.BytesIO(self.data.to_csv(index=False).encode()), "customers.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["files"][0]["dataset_type"], "tabular")
        self.assertIn("column_profiles", body["files"][0])
        self.assertGreaterEqual(body["data_quality"]["score"], 0)

    def test_prepare_endpoint_uses_inferred_target_automatically(self):
        response = self.client.post(
            "/api/prepare",
            data={"files": (io.BytesIO(self.data.to_csv(index=False).encode()), "customers.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        encoded = response.headers["X-DataOpti-Report"]
        report = json.loads(base64.b64decode(encoded).decode("utf-8"))
        self.assertEqual(report["items"][0]["status"], "ml_ready")
        self.assertEqual(report["items"][0]["target_column"], "primary_category")
        self.assertEqual(report["items"][0]["model"]["status"], "complete")
        self.assertEqual(len(report["items"][0]["model"]["pipelines"]), 3)
        self.assertIn("data_quality_before", report["items"][0])
        self.assertGreater(report["items"][0]["feature_selected_export"]["feature_count"], 0)
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            self.assertTrue(any(name.endswith("_feature_selected.csv") for name in archive.namelist()))

    def test_prepare_compares_uploaded_tables_before_and_after_cleaning(self):
        rows = 40
        reference = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="s").astype(str),
            "accelerometer_x": np.arange(rows, dtype=float),
            "accelerometer_y": np.arange(rows, dtype=float) * 2,
            "accelerometer_z": np.arange(rows, dtype=float) * 3,
        })
        current = reference.copy()
        current["accelerometer_x"] += 250
        response = self.client.post(
            "/api/prepare",
            data={
                "files": [
                    (io.BytesIO(reference.to_csv(index=False).encode()), "reference.csv"),
                    (io.BytesIO(current.to_csv(index=False).encode()), "current.csv"),
                ]
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        report = json.loads(base64.b64decode(response.headers["X-DataOpti-Report"]).decode("utf-8"))
        self.assertEqual(len(report["dataset_comparisons"]), 1)
        comparison = report["dataset_comparisons"][0]
        self.assertEqual(comparison["reference_filename"], "reference.csv")
        self.assertEqual(comparison["current_filename"], "current.csv")
        self.assertEqual(comparison["before_cleaning"]["status"], "high")
        self.assertIn("accelerometer_x", comparison["before_cleaning"]["drifted_columns"])
        self.assertIn("status", comparison["after_preprocessing"])

    def test_dynamic_api_compares_two_uploaded_versions(self):
        baseline = pd.DataFrame({"age": np.arange(60), "category": ["A", "B"] * 30})
        current = pd.DataFrame({"age": np.arange(60) + 500, "category": ["C", "D"] * 30})
        response = self.client.post(
            "/api/dynamic/analyze",
            data={
                "reference": (io.BytesIO(baseline.to_csv(index=False).encode()), "reference.csv"),
                "current": (io.BytesIO(current.to_csv(index=False).encode()), "current.csv"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json()["status"], "high")
        self.assertTrue(response.get_json()["retraining_recommended"])

    def test_drift_comparison_reports_real_distribution_changes(self):
        reference = pd.DataFrame({"age": np.arange(100), "city": ["A", "B"] * 50})
        current = pd.DataFrame({"age": np.arange(100) + 1000, "city": ["C", "D"] * 50})
        result = compare_datasets(reference, current)
        self.assertEqual(result["status"], "high")
        self.assertTrue(result["retraining_recommended"])
        self.assertEqual(set(result["drifted_columns"]), {"age", "city"})

    def test_anomaly_detection_flags_numeric_rows_without_removing_them(self):
        frame = pd.DataFrame({"measurement": [0.0] * 19 + [1000.0], "other": np.arange(20)})
        result = detect_anomalies(frame)
        self.assertEqual(result["status"], "complete")
        self.assertGreater(result["anomaly_count"], 0)
        self.assertEqual(result["rows_analyzed"], len(frame))
        self.assertIn("measurement", result["columns_analyzed"])


if __name__ == "__main__":
    unittest.main()
