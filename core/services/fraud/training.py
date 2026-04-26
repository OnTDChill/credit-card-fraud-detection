import json
import os
from datetime import datetime
from time import perf_counter
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def _extract_column_groups(frame: pd.DataFrame) -> Dict[str, List[str]]:
    categorical_columns = frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column_name for column_name in frame.columns if column_name not in categorical_columns]
    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
    }


def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    column_groups = _extract_column_groups(frame)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, column_groups["numeric"]),
            ("cat", categorical_pipeline, column_groups["categorical"]),
        ],
        remainder="drop",
    )


def _build_candidate_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            n_jobs=-1,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def _build_model_pipeline(preprocessor: ColumnTransformer, estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ]
    )


def _compute_binary_metrics(
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)

    try:
        roc_auc_value = float(roc_auc_score(labels, probabilities))
    except ValueError:
        roc_auc_value = 0.5

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": roc_auc_value,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def _build_prediction_frame(
    pipeline: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    split_name: str,
) -> pd.DataFrame:
    probabilities = pipeline.predict_proba(features)[:, 1]
    prediction_frame = pd.DataFrame(
        {
            "split": split_name,
            "y_true": labels.astype(int).to_numpy(),
            "fraud_probability": probabilities,
        }
    )
    prediction_frame["y_pred_default"] = (prediction_frame["fraud_probability"] >= 0.5).astype(int)
    prediction_frame["confidence"] = np.maximum(
        prediction_frame["fraud_probability"],
        1 - prediction_frame["fraud_probability"],
    )
    prediction_frame["uncertainty"] = 1 - prediction_frame["confidence"]
    return prediction_frame


def _summarize_confidence(prediction_frame: pd.DataFrame) -> Dict[str, float]:
    if prediction_frame.empty:
        return {
            "mean_confidence": 0.0,
            "median_confidence": 0.0,
            "high_confidence_rate": 0.0,
            "low_confidence_rate": 0.0,
            "mean_uncertainty": 0.0,
            "brier_score": 0.0,
        }

    y_true = prediction_frame["y_true"].astype(int).to_numpy()
    y_prob = prediction_frame["fraud_probability"].astype(float).to_numpy()
    confidence = prediction_frame["confidence"].astype(float).to_numpy()
    uncertainty = prediction_frame["uncertainty"].astype(float).to_numpy()

    return {
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "high_confidence_rate": float(np.mean(confidence >= 0.9)),
        "low_confidence_rate": float(np.mean(confidence <= 0.6)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def _summarize_split(prediction_frame: pd.DataFrame, threshold: float = 0.5) -> Dict[str, object]:
    labels = prediction_frame["y_true"].astype(int)
    probabilities = prediction_frame["fraud_probability"].astype(float).to_numpy()
    metrics = _compute_binary_metrics(labels=labels, probabilities=probabilities, threshold=threshold)
    return {
        "metrics": metrics,
        "confidence": _summarize_confidence(prediction_frame),
    }


def _compute_overfit_summary(
    train_metrics: Dict[str, object],
    val_metrics: Dict[str, object],
    test_metrics: Dict[str, object],
) -> Dict[str, object]:
    f1_train = float(train_metrics.get("f1_score", 0.0))
    f1_val = float(val_metrics.get("f1_score", 0.0))
    f1_test = float(test_metrics.get("f1_score", 0.0))

    roc_train = float(train_metrics.get("roc_auc", 0.0))
    roc_val = float(val_metrics.get("roc_auc", 0.0))
    roc_test = float(test_metrics.get("roc_auc", 0.0))

    f1_gap_train_val = f1_train - f1_val
    f1_gap_val_test = f1_val - f1_test
    roc_gap_train_val = roc_train - roc_val
    roc_gap_val_test = roc_val - roc_test

    max_abs_gap = max(abs(f1_gap_train_val), abs(roc_gap_train_val))
    if max_abs_gap >= 0.10:
        risk_level = "high"
    elif max_abs_gap >= 0.05:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_level": risk_level,
        "f1_gap_train_minus_val": float(f1_gap_train_val),
        "f1_gap_val_minus_test": float(f1_gap_val_test),
        "roc_auc_gap_train_minus_val": float(roc_gap_train_val),
        "roc_auc_gap_val_minus_test": float(roc_gap_val_test),
    }


def _serialize_class_distribution(labels: pd.Series) -> Dict[str, int]:
    raw_counts = labels.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in raw_counts.items()}


def evaluate_binary_classifier(pipeline: Pipeline, features: pd.DataFrame, labels: pd.Series) -> Dict[str, object]:
    probabilities = pipeline.predict_proba(features)[:, 1]
    return _compute_binary_metrics(labels=labels, probabilities=probabilities, threshold=0.5)


def estimate_latency_ms(
    pipeline: Pipeline,
    features: pd.DataFrame,
    max_samples: int = 120,
    batch_size: int = 16,
) -> Dict[str, float]:
    if features.empty:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    sampled_features = features.head(min(max_samples, len(features)))
    elapsed_ms = []

    # {'mục_đích': 'Ước lượng latency suy luận nhanh và ổn định bằng mini-batch, tránh nghẽn khi feature rất lớn', 'đầu_vào': 'tập mẫu validation/test', 'đầu_ra': 'p50/p95/p99 per-transaction latency (ms)'}
    effective_batch_size = max(1, min(batch_size, len(sampled_features)))

    for start_index in range(0, len(sampled_features), effective_batch_size):
        batch_frame = sampled_features.iloc[start_index:start_index + effective_batch_size]
        start_time = perf_counter()
        pipeline.predict_proba(batch_frame)
        per_transaction_ms = ((perf_counter() - start_time) * 1000) / len(batch_frame)
        elapsed_ms.extend([per_transaction_ms] * len(batch_frame))

    return {
        "p50": float(np.percentile(elapsed_ms, 50)),
        "p95": float(np.percentile(elapsed_ms, 95)),
        "p99": float(np.percentile(elapsed_ms, 99)),
    }


def _compute_utility_score(
    val_metrics: Dict[str, object],
    val_latency: Dict[str, float],
    latency_budget_ms: float,
) -> float:
    recall_value = float(val_metrics.get("recall", 0.0))
    precision_value = float(val_metrics.get("precision", 0.0))
    f1_value = float(val_metrics.get("f1_score", 0.0))
    pr_auc_value = float(val_metrics.get("pr_auc", 0.0))
    p95_latency = float(val_latency.get("p95", 0.0))

    latency_penalty = max(0.0, (p95_latency - latency_budget_ms) / max(latency_budget_ms, 1.0))
    utility_score = (
        0.45 * recall_value
        + 0.25 * f1_value
        + 0.15 * precision_value
        + 0.15 * pr_auc_value
        - 0.10 * latency_penalty
    )
    return float(utility_score)


def _evaluate_candidate(
    candidate_name: str,
    estimator: object,
    preprocessor: ColumnTransformer,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    latency_budget_ms: float,
) -> Tuple[Pipeline, Dict[str, object]]:
    pipeline = _build_model_pipeline(preprocessor=preprocessor, estimator=estimator)

    train_start = perf_counter()
    pipeline.fit(train_x, train_y)
    train_duration_s = perf_counter() - train_start

    val_metrics = evaluate_binary_classifier(pipeline, val_x, val_y)
    test_metrics = evaluate_binary_classifier(pipeline, test_x, test_y)
    val_latency = estimate_latency_ms(pipeline, val_x)
    test_latency = estimate_latency_ms(pipeline, test_x)

    utility_score = _compute_utility_score(
        val_metrics=val_metrics,
        val_latency=val_latency,
        latency_budget_ms=latency_budget_ms,
    )

    candidate_report = {
        "candidate_name": candidate_name,
        "model_class": pipeline.named_steps["model"].__class__.__name__,
        "train_duration_seconds": float(train_duration_s),
        "validation": val_metrics,
        "test": test_metrics,
        "latency_ms": {
            "validation": val_latency,
            "test": test_latency,
        },
        "utility_score": utility_score,
        "latency_budget_ms": float(latency_budget_ms),
    }

    return pipeline, candidate_report


def train_and_evaluate(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    random_state: int = 42,
    latency_budget_ms: float = 120.0,
) -> Dict[str, object]:
    preprocessor = _build_preprocessor(train_x)
    candidates = _build_candidate_models(random_state=random_state)

    best_pipeline = None
    best_report = None
    candidate_reports = []

    for candidate_name, estimator in candidates.items():
        candidate_pipeline, candidate_report = _evaluate_candidate(
            candidate_name=candidate_name,
            estimator=estimator,
            preprocessor=preprocessor,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            latency_budget_ms=latency_budget_ms,
        )
        candidate_reports.append(candidate_report)

        if best_report is None or candidate_report["utility_score"] > best_report["utility_score"]:
            best_pipeline = candidate_pipeline
            best_report = candidate_report

    if best_pipeline is None or best_report is None:
        raise RuntimeError("Không thể huấn luyện bất kỳ candidate model nào.")

    train_predictions = _build_prediction_frame(best_pipeline, train_x, train_y, split_name="train")
    val_predictions = _build_prediction_frame(best_pipeline, val_x, val_y, split_name="validation")
    test_predictions = _build_prediction_frame(best_pipeline, test_x, test_y, split_name="test")

    train_summary = _summarize_split(train_predictions, threshold=0.5)
    val_summary = _summarize_split(val_predictions, threshold=0.5)
    test_summary = _summarize_split(test_predictions, threshold=0.5)

    overfit_summary = _compute_overfit_summary(
        train_metrics=train_summary["metrics"],
        val_metrics=val_summary["metrics"],
        test_metrics=test_summary["metrics"],
    )

    model_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    report = {
        "model": best_report["model_class"],
        "model_version": model_version,
        "champion_candidate": best_report["candidate_name"],
        "random_state": random_state,
        "latency_budget_ms": float(latency_budget_ms),
        "class_distribution": {
            "train": _serialize_class_distribution(train_y),
            "val": _serialize_class_distribution(val_y),
            "test": _serialize_class_distribution(test_y),
        },
        "train": train_summary["metrics"],
        "validation": val_summary["metrics"],
        "test": test_summary["metrics"],
        "split_results": {
            "train": train_summary,
            "validation": val_summary,
            "test": test_summary,
        },
        "confidence": {
            "train": train_summary["confidence"],
            "validation": val_summary["confidence"],
            "test": test_summary["confidence"],
        },
        "overfit": overfit_summary,
        "latency_ms": best_report["latency_ms"],
        "utility_score": best_report["utility_score"],
        "candidates": candidate_reports,
        "default_threshold": 0.5,
    }

    return {
        "pipeline": best_pipeline,
        "report": report,
        "prediction_frames": {
            "train": train_predictions,
            "validation": val_predictions,
            "test": test_predictions,
        },
    }


def save_training_artifacts(
    pipeline: Pipeline,
    report: Dict[str, object],
    artifacts_dir: str,
    prediction_frames: Dict[str, pd.DataFrame] | None = None,
    model_file_name: str = "fraud_model.joblib",
    report_file_name: str = "training_report.json",
) -> Dict[str, str]:
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, model_file_name)
    report_path = os.path.join(artifacts_dir, report_file_name)

    joblib.dump(pipeline, model_path)

    prediction_paths: Dict[str, str] = {}
    if prediction_frames:
        for split_name, frame in prediction_frames.items():
            prediction_file_name = f"predictions_{split_name}.csv"
            prediction_path = os.path.join(artifacts_dir, prediction_file_name)
            frame.to_csv(prediction_path, index=False)
            prediction_paths[split_name] = prediction_path

        report["prediction_files"] = {
            split_name: os.path.basename(path)
            for split_name, path in prediction_paths.items()
        }

    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    output_paths = {
        "model_path": model_path,
        "report_path": report_path,
    }
    for split_name, path in prediction_paths.items():
        output_paths[f"predictions_{split_name}_path"] = path

    return output_paths
