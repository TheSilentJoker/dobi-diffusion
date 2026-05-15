from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breastdiffusion.image_features import extract_feature_matrix
from breastdiffusion.metrics import binary_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tumor/no-tumor classifier on DOBI NIR images.")
    parser.add_argument("--manifest", default="data/manifests/dobi_prompts.csv")
    parser.add_argument("--synthetic-manifest", default=None, help="Optional generated-image manifest for augmentation.")
    parser.add_argument("--output", default="reports/classification_baseline.json")
    parser.add_argument("--threshold-metric", choices=["f1", "balanced_accuracy"], default="balanced_accuracy")
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--model", choices=["all", "logreg", "svm", "random_forest"], default="all")
    return parser.parse_args()


def _load_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    paths = df["image_path"].astype(str).tolist()
    x = extract_feature_matrix(paths)
    y = df["label"].astype(int).to_numpy()
    return x, y


def _best_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_value = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (y_score >= threshold).astype(int)
        current = binary_classification_metrics(y_true, y_pred, y_score)
        if current[metric] > best_value:
            best_value = current[metric]
            best_threshold = float(threshold)
            best_metrics = current
    assert best_metrics is not None
    return best_threshold, best_metrics


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest["split"] == "train"].copy()

    if args.synthetic_manifest:
        synthetic = pd.read_csv(args.synthetic_manifest)
        synthetic = synthetic[synthetic["split"].fillna("train") == "train"].copy()
        train_df = pd.concat([train_df, synthetic], ignore_index=True)

    val_df = manifest[manifest["split"] == "val"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    x_train, y_train = _load_features(train_df)
    x_val, y_val = _load_features(val_df)
    x_test, y_test = _load_features(test_df)

    n_components = min(args.pca_components, x_train.shape[0] - 1, x_train.shape[1])
    candidates = {
        "logreg": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=42)),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=42)),
                (
                    "svm",
                    SVC(
                        C=2.0,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    if args.model != "all":
        candidates = {args.model: candidates[args.model]}

    candidate_reports: dict[str, dict[str, object]] = {}
    best_name = ""
    best_value = -1.0
    best_ap = -1.0
    best_model = None
    best_threshold = 0.5

    for name, clf in candidates.items():
        clf.fit(x_train, y_train)
        val_score = clf.predict_proba(x_val)[:, 1]
        threshold, val_metrics = _best_threshold(y_val, val_score, args.threshold_metric)
        candidate_reports[name] = {
            "threshold": threshold,
            "validation": val_metrics,
        }
        selected_value = val_metrics[args.threshold_metric]
        selected_ap = val_metrics["average_precision"]
        if selected_value > best_value or (selected_value == best_value and selected_ap > best_ap):
            best_name = name
            best_value = selected_value
            best_ap = selected_ap
            best_model = clf
            best_threshold = threshold

    assert best_model is not None

    test_score = best_model.predict_proba(x_test)[:, 1]
    test_pred = (test_score >= best_threshold).astype(int)
    test_metrics = binary_classification_metrics(y_test, test_pred, test_score)

    report = {
        "model": best_name,
        "candidate_models": candidate_reports,
        "threshold_metric": args.threshold_metric,
        "selected_threshold": best_threshold,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_positive": int(y_train.sum()),
        "val_positive": int(y_val.sum()),
        "test_positive": int(y_test.sum()),
        "validation": candidate_reports[best_name]["validation"],
        "test": test_metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
