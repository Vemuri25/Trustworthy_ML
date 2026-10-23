import json
import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "adult.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

for folder in (MODEL_DIR, REPORT_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python -m src.trustworthy_ml.data"
        )

    df = pd.read_csv(DATA_PATH)

    # Normalize missing markers and categorical whitespace.
    df = df.replace(r"^\s*\?\s*$", pd.NA, regex=True)

    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    target = "class"
    if target not in df.columns:
        raise ValueError(f"Expected target column '{target}'.")

    # Encode the higher-income category as the positive class.
    y = (df[target].astype(str).str.strip() == ">50K").astype(int)

    # Retain sensitive attributes separately for subgroup auditing.
    audit_columns = [
        col for col in ("sex", "race") if col in df.columns
    ]
    audit_data = df[audit_columns].copy()

    # Exclude target, sensitive attributes, and census sampling weight.
    excluded = {target, "sex", "race", "fnlwgt"}
    X = df.drop(columns=[c for c in excluded if c in df.columns])

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # Stratified split preserves the class distribution.
    X_train, X_test, y_train, y_test, idx_train, idx_test = (
        train_test_split(
            X,
            y,
            df.index,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    # All learned preprocessing is fitted on training data only.
    model.fit(X_train, y_train)

    # Save the fitted pipeline for future predictions.
    joblib.dump(model, MODEL_DIR / "income_model.joblib")

    # Build a reference profile from training data only.
    reference_profile = {}
    for column in X_train.columns:
        series = X_train[column]
        missing_rate = float(series.isna().mean())

        if pd.api.types.is_numeric_dtype(series):
            edges = sorted(set(
                float(value)
                for value in series.quantile(
                    [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
                ).dropna().tolist()
            ))

            if len(edges) >= 2:
                counts = (
                    pd.cut(
                        series,
                        bins=edges,
                        include_lowest=True
                    ).value_counts(sort=False).tolist()
                )
            else:
                counts = [int(series.notna().sum())]

            reference_profile[column] = {
                "type": "numeric",
                "edges": edges,
                "reference_counts": counts,
                "reference_missing_rate": missing_rate,
            }
        else:
            values = series.fillna("__MISSING__").astype(str)
            frequencies = values.value_counts()

            reference_profile[column] = {
                "type": "categorical",
                "categories": frequencies.index.tolist(),
                "reference_counts": frequencies.tolist(),
                "reference_missing_rate": missing_rate,
            }

    (REPORT_DIR / "reference_profile.json").write_text(
        json.dumps(reference_profile, indent=2)
    )


    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "positive_class": ">50K",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_before_encoding": int(X.shape[1]),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "excluded_model_features": [
            c for c in ("sex", "race", "fnlwgt") if c in df.columns
        ],
        "random_state": 42,
        "test_size": 0.20,
    }

    # Save evaluation results.
    with (REPORT_DIR / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["<=50K", ">50K"],
        zero_division=0,
    )
    (REPORT_DIR / "classification_report.txt").write_text(report)

    # Save confusion matrix.
    display = ConfusionMatrixDisplay(
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        display_labels=["<=50K", ">50K"],
    )
    display.plot(values_format="d")
    plt.title("Income Classification: Test Set")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Audit recall by recorded demographic groups.
    audit_results = {}
    test_audit = audit_data.loc[idx_test]

    for column in audit_columns:
        audit_results[column] = {}
        for group in sorted(
            test_audit[column].dropna().astype(str).unique()
        ):
            mask = test_audit[column].astype(str).to_numpy() == group
            actual = y_test.to_numpy()[mask]
            predicted = predictions[mask]

            if len(actual) == 0:
                continue

            audit_results[column][group] = {
                "count": int(len(actual)),
                "accuracy": float(
                    accuracy_score(actual, predicted)
                ),
                "positive_class_recall": float(
                    recall_score(actual, predicted, zero_division=0)
                ),
                "positive_class_precision": float(
                    precision_score(actual, predicted, zero_division=0)
                ),
            }

    (REPORT_DIR / "subgroup_metrics.json").write_text(
        json.dumps(audit_results, indent=2)
    )

    print("\n=== TEST SET METRICS ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    print("\n=== SUBGROUP AUDIT ===")
    print(json.dumps(audit_results, indent=2))

    print("\nSaved results:")
    print(REPORT_DIR / "metrics.json")
    print(REPORT_DIR / "classification_report.txt")
    print(REPORT_DIR / "subgroup_metrics.json")
    print(FIGURE_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
