import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "adult.csv"
PROFILE_PATH = ROOT / "reports" / "reference_profile.json"
OUTPUT_PATH = ROOT / "reports" / "drift_report.json"

PSI_WARNING = 0.10
PSI_CRITICAL = 0.20


def clean_data(df):
    df = df.replace(r"^\s*\?\s*$", pd.NA, regex=True)

    for col in df.columns:
        if (
            pd.api.types.is_object_dtype(df[col].dtype)
            or isinstance(df[col].dtype, pd.StringDtype)
        ):
            df[col] = df[col].map(
                lambda value: value.strip()
                if isinstance(value, str) else value
            )
    return df


def calculate_psi(reference_counts, current_counts):
    reference = np.asarray(reference_counts, dtype=float)
    current = np.asarray(current_counts, dtype=float)

    if reference.sum() == 0 or current.sum() == 0:
        return None

    # Smoothing prevents zero-count bins from causing infinite PSI.
    epsilon = 1e-6
    reference = (reference + epsilon) / (
        reference.sum() + epsilon * len(reference)
    )
    current = (current + epsilon) / (
        current.sum() + epsilon * len(current)
    )

    return float(
        np.sum((current - reference) * np.log(current / reference))
    )


def categorical_counts(series, categories):
    values = series.fillna("__MISSING__").astype(str)
    known = set(categories)
    values = values.map(
        lambda value: value if value in known else "__OTHER__"
    )

    labels = list(categories)
    if "__OTHER__" not in labels:
        labels.append("__OTHER__")

    return values.value_counts().reindex(labels, fill_value=0).tolist()


def feature_drift(series, reference):
    missing_rate = float(series.isna().mean())
    reference_missing = float(reference["reference_missing_rate"])

    if reference["type"] == "numeric":
        edges = reference["edges"]
        if len(edges) < 2:
            return {
                "psi": None,
                "status": "not_evaluable",
                "current_missing_rate": missing_rate,
                "reference_missing_rate": reference_missing,
            }

        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.dropna()

        # Include values beyond reference extrema in the edge bins.
        bin_edges = [-np.inf] + edges[1:-1] + [np.inf]
        current_counts = (
            pd.cut(numeric, bins=bin_edges, include_lowest=True)
            .value_counts(sort=False)
            .tolist()
        )
        reference_counts = reference["reference_counts"]

    else:
        categories = reference["categories"]
        labels = list(categories)
        if "__OTHER__" not in labels:
            labels.append("__OTHER__")

        current_counts = categorical_counts(series, categories)

        ref_lookup = dict(zip(
            categories, reference["reference_counts"]
        ))
        reference_counts = [
            ref_lookup.get(label, 0) for label in labels
        ]

        # Add an empty reference bin for unseen categories.
        if "__OTHER__" in labels and "__OTHER__" not in categories:
            reference_counts[-1] = 0

    psi = calculate_psi(reference_counts, current_counts)

    if psi is None:
        status = "not_evaluable"
    elif psi >= PSI_CRITICAL:
        status = "critical"
    elif psi >= PSI_WARNING:
        status = "warning"
    else:
        status = "low"

    return {
        "psi": psi,
        "status": status,
        "current_missing_rate": missing_rate,
        "reference_missing_rate": reference_missing,
        "missing_rate_change": missing_rate - reference_missing,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Reference profile not found: {PROFILE_PATH}. "
            "Run python -m src.trustworthy_ml.train first."
        )

    df = clean_data(pd.read_csv(DATA_PATH))
    profile = json.loads(PROFILE_PATH.read_text())

    X = df.drop(columns=["class", "sex", "race", "fnlwgt"])
    y = (df["class"].astype(str).str.strip() == ">50K").astype(int)

    X_train, X_test, _, _ = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Build two reproducible test batches.
    baseline_batch = X_test.copy()
    shifted_batch = X_test.copy()

    # Simulate distribution changes, without modifying the source CSV.
    shifted_batch["age"] = (
        pd.to_numeric(shifted_batch["age"], errors="coerce") + 15
    )
    shifted_batch["hours-per-week"] = (
        pd.to_numeric(
            shifted_batch["hours-per-week"], errors="coerce"
        ) + 10
    )

    results = {}
    for batch_name, batch in (
        ("baseline_test_batch", baseline_batch),
        ("simulated_shift", shifted_batch),
    ):
        feature_results = {}
        for column, reference in profile.items():
            feature_results[column] = feature_drift(
                batch[column], reference
            )

        evaluable = [
            item["psi"] for item in feature_results.values()
            if item["psi"] is not None
        ]
        results[batch_name] = {
            "rows": int(len(batch)),
            "features": feature_results,
            "features_warning_or_critical": sum(
                item["status"] in ("warning", "critical")
                for item in feature_results.values()
            ),
            "mean_psi": (
                float(np.mean(evaluable)) if evaluable else None
            ),
        }

    report = {
        "method": "Population Stability Index (PSI)",
        "thresholds": {
            "low": "PSI < 0.10",
            "warning": "0.10 <= PSI < 0.20",
            "critical": "PSI >= 0.20",
        },
        "note": (
            "Simulated shifts are controlled tests, not evidence of "
            "real-world drift or model performance degradation."
        ),
        "batches": results,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved drift report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
