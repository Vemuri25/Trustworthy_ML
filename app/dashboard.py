import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "drift_report.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"
SUBGROUP_PATH = ROOT / "reports" / "subgroup_metrics.json"

st.set_page_config(
    page_title="Trustworthy ML Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("Trustworthy ML Monitor")
st.caption(
    "Monitor model quality, data drift, and subgroup performance."
)


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def refresh_drift():
    result = subprocess.run(
        [sys.executable, "-m", "src.trustworthy_ml.drift"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        st.error("Drift monitoring failed.")
        st.code(result.stderr or result.stdout)
        return False
    return True


with st.sidebar:
    st.header("Controls")
    st.write("Recalculate drift using the project's configured test batches.")
    if st.button("Run drift monitor", type="primary"):
        with st.spinner("Calculating drift..."):
            if refresh_drift():
                st.success("Drift report updated.")
                st.rerun()

    st.divider()
    st.caption("PSI thresholds")
    st.write("Low: below 0.10")
    st.write("Warning: 0.10 to below 0.20")
    st.write("Critical: 0.20 or above")

report = load_json(REPORT_PATH)
metrics = load_json(METRICS_PATH)
subgroups = load_json(SUBGROUP_PATH)

if report is None:
    st.warning("No drift report found. Run the drift monitor first.")
    st.stop()

if metrics is None:
    st.warning("No model metrics found. Train the model first.")
    st.stop()

baseline = report["batches"]["baseline_test_batch"]
shifted = report["batches"]["simulated_shift"]

tab_overview, tab_features, tab_model, tab_fairness = st.tabs([
    "Overview",
    "Feature Drift",
    "Model Performance",
    "Subgroup Audit",
])

with tab_overview:
    st.subheader("Monitoring overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test records", f'{baseline["rows"]:,}')
    c2.metric("Baseline mean PSI", f'{baseline["mean_psi"]:.4f}')
    c3.metric("Shifted mean PSI", f'{shifted["mean_psi"]:.4f}')
    c4.metric(
        "Critical shifted features",
        sum(
            item["status"] == "critical"
            for item in shifted["features"].values()
        ),
    )

    st.subheader("Baseline vs simulated shift")

    comparison = pd.DataFrame({
        "Baseline PSI": {
            name: values["psi"]
            for name, values in baseline["features"].items()
        },
        "Simulated shift PSI": {
            name: values["psi"]
            for name, values in shifted["features"].items()
        },
    })
    st.bar_chart(comparison)

    st.info(
        "The simulated batch intentionally modifies age and hours-per-week. "
        "This validates detection; it is not evidence of real-world drift "
        "or a measured decline in model accuracy."
    )

with tab_features:
    st.subheader("Feature-level drift")

    batch_choice = st.radio(
        "Choose batch",
        ["Baseline test batch", "Simulated shift"],
        horizontal=True,
    )
    selected = (
        baseline if batch_choice == "Baseline test batch" else shifted
    )

    rows = []
    for feature, values in selected["features"].items():
        rows.append({
            "Feature": feature,
            "PSI": values["psi"],
            "Status": values["status"],
            "Reference missing rate": values["reference_missing_rate"],
            "Current missing rate": values["current_missing_rate"],
            "Missing rate change": values.get("missing_rate_change"),
        })

    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format({
            "PSI": "{:.4f}",
            "Reference missing rate": "{:.2%}",
            "Current missing rate": "{:.2%}",
            "Missing rate change": "{:+.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    flagged = table[table["Status"].isin(["warning", "critical"])]
    if flagged.empty:
        st.success("No features crossed the configured PSI thresholds.")
    else:
        st.warning(f"{len(flagged)} feature(s) crossed drift thresholds.")
        st.dataframe(flagged, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("Held-out model performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f'{metrics["accuracy"]:.2%}')
    c2.metric("Balanced accuracy", f'{metrics["balanced_accuracy"]:.2%}')
    c3.metric("ROC-AUC", f'{metrics["roc_auc"]:.3f}')

    c4, c5, c6 = st.columns(3)
    c4.metric("Positive-class precision", f'{metrics["precision"]:.2%}')
    c5.metric("Positive-class recall", f'{metrics["recall"]:.2%}')
    c6.metric("F1 score", f'{metrics["f1"]:.3f}')

    st.caption(
        "These metrics come from the original held-out test set. "
        "They are not recomputed for the simulated shifted batch."
    )

with tab_fairness:
    st.subheader("Subgroup performance audit")

    if not subgroups:
        st.warning("No subgroup metrics found.")
    else:
        group_rows = []
        for attribute, groups in subgroups.items():
            for group, values in groups.items():
                group_rows.append({
                    "Attribute": attribute,
                    "Group": group,
                    "Records": values["count"],
                    "Accuracy": values["accuracy"],
                    "Positive recall": values["positive_class_recall"],
                    "Positive precision": values["positive_class_precision"],
                })

        group_table = pd.DataFrame(group_rows)
        st.dataframe(
            group_table.style.format({
                "Accuracy": "{:.2%}",
                "Positive recall": "{:.2%}",
                "Positive precision": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Differences are descriptive audit signals, not proof of "
            "unfairness. Consider sample sizes, uncertainty, and context."
        )

st.divider()
st.caption("Trustworthy ML Monitor · Local development dashboard")
