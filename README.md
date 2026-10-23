# Trustworthy ML Monitor

A machine learning monitoring project focused on model evaluation, feature drift detection, subgroup performance auditing, and automated testing.

The project trains a Logistic Regression classifier on the Adult Census Income dataset and provides a Streamlit dashboard to explore model performance and simulated changes in feature distributions.

## Project Highlights

- **Machine learning pipeline:** preprocessing and classification using scikit-learn.
- **Model evaluation:** accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, and average precision.
- **Feature drift monitoring:** Population Stability Index (PSI) with low, warning, and critical thresholds.
- **Subgroup auditing:** descriptive performance comparisons across sex and race groups.
- **Interactive dashboard:** Streamlit interface for exploring model metrics, drift reports, and subgroup results.
- **Automated testing:** pytest tests executed through GitHub Actions.

## Architecture

```text
Adult Census Income Dataset
            |
            v
 Data Cleaning and Preparation
            |
            v
  Train/Test Split and Pipeline
            |
            v
 Logistic Regression Classifier
            |
            +----------------------+
            |                      |
            v                      v
     Model Evaluation       Subgroup Audit
            |
            v
    Reference Profile
            |
            v
   PSI Drift Detection
            |
            v
     JSON Reports
            |
            v
    Streamlit Dashboard

       pytest + GitHub Actions
       validate automated tests
```

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- Matplotlib
- joblib
- pytest
- Git and GitHub Actions

## Repository Structure

```text
trustworthy-ml/
├── .github/
│   └── workflows/
│       └── tests.yml
├── app/
│   └── dashboard.py
├── data/
│   └── raw/                  # Local dataset; not tracked
├── models/                   # Local trained model; not tracked
├── reports/                  # Local generated reports; not tracked
├── src/
│   └── trustworthy_ml/
│       ├── data.py
│       ├── train.py
│       └── drift.py
├── tests/
│   └── test_drift.py
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vemuri25/Trustworthy_ML.git
cd Trustworthy_ML
```

### 2. Create a virtual environment

Python 3.12 is the version used by the GitHub Actions workflow.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare the dataset

The Adult Census Income dataset is used for model training and drift analysis.

Place the dataset CSV at:

```text
data/raw/adult.csv
```

The dataset is intentionally excluded from version control. Obtain it from a legitimate dataset source and ensure the CSV has the expected columns before running the pipeline.

### 4. Train the model

```bash
python -m src.trustworthy_ml.train
```

This step generates the local trained model, evaluation results, subgroup metrics, and reference feature profile.

### 5. Run drift detection

```bash
python -m src.trustworthy_ml.drift
```

The drift detector compares a baseline test batch with a controlled simulated shift and writes a local drift report.

### 6. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### 7. Run automated tests

```bash
python -m pytest tests/ -v
```

GitHub Actions automatically runs the test suite when changes are pushed to the repository or submitted through a pull request.

## Model Evaluation

The recorded evaluation results from the current training run are:

| Metric | Result |
|---|---:|
| Accuracy | 80.64% |
| Balanced accuracy | 81.91% |
| Precision for income >50K | 56.39% |
| Recall for income >50K | 84.35% |
| F1-score | 67.59% |
| ROC-AUC | 90.34% |
| Average precision | 76.08% |

The model uses a preprocessing pipeline with numeric imputation and scaling, categorical imputation and one-hot encoding, and class-weighted Logistic Regression.

The features `sex`, `race`, and `fnlwgt` are excluded from the classifier inputs. Sex and race are retained for separate descriptive subgroup evaluation.

These results describe one recorded train/test evaluation. They do not establish performance on future populations or prove that the model is suitable for consequential income-related decisions.

## Drift Monitoring

The drift detector uses Population Stability Index (PSI) to compare feature distributions.

| PSI | Interpretation |
|---|---|
| PSI < 0.10 | Low drift |
| 0.10 ≤ PSI < 0.20 | Warning |
| PSI ≥ 0.20 | Critical |

The project includes a controlled simulated shift that increases age by 15 and hours per week by 10 in a test batch. This checks whether the monitoring logic detects deliberately changed distributions.

**Important:** Distribution drift alone does not prove that model accuracy has declined. A production monitoring system should also evaluate incoming data quality, prediction behavior, and performance when ground-truth labels become available.

## Subgroup Auditing

The project reports descriptive classification metrics for sex and race groups to help identify potential differences in model behavior.

Subgroup metrics should be interpreted alongside sample sizes, uncertainty, data limitations, and the context of the prediction task. Differences alone do not establish discrimination or explain its cause.

## Testing and Continuous Integration

The test suite checks:

- PSI is approximately zero for identical distributions.
- Different distributions produce positive PSI.
- PSI remains finite when a histogram bin is empty.
- The drift detector identifies a deliberately shifted distribution.

The repository's GitHub Actions workflow runs these tests automatically. The current workflow has a successful run for commit `16041d1`.

## Limitations and Future Improvements

- Add monitoring for real incoming batches rather than only controlled simulations.
- Track data quality, missing values, and changes in prediction distributions.
- Evaluate model performance over time when ground-truth labels are available.
- Expand fairness evaluation with additional metrics and uncertainty estimates.
- Add dashboard screenshots and reproducible visual evaluation reports.
- Pin and test dependencies across supported Python versions.

## Responsible Use

This is an educational and portfolio project. The Adult Census Income dataset contains historical demographic and socioeconomic information. Model outputs and subgroup metrics can reflect limitations or biases in the source data.

This project is not intended to make employment, lending, or other high-impact decisions.

## Author

**Charan Vemuri**

GitHub: [@Vemuri25](https://github.com/Vemuri25)

---

*Built as a machine learning portfolio project exploring evaluation, drift monitoring, subgroup auditing, and automated testing.*
