# Bank AI Churn & Retention Platform

An end-to-end machine-learning portfolio project for predicting bank customer
churn and turning the prediction into an actionable retention workflow.

The project combines:

- A reproducible scikit-learn training pipeline.
- A persisted Random Forest churn model.
- A Streamlit decision-support application.
- Individual and batch prediction workflows.
- Early-warning queues, explainable drivers, and what-if scenarios.
- A transparent retention Opportunity Engine.
- A safe in-process demo bank integration with SQLite persistence.

> **Demo disclaimer:** This project uses an educational dataset and simulated
> bank records. It does not connect to a real bank and its value-at-risk
> figures are illustrative proxies, not financial forecasts.

## Business Problem

Customer churn is costly because a bank often has an opportunity to intervene
before a customer leaves. A conventional classifier answers:

> Which customers may leave?

This project extends that answer into an operating workflow:

> Which customers should a retention team contact first, why, and what
> hypothetical action could be considered?

## Results

Three models were evaluated on a held-out test set:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8170 | 0.5380 | 0.7125 | 0.6131 | 0.8612 |
| Decision Tree | 0.7615 | 0.4497 | 0.7690 | 0.5675 | 0.8398 |
| Logistic Regression | 0.7135 | 0.3872 | 0.7002 | 0.4987 | 0.7772 |

Random Forest was selected because it achieved the strongest F1-score and
ROC-AUC in the project evaluation. Recall is especially relevant here because
missing a customer who later churns can cause the bank to miss an intervention
opportunity.

## Application Workflow

```text
Customer data
    -> churn probability
    -> 0-100 risk score and risk band
    -> prioritized early-warning queue
    -> explainable retention action
```

The main application areas are:

- **Retention Command Center:** ranks opportunities using churn risk, a
  clearly labelled demo relationship-value proxy, and intervention signals.
- **Executive Overview:** summarizes the portfolio and key churn patterns.
- **Bank Sync:** runs a safe in-process mock synchronization and stores local
  snapshots.
- **Early Warning:** filters the prioritized queue by risk band and score.
- **Customer Profile:** shows an individual prediction, drivers, action, and
  a what-if scenario.
- **Advanced Tools:** individual and batch prediction, data quality, model
  performance, explainability, analytics, audit, and integration demos.

## Quick Start

### Option 1: One-click Windows launcher

1. Install Python 3.10 or newer.
2. Double-click `تشغيل_الواجهة.bat`.
3. The launcher installs `requirements.txt` and starts Streamlit.
4. Open `http://localhost:8501` if the browser does not open automatically.

### Option 2: Terminal

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Reproduce Training

```powershell
py train_final.py
```

The final training script:

1. Removes identifier-only columns.
2. Performs stratified train/test and train/validation splits.
3. Scales numerical features and one-hot encodes categorical features.
4. Trains Logistic Regression, Decision Tree, and Random Forest pipelines.
5. Selects the decision threshold on validation data using F1-score.
6. Evaluates on the untouched test set.
7. Saves `best_model.joblib` and `outputs/metrics.json`.

## Validation

```powershell
py -m unittest discover -s tests -v
py -m py_compile app.py train_final.py src\*.py tests\*.py
```

## Repository Structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application and user workflows |
| `train_final.py` | Reproducible model training and evaluation |
| `Churn_Project_Final.ipynb` | Exploratory analysis and model comparison |
| `best_model.joblib` | Persisted production-style model pipeline |
| `data/` | Educational churn dataset |
| `src/` | Prediction, sync, persistence, reporting, audit, and scoring services |
| `outputs/metrics.json` | Evaluation metadata and saved model results |
| `tests/` | Automated project tests |
| `requirements.txt` | Python dependencies |
| `تشغيل_الواجهة.bat` | Windows one-click launcher |

## Technical Highlights

- Prevents preprocessing mismatch by keeping transformations and estimators
  in a single scikit-learn `Pipeline`.
- Uses `class_weight="balanced"` to address the imbalanced churn target.
- Separates validation threshold selection from final test evaluation.
- Supports CSV batch scoring and downloadable CSV/PDF reports.
- Exposes feature importance and model drivers for explainability.
- Uses SQLite and a mock client to demonstrate integration without credentials.
- Treats Opportunity Engine calculations as transparent scenarios rather than
  unsupported financial claims.

## Portfolio Summary

This project demonstrates practical skills in:

- Python, Pandas, scikit-learn, and model evaluation.
- Classification under class imbalance.
- Feature preprocessing and reproducible pipelines.
- Streamlit product development.
- Explainable decision support.
- Batch inference and report generation.
- Lightweight persistence, audit logging, and service integration.

## Suggested Resume Description

> Built an end-to-end bank customer churn and retention platform using
> Python, scikit-learn, and Streamlit; compared three classifiers, selected a
> Random Forest using F1/ROC-AUC, implemented validation-based thresholding,
> and delivered explainable early-warning and retention workflows with batch
> scoring, what-if analysis, and a simulated bank integration layer.
