# Random Forest Hyperparameter Optimization

> Individual five-week Introduction to Optimization portfolio project. Reproducible experiment using the public Breast Cancer Wisconsin (Diagnostic) dataset.

## Problem
Determine whether a fixed-budget Random Search can outperform a default Random Forest classifier on a small binary classification benchmark.

## Optimization Formulation
Decision vector `x = (n_estimators, max_depth, min_samples_split, min_samples_leaf)`.
Maximize `f(x) = mean 5-fold training-CV macro-F1(x)`, subject to the discrete search domain specified in `src/run_experiments.py` and a budget of **30 candidate evaluations**.
The **20% stratified test holdout is never used in tuning**. We also measure evaluation runtime as a practical trade-off.

## Dataset
`sklearn.datasets.load_breast_cancer`: 569 samples, 30 numeric features, binary classification. Dataset source and original provenance: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). Data are loaded in memory. No private patient-level identifying information is included.

## Method
- Baseline: `RandomForestClassifier(random_state=42, n_jobs=1)` without parameter tuning.
- Optimizer: `ParameterSampler`, 30 unique sampled configurations, seed 42, identical stratified five-fold CV on training data.
- Primary metric: **macro-F1**; secondary: test accuracy and model fit time.
- One final holdout-test evaluation per selected method, not used for optimization.
- Sensitivity: change `max_depth` around the best CV configuration.
- Stability: repeat model random-state values 11, 23, 42 on identical training CV folds. This tests estimator randomness, **not** independent random-search repetitions.

## Key Results

| Method | CV macro-F1 | Held-out test macro-F1 | Test accuracy |
|---|---:|---:|---:|
| Default RF | 0.9603 | 0.9526 | 0.9561 |
| Random Search (best CV) | 0.9576 | 0.9526 | 0.9561 |

Best searched parameters: `{'n_estimators': 80, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_depth': 5}`.

**Decision:** Keep the default Random Forest for this benchmark and protocol: its CV score is higher, held-out test macro-F1 is tied, and optimization added 14.9 seconds of candidate evaluation. This does not imply Random Search cannot improve other datasets/budgets.

![Search history](figures/convergence.png)

![Comparison](figures/comparison.png)

![Trade-off](figures/tradeoff.png)

## How to Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiments.py
```

Run from repository root. This regenerates all CSV, JSON and PNG outputs. On different computers exact runtime and some library-version-dependent scores may vary. Execution was checked with Python 3.13.5 and scikit-learn 1.8.0.

## Repository Structure
- `src/`: runnable scripts; `results/`: authentic generated CSV, JSON and logs
- `figures/`: plots; `report/`: Vietnamese report; `slides/`: oral defense slides
- `data/README.md`: dataset source; `requirements.txt`: environment dependencies

## Limitations
Single relatively small dataset; only one search seed and one train/test split; hyperparameter search does not cover all settings; test scores are subject to sampling uncertainty; no claim of clinical deployment. Runtime is hardware-dependent. The stability study varies RF model seeds only.

## AI Assistance
AI was used to help draft project structure, Python scripts, documentation and presentation. All reported metrics were computed by executing the provided code; students should independently inspect, rerun and explain the model, formulation and results. Review and edit before submission.

## Project Development
For the actual course timeline, commit work as it is completed across the five weeks. **Do not manufacture historical commits**. Once verified, optionally create tag `v1.0` and add the public repository URL to the report and LMS submission.
