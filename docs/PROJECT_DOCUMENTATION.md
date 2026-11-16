# Project Documentation
## Diabetes Risk Prediction — Explainable ML System

**Author:** Ankit Kumar  
**Organisation:** Academy of Skill Development, Kolkata  
**Domain:** Data Science, AI & ML Using Python  
**Duration:** May – July 2025

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Dataset Description](#2-dataset-description)
3. [Project Architecture](#3-project-architecture)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Feature Engineering](#5-feature-engineering)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Model Evaluation Strategy](#8-model-evaluation-strategy)
9. [Threshold Optimisation](#9-threshold-optimisation)
10. [SHAP Explainability](#10-shap-explainability)
11. [Learning Curves](#11-learning-curves)
12. [Results](#12-results)
13. [Streamlit Web Application](#13-streamlit-web-application)
14. [How to Run](#14-how-to-run)
15. [Interview Preparation](#15-interview-preparation)
16. [Limitations and Future Work](#16-limitations-and-future-work)

---

## 1. Project Objective

Diabetes is one of the leading chronic diseases globally. Early identification of at-risk individuals enables timely lifestyle interventions and medical care, reducing long-term complications such as heart disease, kidney failure, and nerve damage.

This project builds a **supervised binary classification system** that predicts whether a person is at risk of diabetes based on 21 health and lifestyle survey indicators collected by the US Centers for Disease Control and Prevention (CDC).

**Primary goals:**
- Accurate identification of diabetic individuals (high recall)
- Model transparency — explain *why* a prediction was made (SHAP)
- Practical deployment via an interactive web application

**Why recall matters more than accuracy here:**  
Missing a diabetic patient (false negative) is clinically more costly than a false alarm (false positive). A false alarm leads to unnecessary follow-up. A missed diagnosis leads to untreated disease progression. The evaluation strategy and threshold optimisation reflect this priority.

---

## 2. Dataset Description

### Source
**CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015**  
Published on Kaggle by Alex Teboul:  
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

The BRFSS is an annual telephone survey conducted by the CDC across all 50 US states, collecting health behaviour and risk factor data from over 400,000 adults per year.

### Files Available

| File | Rows | Description |
|---|---|---|
| `diabetes_binary_health_indicators_BRFSS2015.csv` | 253,680 | Main dataset — binary target, natural class imbalance |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70,692 | Pre-balanced 50/50 version |
| `diabetes_012_health_indicators_BRFSS2015.csv` | 253,680 | 3-class version (No / Pre-diabetic / Diabetic) |

**This project uses the first file** (natural imbalance) to build a realistic model.

### Features

| Feature | Type | Description |
|---|---|---|
| `Diabetes_binary` | Target | 0 = No Diabetes, 1 = Pre-diabetic or Diabetic |
| `HighBP` | Binary | Ever told blood pressure is high |
| `HighChol` | Binary | Ever told cholesterol is high |
| `CholCheck` | Binary | Cholesterol check in past 5 years |
| `BMI` | Continuous | Body Mass Index |
| `Smoker` | Binary | Smoked at least 100 cigarettes in lifetime |
| `Stroke` | Binary | Ever had a stroke |
| `HeartDiseaseorAttack` | Binary | Coronary heart disease or myocardial infarction |
| `PhysActivity` | Binary | Physical activity in past 30 days |
| `Fruits` | Binary | Consumes fruit 1+ times per day |
| `Veggies` | Binary | Consumes vegetables 1+ times per day |
| `HvyAlcoholConsump` | Binary | Heavy alcohol consumption |
| `AnyHealthcare` | Binary | Any healthcare coverage |
| `NoDocbcCost` | Binary | Could not see doctor due to cost in past year |
| `GenHlth` | Ordinal | General health rating (1=Excellent to 5=Poor) |
| `MentHlth` | Continuous | Days of poor mental health in past 30 |
| `PhysHlth` | Continuous | Days of poor physical health in past 30 |
| `DiffWalk` | Binary | Difficulty walking or climbing stairs |
| `Sex` | Binary | 0=Female, 1=Male |
| `Age` | Ordinal | Age category (1=18–24 to 13=80+) |
| `Education` | Ordinal | Education level (1=Never attended to 6=College grad) |
| `Income` | Ordinal | Income scale (1=<$10K to 8=≥$75K) |

### Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Non-Diabetic (0) | 218,334 | 86.1% |
| Diabetic (1) | 35,346 | 13.9% |

This imbalance is **real and meaningful** — it reflects the actual prevalence of diabetes in the US adult population. We preserve it in our validation and test sets.

---

## 3. Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_pipeline.py                             │
│  Orchestrates the full 13-step pipeline end-to-end             │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────────┐
        ▼                ▼                    ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│preprocessing │  │   models.py  │  │   tuning.py      │
│    .py       │  │              │  │                  │
│ - load_data  │  │ - get_models │  │ - tune_xgboost   │
│ - clean      │  │ - train      │  │ - tune_lightgbm  │
│ - engineer   │  │ - evaluate   │  │ - learning_curve │
│ - split      │  │ - threshold  │  └──────────────────┘
│ - SMOTE      │  │ - stacking   │
│ - scale      │  └──────────────┘
└──────────────┘
        │
        ▼
┌──────────────────┐    ┌──────────────────────┐
│explainability.py │    │  visualizations.py   │
│                  │    │                      │
│ - SHAP summary   │    │ - EDA plots          │
│ - SHAP bar       │    │ - ROC curves         │
│ - SHAP waterfall │    │ - Confusion matrices │
│ - top features   │    │ - Learning curves    │
└──────────────────┘    │ - Threshold plots    │
                        │ - Tuning comparison  │
                        └──────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   app/streamlit_app.py   │
                    │                         │
                    │  Tab 1: Predict          │
                    │  Tab 2: Model Results    │
                    │  Tab 3: Plots            │
                    │  Tab 4: About            │
                    └─────────────────────────┘
```

---

## 4. Data Preprocessing

### Step 1 — Loading
The CSV is loaded with pandas. Shape is verified (253,680 × 22). No missing values exist in this dataset — the CDC pre-processes BRFSS responses before publishing.

### Step 2 — Missing Value Check
```python
df.isnull().sum()  # → all zeros
```
Result: **0 missing values across all 22 columns.**

### Step 3 — Stratified Train / Val / Test Split

| Set | Size | Rows | Purpose |
|---|---|---|---|
| Training | 80% | ~202,905 | Model training + SMOTE |
| Validation | 5% | ~12,723 | Threshold optimisation only |
| Test | 15% | ~38,052 | Final evaluation (held out completely) |

**Critical design decision:** Validation and test sets are split **before** SMOTE, preserving the real 86/14 class distribution. This prevents overly optimistic threshold search and evaluation.

### Step 4 — SMOTE (Training Set Only)

**Problem:** 86/14 class imbalance causes models to predict "No Diabetes" almost always at default threshold, missing diabetic patients.

**Solution:** SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic diabetic samples by interpolating between existing ones in feature space.

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_trainval, y_trainval)
# Result: balanced 50/50 → [174,634 non-diabetic, 174,634 diabetic]
```

**Why SMOTE only on training data:**  
Applying SMOTE to validation/test would constitute data leakage — the model would be evaluated on data that doesn't represent reality.

### Step 5 — Feature Scaling

StandardScaler is fit on the SMOTE-augmented training set and applied to all three splits:

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform
X_val_sc   = scaler.transform(X_val)          # transform only
X_test_sc  = scaler.transform(X_test)         # transform only
```

**Tree-based models** (Random Forest, XGBoost, LightGBM) are scale-invariant and use **raw (unscaled) data**.  
**Linear models** (Logistic Regression, SVM) use **scaled data**.

---

## 5. Feature Engineering

Three composite features were added to give the model denser, clinically-motivated signal:

### MetabolicRisk
```python
MetabolicRisk = BMI × 0.4 + HighBP × 10 + HighChol × 8
              + HeartDiseaseorAttack × 12 + Stroke × 10
```
**Rationale:** Combines the five strongest cardiometabolic risk factors into one score. High BMI, blood pressure, and cholesterol are the core triad of metabolic syndrome which directly precedes Type 2 diabetes. Heart disease and stroke indicate advanced vascular damage.

**Validation:** MetabolicRisk ranked **3rd globally** in SHAP feature importance (Mean |SHAP| = 0.378), above individual features like BMI (0.309) and HighBP (0.183). This confirms it carries real combined signal.

### LifestyleScore
```python
LifestyleScore = PhysActivity × 3 + Fruits + Veggies
               - Smoker × 2 - HvyAlcoholConsump × 2
```
**Rationale:** A healthy lifestyle proxy. Physical activity has the strongest protective effect (weight × 3). Fruit and vegetable intake contribute positively. Smoking and heavy alcohol consumption reduce the score. Regular physical activity is one of the most evidence-based interventions for diabetes prevention.

### HealthcareBurden
```python
HealthcareBurden = MentHlth + PhysHlth
```
**Rationale:** Total days of poor health (mental + physical) in the past 30 days. Captures overall health impairment. High scores indicate chronic illness burden which correlates strongly with diabetes.

---

## 6. Machine Learning Models

### Why these models?

| Model | Rationale |
|---|---|
| **Logistic Regression** | Interpretable linear baseline. Establishes minimum expected performance. Coefficients directly show feature relationships. |
| **Random Forest** | Strong ensemble that handles non-linearity, robust to outliers, provides Gini feature importance. Good at capturing interaction effects between features. |
| **XGBoost** | State-of-the-art gradient boosting. Sequential tree building with regularisation. Consistently top performer in tabular ML benchmarks and healthcare prediction tasks. |
| **LightGBM** | Leaf-wise tree growth (vs. level-wise in XGBoost). Faster on large datasets (253K rows), comparable accuracy. Best performer in this project. |
| **SVM (Linear)** | Maximum-margin linear classifier. LinearSVC wrapped with CalibratedClassifierCV for probability estimates. Fast on large data. |
| **Stacking Ensemble** | Meta-learning: RF + XGBoost + LightGBM as base learners → Logistic Regression meta-learner. Uses 5-fold CV to generate unbiased meta-features. |

### Model Parameters (before tuning)

```python
RandomForestClassifier(
    n_estimators=300, max_depth=10,
    min_samples_split=5, class_weight='balanced'
)

XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8
)

LGBMClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, num_leaves=50, class_weight='balanced'
)
```

---

## 7. Hyperparameter Tuning

### Method: RandomizedSearchCV

RandomizedSearchCV was chosen over GridSearchCV because:
- The search space has ~50,000+ possible combinations
- Random sampling of 40 combinations finds near-optimal solutions in a fraction of the time
- 5-fold stratified CV ensures reliable estimates

```python
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=40,           # 40 random combinations
    cv=5,                # 5-fold stratified CV
    scoring='f1',        # optimise F1, not accuracy
    n_jobs=-1,           # all CPU cores
)
```

**Why F1 as the scoring metric?**  
F1 balances precision and recall. For imbalanced medical data, optimising accuracy would reward models that rarely predict positive. F1 forces the search to find models that actually identify diabetic patients.

### Best Parameters Found

**XGBoost** (CV F1 = 0.9129):
```
learning_rate=0.08, max_depth=7, subsample=0.6,
colsample_bytree=0.7, gamma=0.1, reg_alpha=0.5,
reg_lambda=0.5, min_child_weight=3, n_estimators=200,
scale_pos_weight=1
```

**LightGBM** (CV F1 = 0.9140):
```
learning_rate=0.08, num_leaves=127, max_depth=4,
subsample=0.9, colsample_bytree=0.6, reg_alpha=0.1,
reg_lambda=1.0, min_child_samples=10, n_estimators=500,
class_weight='balanced'
```

> Note: CV F1 of 0.91 is computed on the SMOTE-balanced training set, which explains the high value. Test set F1 at default threshold is lower (~0.27) due to the return to real 86/14 distribution — this is expected and correct behaviour.

---

## 8. Model Evaluation Strategy

### Metrics Used

| Metric | Formula | Why Used |
|---|---|---|
| **ROC-AUC** | Area under ROC curve | Primary metric. Threshold-independent measure of ranking ability. Robust to class imbalance. |
| **F1-Score** | 2×P×R / (P+R) | Harmonic mean of precision and recall. Main metric for comparing models at a fixed threshold. |
| **Recall (Sensitivity)** | TP / (TP+FN) | % of actual diabetics correctly identified. Most important clinically. |
| **Precision** | TP / (TP+FP) | % of predicted diabetics who are actually diabetic. |
| **Accuracy** | (TP+TN) / Total | Reported for completeness but misleading with 86/14 imbalance. |

### Why ROC-AUC is the primary metric

Consider a naive model that always predicts "No Diabetes":
- Accuracy: 86.1% ← looks good
- Recall: 0% ← useless
- ROC-AUC: 0.5 ← correctly identifies it as random

ROC-AUC measures whether the model assigns a higher probability to a randomly-chosen diabetic patient than to a randomly-chosen non-diabetic patient. It is completely independent of the decision threshold.

### Cross-Validation

5-Fold Stratified K-Fold CV on the SMOTE training set:

| Model | CV ROC-AUC |
|---|---|
| XGBoost | 0.9714 ± 0.0006 |
| LightGBM | 0.9714 ± 0.0006 |
| Random Forest | 0.9480 ± 0.0012 |
| SVM (Linear) | 0.8267 ± 0.0005 |
| Logistic Regression | 0.8269 ± 0.0004 |

The high CV scores on SMOTE-balanced data confirm the models have learned genuine signal, not noise.

---

## 9. Threshold Optimisation

### The Problem

At the default threshold of 0.50, XGBoost achieves:
- Accuracy: 86.4% (looks good)
- Recall: 19.0% (misses 81% of diabetic patients)

This happens because the model correctly learns that ~86% of people are not diabetic, so it assigns low probabilities to positive predictions.

### The Solution

We scan all decision thresholds from 0 to 1 and select the one that maximises F1-Score on the **real-distribution validation set** (5% of data, pre-SMOTE, 86/14 split).

```python
precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
f1_scores = 2 * precisions * recalls / (precisions + recalls)
best_threshold = thresholds[np.argmax(f1_scores)]
```

### Why Use the Real-Distribution Validation Set

If we searched for thresholds on the SMOTE-balanced training data, the optimal threshold found (~0.50) would be wrong for the real imbalanced test data. The validation set must have the same distribution as the deployment environment.

### Optimal Thresholds Found

| Model | Threshold | Recall at Threshold |
|---|---|---|
| Tuned LightGBM | 0.243 | 61.7% |
| XGBoost | 0.237 | 64.9% |
| LightGBM | 0.275 | 57.5% |
| Tuned XGBoost | 0.269 | 56.9% |
| Logistic Regression | 0.662 | 56.6% |
| Random Forest | 0.476 | 59.3% |

---

## 10. SHAP Explainability

### What is SHAP?

SHAP (SHapley Additive exPlanations) is based on game theory's Shapley values. For each prediction, SHAP assigns each feature a contribution score showing how much it pushed the prediction toward "Diabetic" or away from it.

**Key properties:**
- **Locally accurate:** SHAP values sum to the model's prediction
- **Consistent:** If a feature matters more in one model, its SHAP value is higher
- **Model-agnostic:** Works for any model (though TreeExplainer is exact and fast for tree models)

### Implementation

```python
explainer   = shap.TreeExplainer(model)   # exact for tree models
shap_values = explainer(X_sample)         # shape: (n_samples, n_features)
```

### Plot Types Generated

**1. Summary Plot (Beeswarm)**  
Shows every sample as a dot. X-axis = SHAP value (impact on prediction). Colour = feature value (red=high, blue=low). Reveals not just which features matter but how their values affect risk.

**2. Bar Plot**  
Mean absolute SHAP values per feature — clean global ranking.

**3. Waterfall Plot**  
Single-patient explanation. Shows exactly which features pushed one specific prediction toward or away from "Diabetic", starting from the base rate.

### Top 10 Features (Tuned XGBoost)

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | GenHlth | 0.766 | Poor general health is the strongest individual predictor |
| 2 | Age | 0.451 | Risk increases substantially with age |
| 3 | MetabolicRisk | 0.378 | Engineered feature — combines BMI, BP, cholesterol |
| 4 | BMI | 0.309 | High BMI independently contributes |
| 5 | Education | 0.209 | Lower education correlates with higher risk |
| 6 | Sex | 0.184 | Males show higher risk in this dataset |
| 7 | HighBP | 0.183 | High blood pressure is a direct risk factor |
| 8 | Income | 0.176 | Lower income associated with higher risk |
| 9 | Fruits | 0.129 | Regular fruit intake is protective |
| 10 | DiffWalk | 0.118 | Difficulty walking indicates physical deterioration |

**Key insight:** The engineered `MetabolicRisk` feature ranks 3rd, outperforming individual features like HighBP (rank 7) and HighChol. This validates the feature engineering decision.

---

## 11. Learning Curves

Learning curves plot model performance (ROC-AUC) as a function of training set size, computed with 3-fold CV.

### How to Read Them

| Pattern | Meaning | Fix |
|---|---|---|
| Large gap between train and val | Overfitting | Regularise, reduce complexity |
| Both curves low and converged | Underfitting | More features, more complex model |
| Val curve still rising at max size | More data would help | Collect more data |
| Val curve plateaued | Model has learned what it can | Try different architecture |

### Observations

- **XGBoost / LightGBM:** Small train-val gap, high val AUC — well-fitted
- **Logistic Regression:** Both curves converge early — the linear model has reached its capacity on this non-linear data
- **Random Forest:** Slight overfitting visible — gap reduces with more data

---

## 12. Results

### Final Results Table (Optimised Thresholds)

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **🏆 Tuned LightGBM** | **0.243** | **80.3%** | **37.4%** | **61.7%** | **0.4661** | **0.8264** |
| XGBoost | 0.237 | 79.2% | 36.2% | 64.9% | 0.4646 | 0.8255 |
| LightGBM | 0.275 | 81.3% | 38.6% | 57.5% | 0.4617 | 0.8254 |
| Tuned XGBoost | 0.269 | 81.5% | 38.8% | 56.9% | 0.4610 | 0.8253 |
| SVM (Linear) | 0.603 | 78.4% | 35.0% | 64.6% | 0.4539 | 0.8184 |
| Logistic Regression | 0.662 | 80.9% | 37.7% | 56.6% | 0.4527 | 0.8184 |
| Random Forest | 0.476 | 79.9% | 36.5% | 59.3% | 0.4518 | 0.8188 |

### F1 Improvement After Threshold Optimisation

| Model | Default F1 | Tuned F1 | Gain |
|---|---|---|---|
| XGBoost | 0.279 | **0.465** | ▲ +0.186 |
| Tuned LightGBM | 0.262 | **0.466** | ▲ +0.204 |
| LightGBM | 0.285 | **0.462** | ▲ +0.177 |
| Tuned XGBoost | 0.269 | **0.461** | ▲ +0.192 |

---

## 13. Streamlit Web Application

The app is built with Streamlit and organised into 4 tabs:

### Tab 1 — Predict
- User enters 21 health indicators via sliders and dropdowns
- 3 composite features auto-calculated
- Select any of the 8 trained models
- Output: prediction, risk probability %, optimised threshold used
- All models compared side-by-side
- SHAP waterfall plot for tree models

### Tab 2 — Model Results
- Full results table with best values highlighted in green
- Metric explanations (ROC-AUC, F1, Recall, threshold)

### Tab 3 — Plots
- All 34 generated plots viewable in-app, organised by section:
  - EDA, Model Performance, Hyperparameter Tuning,
    Threshold Optimisation, Learning Curves, SHAP

### Tab 4 — About
- Dataset summary, tech stack, project description

### Run the App
```bash
python -m streamlit run app/streamlit_app.py
```

---

## 14. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Download Dataset
1. Go to: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
2. Download and extract
3. Place `diabetes_binary_health_indicators_BRFSS2015.csv` in `data/`

### Run Full Pipeline
```bash
python run_pipeline.py
```
This runs all 13 steps (~25–30 minutes on a standard laptop):
1. Load data
2. EDA plots
3. Feature engineering
4. Split + SMOTE + scaling
5. Base model training
6. Default evaluation
7. Cross-validation
8. Hyperparameter tuning (40×5-fold for XGBoost + LightGBM)
9. Stacking ensemble
10. Learning curves
11. Threshold optimisation
12. SHAP explainability
13. Save all models and results

### Launch Web App
```bash
python -m streamlit run app/streamlit_app.py
```
Open: http://localhost:8501

---

## 15. Interview Preparation

### Beginner Questions

**Q: What is the target variable?**  
`Diabetes_binary` — 0 for no diabetes, 1 for pre-diabetic or diabetic. Binary classification problem.

**Q: Why not use the PIMA Indians dataset?**  
PIMA has only 768 rows of data from one specific ethnic group (Native American women). The CDC BRFSS has 253,680 real survey records from a diverse US adult population across all 50 states. It's more realistic, more generalisable, and matches what real healthcare ML systems use.

**Q: What is SMOTE?**  
Synthetic Minority Over-sampling Technique. It creates new synthetic minority-class samples by interpolating between existing ones in feature space. For example, if two diabetic patients have BMI values of 30 and 34, SMOTE might create a synthetic patient with BMI 32. This balances the training data without duplicating existing samples.

### Intermediate Questions

**Q: Why is ROC-AUC better than accuracy for this dataset?**  
With 86% non-diabetic, a model predicting "No Diabetes" always gets 86% accuracy but zero utility. ROC-AUC measures the probability that a randomly chosen diabetic patient is ranked higher than a randomly chosen non-diabetic patient — independent of threshold. A value of 0.826 means the model correctly ranks a diabetic above a non-diabetic 82.6% of the time.

**Q: What is the bias-variance tradeoff and where does it appear in this project?**  
Bias = error from wrong assumptions (underfitting). Variance = error from sensitivity to training data (overfitting). Logistic Regression has high bias on this non-linear data — it underfits. Deep Random Forest with 300 trees can overfit, which appears as a gap in learning curves. We control variance through regularisation (XGBoost's `reg_alpha`, `gamma`) and hyperparameter tuning.

**Q: Why does the Stacking Ensemble not outperform individual models?**  
The stacking ensemble is only as good as the diversity of its base learners and the quality of its meta-features. Here, RF, XGBoost and LightGBM are all tree-based and highly correlated — they make similar errors on the same samples. True ensemble diversity requires architecturally different models. Additionally, the meta-learner (Logistic Regression) is linear, limiting its ability to combine non-linear interactions.

### Advanced Questions

**Q: Explain how SHAP values are computed for tree models.**  
TreeExplainer uses the tree structure to compute exact Shapley values in polynomial time. For each prediction, it computes how each feature's value changes the output compared to the expected output. It does this by considering all possible subsets of features and averaging the marginal contribution of each feature across all orderings — the game-theory definition of Shapley values.

**Q: What is data leakage and how did you prevent it?**  
Data leakage occurs when information from the test set influences model training or evaluation. We prevented it three ways: (1) SMOTE applied only to training data, (2) StandardScaler fit only on training data, (3) threshold optimisation done on a held-out validation set that was split before SMOTE.

**Q: What would you do to further improve this model?**  
Five concrete steps: (1) Optuna for more efficient Bayesian hyperparameter optimisation, (2) Platt scaling / isotonic regression for better probability calibration, (3) Feature selection using Recursive Feature Elimination to reduce noise, (4) Testing on BRFSS 2016/2017 data for temporal generalisability, (5) A TabNet or gradient-boosted neural network baseline for comparison.

---

## 16. Limitations and Future Work

### Current Limitations

| Limitation | Detail |
|---|---|
| Self-reported data | BRFSS is based on telephone surveys — responses may not be medically accurate |
| No clinical measurements | No blood glucose, HbA1c, or insulin levels — the most direct diabetes indicators |
| Cross-sectional | One point in time — cannot capture disease progression |
| US population only | May not generalise to other countries or ethnic groups |
| Stacking ensemble | Collapsed at threshold tuning; needs probability recalibration |
| Moderate F1 (~0.47) | Real limitation of survey-based features vs. clinical measurements |

### Future Work

1. **Probability Calibration** — Apply Platt scaling or isotonic regression to ensure predicted probabilities are reliable (a predicted 70% should mean ~70% of those patients are actually diabetic)

2. **Optuna Hyperparameter Optimisation** — Replace RandomizedSearchCV with Bayesian optimisation for more efficient search

3. **Neural Network Comparison** — Train a TabNet or simple MLP for comparison

4. **Temporal Validation** — Train on BRFSS 2014/2015, test on BRFSS 2016/2017 to validate temporal stability

5. **Multi-class Extension** — Retrain on `diabetes_012` file to distinguish No Diabetes / Pre-diabetic / Full Diabetes

6. **Clinical Feature Integration** — Combine survey data with available lab values for stronger prediction signal

7. **Model Deployment** — Deploy the Streamlit app on Streamlit Community Cloud or Hugging Face Spaces for public access

---

*Documentation prepared by Ankit Kumar | Academy of Skill Development, Kolkata | May–July 2025*
