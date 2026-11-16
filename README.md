# 🩺 Diabetes Risk Prediction — Explainable ML System

**Developed by:** Ankit Kumar  
**Internship:** Academy of Skill Development, Kolkata — Data Science, AI & ML (May–July 2025)

![Python](https://img.shields.io/badge/Python-3.14-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-green)
![SHAP](https://img.shields.io/badge/SHAP-0.54-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-ff4b4b)

---

## What Makes This Project Stand Out

| Feature | Detail |
|---|---|
| **Dataset** | CDC BRFSS 2015 — 253,680 real US adult survey records (not the overused 768-row PIMA dataset) |
| **Models** | 8 models including Tuned XGBoost, Tuned LightGBM and a Stacking Ensemble |
| **Hyperparameter Tuning** | RandomizedSearchCV (40 iterations × 5-fold) on XGBoost and LightGBM |
| **Class Imbalance** | SMOTE applied to training set only — val and test sets kept at real 86/14 split |
| **Threshold Optimisation** | Per-model optimal threshold from F1 optimisation on real-distribution validation set |
| **Explainability** | SHAP TreeExplainer — global summary + local waterfall plots per patient |
| **Feature Engineering** | 3 composite features: MetabolicRisk, LifestyleScore, HealthcareBurden |
| **Learning Curves** | Overfitting/underfitting diagnosis for every model |
| **Web App** | Streamlit app with 4 tabs: Predict, Results, Plots, About |
| **34 saved plots** | Full audit trail from EDA through SHAP |

---

## Dataset

**CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015**  
Source: [Kaggle — Alex Teboul](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

| Property | Value |
|---|---|
| Records | 253,680 |
| Features | 21 original + 3 engineered = 24 |
| Target | `Diabetes_binary` (0 = No, 1 = Diabetic/Pre-diabetic) |
| Class split | 86.1% Non-diabetic / 13.9% Diabetic |

**Key features:** High Blood Pressure, High Cholesterol, BMI, Age, General Health,
Physical Activity, Income, Education, Smoker, Mental Health Days, Stroke, Heart Disease

**Engineered features:**
- `MetabolicRisk` = BMI × 0.4 + HighBP × 10 + HighChol × 8 + HeartDisease × 12 + Stroke × 10
- `LifestyleScore` = PhysActivity × 3 + Fruits + Veggies − Smoker × 2 − HeavyAlcohol × 2
- `HealthcareBurden` = MentalHealthDays + PhysicalHealthDays

---

## Results

### Optimised Threshold Results (Primary)

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **🏆 Tuned LightGBM** | **0.243** | **80.3%** | **37.4%** | **61.7%** | **0.4661** | **0.8264** |
| XGBoost | 0.237 | 79.2% | 36.2% | 64.9% | 0.4646 | 0.8255 |
| LightGBM | 0.275 | 81.3% | 38.6% | 57.5% | 0.4617 | 0.8254 |
| Tuned XGBoost | 0.269 | 81.5% | 38.8% | 56.9% | 0.4610 | 0.8253 |
| SVM (Linear) | 0.603 | 78.4% | 35.0% | 64.6% | 0.4539 | 0.8184 |
| Logistic Regression | 0.662 | 80.9% | 37.7% | 56.6% | 0.4527 | 0.8184 |
| Random Forest | 0.476 | 79.9% | 36.5% | 59.3% | 0.4518 | 0.8188 |

### Cross-Validation ROC-AUC (5-Fold, post-SMOTE train)

| Model | CV ROC-AUC |
|---|---|
| XGBoost | 0.9714 ± 0.0006 |
| LightGBM | 0.9714 ± 0.0006 |
| Random Forest | 0.9480 ± 0.0012 |
| Logistic Regression | 0.8269 ± 0.0004 |

> **Why ROC-AUC matters more than accuracy:** With 86/14 class imbalance, a model that always predicts "No Diabetes" gets 86% accuracy. ROC-AUC measures ability to rank diabetics above non-diabetics at any threshold — a far more honest measure.

### Hyperparameter Tuning Results

| Model | Default CV F1 | Tuned CV F1 | Gain |
|---|---|---|---|
| XGBoost | — | **0.9129** | Best params found |
| LightGBM | — | **0.9140** | Best params found |

Best XGBoost params: `lr=0.08, max_depth=7, subsample=0.6, colsample=0.7, gamma=0.1, reg_alpha=0.5`  
Best LightGBM params: `lr=0.08, num_leaves=127, max_depth=4, subsample=0.9, n_estimators=500`

### Top 10 Features (SHAP — Tuned XGBoost)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | GenHlth (General Health) | 0.766 |
| 2 | Age | 0.451 |
| 3 | MetabolicRisk *(engineered)* | 0.378 |
| 4 | BMI | 0.309 |
| 5 | Education | 0.209 |
| 6 | Sex | 0.184 |
| 7 | HighBP | 0.183 |
| 8 | Income | 0.176 |
| 9 | Fruits | 0.129 |
| 10 | DiffWalk | 0.118 |

The engineered `MetabolicRisk` feature ranks **3rd globally** — confirming it carries real signal.

---

## Project Structure

```
Diabetes Prediction Using Machine Learning/
├── data/
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv   ← main dataset (253K rows)
│   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   └── diabetes_012_health_indicators_BRFSS2015.csv
├── notebooks/
│   └── diabetes_prediction_full.ipynb    ← step-by-step walkthrough
├── src/
│   ├── preprocessing.py      ← load, clean, engineer features, SMOTE, scale
│   ├── models.py             ← all model definitions, train, evaluate, threshold
│   ├── tuning.py             ← RandomizedSearchCV + learning curves
│   ├── visualizations.py     ← all 16 plot functions
│   ├── explainability.py     ← SHAP summary, bar, waterfall
│   └── download_data.py      ← dataset download helper
├── app/
│   └── streamlit_app.py      ← 4-tab interactive web app
├── outputs/
│   ├── plots/                ← 34 saved figures
│   └── models/               ← 13 trained model .pkl files + thresholds.json
├── run_pipeline.py           ← single script to run everything
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run full pipeline (trains all models, generates all plots)
python run_pipeline.py

# 3. Launch web app
python -m streamlit run app/streamlit_app.py
```

---

## Pipeline Steps

```
Data (253K rows)
    │
    ├─ EDA (4 plots)
    │
    ├─ Feature Engineering (+3 composite features)
    │
    ├─ Stratified Split: 80% train / 5% val / 15% test
    │      └─ SMOTE on train only (balanced 50/50)
    │         Val + Test stay at real 86/14 distribution
    │
    ├─ 5 Base Models trained
    │
    ├─ 5-Fold CV evaluation
    │
    ├─ Hyperparameter Tuning (XGBoost + LightGBM, 40 iters × 5-fold)
    │
    ├─ Stacking Ensemble (RF + XGBoost + LightGBM → Logistic Regression)
    │
    ├─ Learning Curves (overfitting diagnosis)
    │
    ├─ Threshold Optimisation (F1 on real-distribution val set)
    │
    └─ SHAP Explainability (global + local)
```

---

## Interview Q&A

**Q: Why use ROC-AUC as the primary metric?**  
The dataset is 86/14 imbalanced. Accuracy is misleading — predicting "No Diabetes" always gives 86%. ROC-AUC measures ranking ability across all thresholds and is robust to imbalance.

**Q: Why SMOTE only on training data?**  
Applying SMOTE to validation or test data would create data leakage — the model would be evaluated on artificially balanced data that doesn't represent reality. SMOTE is a training-time technique only.

**Q: Why threshold = 0.24 for Tuned LightGBM instead of 0.5?**  
The default 0.5 threshold is designed for balanced classes. With 86/14 imbalance, models assign low probabilities to the minority class even when they're genuinely diabetic. Lowering the threshold to 0.24 trades some precision for significantly better recall — catching 61.7% of diabetics instead of 17%.

**Q: Why does your engineered MetabolicRisk feature rank 3rd in SHAP?**  
It combines four correlated risk factors (BMI, blood pressure, cholesterol, heart disease) into one composite score. This reduces multicollinearity and gives the model a denser signal than any individual feature alone.

**Q: What would you do next to improve this further?**  
Four things: (1) Optuna for more efficient hyperparameter search than RandomizedSearchCV, (2) calibrated probabilities (Platt scaling / isotonic regression) so the raw probability scores are reliable, (3) testing on the 2016/2017 BRFSS data to check temporal generalisability, (4) a neural network baseline (TabNet or simple MLP) for comparison.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.14 | Language |
| pandas | 3.0.1 | Data manipulation |
| numpy | 2.4.4 | Numerical ops |
| scikit-learn | 1.9.1 | Models, CV, metrics |
| XGBoost | 3.4.1 | Gradient boosting |
| LightGBM | 4.7.0 | Gradient boosting |
| SHAP | 0.54.1 | Explainability |
| imbalanced-learn | 0.14.2 | SMOTE |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Statistical plots |
| Streamlit | 1.67.0 | Web app |

---

> ⚠️ **Medical Disclaimer:** This project is for educational and portfolio purposes only. It does not constitute medical advice. Consult a qualified healthcare professional for any health concerns.
