# Interview Preparation Guide
## Diabetes Risk Prediction Project

---

## How to Introduce This Project (60-second pitch)

> "I built an end-to-end diabetes risk prediction system using 253,680 real CDC survey records — not the standard PIMA dataset most projects use. I trained 8 models including tuned XGBoost and LightGBM with RandomizedSearchCV, handled class imbalance with SMOTE on training data only, and added SHAP explainability so the model can explain any individual prediction. The best model achieves ROC-AUC of 0.826 on the real 86/14 imbalanced test set. I also deployed it as a Streamlit web app with 4 tabs. The unique part is I engineered a MetabolicRisk composite feature that ranked 3rd globally in SHAP — above individual features like blood pressure and cholesterol."

---

## Question Bank

### Level 1 — Basics (Any Interviewer)

**Q1: What problem does this project solve?**
Early detection of diabetes risk from health and lifestyle survey data. The model can flag high-risk individuals who should seek medical screening, enabling preventive intervention before clinical diagnosis.

**Q2: What dataset did you use and why?**
CDC BRFSS 2015 — 253,680 real US adult survey responses. I chose it over the commonly used PIMA Indians dataset (768 rows) because it's larger, more diverse, and uses real-world lifestyle features rather than clinical lab measurements. It's the same dataset cited in peer-reviewed papers published in Nature and NIH journals in 2024–2025.

**Q3: What is the target variable?**
`Diabetes_binary` — 1 means the person is pre-diabetic or diabetic, 0 means no diabetes. Binary classification.

**Q4: What models did you use?**
Eight models: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM, Linear SVM, Tuned XGBoost, Tuned LightGBM, and a Stacking Ensemble. The best performer was Tuned LightGBM with ROC-AUC 0.826 and F1 0.466.

**Q5: What accuracy did you get?**
Accuracy alone is misleading here — with 86% non-diabetic, any model predicting "No Diabetes" always gets 86% accuracy. I report ROC-AUC (0.826 for best model) and F1-Score (0.466), which are meaningful for imbalanced data. The model correctly identifies 61.7% of diabetic patients (recall).

---

### Level 2 — Technical (Technical Interviewer)

**Q6: Why is accuracy misleading for this dataset?**
The dataset has 86.1% non-diabetic and 13.9% diabetic. A completely useless model that always predicts "No Diabetes" achieves 86.1% accuracy. ROC-AUC measures ranking ability (can the model assign higher probability to a diabetic than a non-diabetic?) independent of threshold. F1 balances precision and recall and doesn't reward ignoring the minority class.

**Q7: What is SMOTE and why did you apply it only to training data?**
SMOTE creates synthetic minority-class samples by interpolating between existing ones in feature space. For each minority sample, it finds k nearest neighbours and creates new points along the line between them. I applied it only to training data because validation and test sets must represent the real class distribution (86/14). Applying SMOTE to test data would create artificially optimistic evaluation results — a form of data leakage.

**Q8: Explain your train/val/test split design.**
- 80% training (then SMOTE-balanced to 50/50)
- 5% validation — pre-SMOTE, real 86/14 distribution — used only for threshold optimisation
- 15% test — pre-SMOTE, real 86/14 distribution — held out completely until final evaluation

The key design decision was splitting the validation set before SMOTE. This ensures threshold search happens at the same distribution as deployment.

**Q9: What is threshold optimisation and why did you do it?**
At default threshold 0.50, XGBoost has only 19% recall — it misses 81% of diabetics. This happens because SMOTE training teaches the model 50/50 class balance, but at test time the class balance returns to 86/14. By scanning the precision-recall curve and finding the threshold that maximises F1 on the real-distribution validation set, we can recover recall from 19% to 65% while keeping F1 higher than the default.

**Q10: Explain the bias-variance tradeoff and where it appears in this project.**
Bias is error from oversimplified model assumptions. Variance is error from sensitivity to training data fluctuations. Logistic Regression shows high bias — it underfits because the relationship between features and diabetes is non-linear. Deep trees show high variance — they overfit to training data. The learning curves show this: LR converges early (bias-limited), while RF shows a gap between training and validation AUC (variance). XGBoost's regularisation parameters (gamma, reg_alpha, reg_lambda) control the bias-variance tradeoff, which is why hyperparameter tuning improved it.

**Q11: What is cross-validation and why did you use 5-fold?**
Cross-validation splits training data into k folds, training on k-1 and validating on the remaining fold, rotating k times. The mean score across folds gives a robust estimate of generalisation performance. 5-fold is the standard choice — it balances computation cost against estimate variance. I used stratified folds to maintain class proportion in each fold.

**Q12: Explain feature importance. What is the difference between Gini importance and SHAP?**
Gini importance (Random Forest built-in) measures how much each feature reduces impurity across all trees — it's biased toward high-cardinality features and doesn't account for feature interactions. SHAP computes exact Shapley values based on marginal contributions across all feature subsets. SHAP is model-agnostic, locally accurate (SHAP values sum to prediction), and doesn't have the high-cardinality bias. GenHlth ranks first in both, confirming consistency, but SHAP gives far more nuanced information about direction and magnitude per sample.

---

### Level 3 — Advanced (Senior Interviewer / ML Engineer Role)

**Q13: How does XGBoost work mathematically?**
XGBoost builds trees sequentially. Each tree fits the negative gradient of the loss function from the previous iteration — this is gradient descent in function space. The objective is:
```
L = Σ l(ŷᵢ, yᵢ) + Σ Ω(fₖ)
```
where l is the loss (log-loss for classification), Ω is the regularisation term: `γT + ½λ||w||²` (T = number of leaves, w = leaf weights). XGBoost uses a second-order Taylor approximation of the loss, allowing efficient closed-form computation of optimal leaf weights.

**Q14: How does SHAP compute exact values for tree models?**
TreeExplainer exploits the tree structure to compute exact Shapley values in O(TLD²) time (T=trees, L=leaves, D=depth) rather than the exponential time of brute-force Shapley computation. For each prediction, it uses the tree's conditional expectation structure to efficiently compute how each feature changes the prediction compared to the expected prediction, averaged across all possible feature orderings.

**Q15: What is data leakage and give three ways you prevented it?**
Data leakage is when information from outside the training set improperly influences the model, leading to overly optimistic evaluation. Three ways prevented: (1) StandardScaler fit only on training data, transform applied to val/test — prevents test statistics influencing normalisation; (2) SMOTE applied only to training data — prevents synthetic test samples; (3) Threshold optimisation done on the held-out validation set, never the test set — prevents threshold overfitting to test data.

**Q16: Why did the Stacking Ensemble underperform?**
Two reasons. First, all three base learners (RF, XGBoost, LightGBM) are tree-based — they're correlated and make similar errors on the same samples. True stacking benefit comes from architecturally diverse base learners. Second, the stacking meta-learner (Logistic Regression) sees out-of-fold probability estimates from base learners trained on SMOTE-balanced data, but the meta-learner is evaluated at real distribution — the probability scales don't align. Probability calibration (Platt scaling) would fix this.

**Q17: How would you deploy this to production?**
Four steps: (1) Save models with joblib, wrap in a FastAPI service with `/predict` endpoint, containerise with Docker; (2) Add input validation and monitoring for data drift (the BRFSS distribution may shift over time); (3) Set up a CI/CD pipeline that retrains monthly on new BRFSS data; (4) Implement A/B testing framework to compare new model versions. The Streamlit app demonstrates the concept but FastAPI + Docker is production-grade.

---

## Metrics Quick Reference

| Metric | Formula | Range | Ideal | This Project (Best) |
|---|---|---|---|---|
| Accuracy | (TP+TN)/Total | 0–1 | High (but misleading) | 80.3% |
| Precision | TP/(TP+FP) | 0–1 | High | 37.4% |
| Recall | TP/(TP+FN) | 0–1 | **High (clinical priority)** | 61.7% |
| F1-Score | 2PR/(P+R) | 0–1 | High | 0.466 |
| ROC-AUC | Area under ROC | 0–1 | **>0.80 = good** | **0.826** |

---

## Resume Bullet Points

```
• Built end-to-end diabetes risk prediction system on 253,680 CDC BRFSS records
  using XGBoost, LightGBM, and Stacking Ensemble; achieved ROC-AUC 0.826

• Applied SMOTE for class imbalance (86/14 split), RandomizedSearchCV 
  hyperparameter tuning (40×5-fold), achieving CV F1 of 0.913/0.914

• Engineered 3 composite features (MetabolicRisk, LifestyleScore, HealthcareBurden);
  MetabolicRisk ranked 3rd globally by SHAP feature importance

• Implemented SHAP explainability (global summary + per-patient waterfall plots)
  and threshold optimisation raising XGBoost F1 from 0.279 → 0.465

• Deployed as 4-tab Streamlit web application with real-time prediction and
  SHAP explanations
```

---

*Prepared by Ankit Kumar | Academy of Skill Development, Kolkata | 2025*
