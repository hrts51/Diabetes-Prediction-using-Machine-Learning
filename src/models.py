"""
models.py
---------
Defines, trains, evaluates and threshold-tunes all models + stacking ensemble.
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, StackingClassifier
from sklearn.svm             import SVC, LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import xgboost  as xgb
import lightgbm as lgb


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models() -> dict:
    """Return all five base models with tuned hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
        # NOTE: Full RBF SVM is O(n²) — impractical on 250K rows.
        # We use a Linear SVM via LinearSVC wrapped with CalibratedClassifierCV
        # so it still supports predict_proba and is trainable at scale.
        "SVM (Linear)": CalibratedClassifierCV(
            LinearSVC(C=0.1, max_iter=2000, random_state=42),
            cv=3,
        ),
    }


def get_stacking_ensemble() -> StackingClassifier:
    """
    Stacking ensemble:
      Base learners : Random Forest, XGBoost, LightGBM
      Meta-learner  : Logistic Regression
    Uses 5-fold CV to generate meta-features.
    """
    base_estimators = [
        ("rf",  RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
        ("xgb", xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=50, class_weight="balanced",
            random_state=42, verbose=-1,
        )),
    ]
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, model_name: str = ""):
    """Fit a model and report training time."""
    import time
    print(f"[train] Training {model_name} ...", end=" ", flush=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"done in {time.time()-t0:.1f}s")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    """Compute all classification metrics on the test set."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1-Score":  round(f1_score(y_test, y_pred),        4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_pred_prob), 4),
    }

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


def cross_validate_model(model, X, y, model_name: str = "", cv: int = 5) -> float:
    """Return mean ROC-AUC from stratified k-fold CV."""
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    mean_auc = scores.mean()
    print(f"[cv]  {model_name:25s} | {cv}-Fold ROC-AUC: {mean_auc:.4f} ± {scores.std():.4f}")
    return mean_auc


# ── Threshold optimisation ────────────────────────────────────────────────────

def find_best_threshold(model, X_val: np.ndarray, y_val: np.ndarray,
                         strategy: str = "f1") -> float:
    """
    Scan all Precision-Recall curve thresholds and return the one that
    maximises the chosen strategy:

      'f1'      – best F1-Score          (balanced precision/recall)
      'recall'  – recall ≥ 0.70, then best F1  (clinical: catch more diabetics)
      'youden'  – Youden's J = Sensitivity + Specificity − 1  (ROC-based)

    Parameters
    ----------
    model    : fitted classifier with predict_proba
    X_val    : validation feature matrix
    y_val    : true labels
    strategy : 'f1' | 'recall' | 'youden'

    Returns
    -------
    float : optimal probability threshold
    """
    probs = model.predict_proba(X_val)[:, 1]

    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_val, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = float(thresholds[best_idx])
        print(f"[thresh] Youden's J → threshold = {best_thresh:.4f}  "
              f"(TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")
        return best_thresh

    # Precision-Recall sweep
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    # thresholds has len = len(precisions) - 1; align
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        f1_scores = np.where(
            (precisions + recalls) == 0, 0,
            2 * precisions * recalls / (precisions + recalls)
        )

    if strategy == "f1":
        best_idx    = np.argmax(f1_scores)
        best_thresh = float(thresholds[best_idx])
        print(f"[thresh] Best F1 → threshold = {best_thresh:.4f}  "
              f"(P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f}, "
              f"F1={f1_scores[best_idx]:.3f})")

    elif strategy == "recall":
        # Among thresholds where recall >= 0.70, pick the one with best F1
        mask = recalls >= 0.70
        if mask.sum() == 0:
            mask = recalls >= recalls.max() * 0.90   # fallback
        best_idx    = np.argmax(f1_scores * mask)
        best_thresh = float(thresholds[best_idx])
        print(f"[thresh] Recall≥0.70 + best F1 → threshold = {best_thresh:.4f}  "
              f"(P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f}, "
              f"F1={f1_scores[best_idx]:.3f})")
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'f1', 'recall' or 'youden'.")

    return best_thresh


def evaluate_at_threshold(model, X_test: np.ndarray, y_test: np.ndarray,
                           threshold: float, model_name: str = "") -> dict:
    """
    Evaluate a model using a custom probability threshold instead of 0.5.
    Returns the same metrics dict as evaluate_model.
    """
    probs  = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    metrics = {
        "Model":     model_name,
        "Threshold": round(threshold, 4),
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1-Score":  round(f1_score(y_test, y_pred),        4),
        "ROC-AUC":   round(roc_auc_score(y_test, probs),    4),
    }

    print(f"\n{'='*55}")
    print(f"  {model_name}  [threshold = {threshold:.4f}]")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


def tune_all_thresholds(trained_models: dict, X_val: np.ndarray,
                        y_val: np.ndarray, X_test: np.ndarray,
                        y_test: np.ndarray,
                        strategy: str = "f1") -> tuple[dict, pd.DataFrame]:
    """
    For every model in trained_models:
      1. Find optimal threshold on the validation set.
      2. Evaluate at that threshold on the test set.

    Parameters
    ----------
    trained_models : dict  name → (model, X_test_for_model)
    X_val, y_val   : held-out validation set for threshold search
    X_test, y_test : final test set for reporting
    strategy       : passed to find_best_threshold

    Returns
    -------
    thresholds_dict : dict  name → float
    tuned_results_df: pd.DataFrame  one row per model
    """
    thresholds_dict = {}
    tuned_results   = []

    print(f"\n{'='*65}")
    print(f"  THRESHOLD OPTIMISATION  (strategy = '{strategy}')")
    print(f"{'='*65}")

    for name, (model, _) in trained_models.items():
        print(f"\n── {name} ──")
        thresh = find_best_threshold(model, X_val, y_val, strategy=strategy)
        thresholds_dict[name] = thresh
        metrics = evaluate_at_threshold(model, X_test, y_test, thresh, model_name=name)
        tuned_results.append(metrics)

    tuned_df = pd.DataFrame(tuned_results)
    return thresholds_dict, tuned_df


# ── Persistence ───────────────────────────────────────────────────────────────
def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[save] Model saved → {path}")


def load_model(path: str):
    model = joblib.load(path)
    print(f"[load] Model loaded ← {path}")
    return model
