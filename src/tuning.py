"""
tuning.py
---------
Hyperparameter optimisation for XGBoost and LightGBM using
RandomizedSearchCV with stratified k-fold cross-validation.

Also provides learning-curve generation for any fitted estimator.
"""

import numpy as np
import pandas as pd
import time
import os

from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, learning_curve
)
from sklearn.metrics import make_scorer, f1_score
import xgboost  as xgb
import lightgbm as lgb


# ── Hyperparameter search spaces ──────────────────────────────────────────────

XGB_PARAM_DIST = {
    "n_estimators":      [200, 300, 400, 500],
    "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.10],
    "max_depth":         [4, 5, 6, 7, 8],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  [1, 3, 5, 7],
    "gamma":             [0, 0.1, 0.2, 0.3],
    "reg_alpha":         [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda":        [0.5, 1.0, 1.5, 2.0],
    "scale_pos_weight":  [1, 3, 5, 6],   # handles class imbalance in XGBoost
}

LGB_PARAM_DIST = {
    "n_estimators":      [200, 300, 400, 500],
    "learning_rate":     [0.01, 0.03, 0.05, 0.08, 0.10],
    "max_depth":         [4, 5, 6, 7, 8],
    "num_leaves":        [31, 50, 70, 100, 127],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_samples": [10, 20, 30, 50],
    "reg_alpha":         [0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda":        [0, 0.1, 0.5, 1.0],
    "class_weight":      ["balanced"],
}


# ── Tuning functions ──────────────────────────────────────────────────────────

def tune_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                 n_iter: int = 40, cv: int = 5,
                 random_state: int = 42) -> xgb.XGBClassifier:
    """
    RandomizedSearchCV over XGB_PARAM_DIST.
    Optimises F1-score (better for imbalanced data than accuracy).
    Returns the best fitted estimator.
    """
    base = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )

    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        scoring=scorer,
        cv=skf,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    print(f"[tune] XGBoost — RandomizedSearchCV ({n_iter} iters × {cv}-fold) ...")
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"[tune] XGBoost best params  : {search.best_params_}")
    print(f"[tune] XGBoost best CV F1   : {search.best_score_:.4f}")
    print(f"[tune] Completed in {elapsed:.0f}s")
    return search.best_estimator_


def tune_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                  n_iter: int = 40, cv: int = 5,
                  random_state: int = 42) -> lgb.LGBMClassifier:
    """
    RandomizedSearchCV over LGB_PARAM_DIST.
    Optimises F1-score.
    Returns the best fitted estimator.
    """
    base = lgb.LGBMClassifier(
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )

    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=LGB_PARAM_DIST,
        n_iter=n_iter,
        scoring=scorer,
        cv=skf,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    print(f"[tune] LightGBM — RandomizedSearchCV ({n_iter} iters × {cv}-fold) ...")
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"[tune] LightGBM best params : {search.best_params_}")
    print(f"[tune] LightGBM best CV F1  : {search.best_score_:.4f}")
    print(f"[tune] Completed in {elapsed:.0f}s")
    return search.best_estimator_


def get_tuning_results_df(search_xgb: RandomizedSearchCV,
                           search_lgb: RandomizedSearchCV) -> pd.DataFrame:
    """Return a tidy DataFrame of top-10 candidates from each search."""
    rows = []
    for name, search in [("XGBoost", search_xgb), ("LightGBM", search_lgb)]:
        cv_df = pd.DataFrame(search.cv_results_)
        top   = (cv_df[["params", "mean_test_score", "std_test_score"]]
                 .sort_values("mean_test_score", ascending=False)
                 .head(10))
        top.insert(0, "Model", name)
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


# ── Learning curves ───────────────────────────────────────────────────────────

def compute_learning_curves(model, X: np.ndarray, y: np.ndarray,
                              cv: int = 5,
                              train_sizes: np.ndarray = None,
                              scoring: str = "roc_auc") -> dict:
    """
    Compute learning curves for a fitted (or unfitted) model.
    Uses stratified k-fold CV.

    Returns dict with keys:
        train_sizes, train_scores_mean, train_scores_std,
        val_scores_mean, val_scores_std
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.10, 1.0, 10)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    sizes, train_sc, val_sc = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )

    return {
        "train_sizes":       sizes,
        "train_scores_mean": train_sc.mean(axis=1),
        "train_scores_std":  train_sc.std(axis=1),
        "val_scores_mean":   val_sc.mean(axis=1),
        "val_scores_std":    val_sc.std(axis=1),
    }
