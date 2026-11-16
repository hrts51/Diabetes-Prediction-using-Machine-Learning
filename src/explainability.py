"""
explainability.py
-----------------
SHAP-based global and local explanations for trained models.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os


def get_shap_explainer(model, X_train: np.ndarray, model_name: str):
    """
    Return the appropriate SHAP explainer for the model type:
      - Tree models (RF, XGBoost, LightGBM) → TreeExplainer (fast, exact)
      - Others (LR, SVM)                    → KernelExplainer (sample-based)
    """
    name_lower = model_name.lower()
    if any(t in name_lower for t in ["forest", "xgboost", "lightgbm", "xgb", "lgb"]):
        explainer = shap.TreeExplainer(model)
        print(f"[shap] TreeExplainer selected for {model_name}")
    else:
        # KernelExplainer needs a background sample; use 100 rows
        background = shap.sample(X_train, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        print(f"[shap] KernelExplainer selected for {model_name} (slower)")
    return explainer


def compute_shap_values(explainer, X: np.ndarray):
    """Compute SHAP values; handle both old and new shap API."""
    shap_vals = explainer(X)
    return shap_vals


def plot_shap_summary(shap_values, X: np.ndarray, feature_names: list,
                      model_name: str, save_dir: str = "outputs/plots") -> None:
    """Beeswarm summary plot — global feature importance."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"shap_summary_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Summary plot saved → {fname}")


def plot_shap_bar(shap_values, feature_names: list,
                  model_name: str, save_dir: str = "outputs/plots") -> None:
    """Bar plot of mean absolute SHAP values — cleaner feature ranking."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, feature_names=feature_names,
        plot_type="bar", show=False, plot_size=None,
    )
    plt.title(f"SHAP Feature Importance (Mean |SHAP|) — {model_name}",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"shap_bar_{model_name.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Bar plot saved → {fname}")


def plot_shap_waterfall(shap_values, idx: int, feature_names: list,
                        model_name: str, save_dir: str = "outputs/plots") -> None:
    """
    Waterfall plot for a single prediction — explains WHY the model
    predicted diabetic/non-diabetic for one specific patient.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(shap_values[idx], show=False)
    plt.title(f"SHAP Waterfall — {model_name} | Sample #{idx}",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"shap_waterfall_{model_name.replace(' ', '_')}_idx{idx}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Waterfall plot saved → {fname}")


def get_top_features(shap_values, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """Return a DataFrame of top N features ranked by mean |SHAP| value."""
    # shap_values may be a shap.Explanation object or a numpy array
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values

    # For binary classifiers, TreeExplainer may return shape (n, f, 2)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.abs(vals).mean(axis=0)
    df = pd.DataFrame({
        "Feature":    feature_names,
        "Mean |SHAP|": mean_abs,
    }).sort_values("Mean |SHAP|", ascending=False).head(top_n).reset_index(drop=True)
    df.index += 1
    return df
