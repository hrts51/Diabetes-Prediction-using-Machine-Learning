"""
visualizations.py
-----------------
All plotting helpers: EDA, model comparison, ROC curves, confusion matrices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

SAVE_DIR = "outputs/plots"
PALETTE  = ["#2ecc71", "#e74c3c"]   # green=No Diabetes, red=Diabetes


def _save(fig, fname: str) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]  Saved → {path}")


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_target_distribution(df, target_col: str = "Diabetes_binary") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df[target_col].value_counts()
    labels = ["Non-Diabetic", "Diabetic"]

    # Count bar
    axes[0].bar(labels, counts.values, color=PALETTE, edgecolor="white", linewidth=1.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=PALETTE, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 12})
    axes[1].set_title("Class Proportion", fontsize=14, fontweight="bold")

    fig.suptitle("Target Variable Distribution", fontsize=16, fontweight="bold", y=1.02)
    _save(fig, "01_target_distribution.png")


def plot_feature_distributions(df, feature_names: list,
                                target_col: str = "Diabetes_binary") -> None:
    n_cols = 4
    n_rows = int(np.ceil(len(feature_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        for label, color in zip([0, 1], PALETTE):
            subset = df[df[target_col] == label][feat]
            axes[i].hist(subset, bins=30, alpha=0.6, color=color,
                         label=["Non-Diabetic", "Diabetic"][label], density=True)
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Density")

    # Hide unused subplots
    for j in range(len(feature_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Diabetes Status", fontsize=16, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_feature_distributions.png")


def plot_correlation_heatmap(df) -> None:
    fig, ax = plt.subplots(figsize=(16, 13))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        annot_kws={"size": 7}, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_correlation_heatmap.png")


def plot_boxplots_by_class(df, continuous_features: list,
                            target_col: str = "Diabetes_binary") -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(continuous_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(continuous_features):
        df_plot = df[[feat, target_col]].copy()
        df_plot[target_col] = df_plot[target_col].map({0: "Non-Diabetic", 1: "Diabetic"})
        sns.boxplot(
            x=target_col, y=feat, data=df_plot,
            palette={"Non-Diabetic": PALETTE[0], "Diabetic": PALETTE[1]},
            ax=axes[i],
        )
        axes[i].set_title(feat, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")

    for j in range(len(continuous_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Continuous Feature Distribution by Diabetes Status",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_boxplots_by_class.png")


# ── Model performance plots ───────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """Bar chart comparing all models across five metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    models  = results_df["Model"].tolist()
    x       = np.arange(len(models))
    width   = 0.15

    fig, ax = plt.subplots(figsize=(16, 7))
    colors  = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = results_df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title("Model Performance Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save(fig, "05_model_comparison.png")


def plot_roc_curves(models_dict: dict, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot all ROC curves on one figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for (name, model), color in zip(models_dict.items(), colors):
        probs      = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc    = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "06_roc_curves.png")


def plot_confusion_matrices(models_dict: dict,
                            X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Grid of confusion matrices, one per model."""
    n       = len(models_dict)
    n_cols  = 3
    n_rows  = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes    = axes.flatten()

    for i, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Diabetic", "Diabetic"],
            yticklabels=["Non-Diabetic", "Diabetic"],
            ax=axes[i], linewidths=1,
        )
        axes[i].set_title(name, fontsize=12, fontweight="bold")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "07_confusion_matrices.png")


def plot_feature_importance_rf(model, feature_names: list) -> None:
    """Random Forest built-in Gini importance (quick baseline)."""
    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1][:15]  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
        color="#3498db", alpha=0.85,
    )
    ax.set_title("Random Forest — Feature Importance (Gini)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    _save(fig, "08_rf_feature_importance.png")


# ── Threshold optimisation plots ──────────────────────────────────────────────

def plot_precision_recall_vs_threshold(model, X_val: np.ndarray,
                                        y_val: np.ndarray,
                                        best_threshold: float,
                                        model_name: str) -> None:
    """
    Line plot of Precision, Recall and F1 vs decision threshold.
    Marks the chosen optimal threshold with a vertical line.
    """
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        f1s = np.where(
            (precisions + recalls) == 0, 0,
            2 * precisions * recalls / (precisions + recalls)
        )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, precisions, color="#3498db", lw=2, label="Precision")
    ax.plot(thresholds, recalls,    color="#e74c3c", lw=2, label="Recall")
    ax.plot(thresholds, f1s,        color="#2ecc71", lw=2, label="F1-Score")

    ax.axvline(best_threshold, color="#f39c12", lw=2.5, linestyle="--",
               label=f"Optimal threshold = {best_threshold:.3f}")

    # Annotate the intersection point
    # Find the F1 value at best_threshold
    idx = np.argmin(np.abs(thresholds - best_threshold))
    ax.scatter(best_threshold, f1s[idx], color="#f39c12", s=120, zorder=5)
    ax.annotate(f"F1 = {f1s[idx]:.3f}",
                xy=(best_threshold, f1s[idx]),
                xytext=(best_threshold + 0.03, f1s[idx] - 0.05),
                fontsize=10, color="#f39c12",
                arrowprops=dict(arrowstyle="->", color="#f39c12"))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Precision / Recall / F1 vs Threshold — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, f"09_threshold_curve_{model_name.replace(' ', '_')}.png")


def plot_threshold_comparison(default_df: pd.DataFrame,
                               tuned_df: pd.DataFrame) -> None:
    """
    Side-by-side grouped bar chart comparing default (threshold=0.5)
    vs optimised threshold results for every model.
    Shows Accuracy, Precision, Recall and F1-Score.
    """
    metrics  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models   = default_df["Model"].tolist()
    n_models = len(models)
    n_metrics = len(metrics)

    x     = np.arange(n_models)
    width = 0.18

    fig, axes = plt.subplots(1, n_metrics, figsize=(22, 6), sharey=False)
    colors_default = "#95a5a6"
    colors_tuned   = "#2ecc71"

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        default_vals = default_df.set_index("Model").loc[models, metric].values
        tuned_vals   = tuned_df.set_index("Model").loc[models, metric].values

        b1 = ax.bar(x - width / 2, default_vals, width, label="Default (0.50)",
                    color=colors_default, alpha=0.85, edgecolor="white")
        b2 = ax.bar(x + width / 2, tuned_vals,   width, label="Optimised",
                    color=colors_tuned,   alpha=0.85, edgecolor="white")

        # Value labels
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90,
                    color="#27ae60", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Default Threshold (0.50) vs Optimised Threshold — All Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_threshold_comparison.png")


def plot_tuned_confusion_matrices(models_dict: dict,
                                   thresholds_dict: dict,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray) -> None:
    """Confusion matrices evaluated at each model's optimal threshold."""
    n      = len(models_dict)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models_dict.items()):
        thresh = thresholds_dict.get(name, 0.5)
        probs  = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= thresh).astype(int)
        cm     = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Non-Diabetic", "Diabetic"],
            yticklabels=["Non-Diabetic", "Diabetic"],
            ax=axes[i], linewidths=1,
        )
        axes[i].set_title(f"{name}\n(thresh={thresh:.3f})",
                          fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices at Optimised Thresholds",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "11_tuned_confusion_matrices.png")


def plot_final_model_summary(tuned_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart ranking all models by F1-Score (tuned threshold).
    Clean, presentation-ready.
    """
    df = tuned_df.sort_values("F1-Score", ascending=True).copy()

    colors = ["#e74c3c" if name == df["Model"].iloc[-1] else "#3498db"
              for name in df["Model"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(df["Model"], df["F1-Score"], color=colors, alpha=0.88, edgecolor="white")

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"F1={row['F1-Score']:.3f}  AUC={row['ROC-AUC']:.3f}",
                va="center", fontsize=10)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1-Score (Optimised Threshold)", fontsize=12)
    ax.set_title("Final Model Ranking — Optimised Threshold",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "12_final_model_ranking.png")


# ── Learning curves ───────────────────────────────────────────────────────────

def plot_learning_curves(lc_dict: dict, model_name: str) -> None:
    """
    Plot training vs validation ROC-AUC learning curves.
    Reveals overfitting (gap between curves) and whether more data helps.

    lc_dict: output of tuning.compute_learning_curves()
    """
    sizes  = lc_dict["train_sizes"]
    tr_m   = lc_dict["train_scores_mean"]
    tr_s   = lc_dict["train_scores_std"]
    va_m   = lc_dict["val_scores_mean"]
    va_s   = lc_dict["val_scores_std"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, tr_m, "o-", color="#3498db", lw=2, label="Training ROC-AUC")
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color="#3498db")

    ax.plot(sizes, va_m, "o-", color="#e74c3c", lw=2, label="Validation ROC-AUC")
    ax.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.15, color="#e74c3c")

    gap = tr_m[-1] - va_m[-1]
    if gap > 0.05:
        verdict = f"⚠ Overfitting (gap = {gap:.3f})"
    elif va_m[-1] < 0.75:
        verdict = f"⚠ Underfitting (val AUC = {va_m[-1]:.3f})"
    else:
        verdict = f"✓ Good fit (gap = {gap:.3f})"

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title(f"Learning Curves — {model_name}\n{verdict}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"13_learning_curve_{model_name.replace(' ', '_')}.png"
    _save(fig, fname)


def plot_all_learning_curves(lc_results: dict) -> None:
    """
    Multi-panel learning curves for all models on one figure.
    lc_results: dict  model_name → lc_dict
    """
    n      = len(lc_results)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = axes.flatten()

    for i, (name, lc) in enumerate(lc_results.items()):
        sizes = lc["train_sizes"]
        tr_m  = lc["train_scores_mean"]
        tr_s  = lc["train_scores_std"]
        va_m  = lc["val_scores_mean"]
        va_s  = lc["val_scores_std"]

        axes[i].plot(sizes, tr_m, "o-", color="#3498db", lw=2, label="Train")
        axes[i].fill_between(sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.12, color="#3498db")
        axes[i].plot(sizes, va_m, "o-", color="#e74c3c", lw=2, label="Val")
        axes[i].fill_between(sizes, va_m-va_s, va_m+va_s, alpha=0.12, color="#e74c3c")

        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_ylim(0.5, 1.02)
        axes[i].set_xlabel("Train Size")
        axes[i].set_ylabel("ROC-AUC")
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Learning Curves — All Models", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, "14_all_learning_curves.png")


# ── Hyperparameter tuning results ─────────────────────────────────────────────

def plot_tuning_improvement(before_metrics: dict, after_metrics: dict) -> None:
    """
    Before vs After hyperparameter tuning comparison for XGBoost and LightGBM.
    before_metrics / after_metrics: dict  model_name → metrics_dict
    """
    models  = list(before_metrics.keys())
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5))

    for ax, metric in zip(axes, metrics):
        before_vals = [before_metrics[m].get(metric, 0) for m in models]
        after_vals  = [after_metrics[m].get(metric, 0)  for m in models]
        x = np.arange(len(models))

        ax.bar(x - 0.2, before_vals, 0.35, label="Before tuning",
               color="#95a5a6", alpha=0.85)
        ax.bar(x + 0.2, after_vals,  0.35, label="After tuning",
               color="#e74c3c", alpha=0.85)

        for xi, (bv, av) in enumerate(zip(before_vals, after_vals)):
            delta = av - bv
            ax.text(xi + 0.2, av + 0.01, f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
                    ha="center", fontsize=8, color="#c0392b" if delta < 0 else "#27ae60",
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if metric == metrics[0]:
            ax.legend(fontsize=9)

    fig.suptitle("Hyperparameter Tuning — Before vs After",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "15_tuning_improvement.png")


# ── Feature importance — combined ─────────────────────────────────────────────

def plot_combined_feature_importance(models_dict: dict,
                                      feature_names: list) -> None:
    """
    Side-by-side feature importance from RF (Gini), XGBoost and LightGBM.
    Shows consensus: features that rank high across all three are most reliable.
    """
    tree_models = {k: v for k, v in models_dict.items()
                   if k in ["Random Forest", "XGBoost", "LightGBM",
                             "Tuned XGBoost", "Tuned LightGBM"]}

    n = len(tree_models)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(n * 7, 8))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        try:
            importances = model.feature_importances_
        except AttributeError:
            continue
        idx = np.argsort(importances)[::-1][:15]

        ax.barh([feature_names[i] for i in idx[::-1]],
                importances[idx[::-1]],
                color="#3498db", alpha=0.82)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Feature Importance — Tree Models Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "16_combined_feature_importance.png")
