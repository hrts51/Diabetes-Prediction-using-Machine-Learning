"""
streamlit_app.py
----------------
Diabetes Risk Prediction — Interactive Web App
Run: python -m streamlit run app/streamlit_app.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import shap

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "outputs", "models")
PLOTS_DIR  = os.path.join(ROOT, "outputs", "plots")
SRC_DIR    = os.path.join(ROOT, "src")
sys.path.insert(0, SRC_DIR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border-left: 4px solid #3498db;
    }
    .risk-high   { border-left-color: #e74c3c !important; }
    .risk-medium { border-left-color: #f39c12 !important; }
    .risk-low    { border-left-color: #2ecc71 !important; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_assets():
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest":       "random_forest.pkl",
        "XGBoost":             "xgboost.pkl",
        "LightGBM":            "lightgbm.pkl",
        "SVM (Linear)":        "svm_(linear).pkl",
        "Tuned XGBoost":       "tuned_xgboost.pkl",
        "Tuned LightGBM":      "tuned_lightgbm.pkl",
        "Stacking Ensemble":   "stacking_ensemble.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    thresh_path = os.path.join(MODELS_DIR, "thresholds.json")
    thresholds  = json.load(open(thresh_path)) if os.path.exists(thresh_path) else {}

    results_path = os.path.join(MODELS_DIR, "results_tuned.csv")
    results_df   = pd.read_csv(results_path) if os.path.exists(results_path) else None

    return models, scaler, feature_names, thresholds, results_df


def _plot(fname):
    """Display a saved plot if it exists."""
    p = os.path.join(PLOTS_DIR, fname)
    if os.path.exists(p):
        st.image(p, use_container_width=True)
    else:
        st.info(f"Plot not found: {fname}. Run run_pipeline.py first.")


# ── Sidebar inputs ────────────────────────────────────────────────────────────
def collect_inputs(feature_names):
    st.sidebar.header("👤 Patient Health Inputs")
    inputs = {}

    with st.sidebar:
        st.markdown("**🫀 Cardiovascular**")
        inputs["HighBP"]               = st.selectbox("High Blood Pressure",        [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["HighChol"]             = st.selectbox("High Cholesterol",            [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["CholCheck"]            = st.selectbox("Cholesterol Checked (5yr)",   [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["Stroke"]               = st.selectbox("Ever had Stroke",             [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["HeartDiseaseorAttack"] = st.selectbox("Heart Disease/MI",            [0,1], format_func=lambda x:"Yes" if x else "No")

        st.markdown("---")
        st.markdown("**⚖️ Body & Lifestyle**")
        inputs["BMI"]               = st.slider("BMI",                              10.0, 98.0, 28.0, 0.5)
        inputs["Smoker"]            = st.selectbox("Smoker (100+ cigarettes)",       [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["HvyAlcoholConsump"] = st.selectbox("Heavy Alcohol Use",              [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["PhysActivity"]      = st.selectbox("Physical Activity (30d)",        [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["Fruits"]            = st.selectbox("Fruits Daily",                   [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["Veggies"]           = st.selectbox("Vegetables Daily",               [0,1], format_func=lambda x:"Yes" if x else "No")

        st.markdown("---")
        st.markdown("**🏥 Healthcare**")
        inputs["AnyHealthcare"] = st.selectbox("Healthcare Coverage",    [0,1], format_func=lambda x:"Yes" if x else "No")
        inputs["NoDocbcCost"]   = st.selectbox("Skipped Doctor (Cost)",  [0,1], format_func=lambda x:"Yes" if x else "No")

        st.markdown("---")
        st.markdown("**📊 General Health**")
        inputs["GenHlth"]  = st.slider("General Health  (1=Excellent → 5=Poor)",  1, 5, 3)
        inputs["MentHlth"] = st.slider("Poor Mental Health Days (last 30)",        0, 30, 3)
        inputs["PhysHlth"] = st.slider("Poor Physical Health Days (last 30)",      0, 30, 3)
        inputs["DiffWalk"] = st.selectbox("Difficulty Walking/Climbing", [0,1], format_func=lambda x:"Yes" if x else "No")

        st.markdown("---")
        st.markdown("**👤 Demographics**")
        inputs["Sex"]       = st.selectbox("Sex",        [0,1], format_func=lambda x:"Male" if x else "Female")
        inputs["Age"]       = st.slider("Age Category  (1=18-24 → 13=80+)",  1, 13, 7)
        inputs["Education"] = st.slider("Education  (1=None → 6=College)",   1, 6,  5)
        inputs["Income"]    = st.slider("Income  (1=<$10K → 8=≥$75K)",       1, 8,  5)

    # Engineered features
    inputs["MetabolicRisk"]      = (inputs["BMI"]*0.4 + inputs["HighBP"]*10
                                    + inputs["HighChol"]*8
                                    + inputs["HeartDiseaseorAttack"]*12
                                    + inputs["Stroke"]*10)
    inputs["LifestyleScore"]     = (inputs["PhysActivity"]*3 + inputs["Fruits"]
                                    + inputs["Veggies"] - inputs["Smoker"]*2
                                    - inputs["HvyAlcoholConsump"]*2)
    inputs["HealthcareBurden"]   = inputs["MentHlth"] + inputs["PhysHlth"]

    row = np.array([[inputs.get(f, 0) for f in feature_names]])
    return row, inputs


# ── SHAP explanation ──────────────────────────────────────────────────────────
def shap_waterfall(model, X_raw, feature_names):
    try:
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer(X_raw)
        fig, _      = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_vals[0], show=False)
        plt.title("Why this prediction? (SHAP Explanation)", fontsize=12,
                  fontweight="bold")
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Header
    st.title("🩺 Diabetes Risk Prediction System")
    st.markdown(
        "**Explainable ML** · CDC BRFSS 2015 · 253,680 Records · "
        "6 Models + Stacking Ensemble + SHAP"
    )
    st.markdown("---")

    try:
        models, scaler, feature_names, thresholds, results_df = load_assets()
    except Exception as e:
        st.error(f"⚠️ Models not found. Run `python run_pipeline.py` first.\n\n`{e}`")
        st.stop()

    # ── Tabs
    tab_predict, tab_results, tab_plots, tab_about = st.tabs([
        "🔍 Predict", "📊 Model Results", "📈 Plots", "ℹ️ About"
    ])

    # ════════════════════════════════════════════════════
    # TAB 1 — PREDICT
    # ════════════════════════════════════════════════════
    with tab_predict:
        X_raw, inputs_dict = collect_inputs(feature_names)
        X_sc = scaler.transform(X_raw)

        preferred = ["Tuned XGBoost", "Tuned LightGBM", "Stacking Ensemble",
                     "XGBoost", "LightGBM"]
        default_model = next((m for m in preferred if m in models), list(models.keys())[0])

        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            model_choice = st.selectbox("Select Model", list(models.keys()),
                                        index=list(models.keys()).index(default_model))
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

        if predict_btn:
            model  = models[model_choice]
            thresh = thresholds.get(model_choice, 0.5)
            prob   = model.predict_proba(X_sc)[0][1]
            pred   = int(prob >= thresh)
            pct    = prob * 100

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction",  "🔴 DIABETIC" if pred == 1 else "🟢 NON-DIABETIC")
            c2.metric("Risk Probability", f"{pct:.1f}%")
            c3.metric("Optimal Threshold", f"{thresh:.3f}")
            c4.metric("Model", model_choice)

            if pct < 25:
                st.success(f"✅ **Low Risk** ({pct:.1f}%) — Minimal diabetes risk indicators.")
            elif pct < 50:
                st.warning(f"⚠️ **Moderate Risk** ({pct:.1f}%) — Some risk factors present. Consider lifestyle review.")
            else:
                st.error(f"🚨 **High Risk** ({pct:.1f}%) — Multiple risk factors detected. Consult a physician.")

            st.progress(min(pct / 100, 1.0))

            # All models
            st.markdown("#### All Models — Comparative Prediction")
            rows = []
            for mname, m in models.items():
                p = m.predict_proba(X_sc)[0][1]
                t = thresholds.get(mname, 0.5)
                rows.append({
                    "Model":      mname,
                    "Risk %":     f"{p*100:.1f}%",
                    "Threshold":  f"{t:.3f}",
                    "Prediction": "🔴 Diabetic" if p >= t else "🟢 Non-Diabetic",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            # SHAP
            if model_choice in ["Tuned XGBoost", "XGBoost", "Tuned LightGBM",
                                 "LightGBM", "Random Forest"]:
                st.markdown("#### 🔬 Why this prediction? (SHAP Explanation)")
                st.caption("Red bars push toward *Diabetic*. Blue bars push toward *Non-Diabetic*.")
                with st.spinner("Computing SHAP ..."):
                    X_raw_tree = X_raw  # tree models don't need scaling
                    fig = shap_waterfall(model, X_raw_tree, feature_names)
                    if fig:
                        st.pyplot(fig, use_container_width=True)

            # Input summary
            with st.expander("📋 Full Input Summary"):
                df_in = pd.DataFrame.from_dict(inputs_dict, orient="index",
                                               columns=["Value"])
                df_in.index.name = "Feature"
                st.dataframe(df_in, use_container_width=True)

    # ════════════════════════════════════════════════════
    # TAB 2 — MODEL RESULTS
    # ════════════════════════════════════════════════════
    with tab_results:
        st.subheader("Model Performance — Optimised Thresholds")

        if results_df is not None:
            # Highlight best in each column
            numeric_cols = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
            st.dataframe(
                results_df.sort_values("F1-Score", ascending=False)
                           .style
                           .highlight_max(subset=numeric_cols,
                                          color="#90EE90")
                           .format({c: "{:.4f}" for c in numeric_cols}),
                use_container_width=True,
            )
        else:
            st.info("Run the pipeline to generate results.")

        st.markdown("---")
        st.subheader("Key Metrics Explained")
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
**ROC-AUC** — Ability to rank a diabetic patient above a non-diabetic across all 
thresholds. The primary metric for imbalanced medical data.

**F1-Score** — Harmonic mean of Precision and Recall. Balances catching diabetics 
(recall) vs. avoiding false alarms (precision).
""")
        with col2:
            st.info("""
**Recall (Sensitivity)** — % of actual diabetics correctly identified.
In medical screening, missing a diabetic (false negative) is costlier 
than a false alarm — so recall matters most.

**Threshold** — Custom cutoff per model (not 0.50). Found by optimising 
F1 on a held-out validation set matching real class imbalance.
""")

    # ════════════════════════════════════════════════════
    # TAB 3 — PLOTS
    # ════════════════════════════════════════════════════
    with tab_plots:
        sections = {
            "📊 EDA": [
                ("01_target_distribution.png",   "Class Distribution"),
                ("02_feature_distributions.png", "Feature Distributions by Class"),
                ("03_correlation_heatmap.png",   "Feature Correlation Heatmap"),
                ("04_boxplots_by_class.png",      "Boxplots by Class"),
            ],
            "🤖 Model Performance": [
                ("05_model_comparison.png",        "Model Comparison (Default Threshold)"),
                ("06_roc_curves.png",              "ROC Curves — All Models"),
                ("07_confusion_matrices.png",      "Confusion Matrices (Default)"),
                ("08_rf_feature_importance.png",   "Random Forest Feature Importance"),
            ],
            "⚙️ Hyperparameter Tuning": [
                ("15_tuning_improvement.png",       "Before vs After Tuning"),
                ("16_combined_feature_importance.png", "Feature Importance — All Tree Models"),
            ],
            "🎯 Threshold Optimisation": [
                ("10_threshold_comparison.png",     "Default vs Optimised Threshold"),
                ("11_tuned_confusion_matrices.png", "Confusion Matrices (Optimised)"),
                ("12_final_model_ranking.png",      "Final Model Ranking"),
            ],
            "📉 Learning Curves": [
                ("14_all_learning_curves.png", "Learning Curves — All Models"),
            ],
            "🔬 SHAP Explainability": [
                ("shap_summary_Tuned_XGBoost.png",   "SHAP Summary (Beeswarm)"),
                ("shap_bar_Tuned_XGBoost.png",       "SHAP Feature Importance (Bar)"),
                ("shap_waterfall_Tuned_XGBoost_idx0.png", "SHAP Waterfall (Patient Example)"),
            ],
        }

        for section, plots in sections.items():
            st.markdown(f"### {section}")
            for fname, caption in plots:
                p = os.path.join(PLOTS_DIR, fname)
                if os.path.exists(p):
                    st.image(p, caption=caption, use_container_width=True)
            st.markdown("---")

    # ════════════════════════════════════════════════════
    # TAB 4 — ABOUT
    # ════════════════════════════════════════════════════
    with tab_about:
        st.subheader("Project Overview")
        st.markdown("""
**Diabetes Risk Prediction — Explainable ML System**

| Item | Detail |
|---|---|
| **Dataset** | CDC BRFSS 2015 — 253,680 real US adult survey responses |
| **Features** | 21 original + 3 engineered (MetabolicRisk, LifestyleScore, HealthcareBurden) |
| **Target** | Binary — Diabetic / Non-Diabetic |
| **Class split** | ~86% Non-Diabetic / ~14% Diabetic |
| **Models** | Logistic Regression, Random Forest, XGBoost, LightGBM, SVM (Linear), Tuned XGBoost, Tuned LightGBM, Stacking Ensemble |
| **Imbalance** | SMOTE applied to training set only |
| **Tuning** | RandomizedSearchCV (40 iterations × 5-fold) on XGBoost and LightGBM |
| **Threshold** | Per-model optimal threshold from F1 optimisation on real-distribution val set |
| **Explainability** | SHAP TreeExplainer — global summary + local waterfall plots |

---

**Tech Stack:** Python 3.14 · scikit-learn 1.9 · XGBoost 3.4 · LightGBM 4.7 · 
SHAP 0.54 · imbalanced-learn 0.14 · Streamlit 1.67

**Developed by:** Ankit Kumar  
**Internship:** Academy of Skill Development, Kolkata — Data Science, AI & ML (May–July 2025)
""")
        st.markdown("---")
        st.caption(
            "⚠️ **Medical Disclaimer:** Educational purposes only. "
            "Not a substitute for professional medical advice."
        )


if __name__ == "__main__":
    main()
