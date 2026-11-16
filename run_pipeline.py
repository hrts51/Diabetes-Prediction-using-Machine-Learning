"""
run_pipeline.py
---------------
Complete best-practice ML pipeline:
  1.  Load & validate real CDC BRFSS 2015 data
  2.  EDA plots
  3.  Feature engineering
  4.  Stratified split  →  clean val set  →  SMOTE on train only
  5.  Train 5 base models
  6.  Default-threshold evaluation
  7.  5-Fold cross-validation
  8.  Hyperparameter tuning (XGBoost + LightGBM, 40 iters each)
  9.  Stacking ensemble
  10. Learning curves
  11. Threshold optimisation on real-distribution validation set
  12. SHAP explainability
  13. Save everything

Run from project root:  python run_pipeline.py
"""

import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy  as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from imblearn.over_sampling  import SMOTE

from preprocessing  import (load_data, check_missing, engineer_features,
                             FEATURE_COLS)
from models         import (get_models, get_stacking_ensemble,
                             train_model, evaluate_model, cross_validate_model,
                             find_best_threshold, evaluate_at_threshold)
from tuning         import (tune_xgboost, tune_lightgbm,
                             compute_learning_curves)
from visualizations import (
    plot_target_distribution, plot_feature_distributions,
    plot_correlation_heatmap, plot_boxplots_by_class,
    plot_model_comparison, plot_roc_curves,
    plot_confusion_matrices, plot_feature_importance_rf,
    plot_precision_recall_vs_threshold,
    plot_threshold_comparison, plot_tuned_confusion_matrices,
    plot_final_model_summary, plot_learning_curves,
    plot_all_learning_curves, plot_tuning_improvement,
    plot_combined_feature_importance,
)
from explainability import (get_shap_explainer, compute_shap_values,
                             plot_shap_summary, plot_shap_bar,
                             plot_shap_waterfall, get_top_features)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(ROOT, 'data',
                         'diabetes_binary_health_indicators_BRFSS2015.csv')
PLOT_DIR  = os.path.join(ROOT, 'outputs', 'plots')
MODEL_DIR = os.path.join(ROOT, 'outputs', 'models')
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

import visualizations as _viz
_viz.SAVE_DIR = PLOT_DIR

TREE_MODELS = ['Random Forest', 'XGBoost', 'LightGBM',
               'Tuned XGBoost', 'Tuned LightGBM']

t_pipeline_start = time.time()

print('\n' + '='*65)
print('  DIABETES RISK PREDICTION — BEST-PRACTICE PIPELINE')
print('='*65)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print('\n[1/13] Loading real CDC BRFSS 2015 data ...')
df = load_data(DATA_PATH)
check_missing(df)

# ═══════════════════════════════════════════════════════════════════
# 2. EDA PLOTS
# ═══════════════════════════════════════════════════════════════════
print('\n[2/13] EDA plots ...')
plot_target_distribution(df)
plot_feature_distributions(df, FEATURE_COLS)
plot_correlation_heatmap(df)
continuous = ['BMI','MentHlth','PhysHlth','Age','GenHlth','Education','Income']
plot_boxplots_by_class(df, continuous)

# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
print('\n[3/13] Feature engineering ...')
df_eng = engineer_features(df)

# ═══════════════════════════════════════════════════════════════════
# 4. STRATIFIED SPLIT  (80 / 5 / 15  train / val / test)
#    Val and Test are PRE-SMOTE so they mirror real class imbalance.
#    SMOTE applied to training set only.
# ═══════════════════════════════════════════════════════════════════
print('\n[4/13] Stratified split + SMOTE + scaling ...')

all_features  = FEATURE_COLS + ['MetabolicRisk', 'LifestyleScore', 'HealthcareBurden']
feature_names = [c for c in all_features if c in df_eng.columns]
X_all         = df_eng[feature_names].values
y_all         = df_eng['Diabetes_binary'].values.astype(float)

# 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)

# 5% val from remaining (≈ 5.9% of remaining ≈ 5% of total)
X_trainval, X_val, y_trainval, y_val = train_test_split(
    X_temp, y_temp, test_size=0.059, random_state=42, stratify=y_temp)

print(f'  TrainVal : {X_trainval.shape[0]:,}  '
      f'({np.bincount(y_trainval.astype(int))})')
print(f'  Val      : {X_val.shape[0]:,}  '
      f'({np.bincount(y_val.astype(int))})')
print(f'  Test     : {X_test.shape[0]:,}  '
      f'({np.bincount(y_test.astype(int))})')

# SMOTE on trainval only
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_trainval, y_trainval)
print(f'  After SMOTE train: {np.bincount(y_train.astype(int))}')

# StandardScaler fit on SMOTE-augmented train
scaler         = StandardScaler()
X_train_sc     = scaler.fit_transform(X_train)
X_val_sc       = scaler.transform(X_val)
X_test_sc      = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════════════════
# 5. TRAIN BASE MODELS
# ═══════════════════════════════════════════════════════════════════
print('\n[5/13] Training base models ...')
base_models    = get_models()
trained_models = {}

for name, model in base_models.items():
    X_tr = X_train    if name in TREE_MODELS else X_train_sc
    X_te = X_test     if name in TREE_MODELS else X_test_sc
    m    = train_model(model, X_tr, y_train, model_name=name)
    trained_models[name] = (m, X_te)

# ═══════════════════════════════════════════════════════════════════
# 6. DEFAULT-THRESHOLD EVALUATION
# ═══════════════════════════════════════════════════════════════════
print('\n[6/13] Default-threshold evaluation (0.50) ...')
default_results = []
for name, (m, X_te) in trained_models.items():
    metrics = evaluate_model(m, X_te, y_test, model_name=name)
    default_results.append(metrics)
default_df = pd.DataFrame(default_results)

# ═══════════════════════════════════════════════════════════════════
# 7. 5-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════
print('\n[7/13] 5-Fold Cross-Validation (ROC-AUC) ...')
for name, (m, _) in trained_models.items():
    X_cv = X_train if name in TREE_MODELS else X_train_sc
    cross_validate_model(m, X_cv, y_train, model_name=name)

# Default plots
plot_model_comparison(default_df)
plot_roc_curves({n: m for n,(m,_) in trained_models.items()},
                X_test_sc, y_test)
plot_confusion_matrices({n: m for n,(m,_) in trained_models.items()},
                        X_test_sc, y_test)
plot_feature_importance_rf(trained_models['Random Forest'][0], feature_names)

# ═══════════════════════════════════════════════════════════════════
# 8. HYPERPARAMETER TUNING — XGBoost + LightGBM
# ═══════════════════════════════════════════════════════════════════
print('\n[8/13] Hyperparameter tuning (RandomizedSearchCV 40 iters × 5-fold) ...')

# Use a 30K sample for tuning speed (still stratified)
rng_tune = np.random.default_rng(99)
tune_idx  = rng_tune.choice(len(X_train), size=min(60_000, len(X_train)),
                             replace=False)
X_tune   = X_train[tune_idx]
y_tune   = y_train[tune_idx]

tuned_xgb = tune_xgboost(X_tune,  y_tune, n_iter=40, cv=5)
tuned_lgb = tune_lightgbm(X_tune, y_tune, n_iter=40, cv=5)

# Full retrain on complete training set with best params
print('[8] Retraining tuned models on full training set ...')
t0 = time.time()
tuned_xgb.fit(X_train, y_train)
print(f'  Tuned XGBoost  retrained in {time.time()-t0:.1f}s')
t0 = time.time()
tuned_lgb.fit(X_train, y_train)
print(f'  Tuned LightGBM retrained in {time.time()-t0:.1f}s')

trained_models['Tuned XGBoost']  = (tuned_xgb, X_test)
trained_models['Tuned LightGBM'] = (tuned_lgb, X_test)

# Evaluate tuned models
print('\n--- Tuned model evaluation ---')
for name in ['Tuned XGBoost', 'Tuned LightGBM']:
    m, X_te = trained_models[name]
    metrics = evaluate_model(m, X_te, y_test, model_name=name)
    default_results.append(metrics)

default_df = pd.DataFrame(default_results)

# Tuning improvement plot
before_m = {
    'XGBoost':  default_df[default_df['Model']=='XGBoost'].iloc[0].to_dict(),
    'LightGBM': default_df[default_df['Model']=='LightGBM'].iloc[0].to_dict(),
}
after_m = {
    'XGBoost':  default_df[default_df['Model']=='Tuned XGBoost'].iloc[0].to_dict(),
    'LightGBM': default_df[default_df['Model']=='Tuned LightGBM'].iloc[0].to_dict(),
}
plot_tuning_improvement(before_m, after_m)
plot_combined_feature_importance(
    {n: m for n,(m,_) in trained_models.items()}, feature_names)

# ═══════════════════════════════════════════════════════════════════
# 9. STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════
print('\n[9/13] Training Stacking Ensemble ...')
stack = get_stacking_ensemble()
stack = train_model(stack, X_train, y_train, model_name='Stacking Ensemble')
metrics_stack = evaluate_model(stack, X_test, y_test,
                               model_name='Stacking Ensemble')
default_results.append(metrics_stack)
default_df = pd.DataFrame(default_results)
trained_models['Stacking Ensemble'] = (stack, X_test)

# ═══════════════════════════════════════════════════════════════════
# 10. LEARNING CURVES
# ═══════════════════════════════════════════════════════════════════
print('\n[10/13] Computing learning curves ...')
# Use 20K sample for speed (learning curve retrains model multiple times)
lc_sample = min(20_000, len(X_train))
lc_idx    = rng_tune.choice(len(X_train), size=lc_sample, replace=False)
X_lc      = X_train[lc_idx]
y_lc      = y_train[lc_idx]

lc_results = {}
for name in ['Logistic Regression', 'Random Forest',
             'Tuned XGBoost', 'Tuned LightGBM', 'SVM (Linear)']:
    if name not in trained_models:
        continue
    m, _ = trained_models[name]
    print(f'  Learning curve: {name} ...')
    X_lc_use = X_lc if name in TREE_MODELS else scaler.transform(X_lc)
    try:
        lc = compute_learning_curves(
            m, X_lc_use, y_lc,
            cv=3,
            train_sizes=np.linspace(0.1, 1.0, 8),
            scoring='roc_auc',
        )
        lc_results[name] = lc
        plot_learning_curves(lc, model_name=name)
    except Exception as e:
        print(f'  Skipped {name}: {e}')

if lc_results:
    plot_all_learning_curves(lc_results)

# ═══════════════════════════════════════════════════════════════════
# 11. THRESHOLD OPTIMISATION on real-distribution val set
# ═══════════════════════════════════════════════════════════════════
print('\n[11/13] Threshold optimisation on real-distribution validation set ...')
thresholds_dict = {}
tuned_results   = []

for name, (m, X_te) in trained_models.items():
    print(f'\n── {name} ──')
    X_v = X_val if name in TREE_MODELS else X_val_sc

    thresh = find_best_threshold(m, X_v, y_val, strategy='f1')
    thresholds_dict[name] = thresh
    plot_precision_recall_vs_threshold(m, X_v, y_val, thresh,
                                       model_name=name)
    metrics = evaluate_at_threshold(m, X_te, y_test, thresh,
                                    model_name=name)
    tuned_results.append(metrics)

tuned_df = pd.DataFrame(tuned_results)

plot_threshold_comparison(default_df, tuned_df)
plot_tuned_confusion_matrices(
    {n: m for n,(m,_) in trained_models.items()},
    thresholds_dict, X_test_sc, y_test)
plot_final_model_summary(tuned_df)

# ═══════════════════════════════════════════════════════════════════
# 12. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════
print('\n[12/13] SHAP explainability (Tuned XGBoost) ...')
best_tree_model = trained_models.get('Tuned XGBoost',
                  trained_models.get('XGBoost', None))
if best_tree_model:
    xgb_model  = best_tree_model[0]
    rng_shap   = np.random.default_rng(42)
    samp_idx   = rng_shap.choice(len(X_test), size=2000, replace=False)
    X_samp     = X_test[samp_idx]
    y_samp     = y_test[samp_idx]

    explainer   = get_shap_explainer(xgb_model, X_train[:5000], 'Tuned XGBoost')
    shap_values = compute_shap_values(explainer, X_samp)

    plot_shap_summary(shap_values, X_samp, feature_names,
                      'Tuned XGBoost', save_dir=PLOT_DIR)
    plot_shap_bar(shap_values, feature_names,
                  'Tuned XGBoost', save_dir=PLOT_DIR)

    diab_idx = np.where(y_samp == 1)[0]
    if len(diab_idx) > 0:
        plot_shap_waterfall(shap_values, int(diab_idx[0]),
                            feature_names, 'Tuned XGBoost', save_dir=PLOT_DIR)

    top_df = get_top_features(shap_values, feature_names, top_n=10)
    print('\nTop 10 Features (Mean |SHAP|):\n', top_df.to_string())

# ═══════════════════════════════════════════════════════════════════
# 13. SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════
print('\n[13/13] Saving all models, scaler, thresholds ...')

for name, (m, _) in trained_models.items():
    fname = name.replace(' ', '_').lower() + '.pkl'
    joblib.dump(m, os.path.join(MODEL_DIR, fname))
    print(f'  Saved: {fname}')

joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
print('  Saved: scaler.pkl')

with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
    json.dump(feature_names, f)
print('  Saved: feature_names.json')

with open(os.path.join(MODEL_DIR, 'thresholds.json'), 'w') as f:
    json.dump({k: round(v, 6) for k, v in thresholds_dict.items()}, f, indent=2)
print('  Saved: thresholds.json')

# Save results tables
default_df.to_csv(os.path.join(MODEL_DIR, 'results_default.csv'), index=False)
tuned_df.to_csv(os.path.join(MODEL_DIR, 'results_tuned.csv'), index=False)
print('  Saved: results_default.csv, results_tuned.csv')

elapsed = (time.time() - t_pipeline_start) / 60

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('  DEFAULT THRESHOLD RESULTS')
print('='*65)
print(default_df[['Model','Accuracy','Precision','Recall','F1-Score','ROC-AUC']]
      .sort_values('ROC-AUC', ascending=False).to_string(index=False))

print('\n' + '='*65)
print('  OPTIMISED THRESHOLD RESULTS')
print('='*65)
print(tuned_df[['Model','Threshold','Accuracy','Precision','Recall',
                'F1-Score','ROC-AUC']]
      .sort_values('F1-Score', ascending=False).to_string(index=False))

print('\n' + '='*65)
print('  F1 IMPROVEMENT SUMMARY')
print('='*65)
def_lookup = default_df.set_index('Model')['F1-Score'].to_dict()
for _, row in tuned_df.sort_values('F1-Score', ascending=False).iterrows():
    old = def_lookup.get(row['Model'], 0)
    new = row['F1-Score']
    d   = new - old
    sym = '▲' if d > 0 else ('▼' if d < 0 else '─')
    print(f"  {row['Model']:<27} {old:.4f} → {new:.4f}  "
          f"{sym} {abs(d):.4f}")

best = tuned_df.loc[tuned_df['F1-Score'].idxmax()]
print(f'\n{"="*65}')
print(f'  🏆 BEST MODEL  : {best["Model"]}')
print(f'     Threshold   : {best["Threshold"]:.4f}')
print(f'     Accuracy    : {best["Accuracy"]:.4f}')
print(f'     Precision   : {best["Precision"]:.4f}')
print(f'     Recall      : {best["Recall"]:.4f}')
print(f'     F1-Score    : {best["F1-Score"]:.4f}')
print(f'     ROC-AUC     : {best["ROC-AUC"]:.4f}')
print(f'{"="*65}')
print(f'\nTotal pipeline time : {elapsed:.1f} minutes')
print(f'Plots saved         : {len(os.listdir(PLOT_DIR))} files in outputs/plots/')
print(f'Models saved        : {len(os.listdir(MODEL_DIR))} files in outputs/models/')
print('\nLaunch app: python -m streamlit run app/streamlit_app.py')
