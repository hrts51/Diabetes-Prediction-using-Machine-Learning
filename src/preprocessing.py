"""
preprocessing.py
----------------
All data loading, cleaning, feature engineering and splitting logic
for the CDC BRFSS Diabetes Health Indicators dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "HighBP":            "High Blood Pressure (0=No, 1=Yes)",
    "HighChol":          "High Cholesterol (0=No, 1=Yes)",
    "CholCheck":         "Cholesterol Check in past 5 years (0=No, 1=Yes)",
    "BMI":               "Body Mass Index",
    "Smoker":            "Smoked ≥100 cigarettes in lifetime (0=No, 1=Yes)",
    "Stroke":            "Ever had a stroke (0=No, 1=Yes)",
    "HeartDiseaseorAttack": "Coronary heart disease or MI (0=No, 1=Yes)",
    "PhysActivity":      "Physical activity in past 30 days (0=No, 1=Yes)",
    "Fruits":            "Consumes fruit ≥1 times/day (0=No, 1=Yes)",
    "Veggies":           "Consumes vegetables ≥1 times/day (0=No, 1=Yes)",
    "HvyAlcoholConsump": "Heavy alcohol consumption (0=No, 1=Yes)",
    "AnyHealthcare":     "Any healthcare coverage (0=No, 1=Yes)",
    "NoDocbcCost":       "Could not see doctor due to cost (0=No, 1=Yes)",
    "GenHlth":           "General health (1=Excellent → 5=Poor)",
    "MentHlth":          "Days of poor mental health (0–30)",
    "PhysHlth":          "Days of poor physical health (0–30)",
    "DiffWalk":          "Difficulty walking/climbing stairs (0=No, 1=Yes)",
    "Sex":               "Sex (0=Female, 1=Male)",
    "Age":               "Age category (1=18–24, 13=80+)",
    "Education":         "Education level (1=Never attended → 6=College grad)",
    "Income":            "Income scale (1=<$10K → 8=≥$75K)",
}

TARGET_COL  = "Diabetes_binary"
FEATURE_COLS = list(FEATURE_DESCRIPTIONS.keys())


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and do basic sanity checks."""
    df = pd.read_csv(filepath)
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found."
    print(f"[load]  Shape: {df.shape}")
    print(f"[load]  Target distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(3)}")
    return df


def check_missing(df: pd.DataFrame) -> None:
    """Report any missing values."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[check] No missing values found.")
    else:
        print("[check] Missing values:\n", missing[missing > 0])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three clinically-motivated composite features:

    1. MetabolicRisk  – BMI + blood-pressure + cholesterol combined signal
    2. LifestyleScore – physical activity, fruit/veg intake, alcohol, smoking
    3. HealthcareBurden – mental health days + physical health days
    """
    df = df.copy()

    # 1. Metabolic risk index (higher = worse)
    df["MetabolicRisk"] = (
        df["BMI"] * 0.4
        + df["HighBP"] * 10
        + df["HighChol"] * 8
        + df["HeartDiseaseorAttack"] * 12
        + df["Stroke"] * 10
    )

    # 2. Lifestyle score (higher = healthier)
    df["LifestyleScore"] = (
        df["PhysActivity"] * 3
        + df["Fruits"]
        + df["Veggies"]
        - df["Smoker"] * 2
        - df["HvyAlcoholConsump"] * 2
    )

    # 3. Healthcare burden (days impaired)
    df["HealthcareBurden"] = df["MentHlth"] + df["PhysHlth"]

    print("[feat]  Added 3 engineered features: MetabolicRisk, LifestyleScore, HealthcareBurden")
    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    apply_smote: bool = True,
) -> dict:
    """
    1. Separate features and target.
    2. Stratified train/test split (80/20).
    3. Optionally apply SMOTE on training set to handle class imbalance.
    4. StandardScaler fit on train, applied to both train and test.

    Returns a dict with keys:
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled,
        scaler, feature_names
    """
    all_features = FEATURE_COLS + ["MetabolicRisk", "LifestyleScore", "HealthcareBurden"]
    # Keep only columns that exist (graceful fallback)
    feature_names = [c for c in all_features if c in df.columns]

    X = df[feature_names].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[split] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    if apply_smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"[smote] After SMOTE — {dict(zip(unique, counts))}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("[scale] StandardScaler applied.")

    return {
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "scaler":         scaler,
        "feature_names":  feature_names,
    }
