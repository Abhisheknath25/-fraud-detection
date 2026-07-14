"""
preprocess.py — Data loading, cleaning, feature engineering, and balancing.

Handles the Credit Card Fraud Detection dataset from Kaggle:
- Standardizes Amount and Time features
- Applies SMOTE for class imbalance
- Returns train/test splits ready for model training
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATASET_PATH = os.path.join(DATA_DIR, "creditcard.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the credit card transactions CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Please download 'creditcard.csv' from:\n"
            "  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place it in the 'data/' directory."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[INFO] Fraud ratio: {df['Class'].mean():.4%}")
    return df


def preprocess(df: pd.DataFrame, fit_scaler: bool = True):
    """
    Preprocess the raw dataframe.

    Steps
    -----
    1. Scale `Time` and `Amount` using StandardScaler.
    2. Drop original `Time` and `Amount` columns.
    3. Split into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Raw credit‑card dataset.
    fit_scaler : bool
        If True, fit a new scaler and save it.  If False, load the saved
        scaler (used at inference time).

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    """
    df = df.copy()

    scaler = StandardScaler()

    if fit_scaler:
        df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved to {SCALER_PATH}")
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                "Scaler not found. Train the model first to generate the scaler."
            )
        scaler = joblib.load(SCALER_PATH)
        df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    X = df.drop("Class", axis=1).values
    y = df["Class"].values
    return X, y


def split_data(X: np.ndarray, y: np.ndarray):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[INFO] Train set: {X_train.shape[0]:,} samples")
    print(f"[INFO] Test  set: {X_test.shape[0]:,} samples")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """Apply SMOTE to balance the training set."""
    print("[INFO] Applying SMOTE to handle class imbalance …")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE — class 0: {(y_resampled == 0).sum():,}, "
          f"class 1: {(y_resampled == 1).sum():,}")
    return X_resampled, y_resampled


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def prepare_data(dataset_path: str = DATASET_PATH):
    """
    End‑to‑end data preparation pipeline.

    Returns
    -------
    X_train_sm, X_test, y_train_sm, y_test
    """
    df = load_dataset(dataset_path)
    X, y = preprocess(df, fit_scaler=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    return X_train_sm, X_test, y_train_sm, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print("\n✅ Data preparation complete.")
    print(f"   Training features shape : {X_train.shape}")
    print(f"   Test features shape     : {X_test.shape}")
