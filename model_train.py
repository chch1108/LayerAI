import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = os.path.join("model_artifacts", "rf_model.joblib")

INPUT_FEATURES = [
    '材料黏度 (cps)',
    '抬升高度(μm)',
    '抬升速度(μm/s)',
    '等待時間(s)',
    '下降速度((μm)/s)',
    '面積(mm?)',
    '周長(mm)',
    '水力直徑(mm)'
]

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model
    else:
        return None

def train_and_save_dummy():
    # small dummy training to create a model if none exists (only for demo)
    X = np.random.rand(200, len(INPUT_FEATURES))
    y = np.random.binomial(1, 0.2, size=200)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    return rf

def _prepare_X(df):
    X = df.copy()
    # ensure columns exist and order
    cols = [c for c in INPUT_FEATURES if c in X.columns]
    return X[cols]

def load_model_and_predict(input_df):
    """
    input_df: pandas DataFrame with required numeric columns
    returns: (probability_of_failure (float), importances_dict)
    """
    model = load_model()
    if model is None:
        model = train_and_save_dummy()

    X = _prepare_X(input_df)
    # if model supports predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        prob = float(probs[0])
    else:
        pred = model.predict(X)[0]
        prob = float(pred)

    # feature importances
    try:
        fi = model.feature_importances_
        importances = {col: float(fi[idx]) for idx, col in enumerate(X.columns)}
    except Exception:
        importances = {col: 1.0/len(X.columns) for col in X.columns}

    return prob, importances
