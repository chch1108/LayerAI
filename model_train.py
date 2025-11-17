import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = os.path.join("model_artifacts", "rf_model.joblib")

# 目前模型不使用 shape
FEATURE_COLUMNS = [
    '材料黏度 (cps)',
    '抬升高度(μm)',
    '抬升速度(μm/s)',
    '等待時間(s)',
    '下降速度((μm)/s)',
    '面積(mm?)',
    '周長(mm)',
    '水力直徑(mm)'
]

def _prepare_X(df):
    """
    確保 input_df 只包含 FEATURE_COLUMNS
    """
    X = df.copy()
    cols = [c for c in FEATURE_COLUMNS if c in X.columns]
    return X[cols]

def train_and_save_model():
    """
    使用 data.csv 訓練並存檔模型
    """
    if not os.path.exists("data.csv"):
        raise FileNotFoundError("找不到 data.csv 用於訓練")
    df = pd.read_csv("data.csv")
    X = _prepare_X(df)
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    return rf

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        try:
            return train_and_save_model()
        except Exception as e:
            print(f"[WARN] 無法載入或訓練模型: {e}")
            return None

def load_model_and_predict(input_df):
    """
    input_df: pd.DataFrame with columns matching FEATURE_COLUMNS
    returns: (probability_of_failure (float), importances_dict)
    """
    model = load_model()
    X = _prepare_X(input_df)

    if model is None:
        # fallback heuristic: 用 input 數值簡單計算概率
        X_sum = X.sum(axis=1).values[0]
        prob = float(np.clip(0.2 + 0.00005 * X_sum, 0.01, 0.9))
        importances = {c: 1.0/len(X.columns) for c in X.columns}
        return prob, importances

    # 使用模型預測
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        prob = float(probs[0])
    else:
        pred = model.predict(X)[0]
        prob = 1.0 if pred == 1 else 0.0

    # feature importances
    try:
        fi = model.feature_importances_
        importances = {col: float(fi[idx]) for idx, col in enumerate(X.columns)}
    except:
        importances = {col: 1.0/len(X.columns) for col in X.columns}

    return prob, importances
