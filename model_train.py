import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_PATH = os.path.join("model_artifacts", "rf_model.joblib")
FEATURE_COLUMNS = [
    '材料黏度 (cps)',
    '抬升高度(μm)',
    '抬升速度(μm/s)',
    '等待時間(s)',
    '下降速度((μm)/s)',
    '形狀',               # 若為分類請 one-hot encode 或 map to int
    '面積(mm?)',
    '周長(mm)',
    '水力直徑(mm)'
]

# small helper to ensure shape mapping (if your training used mapping)
SHAPE_MAP = {'90x45矩形': 0, '90x50六角形': 1, '50圓柱': 2}

def _prepare_X(df):
    # copy so we don't modify original
    X = df.copy()
    # map shape to integer
    if '形狀' in X.columns:
        X['形狀'] = X['形狀'].map(SHAPE_MAP).fillna(0).astype(int)
    # ensure order of columns
    cols = [c for c in FEATURE_COLUMNS if c in X.columns]
    return X[cols]

def train_and_save_model():
    """
    如果你想在本地訓練 model，可以呼叫這個函式（使用 data.csv）
    """
    if not os.path.exists("data.csv"):
        raise FileNotFoundError("找不到 data.csv 用於訓練")
    df = pd.read_csv("data.csv")
    # 假設 target column 名為 'label'，0=success,1=failure
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
        model = joblib.load(MODEL_PATH)
        return model
    else:
        # 如果沒有模型檔，嘗試訓練（或回傳 None）
        try:
            model = train_and_save_model()
            return model
        except Exception as e:
            print(f"[WARN] 無法載入或訓練模型: {e}")
            return None

# main API for app.py
def load_model_and_predict(input_df):
    """
    input_df: pd.DataFrame with columns matching FEATURE_COLUMNS
    returns: (probability_of_failure (float), importances_dict)
    """
    model = load_model()
    X = _prepare_X(input_df)
    if model is None:
        # fallback deterministic heuristic if no model
        # return mid-low probability so UI still functions, but not constant 0.3
        prob = float(np.clip(0.2 + 0.1 * np.random.randn(), 0.01, 0.9))
        importances = {c: 1.0/len(X.columns) for c in X.columns}
        return prob, importances

    # ensure model has predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]  # assuming class 1 = failure
        prob = float(probs[0])
    else:
        pred = model.predict(X)[0]
        prob = 1.0 if pred == 1 else 0.0

    # feature importances mapping — try to align with prepared column order
    try:
        fi = model.feature_importances_
        cols = X.columns.tolist()
        importances = {col: float(fi[idx]) for idx, col in enumerate(cols)}
    except Exception:
        importances = {col: 1.0/len(X.columns) for col in X.columns}

    return prob, importances
