import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# 模型儲存位置
# -------------------------
MODEL_PATH = os.path.join("model_artifacts", "dlp_reflow_model.joblib")
os.makedirs("model_artifacts", exist_ok=True)

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

# =========================
# 訓練模型 (一次性)
# =========================
df = pd.read_csv("data.csv")
X = df[FEATURE_COLUMNS]
y = df["回流（完全是0不完全是1）"]

model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
model.fit(X, y)

joblib.dump(model, MODEL_PATH)
print("模型訓練完成，已儲存為", MODEL_PATH)

# =========================
# API 給 app.py 使用
# =========================
def load_model_and_predict(input_df, model_path=MODEL_PATH):
    model = joblib.load(model_path)
    X = input_df[FEATURE_COLUMNS]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        prob = float(probs[0])
    else:
        pred = model.predict(X)[0]
        prob = float(pred)
    try:
        fi = model.feature_importances_
        importances = {col: float(fi[i]) for i, col in enumerate(FEATURE_COLUMNS)}
    except:
        importances = {col: 1.0 / len(FEATURE_COLUMNS) for col in FEATURE_COLUMNS}
    return prob, importances
