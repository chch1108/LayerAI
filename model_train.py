import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

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

# 讀取資料
df = pd.read_csv("data.csv")
X = df[FEATURE_COLUMNS]
y = df["回流（完全是0不完全是1）"]

# 訓練 RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
model.fit(X, y)

# 儲存模型
joblib.dump(model, "dlp_reflow_model.joblib")

print("模型訓練完成，已儲存為 dlp_reflow_model.joblib")
print("型別：", type(model))
