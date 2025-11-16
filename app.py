import streamlit as st
import pandas as pd
import numpy as np
import io, zipfile, os, tempfile

# -------------------------------
# 匯入 LLM 與 RF 模型
# -------------------------------
from llm_recommender import llm_highrisk_feedback
from model_train import load_model_and_predict, INPUT_FEATURES

# 假設你的 features_df 已經包含每層數值特徵
# features_df 必須至少有 'layer', 'filename' 以及 INPUT_FEATURES
# 例如：
# features_df = pd.read_csv("layer_features.csv")

st.set_page_config(layout="wide", page_title="LayerAI - RF + LLM")
st.title("LayerAI — RF 預測高風險層 + LLM 建議")

# -------------------------------
# 1. 上傳特徵 CSV
# -------------------------------
uploaded = st.file_uploader("上傳每層特徵 CSV", type=["csv"])
threshold = st.slider("高風險判定閾值（failure probability）", 0.0, 1.0, 0.5, 0.01)

run_btn = st.button("開始分析")

if uploaded and run_btn:
    st.info("讀取特徵資料...")
    features_df = pd.read_csv(uploaded)
    st.success(f"讀取 {len(features_df)} 層資料")

    # -------------------------------
    # 2. 逐層預測
    # -------------------------------
    st.info("逐層進行 RF 模型預測...")
    results = []
    for idx, row in features_df.iterrows():
        layer_features = row[INPUT_FEATURES].to_frame().T  # 轉成 DataFrame
        prob, _ = load_model_and_predict(layer_features)
        results.append({
            "layer": row['layer'],
            "filename": row['filename'],
            "prob": float(prob)
        })

    results_df = pd.DataFrame(results)
    st.subheader("逐層預測結果")
    st.dataframe(results_df.head(50))

    # -------------------------------
    # 3. Heatmap & 曲線（簡單例子）
    # -------------------------------
    st.info("生成簡易風險曲線...")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(results_df['layer'], results_df['prob'], marker='o')
    ax.axhline(y=threshold, color='r', linestyle='--', label='高風險閾值')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Failure Probability")
    ax.set_title("Layer Failure Probability Curve")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # 4. LLM 建議（僅高風險層）
    # -------------------------------
    st.info("產生 LLM 建議（僅高風險層）...")
    high_risk_layers = results_df[results_df['prob'] >= threshold]

    if len(high_risk_layers) == 0:
        st.markdown("✅ 所有層皆低風險，無需額外 LLM 建議。")
    else:
        st.subheader("LLM 高風險層建議 / 結論")
        for _, row in high_risk_layers.iterrows():
            layer_info = {
                "layer": row["layer"],
                "filename": row["filename"],
                "orig_prob": row["prob"],
                "suggested_params": None,
                "suggested_prob": None
            }
            txt = llm_layer_feedback(layer_info)
            st.markdown(f"### Layer {int(row['layer'])}\n{txt}")

    # -------------------------------
    # 5. 效益分析（簡單示例）
    # -------------------------------
    st.info("計算簡單效益分析...")
    # 假設每層等待時間 1~5s，失敗層浪費時間 = 2x等待時間
    results_df['estimated_time_s'] = results_df['prob'] * 2  # 模擬值
    st.subheader("簡易效益估算")
    st.dataframe(results_df[['layer', 'prob', 'estimated_time_s']])

    st.success("分析完成！")
