import streamlit as st
import pandas as pd
import numpy as np
import os, io, zipfile, tempfile

# -------------------------------
# 匯入 LLM 與 RF 模型
# -------------------------------
from llm_recommender import llm_highrisk_feedback
from model_train import load_model_and_predict, INPUT_FEATURES

# -------------------------------
# 匯入圖片處理模組
# -------------------------------
from image_processor import (
    extract_images_from_zip,
    make_plotly_heatmap_and_curve,
)
from image_editor_level1 import overlay_issue_markers

MODEL_PATH = "model_artifacts/dlp_reflow_model.joblib"

st.set_page_config(layout="wide", page_title="LayerAI - RF + LLM")
st.title("LayerAI — ZIP 切片預測 + 高風險層 LLM 建議")

st.markdown("""
上傳包含切片的 ZIP（每層 png/jpg）。  
系統會依序完成：

1. 逐層生成特徵並預測高風險層  
2. Heatmap + 風險曲線  
3. Level 1 修正版切片（畫框標記高風險層）  
4. LLM 建議（僅高風險層）  
5. 簡單效益分析
""")

col1, col2 = st.columns([1, 2])
with col1:
    uploaded = st.file_uploader("上傳切片 ZIP 檔", type=["zip"])
    threshold = st.slider("高風險判定閾值（failure probability）", 0.0, 1.0, 0.5, 0.01)
    run_btn = st.button("開始分析（全流程）")

if uploaded and run_btn:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "slices.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.info("解壓並讀取切片...")
        imgs, filenames = extract_images_from_zip(zip_path, tmpdir)
        st.success(f"讀取 {len(imgs)} 張切片")

        # -------------------------------
        # 逐層生成特徵 & RF 預測
        # -------------------------------
        st.info("逐層生成特徵並進行 RF 模型預測...")
        results = []
        features_list = []  # 儲存每層特徵

        for idx, (img, fname) in enumerate(zip(imgs, filenames)):
            # ---------------- 這裡需自己實作 extract_features(img) -> dict(INPUT_FEATURES) ----------------
            # 目前用 0 佔位，需換成實際特徵
            feature_dict = {k: 0 for k in INPUT_FEATURES}
            feature_dict['layer'] = idx
            feature_dict['filename'] = fname
            features_list.append(feature_dict)

            # 將單層特徵轉成 DataFrame
            feature_df = pd.DataFrame([feature_dict])[INPUT_FEATURES]
            prob, _ = load_model_and_predict(feature_df)
            results.append({
                "layer": idx,
                "filename": fname,
                "prob": float(prob),
                "features": feature_dict
            })

        results_df = pd.DataFrame(results)
        st.subheader("逐層預測結果")
        st.dataframe(results_df[['layer','filename','prob']])

        # -------------------------------
        # Heatmap / 曲線
        # -------------------------------
        st.info("生成 heatmap 與風險曲線...")
        fig_curve, fig_heatmap = make_plotly_heatmap_and_curve(results_df['prob'].values)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.plotly_chart(fig_curve, use_container_width=True)

        # -------------------------------
        # Level 1 Overlay
        # -------------------------------
        st.info("生成 Level 1 修正版切片（畫框標記高風險層）...")
        modified_images = []
        for img, prob in zip(imgs, results_df['prob']):
            mod_img = overlay_issue_markers(img, prob)
            modified_images.append(mod_img)

        st.subheader("修正後切片 (Level 1 Overlay)")
        for fname, mod_img, prob in zip(filenames, modified_images, results_df['prob']):
            st.image(mod_img, caption=f"{fname} — 風險 {prob:.2f}", use_column_width=True)

        # -------------------------------
        # LLM 高風險層整體建議
        # -------------------------------
        st.info("產生 LLM 高風險層建議...")
        stats_summary = {
            "total_layers": len(results_df),
            "high_risk_layers": len(results_df[results_df['prob'] >= threshold]),
            "avg_prob": results_df['prob'].mean(),
            "max_prob": results_df['prob'].max()
        }

        llm_text = llm_highrisk_feedback(stats_summary, threshold=threshold)
        st.subheader("LLM 高風險層建議 / 結論")
        st.markdown(llm_text)

        # -------------------------------
        # 簡單效益分析
        # -------------------------------
        st.info("簡單效益分析...")
        results_df['estimated_time_s'] = results_df['prob'] * 2  # 模擬浪費時間
        st.subheader("簡易效益估算")
        st.dataframe(results_df[['layer','prob','estimated_time_s']])

        st.success("分析完成！")
