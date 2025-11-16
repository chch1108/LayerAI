import streamlit as st
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import tempfile, os, io, zipfile
import pandas as pd
import matplotlib.pyplot as plt

from image_processor import (
    extract_images_from_zip,
    batch_predict_layers,
    make_plotly_heatmap_and_curve,
    suggest_parameters_for_layers_with_model,
    estimate_time_and_effects
)

# Level 1 Overlay 修圖
from image_editor_level1 import overlay_issue_markers

# LLM 統一建議（所有層）
from llm_recommender import llm_layer_feedback

MODEL_PATH = "model.h5"

st.set_page_config(layout="wide", page_title="LayerAI - Multi-layer Suite")
st.title("LayerAI — 多層逐層預測、Auto-Tune、修正版切片與效益儀表板")

st.markdown("""
上傳包含切片的 ZIP（每層 png/jpg）。  
系統會依序完成：

1. 逐層回流風險預測  
2. Heatmap + 風險曲線  
3. Auto-Tune（高風險層最佳參數）  
4. Level 1 修正版切片（畫框標記風險）  
5. 每層 LLM 意見（不論風險高低）  
6. 成效儀表板：時間節省與成功率提升
""")

col1, col2 = st.columns([1, 2])
with col1:
    uploaded = st.file_uploader("上傳切片 ZIP 檔 (每張為一層)", type=["zip"])
    threshold = st.slider("高風險判定閾值（failure probability）",
                          min_value=0.0, max_value=1.0,
                          value=0.5, step=0.01)
    st.write("Model 檔位置：")
    st.text(MODEL_PATH if MODEL_PATH else "目前使用 mock model")

    run_btn = st.button("開始分析（全流程）")

if uploaded and run_btn:

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "slices.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.info("解壓並讀取切片...")
        imgs, filenames = extract_images_from_zip(zip_path, tmpdir)
        st.success(f"讀取 {len(imgs)} 張切片")

        # ---------------------------------------------
        # Step 1：逐層預測
        # ---------------------------------------------
        st.info("逐層進行模型預測...")
        results_df, model_meta = batch_predict_layers(imgs, filenames, model_path=MODEL_PATH)
        st.dataframe(results_df.head(50))

        risks = results_df["prob"].values

        # ---------------------------------------------
        # Step 2：Heatmap & 曲線
        # ---------------------------------------------
        st.info("生成 heatmap 與風險曲線...")
        heatmap_fig, curve_fig = make_plotly_heatmap_and_curve(risks)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.plotly_chart(curve_fig, use_container_width=True)

        # ---------------------------------------------
        # Step 3：Auto-Tune（高風險層）
        # ---------------------------------------------
        st.info("執行 Auto-Tune（為高風險層生成最佳參數）...")
        suggestion_df = suggest_parameters_for_layers_with_model(results_df, threshold=threshold, model_path=MODEL_PATH)
        st.subheader("建議參數（Auto-Tune 結果）")
        st.dataframe(suggestion_df)

        st.download_button(
            "下載建議參數 CSV",
            data=suggestion_df.to_csv(index=False).encode("utf-8"),
            file_name="layer_suggestions.csv",
            mime="text/csv"
        )

        # ---------------------------------------------
        # Step 4：Level 1 修正（Overlay）
        # ---------------------------------------------
        st.info("生成 Level 1 修正版切片（畫框版）...")

        modified_images = []
        modified_filenames = []

        for img, fname, prob in zip(imgs, filenames, risks):
            mod_img = overlay_issue_markers(img, prob)
            modified_images.append(mod_img)
            modified_filenames.append(fname)

        st.subheader("修正後的切片（Level 1 Overlay）")
        for fname, mod_img, prob in zip(modified_filenames, modified_images, risks):
            st.image(mod_img, caption=f"{fname} — 風險 {prob:.2f}", use_column_width=True)

        # ---------------- ZIP 打包 -------------------
        st.info("壓縮修正版切片 ZIP...")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as z:
            for fname, img in zip(modified_filenames, modified_images):
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                z.writestr(fname, img_bytes.getvalue())

        st.download_button(
            "⬇️ 下載修正版切片 ZIP",
            data=zip_buf.getvalue(),
            file_name="modified_slices.zip",
            mime="application/zip",
        )

        # ---------------------------------------------
        # Step 5：LLM 建議（所有層都輸出）
        # ---------------------------------------------
        st.info("產生 LLM 建議（所有層）...")

        st.subheader("LLM 層級建議 / 結論")
        for _, row in results_df.iterrows():
            layer_info = {
                "layer": row["layer"],
                "filename": row["filename"],
                "orig_prob": row["prob"],
                "suggested_params": None,
                "suggested_prob": None
            }

            # 如果 Auto-Tune 有該層
            match = suggestion_df[suggestion_df["layer"] == row["layer"]]

            if len(match) > 0:
                m = match.iloc[0]
            
                # 若欄位存在才放入（避免 KeyError）
                params = {}
                for key in ["wait_time", "lift_height", "lift_speed"]:
                    if key in m:
                        params[key] = m[key]
            
                # 若找到至少一個參數就存
                layer_info["suggested_params"] = params if len(params) > 0 else None
            
                # 建議後預測機率
                layer_info["suggested_prob"] = m["suggested_prob"] if "suggested_prob" in m else None
            
            else:
                # 沒有 Auto-Tune（低風險層）
                layer_info["suggested_params"] = None
                layer_info["suggested_prob"] = None


            txt = llm_layer_feedback(layer_info)
            st.markdown(f"### Layer {int(row['layer'])}\n{txt}")

        # ---------------------------------------------
        # Step 6：效益儀表板
        # ---------------------------------------------
        st.info("計算時間節省與成功率改善預估...")
        time_report_df = estimate_time_and_effects(results_df, suggestion_df)

        st.subheader("時間與成功率改善預估")
        st.dataframe(time_report_df)

        st.download_button(
            "下載時間效益報告 CSV",
            data=time_report_df.to_csv(index=False).encode('utf-8'),
            file_name="time_effects_report.csv",
            mime="text/csv"
        )

        st.success("分析完成！")
