import streamlit as st
import os
import tempfile
import io
import zipfile
import pandas as pd
import numpy as np
from PIL import Image

# --- 自訂模組 ---
from image_processor import (
    extract_images_from_zip,
    batch_predict_layers,
    make_plotly_heatmap_and_curve,
    estimate_time_and_effects
)
from image_editor_level1 import overlay_issue_markers
from model_train import load_model_and_predict, INPUT_FEATURES
from llm_recommender import llm_highrisk_feedback

# --- Streamlit 設定 ---
st.set_page_config(layout="wide", page_title="LayerAI - Multi-layer Suite")
st.title("LayerAI — 多層逐層預測、Auto-Tune、修正版切片與效益儀表板")

st.markdown("""
上傳包含切片的 ZIP（每層 png/jpg）。  
系統會依序完成：

1. 逐層回流風險預測  
2. Heatmap + 風險曲線  
3. Level 1 修正版切片（畫框標記風險）  
4. 高風險層 LLM 建議 / 結論  
5. 成效儀表板：時間節省與成功率提升
""")

col1, col2 = st.columns([1, 2])
with col1:
    uploaded = st.file_uploader("上傳切片 ZIP 檔 (每張為一層)", type=["zip"])
    threshold = st.slider("高風險判定閾值（failure probability）",
                          min_value=0.0, max_value=1.0,
                          value=0.5, step=0.01)
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

        results = []
        high_risk_count = 0
        total_prob_list = []

        for idx, (img, fname) in enumerate(zip(imgs, filenames)):
            # 假設 extract_features_from_image(img) 會回傳每層的特徵字典
            geo_features = {
                "area": img.size[0] * img.size[1],  # 範例
                "perimeter": 2*(img.size[0]+img.size[1]),
                "hydraulic_diameter": np.sqrt(4*(img.size[0]*img.size[1])/(2*(img.size[0]+img.size[1])))
            }

            # 假設固定範例列印參數（實際可改成從檔名或 metadata 讀取）
            input_data = {
                '材料黏度 (cps)': 500,
                '抬升高度(μm)': 6.0,
                '抬升速度(μm/s)': 2.0,
                '等待時間(s)': 4.5,
                '下降速度((μm)/s)': 5.0,
                '形狀': '方形',
                '面積(mm?)': geo_features['area'],
                '周長(mm)': geo_features['perimeter'],
                '水力直徑(mm)': geo_features['hydraulic_diameter']
            }
            final_input_data = {feat: input_data.get(feat) for feat in INPUT_FEATURES}
            input_df = pd.DataFrame([final_input_data])

            try:
                prediction, importances = load_model_and_predict(input_df)
                prob = float(prediction)  # 0 或 1，示意
                total_prob_list.append(prob)

                if prob >= threshold:
                    high_risk_count += 1

                results.append({
                    "layer": idx+1,
                    "filename": fname,
                    "prob": prob
                })

            except Exception as e:
                st.warning(f"第 {idx+1} 層預測失敗: {e}")
                results.append({
                    "layer": idx+1,
                    "filename": fname,
                    "prob": 0.0
                })
                total_prob_list.append(0.0)

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.head(50))

        # ---------------------------------------------
        # Step 2：Heatmap & 曲線
        # ---------------------------------------------
        st.info("生成 heatmap 與風險曲線...")
        risks = results_df["prob"].values
        heatmap_fig, curve_fig = make_plotly_heatmap_and_curve(risks)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.plotly_chart(curve_fig, use_container_width=True)

        # ---------------------------------------------
        # Step 3：Level 1 修正（Overlay）
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
        # Step 4：LLM 高風險層建議 / 結論
        # ---------------------------------------------
        st.info("產生 LLM 高風險層建議 / 結論...")

        stats_summary = {
            "total_layers": len(results_df),
            "high_risk_layers": high_risk_count,
            "avg_prob": np.mean(total_prob_list) if total_prob_list else 0,
            "max_prob": np.max(total_prob_list) if total_prob_list else 0
        }

        with st.spinner("LLM 正在生成建議，請稍候..."):
            llm_text = llm_highrisk_feedback(stats_summary, threshold=threshold)

        st.subheader("🤖 AI 高風險層建議 / 結論")
        st.markdown(llm_text)

        # ---------------------------------------------
        # Step 5：效益儀表板
        # ---------------------------------------------
        st.info("計算時間節省與成功率改善預估...")
        time_report_df = estimate_time_and_effects(results_df)

        st.subheader("時間與成功率改善預估")
        st.dataframe(time_report_df)

        st.download_button(
            "下載時間效益報告 CSV",
            data=time_report_df.to_csv(index=False).encode('utf-8'),
            file_name="time_effects_report.csv",
            mime="text/csv"
        )

        st.success("分析完成！")
