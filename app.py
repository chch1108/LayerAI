import streamlit as st
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import tempfile, io, zipfile
import pandas as pd
import matplotlib.pyplot as plt

from image_processor import (
    extract_images_from_zip,
    batch_predict_layers,
    make_plotly_heatmap_and_curve,
    suggest_parameters_for_layers_with_model,
    estimate_time_and_effects
)

from image_editor_level1 import overlay_issue_markers
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
5. 整體 LLM 建議（所有層總結）  
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

        # ---------------- Step 1 ----------------
        st.info("解壓並讀取切片...")
        print("Step 1: 解壓並讀取切片")
        imgs, filenames = extract_images_from_zip(zip_path, tmpdir)
        st.success(f"讀取 {len(imgs)} 張切片")
        print(f"Step 1 完成～共 {len(imgs)} 張切片")

        # ---------------- Step 2 ----------------
        st.info("逐層進行模型預測...")
        print("Step 2: 逐層模型預測")
        results_df, model_meta = batch_predict_layers(imgs, filenames, model_path=MODEL_PATH)
        st.dataframe(results_df.head(50))
        print(f"Step 2 完成～results_df rows: {len(results_df)}")

        # ---------------- Step 3 ----------------
        st.info("生成 heatmap 與風險曲線...")
        print("Step 3: heatmap & curve")
        risks = results_df["prob"].values
        heatmap_fig, curve_fig = make_plotly_heatmap_and_curve(risks)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.plotly_chart(curve_fig, use_container_width=True)
        print("Step 3 完成")

        # ---------------- Step 4 ----------------
        st.info("執行 Auto-Tune（為高風險層生成最佳參數）...")
        print("Step 4: Auto-Tune")
        suggestion_df = suggest_parameters_for_layers_with_model(results_df, threshold=threshold, model_path=MODEL_PATH)
        st.subheader("建議參數（Auto-Tune 結果）")
        st.dataframe(suggestion_df)
        print("Step 4 完成～suggestion_df rows:", len(suggestion_df))

        # ---------------- Step 4b: Level 1 Overlay ----------------
        st.info("生成 Level 1 修正版切片（畫框版）...")
        print("Step 4b: Level 1 Overlay")
        modified_images = []
        modified_filenames = []

        for img, fname, prob in zip(imgs, filenames, risks):
            mod_img = overlay_issue_markers(img, prob)
            modified_images.append(mod_img)
            modified_filenames.append(fname)

        st.subheader("修正後的切片（Level 1 Overlay）")
        for fname, mod_img, prob in zip(modified_filenames, modified_images, risks):
            st.image(mod_img, caption=f"{fname} — 風險 {prob:.2f}", use_column_width=True)

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
        print("Step 4b 完成～Overlay ZIP 已生成")

        # ---------------- Step 5 (整體建議) ----------------
        st.info("產生整體 LLM 建議（所有層總結）...")
        print("Step 5: 整體 LLM 建議")

        try:
            total_layers = len(results_df)
            high_risk_layers = results_df[results_df["prob"] >= threshold]
            num_high_risk = len(high_risk_layers)
            max_prob = results_df["prob"].max()
            avg_prob = results_df["prob"].mean()

            prompt_summary = f"""
你是一位具有豐富光固化 3D 列印（DLP/LCD/CLIP）經驗的製程工程師。
請根據以下統計資訊，提供「繁體中文」的整體建議或結論：

- 總層數：{total_layers}
- 高風險層數（失敗機率 >= {threshold:.2f}）：{num_high_risk}
- 最大失敗機率：{max_prob:.2f}
- 平均失敗機率：{avg_prob:.2f}

請給出：
1. 整體回流狀況評估
2. 是否需要調整 Auto-Tune 參數或其他製程設定
3. 可以採取的改善建議
"""

            summary_txt = llm_layer_feedback({
                "layer": "Summary",
                "filename": "N/A",
                "orig_prob": avg_prob,
                "suggested_params": None,
                "suggested_prob": None
            })
            st.markdown(f"### 整體建議 / 結論\n{summary_txt}")
            print("Step 5 完成～已生成整體建議")

        except Exception as e:
            st.markdown(f"(LLM 生成整體建議失敗: {e})")

        # ---------------- Step 6 ----------------
        st.info("計算時間節省與成功率改善預估...")
        print("Step 6: 成效儀表板")
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
        print("Step 6 完成～全流程結束")
