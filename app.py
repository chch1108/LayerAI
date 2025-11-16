import streamlit as st
import tempfile, os, io, zipfile
import pandas as pd
from image_processor import (
    extract_images_from_zip,
    batch_predict_layers,
    make_plotly_heatmap_and_curve,
    suggest_parameters_for_layers_with_model,
    generate_modified_slices_zip,
    estimate_time_and_effects
)
from llm_recommender import llm_textual_suggestions

# CONFIG: 指向模型檔（若無則留空使用 mock）
MODEL_PATH = "model.h5"

st.set_page_config(layout="wide", page_title="LayerAI - Multi-layer Suite")
st.title("LayerAI — 多層逐層預測、Auto-Tune、修正版切片與效益儀表板")

st.markdown("""
上傳包含切片的 ZIP（每層 png/jpg）。系統會逐層預測回流失敗風險、畫出 heatmap 與風險曲線、為高風險層自動測試候選參數找出最佳組合，並產出修正版切片供下載，最後顯示時間/成功率改善預估。
""")

col1, col2 = st.columns([1,2])
with col1:
    uploaded = st.file_uploader("上傳切片 ZIP 檔 (每張為一層)", type=["zip"])
    threshold = st.slider("高風險判定閾值（failure probability）", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.write("Model 檔路徑（若要使用真實模型，請先上傳 model.h5 至 repo 或改此設定）")
    st.text(MODEL_PATH if MODEL_PATH else "使用模擬模型 (mock)")

    run_btn = st.button("開始分析（逐層預測 + Auto-Tune）")

with col2:
    st.empty()

if uploaded and run_btn:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "slices.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.info("解壓並讀取切片...")
        imgs, filenames = extract_images_from_zip(zip_path, tmpdir)
        st.success(f"讀取 {len(imgs)} 張切片")

        st.info("逐層進行模型預測...")
        results_df, model_meta = batch_predict_layers(imgs, filenames, model_path=MODEL_PATH)
        st.dataframe(results_df.head(50))

        st.info("生成 heatmap 與風險曲線...")
        heatmap_fig, curve_fig = make_plotly_heatmap_and_curve(results_df['prob'].values)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.plotly_chart(curve_fig, use_container_width=True)

        st.info("執行 Auto-Tune（為高風險層生成建議參數）...")
        suggestion_df = suggest_parameters_for_layers_with_model(results_df, threshold=threshold, model_path=MODEL_PATH)
        st.subheader("建議參數（Auto-Tune 結果）")
        st.dataframe(suggestion_df)

        csv_bytes = suggestion_df.to_csv(index=False).encode('utf-8')
        st.download_button("下載建議參數 CSV", data=csv_bytes, file_name="layer_suggestions.csv", mime="text/csv")

        st.info("為高風險層生成修正版切片（侵蝕 / 加支撐示範）...")
        modified_zip_bytes = generate_modified_slices_zip(imgs, filenames, results_df, threshold=threshold)
        st.download_button("下載修正版切片 ZIP", data=modified_zip_bytes, file_name="modified_slices.zip", mime="application/zip")

        st.info("產生文字建議（LLM）— 取 top 3 高風險層的解釋")
        top3 = suggestion_df.sort_values('orig_prob', ascending=False).head(3)
        for _, row in top3.iterrows():
            txt = llm_textual_suggestions(row)
            st.markdown(f"**Layer {int(row['layer'])}**: {txt}")

        st.info("計算時間節省與成功率改善預估...")
        time_report_df = estimate_time_and_effects(results_df, suggestion_df)
        st.subheader("時間與成功率改善預估")
        st.dataframe(time_report_df)
        st.download_button("下載時間效益報告 CSV", data=time_report_df.to_csv(index=False).encode('utf-8'),
                           file_name="time_effects_report.csv", mime="text/csv")

        st.success("完成。請檢查結果並下載需要的報表或修正版切片。")
