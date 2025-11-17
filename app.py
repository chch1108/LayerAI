import streamlit as st
import tempfile, os, io
import pandas as pd
import numpy as np
import plotly.express as px

from image_processor import extract_images_from_zip, batch_extract_features
from model_train import load_model_and_predict
from llm_recommender import get_llm_recommendation, get_low_risk_message
from image_editor_level1 import overlay_issue_markers   # 影像 overlay

# -----------------------------------------------------
# Streamlit 設定
# -----------------------------------------------------
st.set_page_config(layout="wide", page_title="LayerAI — 多層樹脂回流預測")
st.title("LayerAI — 多層樹脂回流預測 + 視覺化 + 建議引擎（比賽版）")

# -----------------------------------------------------
# 使用者輸入 — 製程參數
# -----------------------------------------------------
st.sidebar.header("製程參數 (Process Parameters)")

viscosity = st.sidebar.number_input("材料黏度 (cps)", 50, 1000, 150, 10)
lift_height = st.sidebar.number_input("抬升高度 (μm)", 500, 8000, 1500, 100)
lift_speed = st.sidebar.number_input("抬升速度 (μm/s)", 100, 8000, 700, 50)
wait_time = st.sidebar.number_input("等待時間 (s)", 0.0, 5.0, 0.5, 0.1)
down_speed = st.sidebar.number_input("下降速度 (μm/s)", 1000, 10000, 4000, 500)
shape = st.sidebar.selectbox("形狀", ['90x45矩形', '90x50六角形', '50圓柱'])

uploaded = st.file_uploader("上傳切片 ZIP 檔案", type=["zip"])

threshold = st.slider("高風險判定閾值（模型預測機率）", 0.0, 1.0, 0.5, 0.01)
run_btn = st.button("開始分析 (Run)")

# -----------------------------------------------------
# 初始化 session_state
# -----------------------------------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "llm_results" not in st.session_state:
    st.session_state.llm_results = {}   # { layer: 建議 }

# -----------------------------------------------------
# 第一次按下 run_btn 時 — 做完整分析
# -----------------------------------------------------
if run_btn:

    if not uploaded:
        st.error("請上傳 ZIP 檔")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:

        zip_path = os.path.join(tmpdir, "layers.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded.getbuffer())

        imgs, filenames = extract_images_from_zip(zip_path, tmpdir)

        if len(imgs) == 0:
            st.error("❌ ZIP 內沒有有效圖片")
            st.stop()

        features_list = batch_extract_features(imgs, filenames)

        # ---- 做逐層模型預測 ----
        records = []
        overlays = []     # 存高風險圖片 overlay

        for img, feat in zip(imgs, features_list):

            input_data = {
                '材料黏度 (cps)': viscosity,
                '抬升高度(μm)': lift_height,
                '抬升速度(μm/s)': lift_speed,
                '等待時間(s)': wait_time,
                '下降速度((μm)/s)': down_speed,
                '形狀': shape,
                '面積(mm?)': feat['area'],
                '周長(mm)': feat['perimeter'],
                '水力直徑(mm)': feat['hydraulic_diameter'],
            }

            pred, importances = load_model_and_predict(pd.DataFrame([input_data]))

            # ---- importance 取前 3 ----
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            top3_names = [name for name, _ in sorted_imp[:3]]

            record = {
                "layer": feat['layer'],
                "filename": feat['filename'],
                "prob": pred,
                "top3_features": ", ".join(top3_names),
                "params": input_data,        # 不顯示，但 LLM 要用
                "importances": importances   # 不顯示細節，但 LLM 要用
            }
            records.append(record)

            # ------ Overlay for high-risk ------
            if pred >= threshold:
                overlays.append((feat['layer'], overlay_issue_markers(img)))

        # 存進 session_state
        st.session_state.results_df = pd.DataFrame(records)
        st.session_state.llm_results = {}
        st.session_state.overlays = overlays

        st.success("分析完成！請往下看結果 👇")

# -----------------------------------------------------
# 顯示結果（永不消失）
# -----------------------------------------------------
if st.session_state.results_df is not None:

    df = st.session_state.results_df

    st.subheader("📘 逐層模型預測結果（已經簡化欄位）")
    st.dataframe(df[["layer", "filename", "prob", "top3_features"]])

    # -----------------------------------------------------
    # Heatmap（視覺衝擊）
    # -----------------------------------------------------
    st.subheader("🔥 逐層風險 Heatmap")

    fig = px.imshow(
        np.array(df["prob"]).reshape(1, -1),
        color_continuous_scale="RdYlGn_r",
        labels=dict(color="Failure Probability")
    )
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # Overlay 高風險層圖片
    # -----------------------------------------------------
    if len(st.session_state.overlays) > 0:
        st.subheader("⚠️ 高風險層（Overlay 標記）")

        cols = st.columns(3)
        idx = 0
        for layer, overlay_img in st.session_state.overlays:
            with cols[idx % 3]:
                st.image(overlay_img, caption=f"Layer {layer}（高風險）")
            idx += 1

    # -----------------------------------------------------
    # LLM 建議
    # -----------------------------------------------------
    st.subheader("🤖 LLM 建議（高風險才提供按鈕）")

    for _, row in df.iterrows():

        layer = int(row["layer"])
        st.markdown(f"### Layer {layer} — 風險機率：**{row['prob']:.3f}**")

        # ---- 低風險層 ----
        if row["prob"] < threshold:
            st.markdown(get_low_risk_message())
            continue

        # ---- 高風險層：按鈕產生建議 ----
        btn_key = f"gen_btn_{layer}"
        if st.button(f"🔧 生成 Layer {layer} 的 AI 建議", key=btn_key):
            with st.spinner("AI 正在生成建議..."):
                st.session_state.llm_results[layer] = get_llm_recommendation(
                    row["params"], row["importances"]
                )

        # 顯示建議（若已生成）
        if layer in st.session_state.llm_results:
            st.markdown("**AI 建議：**")
            st.markdown(st.session_state.llm_results[layer])

    # -----------------------------------------------------
    # 建議總表
    # -----------------------------------------------------
    st.subheader("📑 所有層建議總覽")

    summary = []
    for _, row in df.iterrows():
        layer = int(row["layer"])
        if layer in st.session_state.llm_results:
            summary.append({
                "layer": layer,
                "prob": row["prob"],
                "top3_features": row["top3_features"],
                "AI_suggestion": st.session_state.llm_results[layer]
            })
        else:
            summary.append({
                "layer": layer,
                "prob": row["prob"],
                "top3_features": row["top3_features"],
                "AI_suggestion": "（低風險，無需調整）"
            })

    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df)
