###############################################
# LayerAI — Competition Edition (Final Version)
###############################################

import streamlit as st
import tempfile
import os
import io
import zipfile
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

# === Internal modules ===
from image_processor import (
    extract_images_from_zip,
    batch_extract_features,
    suggest_parameters_for_layers_with_model,
)
from model_train import load_model_and_predict
from llm_recommender import get_llm_recommendation, get_low_risk_message
from image_editor_level1 import flow_simulation_overlay

############################################################
# 0. Global UI Theme / CSS
############################################################
st.set_page_config(layout="wide", page_title="LayerAI")

st.markdown("""
<style>
/* Page background */
.stApp {
    background: linear-gradient(180deg, #ECF9FF 0%, #FFFFFF 40%);
}

/* Headers */
h1, h2, h3 {
    color: #0b6e6b;
}

/* Buttons */
.stButton>button {
    background-color:#0B9F95;
    color:white;
    border-radius:8px;
    padding:8px 16px;
    font-size:16px;
    border:none;
}
.stButton>button:hover {
    background-color:#08746C;
}

/* Orange download buttons */
.stDownloadButton>button {
    background-color:#FF914D;
    color:white;
    border-radius:8px;
    padding:8px 14px;
    font-size:15px;
}
</style>
""", unsafe_allow_html=True)


############################################################
# 1. Sidebar — Process Params
############################################################
st.sidebar.header("⚙️ 製程參數 (Process Parameters)")

viscosity = st.sidebar.number_input("材料黏度 (cps)", 50, 1000, 150, 10)
lift_height = st.sidebar.number_input("抬升高度 (μm)", 500, 8000, 1500, 100)
lift_speed = st.sidebar.number_input("抬升速度 (μm/s)", 100, 8000, 700, 50)
wait_time = st.sidebar.number_input("等待時間 (s)", 0.0, 5.0, 0.5, 0.1)
down_speed = st.sidebar.number_input("下降速度 (μm/s)", 1000, 10000, 4000, 500)

uploaded = st.sidebar.file_uploader("📁 上傳切片 ZIP", type=["zip"])
threshold = st.sidebar.slider("高風險判定閾值", 0.0, 1.0, 0.5, 0.01)
run_btn = st.sidebar.button("🚀 開始分析")


############################################################
# 2. Session States
############################################################
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "llm_results" not in st.session_state:
    st.session_state.llm_results = {}
if "auto_tune_results" not in st.session_state:
    st.session_state.auto_tune_results = {}
if "overlays" not in st.session_state:
    st.session_state.overlays = []
st.session_state.threshold = threshold


############################################################
# 3. Run Analysis
############################################################
if run_btn:
    if uploaded is None:
        st.sidebar.error("請先上傳 ZIP")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "slices.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded.getbuffer())

            imgs, filenames = extract_images_from_zip(zip_path, tmpdir)
            if len(imgs) == 0:
                st.error("❌ ZIP 內無有效圖片")
            else:
                st.success(f"成功讀取 {len(imgs)} 張切片")
                feats = batch_extract_features(imgs, filenames)

                records = []
                st.session_state.overlays = []

                for img, feat in zip(imgs, feats):
                    input_data = {
                        '材料黏度 (cps)': viscosity,
                        '抬升高度(μm)': lift_height,
                        '抬升速度(μm/s)': lift_speed,
                        '等待時間(s)': wait_time,
                        '下降速度((μm)/s)': down_speed,
                        '面積(mm?)': feat['area'],
                        '周長(mm)': feat['perimeter'],
                        '水力直徑(mm)': feat['hydraulic_diameter'],
                    }


                    pred, importances = load_model_and_predict(pd.DataFrame([input_data]))

                    # Top 3 most important features
                    try:
                        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                        top3 = [name for name, _ in sorted_imp[:3]]
                    except:
                        top3 = []

                    records.append({
                        "layer": feat["layer"],
                        "filename": feat["filename"],
                        "prob": float(pred),
                        "top3_features": ", ".join(top3),
                        "params": input_data,
                        "importances": importances
                    })

                    # overlay for high risk
                    if float(pred) >= st.session_state.threshold:
                        ov = flow_simulation_overlay(img)
                        buf = io.BytesIO()
                        ov.save(buf, format="PNG")
                        st.session_state.overlays.append((feat["layer"], buf.getvalue()))

                st.session_state.results_df = pd.DataFrame(records)
                st.session_state.llm_results = {}
                st.session_state.auto_tune_results = {}

                st.success("分析完成～ 請查看下方結果")


############################################################
# 4. UI Tabs
############################################################
if st.session_state.results_df is not None:

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 預測結果", "✨ 視覺化", "💡 AI 建議", "📄 建議總覽"]
    )

    df = st.session_state.results_df.copy()
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    ############################################################
    # TAB 1 — Prediction Table
    ############################################################
    with tab1:
        st.header("📊 逐層預測結果（依風險排序）")

        show_df = df[["layer", "filename", "prob", "top3_features"]]
        show_df["prob"] = show_df["prob"].apply(lambda x: f"{x:.3f}")

        st.dataframe(show_df, use_container_width=True)


    ############################################################
    # TAB 2 — Visualizations: Heatmap + Overlay
    ############################################################
    with tab2:
        st.header("✨ 視覺化 — Heatmap 與 Overlay")

        # --- Heatmap ---
        st.subheader("Heatmap")
        probs = df["prob"].values
        heat = px.imshow(
            np.array([probs]),
            color_continuous_scale="RdYlGn_r",
            labels={"color": "Failure Probability"}
        )
        heat.update_yaxes(showticklabels=False)
        st.plotly_chart(heat, use_container_width=True)

        # --- Curve ---
        st.subheader("風險折線圖")
        curve = px.line(
            x=df["layer"], y=df["prob"],
            markers=True,
            labels={"x": "Layer", "y": "Failure Probability"}
        )
        st.plotly_chart(curve, use_container_width=True)

        # --- Overlays ---
        st.subheader("高風險層 Overlay")
        if len(st.session_state.overlays) == 0:
            st.info("目前無高風險層")
        else:
            cols = st.columns(3)
            for idx, (layer, img_bytes) in enumerate(st.session_state.overlays):
                with cols[idx % 3]:
                    st.image(img_bytes, caption=f"Layer {layer} (高風險)", use_column_width=True)


    ############################################################
    # TAB 3 — LLM Suggestions
    ############################################################
    with tab3:
        st.header("💡 AI 層級建議")

        for _, row in df.iterrows():
            layer = int(row["layer"])
            prob = float(row["prob"])
            high = (prob >= st.session_state.threshold)

            st.markdown(f"### Layer {layer} — 風險 {prob:.3f}")

            if not high:
                st.markdown(get_low_risk_message())
                continue

            # High risk -> show button
            btn_key = f"llm_btn_{layer}"
            if st.button(f"生成 Layer {layer} 建議", key=btn_key):
                with st.spinner("LLM 正在生成建議..."):
                    txt = get_llm_recommendation(row["params"], row["importances"])
                    st.session_state.llm_results[layer] = txt

            # Show result if generated
            if layer in st.session_state.llm_results:
                st.markdown("**AI 建議：**")
                st.markdown(st.session_state.llm_results[layer])


    ############################################################
    # TAB 4 — Summary Table
    ############################################################
    with tab4:
        st.header("📄 所有層建議總覽")

        summary_rows = []
        for _, row in df.iterrows():
            layer = int(row["layer"])
            prob = float(row["prob"])
            high = (prob >= st.session_state.threshold)

            ai_text = (
                st.session_state.llm_results.get(layer, "（高風險，但尚未生成建議）")
                if high else "（低風險，無需調整）"
            )
            suggested_params = st.session_state.auto_tune_results.get(layer, "—")

            summary_rows.append({
                "layer": layer,
                "prob": round(prob, 3),
                "top3_features": row["top3_features"],
                "suggested_params": suggested_params,
                "ai_suggestion": ai_text
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("prob", ascending=False)
        st.dataframe(summary_df, use_container_width=True)

        # Download CSV
        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 下載建議總表 CSV",
            data=csv_bytes,
            file_name="layerAI_suggestions.csv",
            mime="text/csv"
        )
