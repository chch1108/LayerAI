import streamlit as st
import tempfile, os, io, zipfile
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

# internal modules
from image_processor import (
    extract_images_from_zip,
    batch_extract_features,
    suggest_parameters_for_layers_with_model
)
from model_train import load_model_and_predict
from llm_recommender import get_llm_recommendation, get_low_risk_message
from image_editor_level1 import flow_simulation_overlay

# ----------------- Page config & cute theme -----------------
st.set_page_config(page_title="🎨 LayerAI — Fun Edition", layout="wide")
st.markdown(
    """
    <style>
    /* gradient background */
    .stApp { 
        background: linear-gradient(135deg, #a1c4fd, #c2e9fb); 
        color: #0f1720;
    }
    .block-container { padding: 1rem 2rem; }
    h1, h2, h3 { color: #0f1720; }
    .stButton>button { background-color:#ff8c94; color:white; border-radius:12px; font-weight:bold; }
    .stDownloadButton>button { background-color:#f6cd61; color:#0f1720; border-radius:12px; font-weight:bold; }
    .stSidebar .sidebar-content { background: #f0f4f8; color:#0f1720; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎉 LayerAI — 多層風險分析 & AI 建議 💡")

# ----------------- Sidebar -----------------
st.sidebar.header("⚙️ 控制面板")
st.sidebar.markdown("**輸入製程參數**（將套用到每層）")

viscosity = st.sidebar.number_input("材料黏度 (cps) 🧪", 50, 1000, 150, 10)
lift_height = st.sidebar.number_input("抬升高度 (μm) ⬆️", 500, 8000, 1500, 100)
lift_speed = st.sidebar.number_input("抬升速度 (μm/s) 🚀", 100, 8000, 700, 50)
wait_time = st.sidebar.number_input("等待時間 (s) ⏳", 0.0, 10.0, 0.5, 0.1)
down_speed = st.sidebar.number_input("下降速度 (μm/s) ⬇️", 1000, 10000, 4000, 500)

uploaded = st.sidebar.file_uploader("📦 上傳切片 ZIP 檔", type=["zip"])
threshold = st.sidebar.slider("高風險判定閾值 ⚠️", 0.0, 1.0, 0.5, 0.01)
run_btn = st.sidebar.button("🚀 開始分析")

# ----------------- Page selector -----------------
if "page" not in st.session_state:
    st.session_state.page = "Prediction"
st.session_state.page = st.sidebar.radio("📄 頁面選擇", 
                                         ["Prediction", "Visuals", "AI Suggestions", "Summary"], 
                                         index=["Prediction","Visuals","AI Suggestions","Summary"].index(st.session_state.page))

# ----------------- session_state init -----------------
for key in ["results_df","llm_results","auto_tune_results","overlays"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "results" in key else []
st.session_state.threshold = threshold

# ----------------- fallback auto-tune -----------------
def fallback_suggest_parameters(results_df, threshold=0.5):
    rows = []
    for _, r in results_df.iterrows():
        orig = float(r["prob"])
        layer = int(r["layer"])
        if orig >= threshold:
            cand = {"wait_time": round(wait_time + 0.4,3), "lift_height": lift_height, "lift_speed": max(50, lift_speed-50)}
            suggested_prob = max(0.0, orig - 0.12)
        else:
            cand = {"wait_time": wait_time, "lift_height": lift_height, "lift_speed": lift_speed}
            suggested_prob = orig
        rows.append({"layer": layer, "filename": r["filename"], "orig_prob": orig, "suggested_params": cand, "suggested_prob": suggested_prob})
    return pd.DataFrame(rows)

# ----------------- Run analysis -----------------
if run_btn:
    if uploaded is None:
        st.sidebar.error("⚠️ 請先上傳切片 ZIP")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "slices.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded.getbuffer())

            imgs, filenames = extract_images_from_zip(zip_path, tmpdir)
            if len(imgs) == 0:
                st.error("❌ ZIP 內沒有有效圖片")
            else:
                st.success(f"✅ 讀取 {len(imgs)} 張切片")
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
                    try:
                        top3 = [n for n,_ in sorted(importances.items(), key=lambda x:x[1], reverse=True)[:3]]
                    except:
                        top3 = []

                    records.append({
                        "layer": int(feat["layer"]),
                        "filename": feat["filename"],
                        "prob": float(pred),
                        "top3_features": ", ".join(top3),
                        "params": input_data,
                        "importances": importances
                    })

                    if float(pred) >= threshold:
                        ov = flow_simulation_overlay(img, alpha=0.55)
                        buf = io.BytesIO()
                        ov.save(buf, format="PNG")
                        st.session_state.overlays.append((int(feat["layer"]), buf.getvalue()))

                st.session_state.results_df = pd.DataFrame(records).sort_values("layer").reset_index(drop=True)
                st.session_state.llm_results = {}
                st.session_state.auto_tune_results = {}
                st.success("🎯 逐層預測完成！切換頁面查看詳細結果")
                st.session_state.page = "Prediction"

# ----------------- Render pages -----------------
if st.session_state.results_df is None:
    st.info("ℹ️ 尚未分析，請上傳 ZIP 並按「開始分析」")
else:
    df = st.session_state.results_df.copy()

    # Prediction page
    if st.session_state.page == "Prediction":
        st.header("📝 Prediction — 逐層預測")
        show_df = df[["layer","filename","prob","top3_features"]].copy()
        show_df["prob"] = show_df["prob"].map(lambda x:f"{x:.3f}")
        st.dataframe(show_df, use_container_width=True)

        st.markdown("### 🔧 Auto-Tune 快速操作")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("一鍵 Auto-Tune 高風險層"):
                try:
                    suggestion_df = suggest_parameters_for_layers_with_model(df, threshold=threshold, model_path="")
                except:
                    suggestion_df = fallback_suggest_parameters(df, threshold=threshold)
                for _, r in suggestion_df.iterrows():
                    st.session_state.auto_tune_results[int(r["layer"])] = r["suggested_params"]
                st.success("✅ Auto-Tune 完成")
        with col2:
            st.download_button("💾 下載預測結果 CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="prediction_results.csv", mime="text/csv")

    # Visuals page
    if st.session_state.page == "Visuals":
        st.header("📊 Visuals — Heatmap & Risk Curve & Overlays")
        st.subheader("🔥 Heatmap")
        probs = df["prob"].values
        layers = df["layer"].values
        heat_arr = np.array(probs).reshape(-1,1)
        heat_fig = px.imshow(heat_arr, color_continuous_scale="Turbo", labels={'x':'Failure Prob','y':'Layer'})
        heat_fig.update_yaxes(tickmode="array", tickvals=list(range(len(layers))), ticktext=[str(int(x)) for x in layers])
        heat_fig.update_xaxes(showticklabels=False)
        st.plotly_chart(heat_fig, use_container_width=True)

        st.subheader("📈 Risk Curve")
        curve_fig = px.line(x=df["prob"], y=df["layer"], markers=True, labels={"x":"Failure Probability","y":"Layer"})
        curve_fig.update_yaxes(autorange="reversed")
        st.plotly_chart(curve_fig, use_container_width=True)

        st.subheader("⚠️ High-risk Overlays")
        if len(st.session_state.overlays) == 0:
            st.info("目前無高風險層")
        else:
            cols = st.columns(3)
            for idx,(layer,img_bytes) in enumerate(st.session_state.overlays):
                with cols[idx%3]:
                    st.image(img_bytes, caption=f"Layer {layer} (高風險)", use_column_width=True)

    # AI Suggestions page
    if st.session_state.page == "AI Suggestions":
        st.header("🤖 AI Suggestions — 高風險層建議")
        df_desc = df.sort_values("prob", ascending=False).reset_index(drop=True)
        for _, row in df_desc.iterrows():
            layer = int(row["layer"])
            prob = float(row["prob"])
            st.markdown(f"### Layer {layer} — 風險 {prob:.3f} — Top3: {row['top3_features']}")
            high = (prob >= threshold)
            if not high:
                st.markdown(get_low_risk_message())
                continue

            if layer in st.session_state.llm_results:
                st.markdown("💡 **AI 建議（已生成）：**")
                st.markdown(st.session_state.llm_results[layer])
            else:
                btn_key = f"llm_gen_{layer}"
                if st.button(f"生成 Layer {layer} 建議", key=btn_key):
                    with st.spinner("LLM 生成中..."):
                        txt = get_llm_recommendation(row["params"], row["importances"])
                        st.session_state.llm_results[layer] = txt
                        st.session_state.page = "AI Suggestions"
                        st.experimental_rerun()

    # Summary page
    if st.session_state.page == "Summary":
        st.header("📋 Summary — 所有層建議總覽")
        summary_rows = []
        for _, row in df.iterrows():
            layer = int(row["layer"])
            prob = float(row["prob"])
            high = (prob >= threshold)
            ai_text = st.session_state.llm_results.get(layer, ("（高風險，尚未生成）" if high else "（低風險，無需調整）"))
            suggested_params = st.session_state.auto_tune_results.get(layer, "—")
            summary_rows.append({
                "layer": layer,
                "prob": round(prob,3),
                "top3_features": row["top3_features"],
                "suggested_params": suggested_params,
                "ai_suggestion": ai_text
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("prob", ascending=False).reset_index(drop=True)
        st.dataframe(summary_df, use_container_width=True)
        st.download_button("💾 下載建議總表 CSV", data=summary_df.to_csv(index=False).encode("utf-8"), file_name="suggestions_summary.csv", mime="text/csv")
