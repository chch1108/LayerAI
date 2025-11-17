import streamlit as st
import tempfile, os, io, zipfile
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from PIL import Image

# 你的模組（請確認檔案存在）
from image_processor import (
    extract_images_from_zip,
    batch_extract_features,
    suggest_parameters_for_layers_with_model  # 若 image_processor 無此函式會拋錯，見下方 fallback
)
from model_train import load_model_and_predict
from llm_recommender import get_llm_recommendation, get_low_risk_message
from image_editor_level1 import overlay_issue_markers

# ---------------- Streamlit page ----------------
st.set_page_config(layout="wide", page_title="LayerAI — Competition Edition")
st.title("LayerAI — 多層風險分析、Auto-Tune、Overlay、LLM 建議與報告輸出")

# ---------------- Sidebar: process parameters ----------------
st.sidebar.header("製程參數 (Process Parameters)")
viscosity = st.sidebar.number_input("材料黏度 (cps)", 50, 1000, 150, 10)
lift_height = st.sidebar.number_input("抬升高度 (μm)", 500, 8000, 1500, 100)
lift_speed = st.sidebar.number_input("抬升速度 (μm/s)", 100, 8000, 700, 50)
wait_time = st.sidebar.number_input("等待時間 (s)", 0.0, 5.0, 0.5, 0.1)
down_speed = st.sidebar.number_input("下降速度 (μm/s)", 1000, 10000, 4000, 500)
shape = st.sidebar.selectbox("形狀", ['90x45矩形', '90x50六角形', '50圓柱'])

uploaded = st.sidebar.file_uploader("上傳切片 ZIP 檔", type=["zip"])
threshold = st.sidebar.slider("高風險判定閾值", 0.0, 1.0, 0.5, 0.01)
run_btn = st.sidebar.button("開始分析")

# ---------------- session state init ----------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "llm_results" not in st.session_state:
    st.session_state.llm_results = {}  # layer -> text
if "overlays" not in st.session_state:
    st.session_state.overlays = []     # list of (layer, pil_image_bytes)
if "auto_tune_results" not in st.session_state:
    st.session_state.auto_tune_results = {}  # layer -> suggested params

# ---------------- helper: fallback suggest_parameters ----------------
def fallback_suggest_parameters(results_df, threshold=0.5):
    """
    如果 image_processor 沒有 suggest_parameters_for_layers_with_model，使用簡單 heuristic。
    回傳 DataFrame with columns: layer, filename, orig_prob, suggested_params(dict), suggested_prob
    """
    rows = []
    for _, r in results_df.iterrows():
        orig = float(r["prob"])
        layer = int(r["layer"])
        fname = r["filename"]
        suggested = None
        suggested_prob = orig
        if orig >= threshold:
            # 簡單 heuristic: 增加 wait_time by +0.4s 及 slightly reduce lift_speed
            base_wait = wait_time
            cand = {"wait_time": round(base_wait + 0.4,3), "lift_height": lift_height, "lift_speed": max(50, lift_speed-50)}
            suggested = cand
            suggested_prob = max(0.0, orig - 0.12)
        else:
            suggested = {"wait_time": wait_time, "lift_height": lift_height, "lift_speed": lift_speed}
            suggested_prob = orig
        rows.append({"layer": layer, "filename": fname, "orig_prob": orig, "suggested_params": suggested, "suggested_prob": round(suggested_prob,4)})
    return pd.DataFrame(rows)

# ---------------- Run analysis when user clicks ----------------
if run_btn:
    if uploaded is None:
        st.sidebar.error("請先上傳切片 ZIP")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存 ZIP
            zip_path = os.path.join(tmpdir, "slices.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded.getbuffer())

            imgs, filenames = extract_images_from_zip(zip_path, tmpdir)
            if len(imgs) == 0:
                st.error("ZIP 中未找到可解析影像。請檢查檔案。")
            else:
                st.success(f"讀取 {len(imgs)} 張切片")
                # 提取幾何特徵
                feats = batch_extract_features(imgs, filenames)  # list of dicts with layer, filename, area, perimeter, hydraulic_diameter

                records = []
                overlays = []

                for img, feat in zip(imgs, feats):
                    # 準備 model input
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

                    # top3 feature names
                    try:
                        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                        top3_names = [name for name, _ in sorted_imp[:3]]
                    except Exception:
                        top3_names = []

                    records.append({
                        "layer": feat["layer"],
                        "filename": feat["filename"],
                        "prob": float(pred),
                        "top3_features": ", ".join(top3_names),
                        "params": input_data,
                        "importances": importances
                    })

                    # overlay for high risk
                    if float(pred) >= threshold:
                        ov = overlay_issue_markers(img, float(pred))
                        # save to bytes
                        buf = io.BytesIO()
                        ov.save(buf, format="PNG")
                        st.session_state.overlays.append((feat["layer"], buf.getvalue()))

                st.session_state.results_df = pd.DataFrame(records)
                st.session_state.llm_results = {}
                st.session_state.auto_tune_results = {}

                st.success("逐層預測完成，請向下查看結果。")

# -------------- UI: show results if available ----------------
if st.session_state.results_df is not None:
    df = st.session_state.results_df.copy()

    # sort by prob desc for display
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    # Display simplified table (no params)
    st.subheader("逐層預測結果 (Top: 高風險優先)")
    show_df = df[["layer", "filename", "prob", "top3_features"]].copy()
    show_df["prob"] = show_df["prob"].map(lambda x: f"{x:.3f}")
    st.dataframe(show_df, use_container_width=True)

    # ---------------- Heatmap and Risk Curve ----------------
    st.subheader("Heatmap & 風險折線圖")
    probs = df["prob"].values
    # heatmap as 1xN for simplicity, plus risk curve
    heat_fig = px.imshow(np.array([probs]), color_continuous_scale="RdYlGn_r", labels=dict(color="Failure Prob"))
    heat_fig.update_yaxes(showticklabels=False)
    st.plotly_chart(heat_fig, use_container_width=True)

    curve_fig = px.line(x=list(df["layer"]), y=probs, markers=True, labels={"x":"Layer","y":"Failure Prob"})
    st.plotly_chart(curve_fig, use_container_width=True)

    # ---------------- Overlays display ----------------
    st.subheader("高風險層 Overlay（已標記）")
    if len(st.session_state.overlays) == 0:
        st.info("目前無高風險層。")
    else:
        cols = st.columns(3)
        for idx, (layer, img_bytes) in enumerate(st.session_state.overlays):
            with cols[idx % 3]:
                st.image(img_bytes, caption=f"Layer {layer} (高風險)", use_column_width=True)

    # ---------------- Auto-Tune (batch or per-layer) ----------------
    st.subheader("Auto-Tune（候選參數測試）")
    col_a, col_b = st.columns([1,2])
    with col_a:
        st.caption("按一次會對所有高風險層執行 Auto-Tune（會消耗時間）")
        if st.button("一鍵 Auto-Tune 高風險層"):
            # try to call image_processor.suggest_parameters_for_layers_with_model
            try:
                suggestion_df = suggest_parameters_for_layers_with_model(df, threshold=threshold, model_path="")
            except Exception:
                suggestion_df = fallback_suggest_parameters(df, threshold=threshold)
            # map results to session
            for _, r in suggestion_df.iterrows():
                st.session_state.auto_tune_results[int(r["layer"])] = r["suggested_params"]
            st.success("Auto-Tune 完成（結果已儲存於畫面）")

    with col_b:
        st.caption("選擇要 Auto-Tune 的單層")
        sel_layer = st.selectbox("選取層號（若要單層調整）", options=list(df["layer"].astype(int)))
        if st.button("執行單層 Auto-Tune"):
            try:
                suggestion_df = suggest_parameters_for_layers_with_model(df[df["layer"]==sel_layer], threshold=threshold, model_path="")
            except Exception:
                suggestion_df = fallback_suggest_parameters(df[df["layer"]==sel_layer], threshold=threshold)
            if len(suggestion_df) > 0:
                r = suggestion_df.iloc[0]
                st.session_state.auto_tune_results[int(r["layer"])] = r["suggested_params"]
                st.success(f"Layer {r['layer']} Auto-Tune 建議已儲存。")

    # show auto-tune summary
    if len(st.session_state.auto_tune_results) > 0:
        at_list = [{"layer": k, "suggested_params": v} for k, v in st.session_state.auto_tune_results.items()]
        st.table(pd.DataFrame(at_list))

    # ---------------- LLM suggestions: high-risk layers get a button ----------------
    st.subheader("LLM 建議（高風險層可按按鈕生成）")
    for _, row in df.iterrows():
        layer = int(row["layer"])
        st.markdown(f"**Layer {layer} — 風險 {row['prob']:.3f} — Top3: {row['top3_features']}**")

        if row["prob"] < threshold:
            st.markdown(get_low_risk_message())
        else:
            btn_key = f"llm_gen_{layer}"
            if st.button(f"生成 Layer {layer} 建議", key=btn_key):
                with st.spinner("LLM 正在生成建議..."):
                    txt = get_llm_recommendation(row["params"], row["importances"])
                    st.session_state.llm_results[layer] = txt
            # display if exists
            if layer in st.session_state.llm_results:
                st.markdown("**AI 建議：**")
                st.markdown(st.session_state.llm_results[layer])

    # ---------------- Suggestions summary table (all layers) ----------------
    st.subheader("所有層建議總覽")
    summary_rows = []
    for _, row in df.iterrows():
        layer = int(row["layer"])
        suggestion = st.session_state.llm_results.get(layer, "（低風險，無需調整）")
        suggested_params = st.session_state.auto_tune_results.get(layer, "—")
        summary_rows.append({
            "layer": layer,
            "prob": round(row["prob"],3),
            "top3_features": row["top3_features"],
            "suggested_params": suggested_params,
            "ai_suggestion": suggestion if isinstance(suggestion, str) else suggestion
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("prob", ascending=False)
    st.dataframe(summary_df, use_container_width=True)

    # CSV download
    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("下載建議總表 CSV", data=csv_bytes, file_name="suggestions_summary.csv", mime="text/csv")

    # ---------------- Export PDF report ----------------
    st.subheader("匯出 PDF 報告（包含 overlay 圖與建議）")
    pdf_name = "layerai_report.pdf"
    if st.button("建立並下載 PDF 報告"):
        # create PDF in tmpdir
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, pdf_name)
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Title page
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "LayerAI 逐層分析報告", ln=True, align="C")
            pdf.ln(4)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 8, f"共 {len(df)} 層 — 生成閾值 {threshold}", ln=True)
            pdf.ln(6)

            # Add a table of top risk layers
            topN = summary_df.head(10)
            for _, r in topN.iterrows():
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, f"Layer {int(r['layer'])}  風險: {r['prob']}", ln=True)
                pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 6, f"Top3 features: {r['top3_features']}")
                pdf.multi_cell(0, 6, f"Suggested params: {r['suggested_params']}")
                pdf.multi_cell(0, 6, f"AI Suggestion: {r['ai_suggestion']}")
                pdf.ln(2)

                # embed overlay image if available
                for (lay, img_bytes) in st.session_state.overlays:
                    if int(lay) == int(r["layer"]):
                        img_path = os.path.join(tmpdir, f"layer_{lay}.png")
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        pdf.image(img_path, w=120)
                        pdf.ln(4)
                        break

            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                pdf_data = f.read()

            st.download_button("下載 PDF 報告", data=pdf_data, file_name=pdf_name, mime="application/pdf")

# end
