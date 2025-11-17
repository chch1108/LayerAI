import streamlit as st
import tempfile, os, io
import pandas as pd

from image_processor import extract_images_from_zip, batch_extract_features
from model_train import load_model_and_predict
from llm_recommender import get_llm_recommendation, get_low_risk_message

# -----------------------------------------------------
# Streamlit 設定
# -----------------------------------------------------
st.set_page_config(layout="wide", page_title="LayerAI — 多層樹脂回流預測")
st.title("LayerAI — 多層樹脂回流預測 + 風險分析")

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
# 主流程
# -----------------------------------------------------
if uploaded and run_btn:

    with tempfile.TemporaryDirectory() as tmpdir:

        # ---------------- 解壓 ZIP ----------------
        zip_path = os.path.join(tmpdir, "layers.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded.getbuffer())

        imgs, filenames = extract_images_from_zip(zip_path, tmpdir)

        if len(imgs) == 0:
            st.error("❌ ZIP 內沒有有效的圖片，請檢查檔案內容。")
            st.stop()

        st.success(f"📂 成功讀取 {len(imgs)} 層切片")

        # ---------------- 提取每層特徵 ----------------
        st.info("🔍 正在提取幾何特徵...")
        features_list = batch_extract_features(imgs, filenames)

        records = []
        st.info("🤖 正在逐層執行模型預測...")

        # -----------------------------------------------------
        # 每層 = 幾何特徵 + 使用者輸入的製程參數
        # -----------------------------------------------------
        for feat in features_list:

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

            # 隨機森林預測
            pred, importances = load_model_and_predict(pd.DataFrame([input_data]))

            records.append({
                "layer": feat['layer'],
                "filename": feat['filename'],
                "prob": pred,
                "params": input_data,
                "importances": importances
            })

        df = pd.DataFrame(records)

        st.subheader("📘 逐層模型預測結果")
        st.dataframe(df)


        # -----------------------------------------------------
        # LLM 建議功能
        # -----------------------------------------------------
        st.subheader("🤖 LLM 建議（高風險才提供按鈕）")

        # 存放每層的 LLM 建議
        if "llm_results" not in st.session_state:
            st.session_state.llm_results = {}

        for _, row in df.iterrows():

            layer = int(row["layer"])
            st.markdown(f"### Layer {layer} — 風險機率：**{row['prob']:.3f}**")

            # ---------------- 低風險層：固定結論 ----------------
            if row["prob"] < threshold:
                st.markdown(get_low_risk_message())
                continue

            # ---------------- 高風險層：提供按鈕 ----------------
            btn_key = f"gen_btn_{layer}"

            if st.button(f"🔧 生成 Layer {layer} 的 AI 建議", key=btn_key):
                with st.spinner("AI 正在生成建議..."):
                    st.session_state.llm_results[layer] = get_llm_recommendation(
                        row["params"], row["importances"]
                    )

            # 若已生成 → 持續顯示，不會消失
            if layer in st.session_state.llm_results:
                st.markdown("**AI 建議：**")
                st.markdown(st.session_state.llm_results[layer])
