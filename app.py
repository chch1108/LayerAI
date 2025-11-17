import streamlit as st
import tempfile, os, io
import pandas as pd

from image_processor import extract_images_from_zip, batch_extract_features
from model_train import load_model_and_predict   # 使用隨機森林
from llm_recommender import get_llm_recommendation, get_low_risk_message

st.set_page_config(layout="wide", page_title="LayerAI — 多層樹脂回流預測")
st.title("LayerAI — 多層樹脂回流預測 + 風險分析")

# ----------- 使用者輸入參數 ----------------
st.sidebar.header("製程參數 (Process Parameters)")

viscosity = st.sidebar.number_input("材料黏度 (cps)", 50, 1000, 150, 10)
lift_height = st.sidebar.number_input("抬升高度 (μm)", 500, 8000, 1500, 100)
lift_speed = st.sidebar.number_input("抬升速度 (μm/s)", 100, 8000, 700, 50)
wait_time = st.sidebar.number_input("等待時間 (s)", 0.0, 5.0, 0.5, 0.1)
down_speed = st.sidebar.number_input("下降速度 (μm/s)", 1000, 10000, 4000, 500)
shape = st.sidebar.selectbox("形狀", ['90x45矩形', '90x50六角形', '50圓柱'])

uploaded = st.file_uploader("上傳切片 ZIP", type=["zip"])

threshold = st.slider("高風險判定閾值", 0.0, 1.0, 0.5, 0.01)
run_btn = st.button("開始分析 (Run)")


# =====================================================
# -------------------- 處理流程 -----------------------
# =====================================================
if uploaded and run_btn:
    with tempfile.TemporaryDirectory() as tmpdir:

        # 解壓縮
        zip_path = os.path.join(tmpdir, "layers.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded.getbuffer())

        imgs, filenames = extract_images_from_zip(zip_path, tmpdir)

        st.success(f"讀取 {len(imgs)} 層切片成功")

        # ---------- 對每層提取幾何特徵 ----------
        features_list = batch_extract_features(imgs, filenames)

        records = []
        st.info("逐層執行模型預測中...")

        # ---------- 每層都加入「製程參數」 ----------
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

            pred, importances = load_model_and_predict(pd.DataFrame([input_data]))

            records.append({
                "layer": feat['layer'],
                "filename": feat['filename'],
                "prob": pred,
                "params": input_data,
                "importances": importances
            })

        df = pd.DataFrame(records)
        st.subheader("逐層預測結果")
        st.dataframe(df)

        # =====================================================
        # ------------------ LLM 建議 / 結論 -------------------
        # =====================================================
        st.subheader("LLM 建議（高風險才提供按鈕）")

        for _, row in df.iterrows():
            st.markdown(f"### Layer {int(row['layer'])} — 風險：{row['prob']:.3f}")

            if row["prob"] >= threshold:
                # 高風險
                key = f"btn_{row['layer']}"
                if st.button(f"🔧 生成 Layer {int(row['layer'])} 的建議", key=key):
                    with st.spinner("AI 正在生成建議..."):
                        txt = get_llm_recommendation(row["params"], row["importances"])
                        st.markdown(txt)
            else:
                # 低風險
                st.markdown(get_low_risk_message())
