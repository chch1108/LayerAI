import streamlit as st
import pandas as pd
import os
from PIL import Image
import google.generativeai as genai

# Import 模組
from image_processor import extract_geometric_features
from model_train import load_model_and_predict, INPUT_FEATURES, CATEGORICAL_FEATURES
from llm_recommender import get_llm_recommendation

# --- Page Configuration ---
st.set_page_config(
    page_title="AI 決策支持系統 (DLP 3D列印)",
    page_icon="🤖",
    layout="wide"
)

# --- Application State ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- UI Layout ---
st.title("🤖 AI 決策支持系統：DLP 樹脂回流預測")
st.write("根據您輸入的 **單層圖像** 與 **製程參數**，本系統將預測樹脂回流是否完全。若預測失敗，將由 AI 提供優化建議。")

col1, col2 = st.columns(2)

# --- Column 1: User Inputs ---
with col1:
    st.header("1. 輸入參數")
    uploaded_file = st.file_uploader(
        "上傳單層切片圖像 (Upload Layer Image)", 
        type=['png', 'jpg', 'jpeg', 'bmp']
    )
    st.subheader("製程參數 (Process Parameters)")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        viscosity = p_col1.number_input("材料黏度 (cps)", min_value=50, max_value=1000, value=150, step=10)
        lift_height = p_col1.number_input("抬升高度 (μm)", min_value=500, max_value=8000, value=1500, step=100)
        lift_speed = p_col1.number_input("抬升速度 (μm/s)", min_value=100, max_value=8000, value=700, step=50)
    with p_col2:
        wait_time = p_col2.number_input("等待時間 (s)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        down_speed = p_col2.number_input("下降速度 (μm/s)", min_value=1000, max_value=10000, value=4000, step=500)
        shape = p_col2.selectbox("形狀 (Shape)", options=['90x45矩形', '90x50六角形', '50圓柱'])
    predict_button = st.button("執行預測 (Run Prediction)", type="primary")

# --- Column 2: Prediction and Recommendation ---
with col2:
    st.header("2. 預測結果與建議")
    if predict_button:
        # --- Step 1: 讀取並初始化 GenAI SDK ---
        genai_api_key = st.secrets.get("GENAI_API_KEY") or os.getenv("GENAI_API_KEY")
        if not genai_api_key:
            st.error("GenAI API Key 未設定！請在 Streamlit Secrets 或環境變數中設定。")
        else:
            try:
                genai.configure(api_key=genai_api_key)
            except Exception as e:
                st.error(f"初始化 GenAI SDK 發生錯誤: {e}")

            # --- Step 2: 上傳圖像檢查 ---
            if uploaded_file is None:
                st.error("請先上傳圖像文件。")
            else:
                with st.spinner("處理中... 正在分析圖像並執行預測..."):
                    temp_image_path = f"temp_{uploaded_file.name}"
                    with open(temp_image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    geo_features = extract_geometric_features(temp_image_path)
                    os.remove(temp_image_path)

                    if geo_features is None:
                        st.error("圖像處理失敗，請檢查圖像文件是否有效。")
                    else:
                        st.info(f"圖像特徵提取成功：\n" 
                                f"- 面積: {geo_features['area']:.2f} mm²\n" 
                                f"- 周長: {geo_features['perimeter']:.2f} mm\n" 
                                f"- 水力直徑: {geo_features['hydraulic_diameter']:.2f} mm")
                        
                        # --- Prepare Data for Model ---
                        input_data = {
                            '材料黏度 (cps)': viscosity,
                            '抬升高度(μm)': lift_height,
                            '抬升速度(μm/s)': lift_speed,
                            '等待時間(s)': wait_time,
                            '下降速度((μm)/s)': down_speed,
                            '形狀': shape,
                            '面積(mm?)': geo_features['area'],
                            '周長(mm)': geo_features['perimeter'],
                            '水力直徑(mm)': geo_features['hydraulic_diameter']
                        }
                        final_input_data = {feat: input_data.get(feat) for feat in INPUT_FEATURES}
                        input_df = pd.DataFrame([final_input_data])

                        # --- Run Prediction ---
                        try:
                            prediction, importances = load_model_and_predict(input_df)
                            if prediction == 0:
                                st.success("✅ **預測成功：樹脂回流完全**")
                                st.write("目前的參數設定在此層是安全的，可以繼續列印。")
                            else:
                                st.error("🚨 **預測失敗：樹脂回流不完全**")
                                st.write("偵測到潛在的列印失敗風險。正在向 AI 尋求優化建議...")
                                with st.spinner("正在生成 AI 建議..."):
                                    recommendation = get_llm_recommendation(final_input_data, importances)
                                    st.markdown("---")
                                    st.subheader("🤖 AI 優化建議")
                                    st.markdown(recommendation)
                        except FileNotFoundError as e:
                            st.error(f"模型文件遺失：{e}\n請先執行 `python model_train.py` 訓練模型。")
                        except Exception as e:
                            st.error(f"預測時發生錯誤：{e}")
