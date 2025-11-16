# app.py
import streamlit as st
import pandas as pd
import zipfile
import os
from model_train import load_model_and_predict, INPUT_FEATURES
from llm_recommender import get_llm_recommendation

st.set_page_config(page_title="3D 列印回流檢測", layout="wide")
st.title("⚙️ 3D 列印回流檢測與 AI 優化建議")

uploaded_file = st.file_uploader("請上傳列印圖檔 ZIP (.zip)", type=['zip'])

if uploaded_file:
    try:
        with zipfile.ZipFile(uploaded_file) as zf:
            # 取出所有 PNG 檔（忽略大小寫），只取檔名
            png_files = [os.path.basename(name) for name in zf.namelist() if name.lower().endswith(".png")]
            # 去除重複檔名
            png_files = list(dict.fromkeys(png_files))

            if not png_files:
                st.error("ZIP 內沒有 PNG 檔案")
            else:
                st.success(f"找到 {len(png_files)} 張圖片")
                layer_choice = st.selectbox("選擇要檢測的圖層", png_files)

                # 模擬讀取圖檔對應特徵數據
                input_data = {
                    '材料黏度 (cps)': 1000,
                    '抬升高度(μm)': 50,
                    '抬升速度(μm/s)': 20,
                    '等待時間(s)': 5,
                    '下降速度((μm)/s)': 15,
                    '形狀': '方形',
                    '面積(mm?)': 200,
                    '周長(mm)': 60,
                    '水力直徑(mm)': 10
                }
                final_input_data = {feat: input_data.get(feat) for feat in INPUT_FEATURES}
                input_df = pd.DataFrame([final_input_data])

                try:
                    prediction, importances = load_model_and_predict(input_df)

                    if prediction == 0:
                        st.success("✅ **預測成功：樹脂回流完全**")
                        st.write("目前的參數設定安全，可以繼續列印。")
                    else:
                        st.error("🚨 **預測失敗：樹脂回流不完全**")
                        st.write("偵測到潛在列印失敗風險，正在生成 AI 建議...")
                        with st.spinner("正在生成 AI 建議..."):
                            recommendation = get_llm_recommendation(final_input_data, importances)
                            st.markdown("---")
                            st.subheader("🤖 AI 優化建議")
                            st.markdown(recommendation)

                except FileNotFoundError as e:
                    st.error(f"模型文件遺失：{e}\n請先執行 `python model_train.py` 訓練模型。")
                except Exception as e:
                    st.error(f"預測時發生錯誤：{e}")

    except Exception as e:
        st.error(f"ZIP 檔案讀取失敗：{e}")
