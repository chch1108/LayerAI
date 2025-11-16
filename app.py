import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
from model_train import load_model_and_predict, INPUT_FEATURES
from llm_recommender import get_llm_recommendation

st.set_page_config(page_title="3D 列印回流檢測", layout="wide")
st.title("⚙️ 3D 列印回流檢測與 AI 優化建議")

uploaded_file = st.file_uploader("請上傳列印圖檔 ZIP (.zip)", type=['zip'])

def extract_images_from_zip(zip_path, extract_dir):
    """解壓 ZIP 並取得 PNG 圖片路徑"""
    imgs = []
    filenames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".png"):
                # 保留原始名稱，避免重複時加上 _1, _2
                base = os.path.basename(name)
                if base in filenames:
                    continue  # 已存在則跳過
                zf.extract(name, path=extract_dir)
                imgs.append(os.path.join(extract_dir, name))
                filenames.append(base)
    return imgs, filenames

if uploaded_file:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "slices.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info("解壓並讀取切片中...")
            imgs, filenames = extract_images_from_zip(zip_path, tmpdir)
            if not imgs:
                st.error("ZIP 內沒有 PNG 圖片")
            else:
                st.success(f"讀取 {len(imgs)} 張切片")
                # 選擇圖層
                layer_choice_base = st.selectbox("選擇要檢測的圖層", filenames)
                layer_choice_full = imgs[filenames.index(layer_choice_base)]

                # --- 模擬讀取圖檔對應特徵 ---
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

                # --- Run Prediction ---
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
