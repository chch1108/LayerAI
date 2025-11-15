# AI 決策支持系統專案說明書

## 1. Purpose
本專案旨在開發一個 **AI 決策支持系統**，用於 **DLP 3D 列印前**，根據使用者輸入的每一層物件圖像及製程參數，預測該層的樹脂回流是否會不完全。  

若預測結果顯示失敗，系統將透過 **免費 GPT API**（Hugging Face 或 OpenAI GPT-3.5 免費額度）提供可執行的 **優化建議數值**，幫助使用者快速調整列印參數，提升列印成功率與效率。

---

## 2. Tech Stack
- **語言**: Python 3.10+
- **數據處理**: pandas, numpy
- **圖像處理 / 特徵提取**: OpenCV (或 Pillow, scikit-image)
- **機器學習**: scikit-learn
- **語言模型 (LLM)**: 免費 GPT API（Hugging Face Inference API 或 OpenAI GPT-3.5 免費額度）
- **介面 (UI/UX)**: Streamlit (快速建立展示原型)
- **模型儲存**: joblib

---

## 3. Project Conventions

### Code Style
- 遵循 **PEP 8** 標準
- 命名：
  - 變數 / 函式 / 檔案：`snake_case`
  - 類別：`CamelCase`
- 註解：
  - 重要函式和類別需撰寫 **Docstrings** 說明輸入、輸出與邏輯

### Architecture Patterns
- **多模組分離**：程式碼劃分為四大模組，方便單獨開發、測試、維護
- **異步處理 (Potential)**：圖像處理或 LLM 呼叫可能延遲，可使用 Streamlit 緩存 (`@st.cache_data`) 或異步
- **LLM API Wrapper**：在 `llm_recommender.py` 封裝 API 調用，隔離金鑰與邏輯，方便切換免費或付費模型

---

## 4. Domain Context (DLP 製程與圖像輸入)
- **數據顆粒度**：訓練與預測單位為 **單層**
- **圖像角色**：每層上傳的圖片（.png 或 .jpg）用於計算幾何特徵：
  - 面積 (Area)
  - 周長 (Perimeter)
  - 水力直徑 (Hydraulic Diameter)
- **數據集大小**：541 筆資料，需妥善劃分訓練/測試集，避免過度擬合
- **LLM 角色**：
  - **不直接預測數值**
  - 接收 ML 模型的預測失敗結果與參數敏感度分析作為 Prompt
  - 生成專業、具體、可執行的 **優化建議數值**（例如等待時間 0.5s → 1.0s）
  - 使用 **免費 GPT API**，速度可能慢，但可先用於原型展示

---

## 5. 模組化開發清單 (Modular Task List)

| 模組 | 檔案名稱 | 功能與職責 |
|------|----------|-----------|
| I. 圖像特徵提取 | `image_processor.py` | 接收使用者上傳單層圖片，執行影像處理 (邊緣偵測、輪廓分析)，輸出幾何特徵：面積、周長、水力直徑 |
| II. 核心預測模型 | `model_train.py` | 載入訓練好的 ML 模型，輸入 13 個特徵（製程參數 + 幾何特徵 + 物理特徵），輸出回流預測結果 (0/1) |
| III. LLM 建議引擎 | `llm_recommender.py` | 接收「預測失敗標誌」與「影響最大參數」，呼叫 **免費 GPT API**，生成可執行優化建議數值 |
| IV. 整合介面 (Frontend) | `app.py` | 建立 Streamlit 介面，協調 I、II、III 模組，展示輸入、預測結果、優化建議及歷史紀錄 |

---

## 6. External Dependencies
- **數據集**: `data.csv`（541 筆資料）
- **圖像處理套件**: OpenCV 或 scikit-image
- **免費 GPT API Key**: 可用 Hugging Face 免費模型或 OpenAI GPT-3.5 免費額度

---

## 7. Important Constraints
- **即時性 (Latency)**：逐層預測時，圖像處理與免費 GPT API 調用速度需在可接受範圍內  
- **安全性**：API 金鑰不得暴露於程式碼或前端  
- **編碼一致性**：訓練與預測數據需經過相同編碼 (One-Hot) 與標準化 (StandardScaler)

---

## 8. Project Workflow
1. 使用者上傳單層圖片與製程參數
2. `image_processor.py` 計算幾何特徵
3. `model_train.py` 輸入所有特徵，輸出回流預測
4. 若預測失敗：
   - `llm_recommender.py` 呼叫 **免費 GPT API**
   - 輸出優化建議數值（可直接給使用者）
5. `app.py` 整合結果，展示於 Streamlit 介面
