"""
Module: llm_recommender.py
功能: 使用 Gemini-2.5-Flash 模型生成逐層 3D 列印回流優化建議
"""

import os
import requests
import json
from typing import Dict, Any
import streamlit as st

# --- GenAI SDK 配置，從 Streamlit Secrets 讀取 ---
import genai
API_SECRET_NAME = "GENAI_API_KEY"
genai_api_key = st.secrets.get(API_SECRET_NAME) or os.getenv(API_SECRET_NAME)
if not genai_api_key:
    st.error(f"{API_SECRET_NAME} 未設定！請到 Streamlit Secrets 或環境變數設定你的 API Key。")
genai.configure(api_key=genai_api_key)

# --- Constants ---
MODEL_ID = "gemini-2.5-flash"
API_URL = "https://router.huggingface.co/hf-inference"

def get_llm_recommendation(
    input_params: Dict[str, Any],
    feature_importances: Dict[str, float]
) -> str:
    """
    根據每層的製程參數與模型判定的影響因子，使用 Gemini-2.5-Flash 中文模型生成列印優化建議。
    """
    if not genai.api_key:
        return f"**LLM Recommender Error:** {API_SECRET_NAME} 未設定。"

    # --- Prompt 準備 ---
    sorted_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    params_str = "\n".join([f"- {k}: {v}" for k, v in input_params.items()])
    importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_imp[:5]])

    prompt = (
        f"一個 DLP 3D 列印的單層被預測為「樹脂回流不完全」。\n"
        f"請根據下面的製程參數與關鍵影響因素，以繁體中文提供兩項可執行的優化建議："
        f"\n\n列印參數：\n{params_str}"
        f"\n\n最具影響力因素：\n{importances_str}"
        f"\n\n回覆格式：\n"
        f"1. 建議項目：\n"
        f"   - 目前數值：xxx\n"
        f"   - 建議數值：yyy\n"
        f"   - 原因：zzz\n"
        f"\n2. 建議項目：\n"
        f"   - 目前數值：xxx\n"
        f"   - 建議數值：yyy\n"
        f"   - 原因：zzz\n"
    )

    # --- Payload 建構（chat-style）---
    payload = {
        "model": MODEL_ID,
        "inputs": [
            {"role": "system", "content": "你是 3D 列印製程優化專家。"},
            {"role": "user", "content": prompt}
        ],
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    headers = {
        "Authorization": f"Bearer {genai.api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            return f"**LLM Recommender Error:** {result['error']}"
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        return f"**LLM Recommender Error:** Unexpected response format: {result}"

    except requests.exceptions.HTTPError as http_err:
        return f"**LLM Recommender Error:** HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return f"**LLM Recommender Error:** Unexpected exception: {e}"
