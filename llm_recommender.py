"""
Module: llm_recommender.py
功能: 使用 Hugging Face Router API 呼叫「Qwen2.5‑7B‑Instruct（中文推理）」模型，生成逐層 3D 列印回流優化建議
"""

import os
import requests
import json
from typing import Dict, Any
import streamlit as st

# --- Constants ---
MODEL_ID = "RayTsai/chinese-reasoning-qwen2.5-7b"
API_URL = "https://router.huggingface.co/hf-inference"
SECRET_NAME = "HF_TOKEN"

def get_llm_recommendation(
    input_params: Dict[str, Any],
    feature_importances: Dict[str, float]
) -> str:
    """
    根據每層的製程參數與模型判定的影響因子，使用 Qwen2.5‑7B 中文模型生成列印優化建議。
    Args:
        input_params: 製程參數字典
        feature_importances: 各特徵的重要性字典
    Returns:
        建議文字（繁體中文）
    """
    # 讀取 Hugging Face Token
    hf_token = st.secrets.get(SECRET_NAME) or os.getenv(SECRET_NAME)
    if not hf_token:
        return (
            f"**LLM Recommender Error:**\n"
            f"{SECRET_NAME} not set.\n\n"
            "**To fix this:**\n"
            "1. 到 https://huggingface.co/settings/tokens 取得免費 API Token。\n"
            "2. 在 Streamlit 的 Secrets 或環境變數中設定：\n"
            f"   {SECRET_NAME}='你的_token_here'"
        )

    # 準備 Prompt
    sorted_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    params_str = "\n".join([f"- {k}: {v}" for k, v in input_params.items()])
    importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_imp[:5]])

    prompt = (
        f"[USER]\n"
        f"一個 DLP 3D 列印的單層被預測為「樹脂回流不完全」。\n"
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

    # 建構 payload – 參考 Qwen 模型可能需要 chat-style inputs 或 inputs 為 prompt
    payload = {
        "model": MODEL_ID,
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        try:
            result = response.json()
        except json.JSONDecodeError:
            return (
                "**LLM Status:** 模型可能仍在啟動中，請稍候30–60秒後重試。\n"
                "(Received non‑JSON response)"
            )

        # 處理 error 回傳
        if isinstance(result, dict) and "error" in result:
            return f"**LLM Recommender Error:** {result['error']}"

        # 處理正常回傳
        if isinstance(result, list) and "generated_text" in result[0]:
            txt = result[0]["generated_text"].strip()
            # 若模型回傳含特殊 token，如 <|endoftext|>，可加清理
            return txt
        else:
            return f"**LLM Recommender Error:** Unexpected response format: {result}"

    except requests.exceptions.HTTPError as http_err:
        return f"**LLM Recommender Error:** HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return f"**LLM Recommender Error:** Unexpected exception: {e}"
