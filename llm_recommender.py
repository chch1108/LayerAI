"""
Module: llm_recommender.py
功能: 使用 Hugging Face Router API 生成逐層 3D 列印回流優化建議
"""

import streamlit as st
import requests
import json
from typing import Dict, Any

# --- Constants ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
SECRET_NAME = "HF_TOKEN"
API_URL = "https://router.huggingface.co/hf-inference"  # Router endpoint

def get_llm_recommendation(
    input_params: Dict[str, Any], 
    feature_importances: Dict[str, float]
) -> str:
    """
    Generates printing parameter recommendations using Hugging Face Router API.
    """

    # --- 1. 讀取 Hugging Face Token ---
    hf_token = st.secrets.get(SECRET_NAME)
    if not hf_token:
        return (
            f"**LLM Recommender Error:**\n"
            f"{SECRET_NAME} not set in Streamlit Secrets.\n\n"
            "**To fix this:**\n"
            "1. Get a free API Token from Hugging Face (https://huggingface.co/settings/tokens).\n"
            "2. In your Streamlit app's Settings -> Secrets, set the secret as:\n"
            f"   `{SECRET_NAME}='your_token_here'`"
        )

    # --- 2. 準備 Prompt ---
    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    params_str = "\n".join([f"- {key}: {value}" for key, value in input_params.items()])
    importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_importances[:5]])

    prompt = f"""
[INST]
You are an expert AI assistant for DLP 3D printing.
A machine learning model has predicted a "resin reflow failure" for a single print layer with the following parameters.
This means the resin might not have enough time or space to flow back properly before the next layer, causing print defects.

**Printing Parameters Used:**
{params_str}

**Most Influential Factors (according to the model):**
{importances_str}

**Your Task:**
Provide concise, actionable, and numerical recommendations to fix this reflow issue.
Please suggest specific new values for 1-2 of the most critical parameters.
Explain *why* you are suggesting the change.
Keep the format clean and easy to read.

**Example Response Format:**
**1. Increase Wait Time:**
   - **Current:** 0.5s
   - **Suggested:** 1.0s
   - **Reason:** Increasing the wait time gives the resin more time to settle, which is the most direct way to resolve reflow issues.
[/INST]
"""

    # --- 3. 呼叫 Hugging Face Router API ---
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.6,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        try:
            result = response.json()
        except json.JSONDecodeError:
            return (
                "**LLM Status:** The model may be loading. "
                "Try again in 30-60 seconds.\n"
                "(Received non-JSON response)"
            )

        # 處理 API 回傳錯誤
        if isinstance(result, dict) and "error" in result:
            error_message = result.get("error")
            if "loading" in error_message.lower():
                return (
                    f"**LLM Status:** Model is loading on the server. "
                    "Please retry in ~30-60 seconds.\n"
                    f"(Details: {error_message})"
                )
            else:
                return f"**LLM Recommender Error:** {error_message}"

        # 處理正常回傳
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        else:
            return f"**LLM Recommender Error:** Unexpected response format: {result}"

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 401:
            return (
                "**LLM Recommender Error:**\n"
                "Authentication failed. Your Hugging Face API Token may be invalid or expired.\n"
                f"Check Streamlit Secrets: {SECRET_NAME}"
            )
        return f"**LLM Recommender Error:** HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return f"**LLM Recommender Error:** Unexpected exception: {e}"


# --- 測試程式 ---
if __name__ == '__main__':
    print("--- Running a test recommendation ---")
    if not st.secrets.get(SECRET_NAME):
        print(f"Skipping test: {SECRET_NAME} not set.")
    else:
        sample_params = {
            '形狀': '90x45矩形',
            '材料黏度 (cps)': 150,
            '抬升高度(μm)': 2000,
            '抬升速度(μm/s)': 1000,
            '等待時間(s)': 0.5,
            '下降速度((μm)/s)': 4000,
            '面積(mm?)': 4034.83,
            '周長(mm)': 269.6,
            '水力直徑(mm)': 59.86
        }
        sample_importances = {
            '抬升速度(μm/s)': 0.35,
            '等待時間(s)': 0.25,
            '水力直徑(mm)': 0.15,
            '面積(mm?)': 0.12,
            '材料黏度 (cps)': 0.08,
            '抬升高度(μm)': 0.05
        }

        recommendation = get_llm_recommendation(sample_params, sample_importances)
        print("\n--- LLM Recommendation ---")
        print(recommendation)
        print("--------------------------")
