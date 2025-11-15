import os
import requests
import json
from typing import Dict, Any

# --- Constants ---
MODEL_ID = "RayTsai/chinese-reasoning-qwen2.5-7b"  # 中文推理模型
API_URL = "https://router.huggingface.co/hf-inference"  # Router API endpoint
SECRET_ENV_VAR = "HF_TOKEN"

def get_llm_recommendation(
    input_params: Dict[str, Any], 
    feature_importances: Dict[str, float]
) -> str:
    """
    Generates printing parameter recommendations using a Hugging Face
    Router API with a Chinese reasoning model (Qwen2.5-7B).
    """
    hf_token = os.getenv(SECRET_ENV_VAR)
    if not hf_token:
        return (
            f"**LLM Recommender Error:**\n"
            f"{SECRET_ENV_VAR} environment variable not set.\n\n"
            "**To fix this:**\n"
            "1. Get a free API Token from Hugging Face (`https://huggingface.co/settings/tokens`).\n"
            f"2. In your Streamlit app's 'Settings' -> 'Secrets', set the secret as:\n"
            f"   `{SECRET_ENV_VAR}='your_token_here'`"
        )

    # --- 1. Prepare Prompt ---
    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    params_str = "\n".join([f"- {key}: {value}" for key, value in input_params.items()])
    importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_importances[:5]])

    user_content = f"""
一個 DLP 3D 列印單層預測失敗，原因為樹脂回流不完全。
根據下列參數，請提供兩項簡明可執行、帶數值的建議來改善此問題，請以繁體中文說明。

**列印參數:**
{params_str}

**影響最大因素:**
{importances_str}

**請求內容:**
對每個建議，提供：
- **目前數值**
- **建議數值**
- **簡短原因**

**範例格式:**
**1. 增加等待時間:**
   - **目前數值:** 0.5s
   - **建議數值:** 1.0s
   - **原因:** 增加等待時間能給予樹脂更多時間回流，是解決回流問題最直接的方法。
"""

    # Qwen2 Chat Template
    prompt = f"<|im_start|>system\n你是專家 AI 助手，專門提供 DLP 3D 列印優化建議。<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    # --- 2. Call Hugging Face Router API ---
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.6,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # 檢查返回格式
        if isinstance(result, dict) and "error" in result:
            return f"**LLM Recommender Error:** {result['error']}"

        if isinstance(result, list) and "generated_text" in result[0]:
            recommendation = result[0]["generated_text"].strip()
            if "<|im_end|>" in recommendation:
                recommendation = recommendation.split("<|im_end|>")[0].strip()
            return recommendation
        else:
            return f"**LLM Recommender Error:** Unexpected API response format: {result}"

    except requests.exceptions.HTTPError as http_err:
        return f"**LLM Recommender Error:** HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return f"**LLM Recommender Error:** An unexpected error occurred: {e}"


# --- 測試範例 ---
if __name__ == '__main__':
    if not os.getenv(SECRET_ENV_VAR):
        print(f"Skipping test: {SECRET_ENV_VAR} not set.")
    else:
        sample_params = {
            '形狀': '90x45矩形', '材料黏度 (cps)': 150, '抬升高度(μm)': 2000,
            '抬升速度(μm/s)': 1000, '等待時間(s)': 0.5, '下降速度((μm)/s)': 4000,
            '面積(mm?)': 4034.83, '周長(mm)': 269.6, '水力直徑(mm)': 59.86
        }
        sample_importances = {
            '抬升速度(μm/s)': 0.35, '等待時間(s)': 0.25, '水力直徑(mm)': 0.15,
            '面積(mm?)': 0.12, '材料黏度 (cps)': 0.08, '抬升高度(μm)': 0.05
        }
        recommendation = get_llm_recommendation(sample_params, sample_importances)
        print("\n--- LLM Recommendation ---")
        print(recommendation)
        print("--------------------------")
