import os
import requests
import json
from typing import Dict, Any

# --- Constants ---
# Switching to a more standard, widely available model and the standard API endpoint structure.
MODEL_ID = "google/flan-t5-large"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
SECRET_ENV_VAR = "HF_TOKEN"

def get_llm_recommendation(
    input_params: Dict[str, Any], 
    feature_importances: Dict[str, float]
) -> str:
    """
    Generates printing parameter recommendations using a Hugging Face
    Inference API model (google/flan-t5-large).
    """
    hf_token = os.getenv(SECRET_ENV_VAR)
    if not hf_token:
        return (
            f"**LLM Recommender Error:**\n"
            f"{SECRET_ENV_VAR} environment variable not set.\n\n"
            "**To fix this:**\n"
            "1. Get a free API Token from Hugging Face (`https://huggingface.co/settings/tokens`).\n"
            "2. In your Streamlit app's 'Settings' -> 'Secrets', set the secret as:\n"
            f"   `{SECRET_ENV_VAR}='your_token_here'`"
        )

    # --- 1. Prepare the Prompt for Flan-T5 ---
    # Flan-T5 is a text-to-text model, so we create a clear, direct instruction.
    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    params_str = "\n".join([f"- {key}: {value}" for key, value in input_params.items()])
    importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_importances[:5]])

    prompt = f"""
Context: A DLP 3D printing process for a single layer is predicted to fail due to incomplete resin reflow.

Printing Parameters Used:
{params_str}

Most Influential Factors:
{importances_str}

Task: Based on the data above, provide two concise, actionable, and numerical recommendations to fix the resin reflow issue. For each recommendation, provide the current value, the suggested new value, and a brief reason.
"""

    # --- 2. Call the Hugging Face API ---
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200, # Flan-T5 can be a bit more verbose
            "temperature": 0.7,
        }
    }

    try:
        print(f"Sending request to Hugging Face endpoint: {API_URL}...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
        
        response.raise_for_status()
        
        try:
            result = response.json()
        except json.JSONDecodeError:
            return (
                "**LLM Status:** The model is likely loading on the server. "
                "This is common on the first request. **Please try again in about 30-60 seconds.**\n\n"
                f"(Details: Received a non-JSON response from the API)"
            )

        if isinstance(result, dict) and "error" in result:
            error_message = result.get("error")
            if isinstance(error_message, str) and "loading" in error_message.lower():
                estimated_time = result.get("estimated_time", "30")
                return (
                    f"**LLM Status:** The model is currently loading on the server. "
                    f"**Please try again in about {estimated_time} seconds.**\n\n"
                    f"(Details: {error_message})"
                )
            else:
                raise ValueError(f"API returned an error in the JSON body: {error_message}")

        if result and isinstance(result, list) and "generated_text" in result[0]:
            recommendation = result[0]["generated_text"].strip()
            print("LLM recommendation received.")
            return recommendation
        else:
            raise ValueError(f"Unexpected API response format: {result}")

    except requests.exceptions.HTTPError as http_err:
        error_body = response.text
        if response.status_code == 401:
             return (
                "**LLM Recommender Error:**\n"
                "Authentication failed. Your Hugging Face API Token is likely invalid or expired."
            )
        return f"**LLM Recommender Error:**\nHTTP Error {response.status_code}: {error_body}"
    except Exception as e:
        print(f"An error occurred with the LLM API call: {e}")
        return f"**LLM Recommender Error:**\nAn unexpected error occurred: {e}"


if __name__ == '__main__':
    print("--- Running a test recommendation with Hugging Face API ---")
    
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
