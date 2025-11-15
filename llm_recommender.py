import os
import requests
import json
from typing import Dict, Any

# --- Constants ---
# The new recommended endpoint for Hugging Face Serverless Inference
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
SECRET_ENV_VAR = "HF_TOKEN"

def get_llm_recommendation(
    input_params: Dict[str, Any], 
    feature_importances: Dict[str, float]
) -> str:
    """
    Generates printing parameter recommendations using a Hugging Face
    Inference API model (Mistral-7B-Instruct).

    Args:
        input_params (Dict[str, Any]): The dictionary of input parameters that
                                       led to the predicted failure.
        feature_importances (Dict[str, float]): A dictionary mapping feature
                                                names to their importance scores.

    Returns:
        str: A string containing the optimization advice from the LLM.
             Returns an error message if the API call fails.
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

    # --- 1. Prepare the Prompt for Mistral ---
    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    params_str = "\n".join([f"- {key}: {value}" for key, value in input_params.items()])
    importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_importances[:5]])

    # Using Mistral's instruction format for better results
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

    # --- 2. Call the Hugging Face API ---
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    # The payload now includes the model name, as we are using a general router endpoint.
    payload = {
        "model": MODEL_NAME,
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.6,
            "return_full_text": False, # Only return the generated part
        }
    }

    # The new endpoint URL from the error message
    new_api_url = "https://router.huggingface.co/hf-inference"

    try:
        print(f"Sending request to new Hugging Face endpoint: {new_api_url}...")
        response = requests.post(new_api_url, headers=headers, json=payload, timeout=45)
        
        # Check for HTTP errors
        if response.status_code >= 400:
            error_body = response.json()
            error_message = error_body.get("error", str(response.text))
            # Specifically handle the old endpoint error message if it appears again
            if "is no longer supported" in error_message:
                 error_message = "The Hugging Face API endpoint is still incorrect. Please check for the latest documentation."
            
            if response.status_code == 401:
                 return (
                    "**LLM Recommender Error:**\n"
                    "Authentication failed. Your Hugging Face API Token is likely invalid or expired.\n\n"
                    "**To fix this:**\n"
                    "1. Verify your token is correct in Streamlit's 'Secrets' settings.\n"
                    f"2. Ensure it is formatted as: `{SECRET_ENV_VAR}='hf_...'`"
                )
            raise requests.exceptions.HTTPError(f"HTTP {response.status_code}: {error_message}")

        result = response.json()
        
        if result and isinstance(result, list) and "generated_text" in result[0]:
            recommendation = result[0]["generated_text"].strip()
            print("LLM recommendation received.")
            return recommendation
        else:
            raise ValueError(f"Unexpected API response format: {result}")

    except requests.exceptions.HTTPError as http_err:
        return f"**LLM Recommender Error:**\n{http_err}"
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