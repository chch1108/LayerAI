import os
import openai
from openai import OpenAI, AuthenticationError, OpenAIError
from typing import Dict, Any

# The new OpenAI v1.0+ library uses a client-based approach.
# The client automatically reads the OPENAI_API_KEY from environment variables.

def get_llm_recommendation(
    input_params: Dict[str, Any], 
    feature_importances: Dict[str, float]
) -> str:
    """
    Generates printing parameter recommendations using an LLM (GPT).
    This function is updated for openai library v1.0+.
    """
    try:
        # Instantiating the client will raise an error if the API key is not found.
        client = OpenAI()
    except OpenAIError:
        return (
            "**LLM Recommender Error:**\n"
            "OPENAI_API_KEY environment variable not set or invalid.\n\n"
            "**To fix this:**\n"
            "1. Go to your Streamlit app's 'Settings' -> 'Secrets'.\n"
            "2. Ensure the secret is correctly set as:\n"
            "   `OPENAI_API_KEY='your_key_here'`"
        )

    # --- 1. Prepare the Prompt ---
    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    params_str = "\n".join([f"- {key}: {value}" for key, value in input_params.items()])
    importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_importances[:5]])

    prompt = f"""
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
    
    **2. Decrease Lifting Speed:**
       - **Current:** 700 μm/s
       - **Suggested:** 400 μm/s
       - **Reason:** A slower lift speed reduces the vacuum force and allows the resin to flow back more gently.
    """

    # --- 2. Call the LLM API (using the new syntax) ---
    try:
        print("Sending request to LLM for recommendation...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert AI assistant for DLP 3D printing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=250
        )
        recommendation = response.choices[0].message.content.strip()
        print("LLM recommendation received.")
        return recommendation

    except AuthenticationError:
        return (
            "**LLM Recommender Error:**\n"
            "Authentication failed. Your OpenAI API key is likely invalid or expired.\n\n"
            "**To fix this:**\n"
            "1. Verify your API key is correct in Streamlit's 'Secrets' settings.\n"
            "2. Ensure it is formatted as: `OPENAI_API_KEY='your_key_here'`"
        )
    except Exception as e:
        print(f"An error occurred with the LLM API call: {e}")
        return f"**LLM Recommender Error:**\nAn unexpected error occurred: {e}"


if __name__ == '__main__':
    # This example remains the same, as the client inside the function
    # will pick up the key from the environment.
    print("--- Running a test recommendation ---")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping test: OPENAI_API_KEY not set.")
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