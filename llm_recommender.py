import os
import openai
from typing import Dict, Any

# --- Security Best Practice ---
# Load the API key from an environment variable for security.
# DO NOT hardcode the API key in the script.
# You can set it in your terminal like this:
# export OPENAI_API_KEY='your_api_key_here'
openai.api_key = os.getenv("OPENAI_API_KEY")

# A fallback for Hugging Face or other free APIs can be added here as well.

def get_llm_recommendation(
    input_params: Dict[str, Any], 
    feature_importances: Dict[str, float]
) -> str:
    """
    Generates printing parameter recommendations using an LLM (GPT).

    This function constructs a detailed prompt for the LLM, including the
    failed parameters and the most influential factors, to get actionable
    advice.

    Args:
        input_params (Dict[str, Any]): The dictionary of input parameters that
                                       led to the predicted failure.
        feature_importances (Dict[str, float]): A dictionary mapping feature
                                                names to their importance scores.

    Returns:
        str: A string containing the optimization advice from the LLM.
             Returns an error message if the API call fails.
    """
    if not openai.api_key:
        return (
            "**LLM Recommender Error:**\n"
            "OPENAI_API_KEY environment variable not set.\n\n"
            "**To fix this:**\n"
            "1. Get an API key from OpenAI.\n"
            "2. Set it in your terminal before running the app:\n"
            "   `export OPENAI_API_KEY='your_api_key_here'`\n"
            "3. Rerun the Streamlit application."
        )

    # --- 1. Prepare the Prompt ---
    # Sort features by importance to highlight the most critical ones.
    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    
    # Format the input parameters and importances for the prompt.
    params_str = "\n".join([f"- {key}: {value}" for key, value in input_params.items()])
    importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_importances[:5]]) # Top 5

    # Construct the detailed prompt.
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

    # --- 2. Call the LLM API ---
    try:
        print("Sending request to LLM for recommendation...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using the specified free-tier-friendly model
            messages=[
                {"role": "system", "content": "You are an expert AI assistant for DLP 3D printing."}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # Lower temperature for more deterministic and factual advice
            max_tokens=250    # Enough for a concise recommendation
        )
        recommendation = response.choices[0].message['content'].strip()
        print("LLM recommendation received.")
        return recommendation

    except openai.error.AuthenticationError:
        return (
            "**LLM Recommender Error:**\n"
            "Authentication failed. Your OpenAI API key is likely invalid or expired.\n\n"
            "**To fix this:**\n"
            "1. Verify your API key is correct.\n"
            "2. Set the environment variable again:\n"
            "   `export OPENAI_API_KEY='your_key_here'`"
        )
    except Exception as e:
        print(f"An error occurred with the LLM API call: {e}")
        return f"**LLM Recommender Error:**\nAn unexpected error occurred: {e}"


if __name__ == '__main__':
    # --- Example of how to use the function ---
    # This block will only run if the script is executed directly.
    # NOTE: This requires the OPENAI_API_KEY to be set.
    
    print("--- Running a test recommendation ---")
    
    # Sample data mimicking a failed prediction
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
