"""
Module: llm_recommender.py
功能: 使用 Gemini-2.5-Flash 模型生成逐層 3D 列印回流優化建議
"""

from typing import Dict, Any
import genai

MODEL_ID = "gemini-2.5-flash"

def get_llm_recommendation(input_params: Dict[str, Any], feature_importances: Dict[str, float]) -> str:
    """
    根據每層的製程參數與模型判定的影響因子，
    使用 Gemini-2.5-Flash 中文模型生成列印優化建議。
    """

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

    try:
        response = genai.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "你是 3D 列印製程優化專家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_output_tokens=300
        )

        text = response.choices[0].content.strip()
        return text

    except Exception as e:
        return f"**LLM Recommender Error:** 發生錯誤: {e}"
