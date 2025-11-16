"""
Module: llm_recommender.py
功能: 使用 Gemini-2.5-Flash 模型生成逐層 3D 列印回流優化建議
"""

from typing import Dict, Any
from google.generativeai import GenerativeModel

MODEL_ID = "gemini-2.5-flash"

# 初始化模型物件 (全域)
model = GenerativeModel(MODEL_ID)

def get_llm_recommendation(input_params: Dict[str, Any], feature_importances: Dict[str, float]) -> str:
    """
    使用 Gemini-2.5-Flash 中文模型生成逐層 3D 列印回流優化建議
    """
    try:
        # --- Prompt 準備 ---
        sorted_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        params_str = "\n".join([f"- {k}: {v}" for k, v in input_params.items()])
        importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_imp[:5]])

        prompt = (
            f"你是一個 DLP 3D 列印製程優化專家。\n"
            f"單層被預測為「樹脂回流不完全」。\n"
            f"請根據下列製程參數與關鍵影響因素，提供兩項可執行優化建議（繁體中文）："
            f"\n\n列印參數：\n{params_str}"
            f"\n\n最具影響力因素：\n{importances_str}"
            f"\n\n回覆格式：\n"
            f"1. 建議項目：\n   - 目前數值：xxx\n   - 建議數值：yyy\n   - 原因：zzz\n"
            f"2. 建議項目：\n   - 目前數值：xxx\n   - 建議數值：yyy\n   - 原因：zzz\n"
        )

        # --- 呼叫 GenerativeModel 生成內容 ---
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"**LLM Recommender Error:** 發生錯誤: {e}"
