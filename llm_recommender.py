"""
Module: llm_recommender.py
功能: 使用 Gemini-2.5-Flash 模型生成高風險層 3D 列印回流優化建議
保證：
- 高風險層才呼叫 LLM
- 全低風險層時直接回傳結論
- 回覆格式與逐層建議範例一致
"""

from typing import Dict, Any
from google.generativeai import GenerativeModel

MODEL_ID = "gemini-2.5-flash"

# 初始化模型物件 (全域)
try:
    model = GenerativeModel(MODEL_ID)
except Exception as e:
    print(f"[ERROR] 初始化 Gemini 模型失敗：{e}")
    model = None

def get_llm_recommendation(input_params: Dict[str, Any], feature_importances: Dict[str, float], risk_level: str = "high") -> str:
    """
    使用 Gemini-2.5-Flash 生成逐層 3D 列印回流優化建議
    input_params: 列印參數字典
    feature_importances: 影響因素重要性字典
    risk_level: "high" 或 "low"，低風險直接回傳結論
    """
    if risk_level.lower() != "high":
        return "✅ 該層回流正常，無需額外優化。"

    if model is None:
        return "(LLM 初始化失敗：請確認 GOOGLE_API_KEY)"

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

        # --- 呼叫 GenerativeModel ---
        response = model.generate_content(prompt)

        # 安全取文字
        if hasattr(response, "text") and response.text:
            return response.text
        elif hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "(LLM 回覆為空，請稍後再試)"

    except Exception as e:
        return f"**LLM Recommender Error:** 發生錯誤: {e}"
