"""
Module: llm_recommender.py
功能：用 Gemini 產生逐層建議（只在高風險時使用）
"""

import os
import google.generativeai as genai
from typing import Dict, Any

API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("[WARNING] 找不到 GENAI_API_KEY，LLM 建議功能將無法使用。")

model = genai.GenerativeModel("gemini-2.5-flash")


def get_llm_recommendation(input_params: Dict[str, Any], feature_importances: Dict[str, float]) -> str:
    """
    給單層產生建議（高風險才使用）
    """
    try:
        sorted_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

        params_str = "\n".join([f"- {k}: {v}" for k, v in input_params.items()])
        importances_str = "\n".join([f"- {feat}: {imp:.3f}" for feat, imp in sorted_imp[:5]])

        prompt = f"""
你是一名 DLP 光固化 3D 列印製程工程師。
此層被預測為「樹脂回流不完全」。

請根據列印參數與關鍵影響因素，提供 2 項可執行優化建議。

【列印參數】
{params_str}

【影響因子（模型判斷）】
{importances_str}

請使用繁體中文回覆，格式為：

1. 建議項目：
   - 目前數值：xxx
   - 建議數值：yyy
   - 原因：zzz

2. 建議項目：
   - 目前數值：xxx
   - 建議數值：yyy
   - 原因：zzz
"""

        reply = model.generate_content(prompt)
        return reply.text or "(LLM 無回覆)"

    except Exception as e:
        return f"(LLM ERROR: {e})"


def get_low_risk_message():
    """
    低風險層固定結論
    """
    return "✔ 此層回流狀況良好，模型判定無明顯風險。\n建議維持目前參數設定。"
