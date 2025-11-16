"""
Module: llm_recommender.py
功能: 使用 Gemini-2.5-Flash 模型生成高風險層 3D 列印回流優化建議
保證：
- 一次呼叫 LLM，針對高風險層產生建議
- 全低風險層時直接回傳結論，速度快
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

def llm_highrisk_feedback(stats_summary: Dict[str, Any], threshold: float = 0.5) -> str:
    """
    針對高風險層生成 LLM 建議
    stats_summary: dict, 包含統計資訊
        {
            "total_layers": int,
            "high_risk_layers": int,
            "avg_prob": float,
            "max_prob": float
        }
    """
    if model is None:
        return "(LLM 初始化失敗：請確認 GOOGLE_API_KEY)"

    high_risk_layers = stats_summary.get("high_risk_layers", 0)

    if high_risk_layers == 0:
        # 全部低風險層
        return "✅ 所有層回流正常，結構穩定，無需額外調整。"

    # 高風險層存在 → 生成建議
    prompt = (
        f"你是一位具有豐富光固化 DLP/LCD 3D 列印經驗的製程工程師。\n"
        f"目前系統共 {stats_summary.get('total_layers', 0)} 層，其中高風險層數 {high_risk_layers}。\n"
        f"平均失敗機率 {stats_summary.get('avg_prob', 0):.3f}，最高失敗機率 {stats_summary.get('max_prob', 0):.3f}。\n"
        f"請提供「繁體中文」的具體建議，重點放在高風險層的優化方向，至少 3~5 行，方便工程師快速調整。"
    )

    try:
        print("[DEBUG] 開始呼叫 LLM 生成高風險建議...")
        response = model.generate_content(prompt)
        print("[DEBUG] LLM 回覆已返回")

        # 安全提取文字
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            text = response.candidates[0].content.parts[0].text
        else:
            text = "(LLM 回覆為空，請稍後再試)"

        return text

    except Exception as e:
        return f"(LLM Error: {e})"
