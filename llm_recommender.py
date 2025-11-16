"""
Module: llm_recommender.py
功能: 針對高風險層生成 3D 列印回流建議或結論。
保證：
- 一定有 API Key（無就提示）
- 一定有輸出文字（避免空白）
- 兼容 Streamlit Cloud / Local
"""

import os
import google.generativeai as genai

# -------------------------------
# 1. API KEY 配置
# -------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY is None:
    print("[WARNING] GOOGLE_API_KEY 未設定，LLM 將無法正常運作。")
else:
    genai.configure(api_key=API_KEY)

# -------------------------------
# 2. 初始化 Gemini 模型
# -------------------------------
MODEL_NAME = "gemini-2.5-flash"
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"[ERROR] 初始化 Gemini 模型失敗：{e}")
    model = None

# -------------------------------
# 3. 安全提取 Gemini 文本
# -------------------------------
def _extract_text_safe(reply):
    try:
        if hasattr(reply, "text") and reply.text:
            return reply.text
        if hasattr(reply, "candidates") and reply.candidates:
            cand = reply.candidates[0]
            if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                part = cand.content.parts[0]
                if hasattr(part, "text") and part.text:
                    return part.text
        return str(reply)
    except Exception:
        return "(LLM 無回覆內容)"

# -------------------------------
# 4. 主函式：針對高風險層生成建議
# -------------------------------
def llm_highrisk_feedback(stats_summary, threshold=0.5):
    """
    針對高風險層生成 LLM 建議，若沒有高風險層則回傳簡單結論。
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
        # 全部低風險
        return "所有層回流正常，結構穩定，無需額外調整。"

    # 高風險層存在，生成建議
    prompt = f"""
你是一位具有豐富光固化 3D 列印經驗的製程工程師。
目前系統共 {stats_summary.get('total_layers', 0)} 層，其中高風險層數 {high_risk_layers}。
平均失敗機率 {stats_summary.get('avg_prob', 0):.3f}，最高失敗機率 {stats_summary.get('max_prob', 0):.3f}。

請提供「繁體中文」的具體建議，重點放在高風險層的優化方向，至少 3~5 行。
"""

    try:
        print("[DEBUG] 開始呼叫 LLM 生成高風險建議...")
        reply = model.generate_content(prompt)
        print("[DEBUG] LLM 回覆已返回")
        text = _extract_text_safe(reply)
        if not text or text.strip() == "":
            return "(LLM 回覆為空，請稍後再試)"
        return text
    except Exception as e:
        return f"(LLM Error: {e})"
