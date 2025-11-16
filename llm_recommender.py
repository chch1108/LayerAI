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
# 4. 主函式：整體 summary
# -------------------------------
def llm_summary_feedback(stats_summary, summary_prompt=None):
    """
    stats_summary: dict, 包含統計資訊，例如：
        {
            "total_layers": 100,
            "high_risk_layers": 5,
            "avg_prob": 0.23,
            "max_prob": 0.72
        }
    summary_prompt: str, 可自訂 summary prompt
    """
    if model is None:
        return "(LLM 初始化失敗：請確認 GOOGLE_API_KEY)"

    if summary_prompt is None:
        # 預設 summary prompt
        summary_prompt = f"""
你是一位具有豐富光固化 3D 列印（DLP / LCD / CLIP）經驗的製程工程師。
請根據下列統計資訊，提供「繁體中文」的整體建議或結論，重點放在可能的高風險層與優化方向：

【統計資訊】
- 總層數：{stats_summary.get('total_layers', 0)}
- 高風險層數（prob >= 0.5）：{stats_summary.get('high_risk_layers', 0)}
- 平均失敗機率：{stats_summary.get('avg_prob', 0):.3f}
- 最高失敗機率：{stats_summary.get('max_prob', 0):.3f}

請給出具體建議或結論，文字需至少 3~5 行。
"""

    # 呼叫 Gemini
    try:
        print("[DEBUG] 開始呼叫 LLM 生成整體 summary...")
        reply = model.generate_content(summary_prompt)
        print("[DEBUG] LLM 回覆已返回")
        text = _extract_text_safe(reply)
        if not text or text.strip() == "":
            return "(LLM 回覆為空，請稍後再試)"
        return text
    except Exception as e:
        return f"(LLM Error: {e})"
