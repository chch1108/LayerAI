import os
import google.generativeai as genai

API_KEY = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception:
    model = None


def _safe_extract_text(reply):
    try:
        if hasattr(reply, "text") and reply.text:
            return reply.text
        if getattr(reply, "candidates", None):
            c = reply.candidates[0]
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                return c.content.parts[0].text
    except Exception:
        pass
    return "(LLM 未回傳內容或 API 未設定)"


def get_llm_recommendation(input_params, feature_importances):
    """
    Generate recommendation text using Gemini.
    If model not accessible, return heuristic text.
    """
    sorted_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_imp = sorted_imp[:5]

    params_str = "\n".join([f"- {k}: {v}" for k, v in input_params.items()])
    imp_str = "\n".join([f"- {k}: {v:.3f}" for k, v in top_imp])

    prompt = f"""
你是光固化 3D 列印製程專家。請以繁體中文根據以下資訊提供 2 項可執行的優化建議：

列印參數：
{params_str}

最重要影響參數：
{imp_str}

請用格式：
1. 建議項目：
 - 目前數值：...
 - 建議數值：...
 - 原因：...
2. 建議項目：
 - 目前數值：...
 - 建議數值：...
 - 原因：...
"""

    # fallback
    if model is None:
        top_feats = [k for k, _ in sorted_imp[:3]]
        return f"(無 LLM，使用 fallback 建議)\n建議參考關鍵特徵：{', '.join(top_feats)}。\n可嘗試：\n- 增加等待時間 0.2~0.6 秒\n- 降低抬升速度 50~200 μm/s\n以改善樹脂回流。"

    try:
        reply = model.generate_content(prompt)
        return _safe_extract_text(reply)
    except Exception as e:
        return f"(LLM Error: {e})"


def get_low_risk_message():
    return "👍 此層風險低，目前參數設定穩定，無需額外調整。"
