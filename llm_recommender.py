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
    # Prepare prompt
    sorted_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    top_imp = sorted_imp[:5]
    params_str = "\n".join([f"- {k}: {v}" for k, v in input_params.items()])
    imp_str = "\n".join([f"- {k}: {v:.3f}" for k, v in top_imp])

    prompt = f"""你是光固化 3D 列印製程專家。請以繁體中文根據以下資訊提供 2 項可執行的優化建議（若風險高請直接給出參數值）。
列印參數：
{params_str}

最重要影響參數：
{imp_str}

請用格式：
1. 建議項目：
 - 目前數值：...
 - 建議數值：...
 - 原因：...
2. 建議項目：...
"""

    # If model not available, return heuristic text
    if model is None:
        # simple heuristic fallback
        top_feats = [k for k,_ in sorted_imp[:3]]
        advice = f"(No LLM) 建議參考 top features: {', '.join(top_feats)}。建議微幅增加等待時間 0.2-0.6s，或降低抬升速率 50-200 μm/s。"
        return advice

    try:
        reply = model.generate_content(prompt)
        return _safe_extract_text(reply)
    except Exception as e:
        return f"(LLM Error: {e})"
