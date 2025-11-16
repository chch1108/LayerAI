"""
Module: llm_recommender.py
功能: 針對每一層（不論風險高低）產生 3D 列印回流建議或結論。
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
    print("[WARNING] GOOGLE_API_KEY 未設定，LLM 將無法正常運作。請至環境變數或 Streamlit Secrets 設定。")
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
    """
    若正常文本不存在，嘗試 fallback，保證一定回傳文字。
    """
    try:
        if hasattr(reply, "text") and reply.text:
            return reply.text

        if hasattr(reply, "candidates") and reply.candidates:
            cand = reply.candidates[0]
            if hasattr(cand, "content") and hasattr(cand.content, "parts") and cand.content.parts:
                part = cand.content.parts[0]
                if hasattr(part, "text") and part.text:
                    return part.text
        # fallback
        return str(reply)
    except Exception:
        return "(LLM 無回覆內容)"


# -------------------------------
# 4. 主函式：輸出層級建議
# -------------------------------
def llm_layer_feedback(layer_info):
    """
    為單一層生成建議。
    """

    if model is None:
        return "(LLM 初始化失敗：請確認 GOOGLE_API_KEY)"

    layer = layer_info.get("layer")
    filename = layer_info.get("filename", "N/A")
    prob = float(layer_info.get("orig_prob", 0.0))
    sug_params = layer_info.get("suggested_params")
    sug_prob = layer_info.get("suggested_prob")

    # 風險分級
    if prob >= 0.50:
        risk_desc = "高風險（樹脂回流可能顯著不足）"
    elif prob >= 0.20:
        risk_desc = "中度風險（可能需要微量調整）"
    else:
        risk_desc = "低風險（回流正常）"

    prompt = f"""
你是一位具有豐富光固化 3D 列印（DLP / LCD / CLIP）經驗的製程工程師。
請根據下列資訊提供「繁體中文」的建議或結論（每層必須有內容）。

【層資訊】
- 層號：{layer}
- 圖片：{filename}

【模型預測】
- 原始失敗機率：{prob:.3f}
- 風險分類：{risk_desc}
"""

    if sug_params:
        prompt += f"""
【AI Auto-Tune 建議】
- wait_time：{sug_params.get('wait_time')}
- lift_height：{sug_params.get('lift_height')}
- lift_speed：{sug_params.get('lift_speed')}
- Auto-Tune 後預期失敗機率：{sug_prob:.3f}

請依據風險分類給出以下形式的回答：
1. 若高風險：提供 2~3 個具體調整建議與原因
2. 若中風險：給出微調方向與可能改善幅度
3. 若低風險：給出「結論」＋「是否需要細微優化」
"""
    else:
        prompt += """
此層無 Auto-Tune 參數建議（通常為低風險）。
請提供適合的結論，例如：
- 回流狀況正常、結構穩定
- 是否仍可透過減少等待時間或微幅調整提升效率
"""

    # 呼叫 Gemini
    try:
        reply = model.generate_content(prompt)
        text = _extract_text_safe(reply)

        if not text or text.strip() == "":
            return "(LLM 回覆為空，請稍後再試)"

        return text

    except Exception as e:
        return f"(LLM Error: {e})"
