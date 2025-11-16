"""
Module: llm_recommender.py
功能: 讓所有層都能獲得意見（不論風險高低）
"""

from typing import Dict, Any
from google.generativeai import GenerativeModel

MODEL_ID = "gemini-2.5-flash"
model = GenerativeModel(MODEL_ID)

def llm_layer_feedback(layer_info: Dict[str, Any]) -> str:
    """
    根據層的預測結果提供建議或總結。
    layer_info 包含:
    {
        "layer": int,
        "filename": str,
        "orig_prob": float,
        "suggested_params": {...} or None,
        "suggested_prob": float or None
    }
    """

    layer = layer_info.get("layer")
    filename = layer_info.get("filename")
    prob = float(layer_info.get("orig_prob", 0.0))

    sug_params = layer_info.get("suggested_params", None)
    sug_prob = layer_info.get("suggested_prob", None)

    # 根據風險分類（你可以自行調整門檻）
    if prob >= 0.50:
        risk_desc = "高風險（樹脂回流不完全可能性大）"
    elif prob >= 0.20:
        risk_desc = "中度風險（可能需部分微調）"
    else:
        risk_desc = "低風險（回流基本正常）"

    # --- Prompt 設計 ---
    prompt = f"""
你是熟悉 DLP/LCD/CLIP 光固化列印的製程工程師。
請根據以下資訊，提供「繁體中文」層級建議或結論。

【層資訊】
- 層號：{layer}
- 檔名：{filename}

【模型預測】
- 原始失敗機率：{prob:.3f}
- 風險評估：{risk_desc}

"""

    if sug_params and sug_prob is not None:
        prompt += f"""
【AI Auto-Tune 建議參數】
- wait_time：{sug_params.get('wait_time')}
- lift_height：{sug_params.get('lift_height')}
- lift_speed：{sug_params.get('lift_speed')}
- 調整後預期失敗機率：{sug_prob:.3f}

請依據風險程度，給出合適的建議或結論：
1. 如果風險高：請提供具體可執行的參數調整建議。
2. 如果風險中等：請給出可微調、可改善的方向。
3. 如果風險低：請給出維持良好狀態的結論，並可附帶「是否還需要優化」的簡短說明。
"""
    else:
        prompt += """
此層沒有 Auto-Tune 資料。
請依據預測機率給出相對應的建議或結論。
"""

    try:
        reply = model.generate_content(prompt)
        return reply.text
    except Exception as e:
        return f"[LLM Error] {e}"
