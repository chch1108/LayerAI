"""
Module: llm_recommender.py
功能: 使用 Gemini-2.5-Flash 產生 Auto-Tune 後的 3D 列印回流層級建議（繁體中文）
"""

from typing import Dict, Any
from google.generativeai import GenerativeModel

MODEL_ID = "gemini-2.5-flash"

# 初始化模型物件 (全域)
model = GenerativeModel(MODEL_ID)


def llm_textual_suggestions(layer_info: Dict[str, Any]) -> str:
    """
    依據 Auto-Tune 結果，產生 LLM 修正建議
    layer_info 包含鍵值:
      - layer: 層號
      - filename: 圖片名稱
      - orig_prob: 原始失敗機率
      - suggested_params: {wait_time, lift_height, lift_speed}
      - suggested_prob: 建議參數後的預期失敗機率
    """

    try:
        layer = layer_info.get("layer")
        filename = layer_info.get("filename")
        orig_prob = layer_info.get("orig_prob")
        sug_prob = layer_info.get("suggested_prob")
        params = layer_info.get("suggested_params", {})

        wt = params.get("wait_time")
        lh = params.get("lift_height")
        ls = params.get("lift_speed")

        # === 建立 prompt ===
        prompt = f"""
你是 DLP / LCD / CLIP 光固化 3D 列印的製程優化工程師。
以下是某一層的回流預測與 AI 自動調參結果，請用「繁體中文」提供 2~3 項明確建議。

【層資訊】
- Layer 編號：{layer}
- 檔名：{filename}

【模型預測】
- 原始失敗機率：{orig_prob:.3f}
- 調整後失敗機率：{sug_prob:.3f}

【Auto-Tune 系統建議參數】
- wait_time（等待時間）：{wt} 秒
- lift_height（抬升高度）：{lh} mm
- lift_speed（抬升速度）：{ls} mm/s

---

請根據以上資訊，用工程師的角度生成「合理可執行」的改善說明，格式如下：

1. 建議項目（例如：增加等待時間）
   - 原因：說明為何此調整可改善樹脂回流（物理或流體原因）
   - 預期改善：描述失敗機率由 {orig_prob:.3f} 降至 {sug_prob:.3f} 的意義

2. 建議項目（例如：調整抬升高度）
   - 原因：
   - 預期改善：

3.（可選）附加建議（例如：結構最佳化 / 模型外形避免大面積連續曝光）

請基本保持簡潔明確，每項用 2–3 行說明。
"""

        # 呼叫 Gemini 生成內容
        reply = model.generate_content(prompt)
        return reply.text

    except Exception as e:
        return f"**LLM Error:** {e}"
