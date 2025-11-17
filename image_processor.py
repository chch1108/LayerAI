import os, io, zipfile
import numpy as np
from PIL import Image
from skimage.measure import perimeter as sk_perimeter


# ---------------------------------------------------------
# 1️⃣ 解壓 ZIP 並讀取圖片
# ---------------------------------------------------------
def extract_images_from_zip(zip_path, tmpdir):
    imgs = []
    filenames = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in sorted(z.namelist()):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                data = z.read(name)
                try:
                    img = Image.open(io.BytesIO(data)).convert("L")
                    imgs.append(img)
                    filenames.append(name)
                except Exception:
                    # 忽略無法開啟的檔案
                    continue

    return imgs, filenames


# ---------------------------------------------------------
# 2️⃣ 計算幾何特徵（面積 / 周長 / 水力直徑）
# ---------------------------------------------------------
def compute_geometry(img):
    arr = np.array(img)
    binary = (arr < 128).astype(np.uint8)  # simple threshold

    area = np.sum(binary)

    try:
        peri = sk_perimeter(binary, neighbourhood=8)
    except Exception:
        peri = 0.0

    if peri > 0:
        hydraulic_d = 4 * area / peri
    else:
        hydraulic_d = 0.0

    return float(area), float(peri), float(hydraulic_d)


# ---------------------------------------------------------
# 3️⃣ 批次特徵擷取（給多層 ZIP 使用）
# ---------------------------------------------------------
def batch_extract_features(imgs, filenames):
    results = []
    for i, (img, fname) in enumerate(zip(imgs, filenames)):
        area, peri, hd = compute_geometry(img)
        results.append({
            "layer": i,
            "filename": fname,
            "area": area,
            "perimeter": peri,
            "hydraulic_diameter": hd
        })
    return results


# ---------------------------------------------------------
# 4️⃣ Auto-Tune（如果 app.py 呼叫但未定義 -> fallback）
# ---------------------------------------------------------
def suggest_parameters_for_layers_with_model(results_df, threshold=0.5, model_path=None):
    """
    這裡使用簡易 heuristic：增加等待時間、微調抬升速度
    app.py 若無需真實模型，這個即可提供基本功能
    """
    rows = []

    for _, r in results_df.iterrows():
        orig = float(r["prob"])
        layer = int(r["layer"])

        if orig >= threshold:
            sugg = {
                "wait_time": 0.8,
                "lift_speed":  max(100,  r["params"]["抬升速度(μm/s)"] - 80),
                "lift_height": r["params"]["抬升高度(μm)"]
            }
            sugg_prob = max(0.0, orig - 0.15)
        else:
            sugg = {
                "wait_time": r["params"]["等待時間(s)"],
                "lift_speed": r["params"]["抬升速度(μm/s)"],
                "lift_height": r["params"]["抬升高度(μm)"]
            }
            sugg_prob = orig

        rows.append({
            "layer": layer,
            "filename": r["filename"],
            "orig_prob": orig,
            "suggested_params": sugg,
            "suggested_prob": sugg_prob
        })

    import pandas as pd
    return pd.DataFrame(rows)
