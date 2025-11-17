"""
image_processor.py
功能：處理多層切片 ZIP、提取幾何特徵（面積/周長/水力直徑）
"""

import zipfile
import cv2
import numpy as np
import os
from PIL import Image
import io


# ============================================================
# 1. 從 ZIP 中讀取每一層圖片
# ============================================================
def extract_images_from_zip(zip_path, extract_dir):
    """
    回傳：
    imgs: [PIL.Image]
    filenames: ["layer001.png" ...]
    """
    imgs = []
    filenames = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in sorted(z.namelist()):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                data = z.read(name)
                img = Image.open(io.BytesIO(data)).convert("L")
                imgs.append(img)
                filenames.append(name)

    return imgs, filenames


# ============================================================
# 2. 計算幾何特徵（面積 / 周長 / 水力直徑）
# ============================================================
def extract_geometric_features_from_image(pil_img):
    """
    與單層版本 extract_geometric_features() 同邏輯
    """

    # PIL → numpy
    img = np.array(pil_img)

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 找輪廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)  # 面積
    perimeter = cv2.arcLength(cnt, True)  # 周長

    # 水力直徑 D_h = 4A/P
    if perimeter > 0:
        hydraulic_diameter = 4 * area / perimeter
    else:
        hydraulic_diameter = 0

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "hydraulic_diameter": float(hydraulic_diameter)
    }


# ============================================================
# 3. 批次特徵提取（多層）
# ============================================================
def batch_extract_features(imgs, filenames):
    """
    回傳：
    [
      {"layer":1, "filename":"xxx.png", "area":..., "perimeter":..., "hydraulic_diameter":...},
      ...
    ]
    """
    feature_list = []

    for i, (img, fname) in enumerate(zip(imgs, filenames)):
        feats = extract_geometric_features_from_image(img)

        if feats is None:
            feats = {"area": 0, "perimeter": 0, "hydraulic_diameter": 0}

        feature_list.append({
            "layer": i + 1,
            "filename": fname,
            **feats
        })

    return feature_list
