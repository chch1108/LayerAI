import numpy as np
import cv2
from PIL import Image

def flow_simulation_overlay(img: Image.Image, alpha=0.55):
    """
    產生回流難易度 Heatmap Overlay（Pseudo Flow Simulation）
    
    img: PIL Image (grayscale)
    alpha: overlay 透明度（0~1）
    """

    # convert to array
    arr = np.array(img)

    # 二值化：假設白色為模型區域
    # 若相反，可改成 THRESH_BINARY_INV
    _, mask = cv2.threshold(arr, 0, 255, cv2.THRESH_OTSU)

    # 去除噪音（很重要）
    mask = cv2.medianBlur(mask, 5)

    # 計算距離變換（distance transform）
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # normalize → [0,1]
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 回流難易度 = 1 - dist_norm
    # 越接近邊界 → dist 小 → risk 高
    risk_map = 1.0 - dist_norm

    # 轉 0~255
    heat = np.uint8(risk_map * 255)

    # 套上顏色圖（COLORMAP_JET 非常有 CFD 感）
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # 原始圖像轉成 BGR
    base = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    # 混合
    blended = cv2.addWeighted(base, 1 - alpha, heat_color, alpha, 0)

    # 轉回 PIL
    return Image.fromarray(blended)
