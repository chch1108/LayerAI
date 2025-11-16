import cv2
import numpy as np
from PIL import Image

def overlay_issue_markers(img: Image.Image, risk_score: float) -> Image.Image:
    """
    Level 1：在影像上畫出風險標記（黃色外框 + 文字）
    用於視覺化「需要修正的層」。
    """

    # 轉成可畫圖格式
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)

    h, w = img_cv.shape[:2]

    # 高風險 → 紅框
    # 中風險 → 黃框
    # 低風險 → 綠框（仍可顯示但表示無需調整）
    if risk_score >= 0.50:
        color = (0, 0, 255)     # Red
        label = "High Risk"
    elif risk_score >= 0.20:
        color = (0, 255, 255)   # Yellow
        label = "Medium Risk"
    else:
        color = (0, 255, 0)     # Green
        label = "Low Risk"

    # 畫框
    cv2.rectangle(img_cv, (5, 5), (w - 5, h - 5), color, 3)

    # 標記文字
    cv2.putText(img_cv, f"{label} ({risk_score:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    return Image.fromarray(img_cv)


def save_modified_images(filenames, images, probs, output_dir):
    """
    儲存所有修改後的圖片，供 ZIP 打包用
    """

    modified_files = []

    for fname, img, p in zip(filenames, images, probs):

        modified = overlay_issue_markers(img, p)
        save_path = f"{output_dir}/{fname}"
        modified.save(save_path)

        modified_files.append(save_path)

    return modified_files
