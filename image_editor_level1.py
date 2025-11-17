import numpy as np
import cv2
from PIL import Image

def flow_simulation_overlay(pil_img: Image.Image, alpha=0.55):
    """
    Create a pseudo-flow heatmap overlay using distance transform.
    pil_img: grayscale PIL Image
    alpha: overlay opacity (0-1)
    returns PIL Image (RGB)
    """
    arr = np.array(pil_img)
    # threshold - assume object darker than background
    _, mask = cv2.threshold(arr, 0, 255, cv2.THRESH_OTSU)
    # invert if needed so mask==255 is object
    # ensure mask is object==255
    # compute distance transform from edges inside the object
    # first get object mask
    obj = cv2.bitwise_not(mask) if np.mean(mask) > 127 else mask
    obj = cv2.medianBlur(obj, 5)
    dist = cv2.distanceTransform(obj, cv2.DIST_L2, 5)
    if dist.max() == 0:
        dist_norm = dist
    else:
        dist_norm = dist / dist.max()
    risk_map = 1.0 - dist_norm  # near boundary high risk
    heat = np.uint8(risk_map * 255)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base, 1.0 - alpha, heat_color, alpha, 0)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
