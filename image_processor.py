import zipfile, io
from PIL import Image
import numpy as np
import cv2
import pandas as pd

def extract_images_from_zip(zip_path, tmpdir=None):
    imgs = []
    filenames = []
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        names = sorted(names)
        for i, name in enumerate(names):
            try:
                data = z.read(name)
                img = Image.open(io.BytesIO(data)).convert("L")
                imgs.append(img)
                filenames.append(name)
            except Exception:
                continue
    return imgs, filenames

def compute_geometry(pil_img):
    arr = np.array(pil_img)
    # binary invert if necessary: assume background white, object darker
    _, binary = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY_INV)
    # find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0, 0.0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    hydraulic = float( (4.0 * area / perimeter) if perimeter > 0 else 0.0 )
    return area, perimeter, hydraulic

def batch_extract_features(imgs, filenames):
    results = []
    for idx, (img, fname) in enumerate(zip(imgs, filenames)):
        area, peri, hd = compute_geometry(img)
        # layer index from 1
        results.append({
            "layer": idx + 1,
            "filename": fname,
            "area": area,
            "perimeter": peri,
            "hydraulic_diameter": hd
        })
    return results

def suggest_parameters_for_layers_with_model(results_df, threshold=0.5, model_path=None):
    # If user provided a custom implementation elsewhere, this function is a safe fallback
    rows = []
    for _, r in results_df.iterrows():
        orig = float(r["prob"])
        layer = int(r["layer"])
        if orig >= threshold:
            sugg = {"wait_time": 0.8, "lift_height": 1500, "lift_speed": max(50, 700 - 50)}
            sugg_prob = max(0.0, orig - 0.12)
        else:
            sugg = {"wait_time": 0.5, "lift_height": 1500, "lift_speed": 700}
            sugg_prob = orig
        rows.append({"layer": layer, "filename": r["filename"], "orig_prob": orig, "suggested_params": sugg, "suggested_prob": sugg_prob})
    return pd.DataFrame(rows)
