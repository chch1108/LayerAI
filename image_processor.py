import zipfile, io, os, re, tempfile
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import pandas as pd
import random
import math
import cv2
import plotly.graph_objs as go

# 若要用真實 Keras model，嘗試載入
def try_load_keras_model(model_path):
    if not model_path:
        return None
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        return model
    except Exception as e:
        print("載入模型失敗，改用 mock。Err:", e)
        return None

def extract_images_from_zip(zip_path: str, extract_to: str):
    """
    從使用者上傳的 ZIP 中讀取所有影像檔 (.png/.jpg/.jpeg/.bmp)
    自動忽略非影像檔，避免 UnidentifiedImageError
    """

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}

    images = []
    filenames = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.namelist():

            ext = os.path.splitext(file)[1].lower()

            # --- 跳過非圖片 ---
            if ext not in valid_ext:
                continue

            try:
                data = z.read(file)
                img = Image.open(io.BytesIO(data)).convert("L")  # 灰階讀取
                images.append(img)
                filenames.append(file)
            except Exception as e:
                print(f"[WARNING] 無法解析影像：{file}。錯誤：{e}")
                continue

    return images, filenames

def preprocess_image_for_model(pil_img, target_size=(128,128)):
    img = pil_img.resize(target_size)
    arr = np.array(img).astype(np.float32)/255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr

def mock_predict(arr):
    # 簡單模擬：以圖面面積占比（白色像素比例）作為風險 proxy
    gray = (arr[...,0] * 255).astype(np.uint8)
    white_ratio = (gray > 30).sum() / gray.size
    base = 0.15 + white_ratio * 0.7
    noise = random.uniform(-0.12, 0.12)
    prob = float(np.clip(base + noise, 0.0, 0.99))
    return prob

def single_predict(pil_img, model=None):
    arr = preprocess_image_for_model(pil_img)
    if model is None:
        return mock_predict(arr)
    # 嘗試處理不同 model input shape
    try:
        x = np.expand_dims(arr, axis=0)
        y = model.predict(x)
        # 支援 model 回傳 [prob] 或 [ [prob] ]
        prob = float(y.ravel()[0])
        if prob < 0 or prob > 1:
            prob = float(1/(1+math.exp(-prob)))  # sigmoid fallback
        return prob
    except Exception as e:
        print("model predict error, fallback to mock:", e)
        return mock_predict(arr)

def batch_predict_layers(pil_images, filenames, model_path=""):
    model = try_load_keras_model(model_path)
    rows = []
    for idx, (im, fname) in enumerate(zip(pil_images, filenames)):
        prob = single_predict(im, model=model)
        pred = 1 if prob >= 0.5 else 0
        rows.append({'layer': idx, 'filename': fname, 'prob': round(prob,4), 'pred': int(pred)})
    df = pd.DataFrame(rows)
    meta = {'model_loaded': model is not None}
    return df, meta

# Plotly heatmap + risk curve
def make_plotly_heatmap_and_curve(probs):
    n = len(probs)
    side = int(np.ceil(np.sqrt(n)))
    mat = np.full((side, side), np.nan)
    matflat = mat.flatten()
    matflat[:n] = probs
    mat = matflat.reshape(side, side)

    heat = go.Figure(data=go.Heatmap(z=mat, colorbar=dict(title="Failure Prob")))
    heat.update_layout(title="Layer Failure Probability Heatmap", height=600)

    curve = go.Figure()
    curve.add_trace(go.Scatter(x=list(range(n)), y=probs, mode='lines+markers', name='Failure Prob'))
    curve.update_layout(title="Layer Failure Probability Curve", xaxis_title="Layer Index", yaxis_title="Failure Probability", height=300)
    return heat, curve

# Auto-tune using model if available; else heuristic simulation
def suggest_parameters_for_layers_with_model(results_df, threshold=0.5, model_path=""):
    model = try_load_keras_model(model_path)
    suggestions = []
    # Candidate grid (示範參數)：wait_time (s), lift_height (mm), lift_speed (mm/s)
    candidates = []
    for wt in [0.8, 1.0, 1.2, 1.4, 1.6]:
        for lh in [0.8, 1.0, 1.2]:
            for ls in [8, 10, 12]:
                candidates.append({'wait_time': wt, 'lift_height': lh, 'lift_speed': ls})

    for _, row in results_df.iterrows():
        layer = int(row['layer'])
        fname = row['filename']
        orig_prob = float(row['prob'])
        best = None
        best_prob = orig_prob
        if orig_prob >= threshold:
            # 用候選參數逐一評估（若 model loaded：理想是把參數輸入模型，但多數模型只看影像；
            # 所以若 model 支援參數輸入，應做適配；此範例：若 model 可用，我們仍用 image-only predict）
            for cand in candidates:
                # 如果真模型存在：我們 *仍然* 呼叫模型只用 image（假設模型學過各參數影響）
                # 若無模型，使用啟發式模擬：每 +0.2s wait_time 降 0.08；每 -? lift_height 調整等
                simulated_prob = None
                if model is not None:
                    # 這裡直接用 image 預測（建議：若你的模型接受附加參數，請修改此段把參數傳給模型）
                    simulated_prob = single_predict(ImageOps.autocontrast(Image.new('L',(1,1))), model=None)  # fallback (placeholder)
                    # 由於我們無法確定你的模型 API，實務上你應修改 single_predict 以接受參數。
                    simulated_prob = max(0.0, orig_prob - 0.05)  # conservative improvement
                else:
                    # 啟發式：等待時間每增加 0.2s，failure prob 降 0.08；lift_height 小幅改善 0.02；提升速度影響小
                    wt_gain = (cand['wait_time'] - 1.0) / 0.2
                    simulated_prob = max(0.0, orig_prob - 0.08 * wt_gain - 0.02 * max(0, (1.0 - cand['lift_height'])))
                if simulated_prob < best_prob:
                    best_prob = simulated_prob
                    best = cand.copy()
            if best is None:
                best = {'wait_time':1.0, 'lift_height':1.0, 'lift_speed':10}
        else:
            best = {'wait_time':1.0, 'lift_height':1.0, 'lift_speed':10}
            best_prob = orig_prob
        suggestions.append({
            'layer': layer,
            'filename': fname,
            'orig_prob': orig_prob,
            'suggested_params': best,
            'suggested_prob': round(best_prob,4)
        })
    return pd.DataFrame(suggestions)

# 影像自動改良：對高風險層做簡單 erosion 與加入支撐線（示範）
def modify_image_for_layer(pil_img, intensity=1):
    # 轉成 numpy binary, 做侵蝕（縮小大面積）、再轉回 PIL
    arr = np.array(pil_img)
    # 門檻二值化（簡單）
    thresh = 30
    bw = (arr > thresh).astype('uint8') * 255
    # kernel size depends on intensity
    ksize = max(1, int(1 + intensity))
    kernel = np.ones((ksize, ksize), np.uint8)
    eroded = cv2.erode(bw, kernel, iterations=1)
    # optional: add vertical supports every 40 pixels
    h, w = eroded.shape
    draw = Image.fromarray(eroded).convert('L')
    d = ImageDraw.Draw(draw)
    for x in range(20, w, 40):
        d.line([(x,0),(x,h)], fill=180, width=2)
    return draw

def generate_modified_slices_zip(pil_images, filenames, results_df, threshold=0.5):
    # 為高風險層生成 modified image，其他層回傳原始
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode='w') as zf:
        for im, fname, row in zip(pil_images, filenames, results_df.to_dict('records')):
            layer_prob = row['prob']
            if layer_prob >= threshold:
                mod = modify_image_for_layer(im, intensity=2)
                buf = io.BytesIO()
                mod.save(buf, format='PNG')
                zf.writestr(f"modified_{fname}", buf.getvalue())
            else:
                buf = io.BytesIO()
                im.save(buf, format='PNG')
                zf.writestr(fname, buf.getvalue())
    mem_zip.seek(0)
    return mem_zip.getvalue()

# Estimate total print time and improvements
def estimate_time_and_effects(results_df, suggestion_df, base_params=None):
    # base per-layer params（範例值，單位秒或 mm）
    if base_params is None:
        base_params = {'exposure': 5.0, 'wait_time': 1.0, 'lift_height':1.0, 'lift_speed':10.0}
    rows = []
    total_orig_time = 0.0
    total_new_time = 0.0
    orig_success = 0.0
    new_success = 0.0
    n = len(results_df)
    for _, r in results_df.iterrows():
        layer = int(r['layer'])
        orig_prob = float(r['prob'])
        orig_success += (1 - orig_prob)
        # find suggested row
        srow = suggestion_df[suggestion_df['layer']==layer]
        if srow.shape[0] > 0:
            suggested = srow.iloc[0]['suggested_params']
            suggested_prob = float(srow.iloc[0]['suggested_prob'])
        else:
            suggested = base_params
            suggested_prob = orig_prob
        # time calc: exposure + wait_time + lift_time (lift_height / lift_speed)
        orig_t = base_params['exposure'] + base_params['wait_time'] + (base_params['lift_height']/base_params['lift_speed'])
        new_t = base_params['exposure'] + float(suggested.get('wait_time', base_params['wait_time'])) + (float(suggested.get('lift_height', base_params['lift_height']))/float(suggested.get('lift_speed', base_params['lift_speed'])))
        total_orig_time += orig_t
        total_new_time += new_t
        new_success += (1 - suggested_prob)
        rows.append({
            'layer': layer,
            'orig_prob': orig_prob,
            'suggested_prob': suggested_prob,
            'orig_time_s': round(orig_t,3),
            'new_time_s': round(new_t,3),
            'time_diff_s': round(orig_t - new_t,3)
        })
    df = pd.DataFrame(rows)
    summary = {
        'total_layers': n,
        'total_orig_time_s': round(total_orig_time,3),
        'total_new_time_s': round(total_new_time,3),
        'time_saved_s': round(total_orig_time - total_new_time,3),
        'orig_expected_success_rate': round((orig_success / n)*100,3),
        'new_expected_success_rate': round((new_success / n)*100,3),
        'expected_success_rate_delta_pct': round(((new_success - orig_success)/n)*100,3)
    }
    summary_df = pd.DataFrame([summary])
    return pd.concat([summary_df, df.head(0)], axis=1).fillna('') if False else df  # return detailed per-layer for UI (caller can compute summary)
