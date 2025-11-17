import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import io

FONT = None

def _load_font(size=16):
    global FONT
    try:
        from PIL import ImageFont
        FONT = ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        FONT = None

def overlay_issue_markers(pil_img, risk_score=None):
    """
    改良版 overlay：
    - 只在有問題的局部區塊畫 box
    - 判斷標準：大面積 (佔比), 長細比 (aspect ratio), 尖角 (approx poly vertices)
    """
    if FONT is None:
        _load_font(14)

    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)
    h, w = img_cv.shape[:2]

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # compute image area
    img_area = w * h
    issues = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:  # ignore tiny specks
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        # aspect ratio
        ar = cw / (ch + 1e-6)
        # polygon vertices
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        verts = len(approx)

        reason = []
        # large area threshold (relative)
        if area / img_area > 0.02:  # >2% of image area
            reason.append("大面積")
        # narrow long piece
        if ar > 3 or ar < 0.33:
            reason.append("細長/高長寬比")
        # many vertices -> complex/尖角
        if verts >= 6:
            reason.append("形狀複雜/尖角")
        if len(reason) > 0:
            issues.append((x, y, cw, ch, reason, area))

    # if no issues, and risk_score low: return small green badge instead
    pil_out = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_out)

    if not issues:
        # draw small corner badge
        badge_text = f"Risk: {risk_score:.2f}" if risk_score is not None else "OK"
        text_pos = (10, 10)
        draw.rectangle([text_pos, (text_pos[0]+140, text_pos[1]+26)], fill=(15,160,90,200))
        draw.text((text_pos[0]+6, text_pos[1]+4), badge_text, fill="white", font=FONT)
        return pil_out

    # draw boxes and labels for each issue
    for (x, y, cw, ch, reason, area) in issues:
        # choose color based on severity (area percentage)
        severity = min(1.0, (area / img_area) * 50)
        # interpolate red->orange
        color = (255, int(180*(1-severity)), 0)
        # draw rectangle
        draw.rectangle([x, y, x+cw, y+ch], outline=color, width=4)
        # label
        label = ",".join(reason)
        # shrink label if too long
        if len(label) > 30:
            label = label[:27] + "..."
        tx = x
        ty = max(0, y-18)
        draw.rectangle([tx, ty, tx+len(label)*7+8, ty+16], fill=(0,0,0,150))
        draw.text((tx+4, ty+1), label, fill="white", font=FONT)

    return pil_out
