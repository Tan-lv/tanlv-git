# -*- coding: utf-8 -*-
r"""\u77e9\u9635\u5bf9\u6bd4\uff1a\u6a21\u578b\u8bad\u7ec3\u65f6\u5403\u7684\u753b\u9762 vs \u6a2a\u5c4f\u4e0b\u4e24\u79cd\u53d6\u666f\u65b9\u5f0f\u3002"""
import os
import numpy as np
import cv2

HERE = r"D:\deeplearning\make_dataset\paoku"
FW, FH = 240, 293


def load(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def save(name, bgr):
    p = os.path.join(HERE, "runs", name)
    cv2.imencode(".png", bgr)[1].tofile(p)
    print("  ->", name, bgr.shape)


def label(img, text, h=24):
    bar = np.full((h, img.shape[1], 3), 28, np.uint8)
    cv2.putText(bar, text, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


# ---- 1) \u8bad\u7ec3\u53e3\u5f84\uff1a\u76f4\u63a5\u7528\u5f53\u5e74\u81ea\u68c0\u7559\u4e0b\u7684\u771f\u5b9e\u88c1\u526a\u6846 720x880 ----
src_crop = load(os.path.join(HERE, "dataset", "_selftest", "crop.png"))
print("\u81ea\u68c0\u88c1\u526a\u6846:", src_crop.shape)
panel_train = cv2.resize(src_crop, (FW, FH), interpolation=cv2.INTER_AREA)

# ---- 2) \u6a2a\u5c4f\uff1a\u7528\u7528\u6237\u622a\u56fe\uff081322x755\uff09 ----
land = load(r"C:\Users\Tanlv\.dsh\attachments\v1\objects\00\00a1df67e3625196abe4b729ff8549e7d3e62d726e43f633b7fc0a92eab4cd20")
print("\u6a2a\u5c4f\u6e90:", land.shape)
H, W = land.shape[:2]
top, right = 35, 54
android = land[top:H, 0:W - right]
ah, aw = android.shape[:2]
print("\u5b89\u5353\u753b\u9762:", (aw, ah))

# 2a) \u4fdd\u6301 240:293 \u6bd4\u4f8b\u7684\u7ad6\u7a97\uff08\u6ee1\u9ad8\uff09\uff0c\u6c34\u5e73\u5bf9\u51c6\u89d2\u8272
cx = int(aw * 0.59)
pw = int(round(FW / FH * ah))
x0 = max(0, min(aw - pw, cx - pw // 2))
cropA = android[:, x0:x0 + pw]
panelA = cv2.resize(cropA, (FW, FH), interpolation=cv2.INTER_AREA)
print("A \u540c\u6bd4\u4f8b\u7ad6\u7a97: %dx%d  x0=%d" % (pw, ah, x0))

# 2b) \u6309\u7ad6\u5c4f\u53d6\u666f\u53cd\u63a8\uff1a\u7ad6\u5c4f\u88c1\u526a\u5360\u5c4f\u5e55\u9ad8\u5ea6 200/1280~1080/1280 = 15.6%~84.4%\uff0c\u5168\u5bbd
y0, y1 = int(ah * 0.156), int(ah * 0.844)
cropB = android[y0:y1, :]
panelB = cv2.resize(cropB, (FW, FH), interpolation=cv2.INTER_AREA)
print("B \u7b49\u6548\u6a2a\u6761: %dx%d -> \u88ab\u62c9\u4f38" % (cropB.shape[1], cropB.shape[0]))

# ---- 3) \u6a2a\u5c4f\u6e90\u56fe + \u6807\u51fa\u4e24\u4e2a\u5019\u9009\u6846 ----
anno = android.copy()
cv2.rectangle(anno, (x0, 0), (x0 + pw, ah - 1), (0, 255, 0), 4)
cv2.rectangle(anno, (0, y0), (aw - 1, y1), (0, 165, 255), 4)
cv2.putText(anno, "A portrait-window", (x0 + 6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(anno, "B old-band (squeezed)", (8, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
sc = FH / float(ah)
anno_s = cv2.resize(anno, (int(aw * sc), FH), interpolation=cv2.INTER_AREA)

gap = np.full((FH + 24, 8, 3), 60, np.uint8)
parts = [label(panel_train, "1 TRAINING crop 720x880 -> 240x293"), gap,
         label(panelA, "2 LAND portrait-window %dx%d" % (pw, ah)), gap,
         label(panelB, "3 LAND old-band %dx%d squeezed" % (cropB.shape[1], cropB.shape[0])), gap,
         label(anno_s, "4 landscape source + candidate crops")]
out = np.concatenate(parts, axis=1)
save("_framing_cmp.png", out)
