# -*- coding: utf-8 -*-
r"""\u7ed5\u8fc7\u5c3a\u5bf8\u4fdd\u62a4\uff0c\u628a\u6a21\u62df\u5668\u7a97\u53e3\u7684**\u6574\u4e2a\u5ba2\u6237\u533a**\u6293\u4e0b\u6765\uff0c\u770b\u6a2a\u5c4f\u5230\u5e95\u957f\u4ec0\u4e48\u6837\u3002"""
import ctypes
from ctypes import wintypes
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from collector.config import Config
from collector.capture import WindowCapture, find_window

cfg = Config()
u32 = ctypes.windll.user32
u32.SetProcessDPIAware()
hwnd = find_window(cfg.window_keyword)
if not hwnd:
    raise SystemExit("\u6ca1\u627e\u5230\u7a97\u53e3")
rc = wintypes.RECT()
u32.GetClientRect(hwnd, ctypes.byref(rc))
cw, ch = rc.right - rc.left, rc.bottom - rc.top
print("\u7a97\u53e3\u5ba2\u6237\u533a = %dx%d   \u914d\u7f6e\u91cc\u7684 game_res=%s crop=%s"
      % (cw, ch, cfg.game_res, cfg.game_crop))

# \u628a game_res \u4f2a\u9020\u6210\u5ba2\u6237\u533a\u5927\u5c0f\uff0ccrop \u53d6\u5168\u5ba2\u6237\u533a -> \u7ed5\u8fc7\u5c3a\u5bf8\u4fdd\u62a4\uff0c\u62ff\u5230\u5b8c\u6574\u753b\u9762
cap = WindowCapture(cfg.window_keyword, (cw, ch), (0, 0, cw, ch), backend=cfg.backend)
full = cap.grab()
print("\u6293\u5230\u5b8c\u6574\u753b\u9762:", full.shape)
bgr = cv2.cvtColor(full, cv2.COLOR_RGB2BGR)
out = os.path.join(HERE, "runs", "_landscape_full.png")
cv2.imencode(".png", bgr)[1].tofile(out)
print("\u5df2\u5b58:", out)

# \u987a\u4fbf\u91cf\u4e00\u4e0b\u201c\u975e\u9ed1\u8272\u5185\u5bb9\u201d\u7684\u8fb9\u754c\uff08\u5206\u8fa8\u662f\u5426\u6709\u9ed1\u8fb9/\u5b57\u6bcd\u6846\uff09
g = bgr.astype(np.float32).mean(axis=2)
cols = np.where(g.max(axis=0) > 25)[0]
rows = np.where(g.max(axis=1) > 25)[0]
if len(cols) and len(rows):
    print("\u975e\u9ed1\u5185\u5bb9\u8303\u56f4: x %d..%d  y %d..%d  (%dx%d)"
          % (cols[0], cols[-1], rows[0], rows[-1], cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1))
cap.close()
