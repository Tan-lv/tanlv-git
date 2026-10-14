# -*- coding: utf-8 -*-
"""记录你在模拟器窗口里的每一次鼠标点击，换算成比例坐标。
   用途：手动点一次复活按钮，就直接得到 --revive-xy 的准确值。"""
import ctypes
from ctypes import wintypes
import sys, time, threading
sys.path.insert(0, r"D:\deeplearning\make_dataset\paoku")
from collector.capture import WindowCapture
from collector.config import Config
from collector.hookstate import HookStateClient, ensure_state_server
from pynput import mouse

cfg = Config(); cfg.validate()
cap = WindowCapture(cfg.window_keyword, cfg.game_res, cfg.game_crop, backend=cfg.backend)
u32 = ctypes.windll.user32
r = wintypes.RECT(); u32.GetClientRect(cap.hwnd, ctypes.byref(r))
cw, ch = r.right - r.left, r.bottom - r.top
pt = wintypes.POINT(0, 0); u32.ClientToScreen(cap.hwnd, ctypes.byref(pt))
print("模拟器窗口 hwnd=%s  客户区 %dx%d  屏幕原点 (%d,%d)" % (cap.hwnd, cw, ch, pt.x, pt.y), flush=True)

hook = None
if ensure_state_server(cfg.hook_state_host, cfg.hook_state_port):
    hook = HookStateClient(cfg.hook_state_host, cfg.hook_state_port)
    hook.wait_ready(8)
t0 = time.time()

def on_click(x, y, button, pressed):
    if not pressed:
        return
    fx = (x - pt.x) / cw
    fy = (y - pt.y) / ch
    st = (hook.read() or {}) if hook else {}
    inside = 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0
    print("%7.2fs  屏幕(%4d,%4d)  比例(%.3f, %.3f) %s   inPlay=%s alive=%s paused=%s chase=%s" % (
          time.time() - t0, x, y, fx, fy, "[窗口内]" if inside else "[窗口外]",
          st.get("inPlay"), st.get("alive"), st.get("paused"), st.get("chasing")), flush=True)

print("=" * 78, flush=True)
print("开始记录鼠标点击。你去玩：撞到被逮捕 -> Continue 弹窗出现 -> **手动点一次复活按钮**", flush=True)
print("我会打印那一下的比例坐标，那就是 --revive-xy 的值。", flush=True)
print("=" * 78, flush=True)
with mouse.Listener(on_click=on_click) as L:
    L.join()