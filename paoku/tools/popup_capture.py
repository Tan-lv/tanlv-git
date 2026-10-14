# -*- coding: utf-8 -*-
"""监听被捕瞬间，自动抓全屏图 —— 用来量复活按钮的准确坐标。"""
import os, sys, time, subprocess
sys.path.insert(0, r"D:\deeplearning\make_dataset\paoku")
from collector.hookstate import HookStateClient, ensure_state_server
from collector.config import Config
ADB = r"D:\leidian\LDPlayer9\adb.exe"
OUT = r"D:\deeplearning\make_dataset\paoku\tools\_popup"
os.makedirs(OUT, exist_ok=True)
cfg = Config(); cfg.validate()
if not ensure_state_server(cfg.hook_state_host, cfg.hook_state_port):
    print("[FAIL] 状态服务起不来"); sys.exit(1)
h = HookStateClient(cfg.hook_state_host, cfg.hook_state_port)
if not h.wait_ready(10):
    print("[FAIL] 读不到状态"); sys.exit(1)
print("=" * 70, flush=True)
print("监听中。你去玩：正常跑 -> 撞 -> 被逮捕。", flush=True)
print("*** 被逮捕后停在弹窗上别动 6 秒 ***", flush=True)
print("=" * 70, flush=True)
prev = 1
idx = 0
t0 = time.time()
while True:
    st = h.read()
    if st:
        ip = int(st.get("inPlay", 1))
        if prev == 1 and ip == 0:
            idx += 1
            print("  [%.1fs] 被捕！连拍 #%d" % (time.time()-t0, idx), flush=True)
            for k in range(4):
                f = os.path.join(OUT, "popup_%d_%d.png" % (idx, k))
                subprocess.run([ADB, "shell", "screencap", "-p", "/sdcard/_p.png"],
                               capture_output=True, timeout=15)
                subprocess.run([ADB, "pull", "/sdcard/_p.png", f], capture_output=True, timeout=15)
                st2 = h.read() or {}
                print("     拍 %d: inPlay=%s alive=%s paused=%s chase=%s" % (
                      k, st2.get("inPlay"), st2.get("alive"), st2.get("paused"),
                      st2.get("chasing")), flush=True)
                time.sleep(0.9)
            subprocess.run([ADB, "shell", "rm", "/sdcard/_p.png"], capture_output=True)
            print("     存到", OUT, flush=True)
        prev = ip
    time.sleep(0.05)