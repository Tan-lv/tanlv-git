# -*- coding: utf-8 -*-
"""实时监听游戏状态：每次 inPlay/alive/paused/chasing 变化都打一行带时间戳。"""
import os, sys, time
sys.path.insert(0, r"D:\deeplearning\make_dataset\paoku")
from collector.hookstate import HookStateClient, ensure_state_server
from collector.config import Config

cfg = Config(); cfg.validate()
if not ensure_state_server(cfg.hook_state_host, cfg.hook_state_port):
    print("[FAIL] 状态服务起不来"); sys.exit(1)
h = HookStateClient(cfg.hook_state_host, cfg.hook_state_port)
if not h.wait_ready(10):
    print("[FAIL] 连上但读不到状态"); sys.exit(1)
print("=" * 72)
print("开始监听。你现在去玩游戏：正常跑 → 撞一次（警察追）→ 再撞到被逮捕")
print("*** 被逮捕后请停在 Continue 弹窗上别动 5 秒 ***")
print("=" * 72, flush=True)
prev = None
t0 = time.time()
while True:
    try:
        st = h.read()
    except Exception:
        st = None
    if st:
        cur = (st.get("inPlay"), st.get("alive"), st.get("paused"), st.get("chasing"))
        if cur != prev:
            print("%8.2fs   inPlay=%s  alive=%s  paused=%s  chasing=%s   dist=%s" % (
                  time.time() - t0, cur[0], cur[1], cur[2], cur[3], st.get("chaseDist")), flush=True)
            prev = cur
    time.sleep(0.05)