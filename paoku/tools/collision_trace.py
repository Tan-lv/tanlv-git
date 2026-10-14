# -*- coding: utf-8 -*-
r"""碰撞内存追踪：把游戏状态高频打到屏幕上，用来分清两种碰撞。

用法:
    python tools\collision_trace.py                 # 一直跟到 Ctrl+C
    python tools\collision_trace.py --seconds 300   # 跟 5 分钟

要区分的两种碰撞（用户 2026-09-23）:
    ① 碰撞被追击：撞一下，警察追出来，但**角色没被抓**（还能继续跑）
    ② 碰撞被抓  ：撞到直接被抓住 -> Continue 弹窗
现在判卷只认 alive 1->0（= 被抓）当"撞车"，所以①根本没被记成撞车。
这个工具把内存里所有相关字段都打出来，看哪个字段能区分①②。

输出：
  * 任何字段变化就立刻打一行（带秒数和墙上时间）
  * 每 5 秒打一次心跳（证明还活着）
  * chasing 0->1 而 alive 还是 1 时特别标 ★★（那就是①）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from collector.config import Config              # noqa: E402
from collector.hookstate import open_client      # noqa: E402

FIELDS = ("inPlay", "alive", "paused", "chasing", "chaseDist",
          "rciN", "gmrN", "adOffer", "keyOffer")


def snap(st: dict):
    return tuple(st.get(k) for k in FIELDS)


def fmt(st: dict, changed, dt: float, t0: float) -> str:
    c = " ".join(f"{k}={st.get(k)}" for k in FIELDS if k != "chaseDist")
    return (f"[{time.time() - t0:7.2f}s {time.strftime('%H:%M:%S')}] {c} "
            f"dist={st.get('chaseDist')} chaser={st.get('chaser')}  "
            f"★变化: {','.join(changed)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="碰撞内存追踪")
    ap.add_argument("--seconds", type=float, default=0.0, help="跟多少秒（0=一直跟）")
    ap.add_argument("--log", default=os.path.join(HERE, "runs", "碰撞追踪.log"),
                    help="同时写一份日志文件")
    args = ap.parse_args()

    cfg = Config()
    print("=" * 100)
    print("碰撞内存追踪 —— 请依次做这两种碰撞，每种之间停 5 秒：")
    print("   ① 撞一下让警察追出来，但别被抓（继续跑）")
    print("   ② 撞到直接被抓住（Continue 弹窗）—— 弹窗上停 5 秒别动")
    print("=" * 100, flush=True)

    hook = open_client(cfg.hook_state_host, cfg.hook_state_port, wait_s=120.0)
    if hook is None:
        print("[FAIL] 拿不到内存状态，先看 runs\\state_server.log")
        return
    print("[就绪] 开始记录…\n", flush=True)

    logf = None
    if args.log:
        try:
            os.makedirs(os.path.dirname(args.log), exist_ok=True)
            logf = open(args.log, "a", encoding="utf-8")
            logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 新的追踪 =====\n")
            logf.flush()
        except Exception as e:
            print(f"[提示] 日志文件打不开（{e}），只打屏幕")

    def out(line: str) -> None:
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    prev = None
    t0 = time.time()
    last_hb = t0
    while True:
        if args.seconds and time.time() - t0 >= args.seconds:
            print(f"\n[停止] 到达 {args.seconds:.0f}s")
            break
        st = hook.read()
        if not st or not st.get("ok") or "inPlay" not in st:
            time.sleep(0.05)
            continue
        cur = snap(st)
        if cur != prev:
            changed = []
            if prev is None:
                changed = ["初始"]
            else:
                for i, k in enumerate(FIELDS):
                    if cur[i] != prev[i]:
                        changed.append(f"{k} {prev[i]}->{cur[i]}")
            line = fmt(st, changed, 0, t0)
            # ★ 重点：警察开始追、但 alive 还是 1 —— 这就是"碰撞被追击"
            if prev is not None and st.get("chasing") == 1 and st.get("alive") == 1 \
                    and cur[FIELDS.index("chasing")] != prev[FIELDS.index("chasing")]:
                line += "   <<<<< ★★ 碰撞被追击：alive 还是 1（现在判卷漏掉的正是这种）"
            out(line)
            prev = cur
        now = time.time()
        if now - last_hb >= 5.0:
            last_hb = now
            out(f"[{now - t0:7.2f}s {time.strftime('%H:%M:%S')}] 心跳 "
                f"inPlay={st.get('inPlay')} alive={st.get('alive')} chasing={st.get('chasing')}"
                f" dist={st.get('chaseDist')}")
        time.sleep(0.02)

    if logf:
        logf.close()
    try:
        hook.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
