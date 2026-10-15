# -*- coding: utf-8 -*-
r"""IL2CPP 现场侦察 —— 找出「警察 / 追击 / 跌倒 / 撞车」相关的类与字段。

一次性逆向工具，不参与采集。用 wasd311 解释器跑：

    D:\Anaconda\envs\wasd311\python.exe -u tools\re_probe.py
    D:\Anaconda\envs\wasd311\python.exe -u tools\re_probe.py --fields PlayerController

前提：模拟器开着、游戏已经启动过（类才会被加载）、frida-server 在跑。
脚本会自己检查并拉起 frida-server。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ADB = r"D:\leidian\LDPlayer9\adb.exe"
FRIDA_HOST = "127.0.0.1:27042"
GAME_KEY = "subway"

# 猜类名用的关键词（大小写敏感，逐个在 metadata 字符串表里搜）
KEYWORDS = [
    "Police", "Cop", "Guard", "Chase", "Chased", "Inspector", "Arrest",
    "Pursu", "Security", "Wanted", "Patrol", "Officer", "Busted", "Caught",
    "Enforcer", "Siren", "Dog",
    "Stumble", "Trip", "Hit", "Crash", "Collide", "Collision",
    "Obstacle", "Barrier", "Barricade", "Train", "Dodge",
    "Death", "Dead", "GameOver", "Fail", "Escape", "Threat", "Danger",
    "PlayerController", "PlayerState", "CharacterState", "GameState",
]


def adb(args, timeout=25):
    try:
        p = subprocess.run([ADB] + args, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    except Exception as e:
        return -1, "", str(e)


def ensure_ready(port: int = 27042) -> bool:
    rc, out, _ = adb(["devices"])
    if "device" not in out:
        print("[FAIL] 没有可用设备，先启动模拟器"); return False
    rc, out, _ = adb(["shell", "ps -A"])
    if "frida-server" not in out:
        print("[前置] frida-server 没在跑，尝试用 root 拉起 ...")
        adb(["shell", "su -c 'nohup /data/local/tmp/frida-server -D > /data/local/tmp/fs.log 2>&1 &'"])
        time.sleep(3)
        rc, out, _ = adb(["shell", "ps -A"])
        if "frida-server" not in out:
            print("[FAIL] frida-server 起不来"); return False
    print("[前置] frida-server 运行中，转发端口", port)
    adb(["forward", f"tcp:{port}", f"tcp:{port}"])
    return True


def main():
    ap = argparse.ArgumentParser(description="IL2CPP 现场侦察")
    ap.add_argument("--fields", default=None, help="只枚举这个类的字段（默认 GameController）")
    ap.add_argument("--dump", type=int, default=0, help="dump 实例前 N 字节的 u32（用于对比变化）")
    ap.add_argument("--no-scan", action="store_true", help="跳过类名关键词搜索")
    args = ap.parse_args()

    try:
        import frida
    except Exception as e:
        print(f"[FAIL] 当前解释器 {sys.executable} 没有 frida: {e}")
        print(r"       请用 D:\Anaconda\envs\wasd311\python.exe 运行")
        return

    if not ensure_ready(27042):
        return

    dev = frida.get_device_manager().add_remote_device(FRIDA_HOST)
    pid = None
    for p in dev.enumerate_processes():
        if GAME_KEY in p.name.lower():
            pid = p.pid; break
    if pid is None:
        print("[FAIL] 没找到地铁跑酷进程"); return
    print(f"[目标] pid={pid}")

    session = dev.attach(pid)
    with open(os.path.join(HERE, "il2cpp_re.js"), encoding="utf-8") as f:
        script = session.create_script(f.read())
    script.on("message", lambda m, d: print("[agent]", m.get("payload", m), flush=True))
    script.load()
    ex = script.exports_sync

    t0 = time.time()
    info = ex.init()
    print(f"\n===== 1. 定位（{time.time()-t0:.1f}s）=====")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info.get("ok"):
        print("\n[FAIL] 定位失败。游戏可能从没进过主界面/关卡 —— 先进一次游戏再跑。")
        return

    print("\n===== 2. GameController 的运行时状态 =====")
    print(json.dumps(ex.read(), ensure_ascii=False))

    print("\n===== 3. 字段表 =====")
    cls = args.fields or "GameController"
    fld = ex.fields(cls)
    if not fld or fld.get("err") or not fld.get("fields"):
        print(f"  (枚举 {cls} 失败: {fld})")
    else:
        print(f"  {cls}: {fld['count']} 个字段")
        for f in fld["fields"]:
            print(f"    +0x{f['off']:<5x}  {f['name']}")
        if cls == "GameController":
            offs = [f["off"] for f in fld["fields"]]
            hit = [h for h in (0x10, 0x14, 0x18) if h in offs]
            print(f"  自检：已知的 3 个状态字段偏移 {['0x%x'%h for h in hit]} "
                  f"({'全部命中 —— 字段表可信' if len(hit)==3 else '只命中 %d/3，字段表可能不可信'%len(hit)})")

    if args.dump:
        print(f"\n===== 4. 实例 dump（前 {args.dump} 字节）=====")
        d = ex.dumpInst(args.dump)
        if isinstance(d, dict) and d.get("u32"):
            for i in range(0, len(d["u32"]), 8):
                row = d["u32"][i:i+8]
                print("  +0x%03x  %s" % (i*4, "  ".join("%08x" % v for v in row)))
        else:
            print("  ", d)

    if not args.no_scan:
        print("\n===== 5. 类名关键词搜索 =====")
        print(f"  关键词 {len(KEYWORDS)} 个，扫 metadata 字符串表 ...")
        t0 = time.time()
        res = ex.scanNames(KEYWORDS)
        print(f"  命中 {len(res)} 个候选名，用时 {time.time()-t0:.1f}s\n")
        live = [r for r in res if r["klass"]]
        dead = [r for r in res if not r["klass"]]
        print(f"  --- 有运行时类的（可按此名调 --fields）: {len(live)} ---")
        for r in live:
            print(f"    ✔ {r['name']}")
        print(f"\n  --- 只是字符串、没有加载的类: {len(dead)} ---")
        for r in dead:
            print(f"      {r['name']}")

    session.detach()


if __name__ == "__main__":
    main()
