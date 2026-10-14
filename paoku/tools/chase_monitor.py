# -*- coding: utf-8 -*-
r"""监听游戏，抓「警察出现 / 撞车」事件（并在启动时顺带搜警察类名）。

houdini 环境下 frida 连续 attach 两三次就会僵死，所以这里全程只 attach 一次，
失败时自动重启设备端 frida-server 再重试。

用法（wasd311 解释器）：
    python tools\chase_monitor.py --out tools\_chase_log.txt --hz 40
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import re_probe as R
import frida

LOG = None
T0 = time.time()


def emit(line: str) -> None:
    out = "%.3f\t%s" % (time.time() - T0, line)
    print(out, flush=True)
    if LOG:
        LOG.write(out + "\n")
        LOG.flush()


def restart_frida_server() -> None:
    def sh(cmd):
        subprocess.run([R.ADB, "shell", cmd], capture_output=True, timeout=30)
    emit("重启 frida-server ...")
    sh("su -c 'killall frida-server'")
    time.sleep(2)
    sh("su -c 'nohup /data/local/tmp/frida-server -D > /data/local/tmp/fs.log 2>&1 &'")
    time.sleep(4)
    subprocess.run([R.ADB, "forward", "--remove-all"], capture_output=True)
    subprocess.run([R.ADB, "forward", "tcp:27042", "tcp:27042"], capture_output=True)


def attach_with_retry(js: str, tries: int = 3):
    for i in range(tries):
        try:
            dev = frida.get_device_manager().add_remote_device(R.FRIDA_HOST)
            procs = dev.enumerate_processes()
            pid = [p.pid for p in procs if R.GAME_KEY in p.name.lower()]
            if not pid:
                emit("没找到游戏进程"); return None, None, None
            session = dev.attach(pid[0])
            script = session.create_script(js)
            script.load()
            return session, script, pid[0]
        except Exception as e:
            emit("注入失败 %d/%d: %s" % (i + 1, tries, str(e)[:70]))
            if i < tries - 1:
                restart_frida_server()
    return None, None, None


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "_chase_log.txt"))
    ap.add_argument("--hz", type=float, default=40.0)
    ap.add_argument("--seconds", type=int, default=0)
    ap.add_argument("--no-names", action="store_true", help="跳过类名搜索")
    args = ap.parse_args()
    LOG = open(args.out, "w", encoding="utf-8")

    if not R.ensure_ready(27042):
        return
    js = open(os.path.join(HERE, "chase_monitor.js"), encoding="utf-8").read()
    session, script, pid = attach_with_retry(js)
    if script is None:
        emit("[FAIL] frida 起不来")
        return
    emit("attach pid=%d" % pid)

    flipped = set()

    def on_message(m, data):
        if m.get("type") != "send":
            emit("agent: %s" % m); return
        p = m["payload"]; k = p.get("type")
        if k == "init":
            emit("INIT  GlobalBool %d 个  inst=%s  state=%s"
                 % (p["gbCount"], p["inst"], p["state"]))
        elif k == "flip":
            flipped.add(p["gb"])
            emit("FLIP  t=%-7d %s  %d -> %d" % (p["t"], p["gb"], p["from"], p["to"]))
        elif k == "gc":
            emit("GC    t=%-7d %s  %s -> %s" % (p["t"], p["off"], p["from"], p["to"]))
        elif k == "watch":
            emit("WATCH " + ", ".join("%s=%d" % (x["name"], x["n"]) for x in p["items"]))
        elif k == "newobj":
            emit("NEWOBJ t=%-7d %s" % (p["t"], p["addr"]))
        elif k == "delobj":
            emit("DELOBJ t=%-7d %s" % (p["t"], p["addr"]))
        elif k == "objdump":
            emit("OBJDUMP t=%-7d %s  %s" % (p["t"], p["addr"], " ".join(p["words"])))
        elif k == "obj":
            emit("OBJ   t=%-7d %s +%s  %s -> %s" % (p["t"], p["addr"], p["off"][2:], p["from"], p["to"]))
        elif k == "count":
            emit("COUNT t=%-7d %-24s %d -> %d" % (p["t"], p["name"], p["from"], p["to"]))
        elif k == "names":
            if p.get("err"):
                emit("names 搜索失败: %s" % p["err"]); return
            live = [x for x in p["items"] if x["klass"]]
            emit("=== metadata 里匹配到的类名：共 %d，其中已加载 %d ===" % (len(p["items"]), len(live)))
            for x in live:
                emit("   LIVE  %s" % x["name"])
            for x in [y for y in p["items"] if not y["klass"]]:
                emit("   str   %s" % x["name"])

    script.on("message", on_message)
    kws = [] if args.no_names else R.KEYWORDS
    info = script.exports_sync.start(args.hz, kws)
    emit("start -> %s" % info)
    if not info.get("ok"):
        return

    t0 = time.time()
    try:
        while True:
            time.sleep(1.0)
            if args.seconds and time.time() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        emit("中断")

    fin = script.exports_sync.stop()
    emit("stop -> %s" % fin)
    if flipped:
        emit("=== 反查翻转标志位所属的类 ===")
        for o in script.exports_sync.owners(sorted(flipped)):
            emit("  %s -> %s" % (o["gb"], o["owners"] if o["owners"] else "(没有持有者)"))
    session.detach()
    if LOG:
        LOG.close()


if __name__ == "__main__":
    main()
