# -*- coding: utf-8 -*-
r"""主机端：把 tools/frida_probe.js 注入到运行中的地铁跑酷进程，打印 x86 侧 hook 计数。

用法（需要雷电模拟器已 root、frida-server 已启动）:
    python tools/hook_probe.py                 # 自动找 subwaysurf 进程
    python tools/hook_probe.py --pid 2945
    python tools/hook_probe.py --raw           # 打印全部原始事件

前置：adb server 必须常驻（DSH/沙箱会回收进程树）:
    D:\leidian\LDPlayer9\adb.exe -L tcp:5037 nodaemon server      # 后台 job
    D:\leidian\LDPlayer9\adb.exe forward tcp:27042 tcp:27042
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ADB = r"D:\leidian\LDPlayer9\adb.exe"
FRIDA_HOST = "127.0.0.1:27042"

# frida 17.x 要求 Python 3.11+，而项目的 wasd_local 是 3.10。
# 所以如果当前解释器导入不了 frida，就自动换成下面这几个能用的解释器重跑。
FALLBACK_PYTHONS = [
    r"D:\Anaconda\envs\wasd311\python.exe",   # 主线：Python 3.11 + torch cu124 + frida
    r"D:\pythonx\python.exe",                  # 备用
]


def reexec_with_working_python() -> bool:
    """当前解释器没有可用的 frida 时，换一个能用的解释器重新执行本脚本。"""
    me = os.path.abspath(sys.executable)
    for exe in FALLBACK_PYTHONS:
        if not os.path.isfile(exe) or os.path.abspath(exe) == me:
            continue
        try:
            chk = subprocess.run([exe, "-c", "import frida; print(frida.__version__)"],
                                 capture_output=True, timeout=60)
        except Exception:
            continue
        if chk.returncode == 0:
            ver = chk.stdout.decode(errors="replace").strip()
            print(f"[切换] 当前解释器 {me} 没有可用的 frida，改用 {exe} (frida {ver})")
            sys.stdout.flush()
            return _spawn_and_forward(exe)
    return False


def _spawn_and_forward(exe: str) -> bool:
    """用能用的解释器重新跑本脚本，并把它的输出转发出来。

    不能用 os.execv：Windows 上 execv 会换掉进程 ID，PyCharm 会以为原进程
    已退出而回收控制台 —— 表现就是"闪一下就没了"。改成子进程 + 转发输出，
    父进程一直活着，IDE 里就不会闪退。
    """
    try:
        proc = subprocess.Popen(
            [exe, "-u", os.path.abspath(__file__)] + sys.argv[1:],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
    except Exception as e:
        print(f"[切换失败] 无法启动 {exe}: {e}", flush=True)
        return False
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        proc.terminate()
    return True



def ensure_adb_forward() -> None:
    try:
        subprocess.run([ADB, "forward", "tcp:27042", "tcp:27042"],
                       capture_output=True, timeout=15)
    except Exception as e:
        print(f"[警告] adb forward 失败: {e}")


def find_pid(dev, keyword: str = "subway"):
    for p in dev.enumerate_processes():
        if keyword.lower() in p.name.lower():
            return p.pid, p.name
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--raw", action="store_true", help="打印全部事件")
    args = ap.parse_args()

    try:
        import frida
    except Exception as e:
        if reexec_with_working_python():
            return
        print(f"[FAIL] 当前解释器 {sys.executable} 用不了 frida: {e}")
        print("       原因：frida 17.x 需要 Python 3.11+，而 wasd_local 是 3.10。")
        print("       解决：改用 D:\\pythonx\\python.exe 运行（那里已装 frida 17.17.0）")
        return

    ensure_adb_forward()
    dev = frida.get_device_manager().add_remote_device(FRIDA_HOST)

    pid = args.pid
    name = None
    if pid is None:
        pid, name = find_pid(dev)
        if pid is None:
            print("[FAIL] 没找到地铁跑酷进程，先启动游戏")
            return
    print(f"[目标] {name or ''}  pid={pid}")

    src = open(os.path.join(HERE, "frida_probe.js"), encoding="utf-8").read()
    session = dev.attach(pid)
    script = session.create_script(src)

    peaks = {}

    def on_msg(msg, data):
        if msg.get("type") != "send":
            print(msg.get("payload", msg))
            return
        p = msg["payload"]
        if p.get("ev") == "tick":
            d = p["delta"]
            hot = {k: v for k, v in d.items() if v}
            if hot or args.raw:
                line = "  ".join(f"{k}+{v}" for k, v in sorted(hot.items()))
                print(f"[{p['sec']:4d}s] {line}")
            for k, v in d.items():
                peaks[k] = max(peaks.get(k, 0), v)
        elif p.get("ev") == "open":
            print(f"         游戏打开文件: {p['path']}")

    script.on("message", on_msg)
    script.load()
    print("[提示] 进关卡开始玩，撞死、重开。Ctrl+C 结束。")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            session.detach()
        except Exception:
            pass
    print("\n===== 各计数器的峰值（每 2 秒）=====")
    for k, v in sorted(peaks.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} 峰值 {v}")


if __name__ == "__main__":
    main()
