# -*- coding: utf-8 -*-
r"""把游戏状态通过本地 TCP 暴露给采集器。

采集器跑在 wasd_local（Python 3.10，没有 frida），而 frida 需要 Python 3.11+。
所以用一个独立进程（D:\pythonx\python.exe）持有 Frida 会话，采集器通过 localhost 轮询。

用法:
    D:\pythonx\python.exe -u tools\state_server.py            # 默认端口 47800
读取:
    {"ok":true,"sv":2,"inPlay":1,"alive":1,"paused":0}
    inPlay: 1=正在跑酷 0=菜单/结算     alive: 1=活着 0=撞死了     paused: 1=暂停

==============================================================================
★ 2026-09-22 改版：这个服务以前会"假活"，把采集器的判卷整死

  老版本的问题（就是"加了 hook 反而检测不到撞车"的根因）：
    ① 先连 frida，最后才 bind 端口。连不上时端口根本不开，
       采集器只能看到一句"状态服务启动超时，改用画面判断" ——
       真实原因（没设备 / 没 frida-server / 游戏没开）全在另一个控制台窗口里。
    ② frida 会话废了（script has been destroyed）以后，进程不退出、端口还开着，
       却永远只回 {"ok":false,"err":"script has been destroyed"} —— 变成僵尸。
       采集器连上来就以为"服务在跑"，于是整局没有判卷，终端上却只显示"撞车 0"。

  现在的做法：
    * **先占端口，再连 frida**：采集器 1 秒内就能连上，看到的是真实进度。
    * **断线自愈**：worker 线程每次读失败都计数，连续失败就 detach + 重新 attach，
      不需要采集器来杀它。
    * 回复里带 sv（协议版本）字段，采集器据此判断端口上是不是旧版/僵尸进程。
==============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ADB = r"D:\leidian\LDPlayer9\adb.exe"
FRIDA_HOST = "127.0.0.1:27042"
GAME_KEY = "subway"
SERVER_VERSION = 2

frida = None        # main() 里 import 成功后赋进来

# frida 17.x 要求 Python 3.11+，而项目的 wasd_local 是 3.10。
# 如果当前解释器导入不了 frida，自动换成下面这个能用的解释器重跑。
FALLBACK_PYTHONS = [r"D:\Anaconda\envs\wasd311\python.exe",   # 主线：Python 3.11 + torch cu124 + frida
                    r"D:\pythonx\python.exe"]                    # 备用


def reexec_with_working_python() -> bool:
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
            print(f"[切换] 当前解释器没有可用的 frida，改用 {exe} (frida {ver})", flush=True)
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


def _adb(args, timeout=25):
    try:
        p = subprocess.run([ADB] + args, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    except Exception as e:
        return -1, "", str(e)


# ★ 设备序列号必须固定下来：
#   LDPlayer 经常同时出现两个条目（emulator-5554 和 127.0.0.1:5555），
#   这时 adb shell 不带 -s 会直接报 "more than one device/emulator"，
#   所有 shell 命令都静默失败 —— frida-server 就是这么"起不来"的。
_DEV = {"serial": None}


def _adb_dev(args, timeout=25):
    s = _DEV["serial"]
    return _adb((["-s", s] if s else []) + list(args), timeout=timeout)


def pick_device():
    """挑一个真的能用的设备并记下来，之后所有 adb 命令都带 -s。"""
    _, out, _ = _adb(["devices"])
    if "List of devices" not in out:
        _adb(["start-server"])
        time.sleep(1)
        _, out, _ = _adb(["devices"])

    def _cands():
        return [l.split()[0] for l in out.splitlines()
                if l.strip() and "List of devices" not in l and "offline" not in l]

    got = _cands()
    if not got:
        print("      adb 里没有可用设备，尝试重连 127.0.0.1:5555 ...", flush=True)
        _adb(["connect", "127.0.0.1:5555"])
        time.sleep(2)
        _, out, _ = _adb(["devices"])
        got = _cands()

    for serial in got:
        rc, so, _ = _adb(["-s", serial, "shell", "echo ok"], timeout=10)
        if rc == 0 and "ok" in so:
            _DEV["serial"] = serial
            print(f"      adb 设备: {serial}", flush=True)
            return serial
    return None


def ensure_device_ready(port: int = 27042) -> bool:
    """把 adb server / 设备连接 / frida-server / 端口转发 全部准备好。

    这一步很关键：**模拟器一重启，frida-server 就没了**，必须重新拉起来。
    所以这里每次都检查一遍，缺什么补什么。

    ★ 所有 adb 命令都走 _adb_dev()（带 -s 序列号）：
      设备同时出现在 emulator-5554 和 127.0.0.1:5555 两条记录里时，
      不带 -s 的 adb shell 会报 "more than one device/emulator" 直接失败，
      表现就是"frida-server 起不来" —— 这是 2026-09-22 事故的第二根因。
    """
    print("      检查 adb server ...", flush=True)
    serial = pick_device()
    if serial is None:
        _, out, _ = _adb(["devices"])
        print("[FAIL] adb 里没有可用设备。请确认：", flush=True)
        print("       1) 雷电模拟器已经启动（能看到安卓桌面）", flush=True)
        print("       2) 模拟器设置里「ADB 调试」是打开的", flush=True)
        print(f"       3) 当前 adb 输出: {out.strip() or '(空)'}", flush=True)
        return False

    print("      检查设备上的 frida-server ...", flush=True)
    _, ps, _ = _adb_dev(["shell", "ps -A | grep frida"])
    if "frida-server" not in ps:
        print("      frida-server 没在跑，用 root 拉起 ...", flush=True)
        _adb_dev(["shell", "su -c 'sh /data/local/tmp/start_frida.sh'"])
        time.sleep(1)
        _adb_dev(["shell", "su -c 'nohup /data/local/tmp/frida-server -D > /data/local/tmp/fs.log 2>&1 &'"])
        time.sleep(3)
        _, ps, _ = _adb_dev(["shell", "ps -A | grep frida"])
        if "frida-server" not in ps:
            print("[FAIL] frida-server 起不来。手动执行一次：", flush=True)
            print(f"       {ADB} -s {serial} shell sh /data/local/tmp/start_frida.sh", flush=True)
            _, lg, _ = _adb_dev(["shell", "cat /data/local/tmp/fs.log"])
            print(f"       设备上的 fs.log: {lg.strip()[:300]}", flush=True)
            return False
    print("      frida-server: 运行中", flush=True)

    _adb_dev(["forward", f"tcp:{port}", f"tcp:{port}"])
    print(f"      端口转发: tcp:{port} -> 设备", flush=True)
    return True


# ======================================================================
#  状态缓存：TCP 线程读它，worker 线程写它
# ======================================================================
_LK = threading.Lock()
_LAST = {"st": None}


def set_state(d: dict) -> None:
    d = dict(d)
    d.setdefault("sv", SERVER_VERSION)
    with _LK:
        _LAST["st"] = d


def get_state() -> dict:
    with _LK:
        s = _LAST["st"]
    if s is None:
        return {"ok": False, "sv": SERVER_VERSION, "err": "状态服务刚启动，正在初始化 ..."}
    return s


# ======================================================================
#  frida 会话
# ======================================================================
def go_live(pid_hint=None):
    """连上游戏并定位到 GameController。失败就抛异常，由 worker 重试。"""
    print("[前置] 检查模拟器 / adb / frida-server", flush=True)
    if not ensure_device_ready(27042):
        raise RuntimeError("adb / frida-server 没准备好（看上面的 [FAIL] 提示）")

    dev = frida.get_device_manager().add_remote_device(FRIDA_HOST)
    pid = pid_hint
    if pid is None:
        for p in dev.enumerate_processes():
            if GAME_KEY in p.name.lower():
                pid = p.pid
                break
    if pid is None:
        raise RuntimeError("没找到地铁跑酷进程（游戏没开？）")
    print(f"[目标] pid={pid}", flush=True)

    session = dev.attach(pid)
    js = open(os.path.join(HERE, "il2cpp_state.js"), encoding="utf-8").read()
    script = session.create_script(js)
    script.on("message", lambda m, d: print("[agent]", m.get("payload", m), flush=True))
    script.load()
    info = script.exports_sync.locate()
    print(f"[定位] {info}", flush=True)
    if not info.get("ok"):
        raise RuntimeError(f"定位失败：{info.get('err')}")
    return session, script


def worker(pid_hint=None) -> None:
    """一个永不退出的后台线程：连不上就重连，读失败就重建会话。"""
    session = None
    script = None
    errs = 0
    while True:
        if script is None:
            set_state({"ok": False, "err": "正在连接游戏（adb / frida-server / 进程）..."})
            try:
                session, script = go_live(pid_hint)
                errs = 0
                set_state({"ok": False, "err": "frida 会话已建立，正在读取状态 ..."})
            except Exception as e:
                msg = str(e).splitlines()[0][:140]
                print(f"[连接失败] {msg}", flush=True)
                set_state({"ok": False, "err": f"连接失败：{msg}"})
                time.sleep(3.0)
                continue
        try:
            r = script.exports_sync.read()
            errs = 0
            if isinstance(r, dict):
                set_state(r)
            else:
                set_state({"ok": False, "err": "读取返回了非字典"})
        except Exception as e:
            errs += 1
            msg = str(e).splitlines()[0][:140]
            if errs in (1, 40, 200) or errs % 1000 == 0:
                print(f"[读取失败] 第 {errs} 次：{msg}", flush=True)
            set_state({"ok": False, "err": f"内存读取中断（第 {errs} 次）：{msg}"})
            if errs >= 40:        # 连续失败约 1 秒 -> 认为会话已废，重建
                print("[重连] frida 会话已废，重新建立 ...", flush=True)
                try:
                    if session:
                        session.detach()
                except Exception:
                    pass
                session = None
                script = None
                errs = 0
                time.sleep(1.5)
        time.sleep(0.02)


def heartbeat() -> None:
    """每 10 秒打一次心跳，让"它还活着"一眼可见（不然会以为跑完了）。"""
    t0 = time.time()
    while True:
        time.sleep(10)
        r = get_state() or {}
        print(f"[心跳] 已运行 {time.time()-t0:6.0f}s   "
              f"inPlay={r.get('inPlay', '?')} alive={r.get('alive', '?')} "
              f"paused={r.get('paused', '?')}   err={str(r.get('err') or '')[:40]}", flush=True)


# ======================================================================
#  TCP 服务
# ======================================================================
def handle(conn) -> None:
    try:
        buf = b""
        while True:
            data = conn.recv(64)
            if not data:
                break
            buf += data
            if b"\n" in buf:
                conn.sendall((json.dumps(get_state(), ensure_ascii=False) + "\n").encode("utf-8"))
                buf = b""
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def serve(srv) -> None:
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                time.sleep(0.2)
                continue
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


def main() -> None:
    global frida
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47800)
    ap.add_argument("--pid", type=int, default=None)
    args = ap.parse_args()

    try:
        import frida as _f
        frida = _f
    except Exception as e:
        if reexec_with_working_python():
            return
        print(f"[FAIL] 当前解释器 {sys.executable} 用不了 frida: {e}", flush=True)
        print("       原因：frida 17.x 需要 Python 3.11+，wasd_local 是 3.10。", flush=True)
        print(r"       解决：双击 tools\run_state_server.bat，或者用 D:\pythonx\python.exe 运行", flush=True)
        return

    # ★ 先占端口，再做耗时的 frida 准备 —— 采集器不用再赌"启动超时"
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("127.0.0.1", args.port))
    except Exception as e:
        print(f"[FAIL] 端口 {args.port} 被占用了：{e}", flush=True)
        print("       已经有另一个状态服务在跑？先关掉它，或者让采集器自己处理。", flush=True)
        return
    srv.listen(8)
    set_state({"ok": False, "err": "状态服务已启动，正在连接 frida ..."})
    print(f"[就绪] 采集器可连 127.0.0.1:{args.port} 取状态。", flush=True)
    print("       这个窗口要一直开着 —— 它是服务，不是跑完就结束的脚本。", flush=True)

    threading.Thread(target=worker, args=(args.pid,), daemon=True).start()
    threading.Thread(target=heartbeat, daemon=True).start()
    serve(srv)


if __name__ == "__main__":
    main()
