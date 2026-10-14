# -*- coding: utf-8 -*-
"""从 tools/state_server.py 读取游戏状态（IL2CPP 内存读取）。

为什么这么绕：frida 需要 Python 3.11+，而采集器跑在 wasd_local(3.10)。
所以由一个独立进程持有 Frida 会话，这里通过 localhost 轮询。

比画面判断强的地方：
  inPlay  = 1 表示游戏自己在跑酷（不是"画面在动"）
  alive   = 0 的那一瞬间就是撞到障碍的时刻 —— 精确到帧，不用等 4 秒猜

==============================================================================
★ 2026-09-22 修的一次事故：「加了 hook 反而检测不到撞车了」

  上一轮跑完留下的状态服务进程还占着 47800，但它内部的 frida 会话已经废了，
  只会一直回 {"ok":false,"err":"script has been destroyed"}。
  老代码只看"端口开没开"（_port_open），于是直接复用了这个僵尸；
  杀掉重开后，新的那个要 20 秒以上才 bind 端口，ensure_state_server 已经
  超时返回 False —— 于是判卷循环继续用**那个已经 close 掉的旧客户端**，
  而它的 read() 会永远返回缓存里的那句 "script has been destroyed"；
  2 秒后循环又判"会话已废"，再去杀端口上的进程 —— 杀的却是它自己刚拉起来的
  那个新服务。如此反复，整局都没有判卷，终端上却只显示「撞车 0」。

  现在的规矩：
    ① 端口上的回复必须带版本号 sv；旧版/僵尸一律杀掉重开。
    ② 状态服务**先占端口、后连 frida**，所以采集器几秒内就能连上；
       到底能不能判卷由 read() 能不能读到 inPlay 决定，不看进程起没起。
    ③ 客户端掉线后 read() 返回 None，绝不再回旧数据。
    ④ 重连只走 open_client()，它一定会换一个**新**客户端或返回 None，
       永远不会留下一个已经 close() 的死客户端。
==============================================================================
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from typing import Optional

# 状态服务协议版本：僵尸/旧版进程没有这个字段，采集器据此强制重开。
SERVER_VERSION = 2

PAOKU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRIDA_PYTHONS = [r"D:\Anaconda\envs\wasd311\python.exe",   # 主线环境（含 frida）
                 r"D:\pythonx\python.exe"]                    # 备用
STATE_LOG = os.path.join(PAOKU_DIR, "runs", "state_server.log")


class HookStateClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 47800,
                 timeout: float = 0.35, poll_hz: float = 50.0):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._state: Optional[dict] = None
        self._last_ok = 0.0
        self._running = True
        self._interval = 1.0 / max(poll_hz, 1.0)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout)
            s.settimeout(self.timeout)
            self._sock = s
            return True
        except Exception:
            self._sock = None
            return False

    def _query(self) -> Optional[dict]:
        if self._sock is None and not self._connect():
            return None
        try:
            self._sock.sendall(b"?\n")
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = self._sock.recv(256)
                if not chunk:
                    raise ConnectionError("closed")
                buf += chunk
                if len(buf) > 65536:
                    raise ValueError("reply too long")
            return json.loads(buf.decode("utf-8"))
        except Exception:
            try:
                if self._sock:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None
            return None

    def _loop(self) -> None:
        while self._running:
            st = self._query()
            if st is not None:
                with self._lock:
                    self._state = st
                    self._last_ok = time.time()
            time.sleep(self._interval)

    # ------------------------------------------------------------------
    def read(self) -> Optional[dict]:
        """当前状态；掉线就返回 None（★ 绝不回旧数据）。

        以前这里不看新鲜度，close() 之后还会把最后一次读到的
        {"ok":false,"err":"script has been destroyed"} 一直喂给判卷循环 ——
        判卷循环因此永远停在"会话已废"分支里出不来。
        """
        with self._lock:
            if self._state is None:
                return None
            if time.time() - self._last_ok > 2.0:
                return None
            return dict(self._state)

    @property
    def alive_link(self) -> bool:
        """最近 1 秒内有没有成功读到状态。"""
        with self._lock:
            return (time.time() - self._last_ok) < 1.0

    def wait_ready(self, timeout: float = 5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.alive_link:
                return True
            time.sleep(0.1)
        return False

    def close(self) -> None:
        self._running = False
        with self._lock:
            self._state = None          # ★ 清掉缓存，免得死客户端继续"说话"
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass


# ======================================================================
#  探测 / 杀僵尸
# ======================================================================
def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def probe_state(host: str = "127.0.0.1", port: int = 47800,
                timeout: float = 0.8) -> Optional[dict]:
    """连上去问一次状态。连不上 / 不回话 / 回的不是 JSON -> None。"""
    try:
        s = socket.create_connection((host, port), timeout=timeout)
    except Exception:
        return None
    try:
        s.settimeout(timeout)
        s.sendall(b"?\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(512)
            if not chunk:
                return None
            buf += chunk
            if len(buf) > 65536:
                return None
        obj = json.loads(buf.decode("utf-8", "replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    finally:
        try:
            s.close()
        except Exception:
            pass


def state_healthy(st) -> bool:
    """这个回复能不能用来判卷（有 inPlay 才算数）。"""
    return bool(isinstance(st, dict) and st.get("ok") and ("inPlay" in st))


def _kill_pid(pid, verbose: bool = True) -> None:
    if not str(pid).isdigit():
        return
    try:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                       capture_output=True, timeout=15)
        if verbose:
            print(f"[IL2CPP] 已杀掉状态服务进程 pid={pid}")
    except Exception:
        pass


def _state_server_pids():
    """所有命令行里带 state_server.py 的 python 进程（包括没 bind 成功的残留）。"""
    cmd = ("Get-CimInstance Win32_Process | "
           "Where-Object { $_.Name -eq 'python.exe' -and "
           "$_.CommandLine -like '*state_server.py*' } | "
           "Select-Object -ExpandProperty ProcessId")
    for sh in ("powershell", "pwsh"):
        try:
            out = subprocess.run([sh, "-NoProfile", "-Command", cmd],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception:
            continue
        return [x.strip() for x in out.split() if x.strip().isdigit()]
    return []


def kill_state_server(host: str = "127.0.0.1", port: int = 47800) -> None:
    """把状态服务彻底清干净。

    两件事都要做：
      ① 杀掉占着端口的那个（netstat + taskkill）；
      ② 杀掉所有残留的 state_server.py 进程 —— 启动超时的那些没 bind 成功，
         netstat 看不到它们，但它们后面随时可能抢到端口，变成新的僵尸。
    """
    p = str(port)
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True, timeout=20).stdout
        pids = set()
        for line in out.splitlines():
            if (":" + p) in line and "LISTENING" in line.upper():
                pids.add(line.split()[-1])
        for pid in pids:
            _kill_pid(pid)
    except Exception as e:
        print(f"[IL2CPP] 杀端口占用失败: {str(e)[:60]}")
    for pid in _state_server_pids():
        _kill_pid(pid, verbose=False)
    time.sleep(0.8)


def _log_tail(n: int = 25) -> str:
    """读状态服务日志的最后几行。

    ★ 注意编码：子进程 stdout 重定向到文件时用的是**本地编码**（中文 Windows 上
      是 GBK），所以不能无脑按 UTF-8 解 —— 解出来一堆 U+FFFD，
      再 print 到 GBK 控制台就直接抛 UnicodeEncodeError。
    """
    try:
        with open(STATE_LOG, "rb") as fh:
            raw = fh.read()
    except Exception:
        return "    （还没有日志）"
    txt = raw.decode("utf-8", "replace")
    if "\ufffd" in txt:
        try:
            txt = raw.decode("gbk", "replace")
        except Exception:
            pass
    lines = [l for l in txt.splitlines() if l.strip()][-n:]
    return "\n".join("    " + l for l in lines) if lines else "    （日志是空的）"


# ======================================================================
#  自动拉起状态服务
# ======================================================================
def _find_frida_python():
    """找一个能 import frida 的解释器。"""
    for exe in FRIDA_PYTHONS:
        if not os.path.isfile(exe):
            continue
        try:
            r = subprocess.run([exe, "-c", "import frida"], capture_output=True, timeout=60)
            if r.returncode == 0:
                return exe
        except Exception:
            continue
    return None


def ensure_state_server(host: str = "127.0.0.1", port: int = 47800,
                        wait_s: float = 45.0) -> bool:
    """保证有一个**新协议**的状态服务在 47800 上跑着。

    返回 True 只代表"服务进程起来了、能收能发 JSON"，
    不代表已经能读到游戏状态（那要等它连上 frida + 定位到 GameController）。
    能不能判卷由调用方拿 probe_state()/client.read() 看 inPlay 决定 ——
    这样就不会再把"服务启动慢"误判成"服务坏了"。
    """
    st = probe_state(host, port)
    if st is not None and st.get("sv") == SERVER_VERSION:
        if state_healthy(st):
            print(f"[IL2CPP] 状态服务已在运行 ({host}:{port})")
        else:
            print(f"[IL2CPP] 状态服务已在运行，正在准备：{str(st.get('err'))[:70]}")
        return True

    # 端口上要么是旧协议/僵尸，要么是不回话的死进程 —— 都得清掉
    if st is not None:
        print(f"[IL2CPP] 端口 {port} 上是旧版/僵死的状态服务"
              f"（{str(st.get('err') or st)[:50]}），杀掉重开")
        kill_state_server(host, port)
    elif _port_open(host, port):
        print(f"[IL2CPP] 端口 {port} 被占着但不回话（僵死），杀掉重开")
        kill_state_server(host, port)

    exe = _find_frida_python()
    if exe is None:
        print("[提示] 没找到能用的 frida 解释器，判卷无法开启")
        return False

    script = os.path.join(PAOKU_DIR, "tools", "state_server.py")
    if not os.path.isfile(script):
        print(f"[提示] 找不到 {script}，判卷无法开启")
        return False

    os.makedirs(os.path.dirname(STATE_LOG), exist_ok=True)
    print(f"[IL2CPP] 拉起状态服务：{exe} -u {script} --port {port}")
    print(f"[IL2CPP] 它的输出写在这个文件里：{STATE_LOG}")
    try:
        logf = open(STATE_LOG, "ab", buffering=0)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"      # 日志固定写成 UTF-8，免得 GBK 乱码
        env["PYTHONUTF8"] = "1"
        subprocess.Popen([exe, "-u", script, "--port", str(port)],
                         stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, cwd=PAOKU_DIR, env=env)
        logf.close()                    # 子进程自己有一份句柄，父进程关掉不影响它
    except Exception as e:
        print(f"[提示] 拉起失败: {e}")
        return False

    t0 = time.time()
    while time.time() - t0 < wait_s:
        st = probe_state(host, port)
        if st is not None and st.get("sv") == SERVER_VERSION:
            print(f"[IL2CPP] 状态服务已就绪 ({host}:{port})，用时 {time.time()-t0:.1f}s")
            return True
        time.sleep(0.4)
    print(f"[提示] 状态服务 {wait_s:.0f} 秒内没就绪。它自己的日志最后几行：")
    print(_log_tail())
    return False


def open_client(host: str = "127.0.0.1", port: int = 47800,
                wait_s: float = 90.0, verbose: bool = True):
    """拿一个**能用**的 HookStateClient；拿不到就返回 None（绝不返回死客户端）。

    先确保服务进程起来，然后等它读到游戏状态（有 inPlay）。
    等的时候把服务端报的原因打出来 —— 以前的"启动超时"什么都说明不了。
    """
    if not ensure_state_server(host, port):
        if verbose:
            print("[hook] ⚠ 状态服务起不来，判卷关闭（撞车/追击都不会被记录）")
        return None
    cli = HookStateClient(host, port)
    t0 = time.time()
    last = None
    while time.time() - t0 < wait_s:
        st = cli.read()
        if state_healthy(st):
            if verbose:
                print(f"[hook]  判卷已开启（用时 {time.time()-t0:.1f}s）")
            return cli
        msg = str((st or {}).get("err") or "连接中，还没读到状态")
        if msg != last:
            last = msg
            if verbose:
                print(f"[hook]  等状态服务就绪：{msg[:90]}")
        time.sleep(0.3)
    if verbose:
        print(f"[hook] ⚠ 等了 {wait_s:.0f} 秒还是读不到游戏状态，判卷关闭（只录不判）")
        print("[hook]  状态服务日志最后几行：")
        print(_log_tail())
    cli.close()
    return None
