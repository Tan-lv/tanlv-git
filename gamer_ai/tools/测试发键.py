# -*- coding: utf-8 -*-
r"""分层测试「发键 -> 模拟器」通道（把模型完全排除在外）。

为什么需要它：
    6_自动玩.py 的日志显示"已发键 N 次"，但模拟器里角色一动不动 ——
    说明按键"发出去"了，但雷电没接收。常见原因必须一种一种排除：
        1) pynput Controller        —— 现在 6_自动玩.py 用的方式（SendInput）
        2) keybd_event              —— 老 API，和 1 略有差别
        3) SendInput + 显式扫描码   —— 有些模拟器只认硬件扫描码
        4) PostMessage(WM_KEYDOWN)  —— 绕过所有键盘钩子，直接给窗口投消息
        5) ADB 触摸滑动             —— 完全不碰 PC 键盘，走模拟器内部触摸事件；
                                      地铁跑酷本来就是滑动手游，这是最可靠的通道
用法：
    python tools\测试发键.py --check           # 只看诊断信息，一个键都不发
    python tools\测试发键.py                   # 依次试 5 种方式（默认用 d 键）
    python tools\测试发键.py --key w --method 3
    python tools\测试发键.py --methods 1,4,5
跑的时候**盯住游戏画面**，记下第几种能让角色动，然后告诉我就行。
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # -> gamer_ai/
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from core.capture import WindowCapture, find_window          # noqa: E402
from core.config import Config                               # noqa: E402

u32 = ctypes.windll.user32
k32 = ctypes.windll.kernel32
adb32 = ctypes.windll.advapi32
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

ADB = r"D:\leidian\LDPlayer9\adb.exe"
ADB_PORTS = (5555, 62001, 5554)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101


class TOKEN_ELEVATION(ctypes.Structure):
    _fields_ = [("TokenIsElevated", wintypes.DWORD)]


# ---------------------------------------------------------------- 诊断
def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def token_elevated(pid: int):
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return None
    try:
        tok = wintypes.HANDLE()
        if not adb32.OpenProcessToken(h, 0x0008, ctypes.byref(tok)):
            return None
        try:
            te = TOKEN_ELEVATION()
            ret = wintypes.DWORD()
            ok = adb32.GetTokenInformation(tok, 20, ctypes.byref(te),
                                           ctypes.sizeof(te), ctypes.byref(ret))
            return bool(te.TokenIsElevated) if ok else None
        finally:
            k32.CloseHandle(tok)
    finally:
        k32.CloseHandle(h)


def win_text(hwnd, which="title") -> str:
    if which == "cls":
        buf = ctypes.create_unicode_buffer(256)
        u32.GetClassNameW(hwnd, buf, 256)
        return buf.value
    n = u32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 2)
    u32.GetWindowTextW(hwnd, buf, n + 2)
    return buf.value


def focus(hwnd):
    if u32.GetForegroundWindow() == hwnd:
        return True
    u32.ShowWindow(hwnd, 9)
    u32.keybd_event(0x12, 0, 0, 0)          # 敲一下 Alt，绕开前台锁定
    u32.SetForegroundWindow(hwnd)
    u32.keybd_event(0x12, 0, 2, 0)
    time.sleep(0.25)
    return u32.GetForegroundWindow() == hwnd


# ---------------------------------------------------------------- ADB
def adb_run(args, timeout=20):
    if not os.path.isfile(ADB):
        return 127, "adb 不存在：%s" % ADB
    try:
        r = subprocess.run([ADB] + args, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return 1, repr(e)


def adb_serial():
    """返回可用的设备序列号；没有就尝试连接常用端口。"""
    for attempt in range(3):
        _, out = adb_run(["devices"])
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                return parts[0]
        if attempt == 0:
            for port in ADB_PORTS:
                adb_run(["connect", "127.0.0.1:%d" % port], timeout=10)
            time.sleep(1.5)
        else:
            time.sleep(1.5)
    return None


def adb_screen(serial):
    _, out = adb_run(["-s", serial, "shell", "wm", "size"])
    for line in out.splitlines():
        if "size:" in line:
            wh = line.split(":")[-1].strip()
            try:
                w, h = [int(v) for v in wh.lower().split("x")]
                return w, h
            except Exception:
                pass
    return None


def adb_swipe(serial, ch, size):
    """把按键翻译成手势。

    ★ 2026-11-05：这里改成**直接复用 core/adbsend.py 的手势表**，
    不再自己写一套 —— 否则测试通过、上线不通过（G/Q 就是这么埋的坑）。
    """
    try:
        import sys as _s
        _g = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _g not in _s.path:
            _s.path.insert(0, _g)
        from core.adbsend import AdbSender
        snd = AdbSender(serial=serial, quiet=True)
        if snd.available():
            ok = snd.send(ch)
            note = "手势表: " + snd.describe()
            snd.close()
            return ok, note
        snd.close()
    except Exception as e:
        print("     （复用 adbsend 失败，退回内置手势：%r）" % e)
    # ---- 兜底：内置手势（老逻辑）----
    if not size:
        return False, "拿不到屏幕分辨率"
    w, h = size
    cx, cy = w // 2, h // 2
    d = max(120, int(min(w, h) * 0.22))
    m = {"w": (cx, cy, cx, cy - d), "s": (cx, cy, cx, cy + d),
         "a": (cx, cy, cx - d, cy), "d": (cx, cy, cx + d, cy)}
    if ch in m:
        x1, y1, x2, y2 = m[ch]
        _, out = adb_run(["-s", serial, "shell", "input", "swipe",
                          str(x1), str(y1), str(x2), str(y2), "120"])
        return True, "swipe %s" % ((x1, y1, x2, y2),)
    if ch in ("q", "n"):
        _, out = adb_run(["-s", serial, "shell", "input", "tap", str(cx), str(cy)])
        return True, "tap center"
    return False, "这个键没有对应的滑动"


# ---------------------------------------------------------------- 五种发键方式
def m_pynput(ch):
    from pynput.keyboard import Controller, KeyCode
    kb = Controller()
    kb.press(KeyCode.from_char(ch))
    time.sleep(0.05)
    kb.release(KeyCode.from_char(ch))


def m_keybd_event(ch):
    vk = ord(ch.upper())
    u32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.05)
    u32.keybd_event(vk, 0, 2, 0)


def m_sendinput_scan(ch):
    vk = ord(ch.upper())
    scan = u32.MapVirtualKeyW(vk, 0)          # MAPVK_VK_TO_VSC
    for up in (False, True):
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.u.ki = KEYBDINPUT(0, scan, KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0), 0, 0)
        u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.05)


def m_postmessage(hwnd, ch):
    vk = ord(ch.upper())
    scan = u32.MapVirtualKeyW(vk, 0)
    u32.PostMessageW(hwnd, WM_KEYDOWN, vk, (scan << 16) | 1)
    time.sleep(0.05)
    u32.PostMessageW(hwnd, WM_KEYUP, vk, (scan << 16) | 0xC0000001)


def m_adb(serial, size, ch):
    ok, note = adb_swipe(serial, ch, size)
    if not ok:
        raise RuntimeError(note)
    time.sleep(0.05)


METHODS = [
    ("1) pynput Controller（现在 6_自动玩.py 用的）", "pynput"),
    ("2) keybd_event 老 API", "keybd"),
    ("3) SendInput + 显式扫描码", "scan"),
    ("4) PostMessage WM_KEYDOWN（绕过键盘钩子）", "post"),
    ("5) ADB 触摸滑动（手游原生操作）", "adb"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="分层测试发键通道")
    ap.add_argument("--key", default="d", help="测试用哪个键（默认 d = 向右）")
    ap.add_argument("--method", type=int, default=None, help="只测某一种（1~5）")
    ap.add_argument("--methods", default=None, help="测哪几种，逗号分隔：1,4,5")
    ap.add_argument("--times", type=int, default=3, help="每种按几次（默认 3）")
    ap.add_argument("--interval", type=float, default=1.5, help="每次间隔秒（默认 1.5）")
    ap.add_argument("--check", action="store_true", help="只看诊断，不发键")
    a = ap.parse_args()

    cfg = Config()
    hwnd = find_window(cfg.window_keyword)
    print("== 诊断 ==")
    print("  这个 python 是管理员 :", is_admin())
    if not hwnd:
        print("  找不到模拟器窗口 —— 先启动雷电并进入游戏")
        return 1
    pid = wintypes.DWORD()
    u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    print("  窗口 hwnd=%s pid=%s 标题=%r 类名=%s" % (hwnd, pid.value, win_text(hwnd), win_text(hwnd, "cls")))
    print("  该进程是管理员      :", token_elevated(pid.value))
    rc = wintypes.RECT()
    u32.GetClientRect(hwnd, ctypes.byref(rc))
    print("  客户区尺寸          : %dx%d" % (rc.right - rc.left, rc.bottom - rc.top))
    print("  当前前台窗口        :", u32.GetForegroundWindow())
    if rc.right - rc.left < 300:
        print("  [注意] 客户区太小 —— 模拟器窗口被最小化/关掉了，先把它显示出来再测")
    serial = adb_serial()
    size = None
    if serial:
        size = adb_screen(serial)
        print("  ADB 设备            : %s   屏幕 %s" % (serial, size))
    else:
        print("  ADB 设备            : 连不上（不影响 1~4 号方式）")
    if a.check:
        return 0

    if a.method:
        todo = [a.method]
    elif a.methods:
        todo = [int(x) for x in a.methods.replace(" ", "").split(",") if x]
    else:
        todo = [1, 2, 3, 4, 5]
    todo = [t for t in todo if 1 <= t <= len(METHODS)]
    if not todo:
        print("没有要测的方式")
        return 1

    ch = a.key.strip().lower()[:1]
    print("\n将要依次测试：%s   （按 %r 键，每种 %d 次，间隔 %.1fs）"
          % ("、".join(str(t) for t in todo), ch, a.times, a.interval))
    print("★ 请盯住游戏画面，看角色有没有反应（d = 向右换道 / a = 向左 / w = 跳 / s = 滚）")
    time.sleep(2.0)

    for t in todo:
        name, kind = METHODS[t - 1]
        print("\n" + "=" * 70)
        print(">>> 方法 %s" % name)
        print("=" * 70)
        for i in range(1, a.times + 1):
            try:
                if kind == "adb":
                    if not serial:
                        print("    跳过：ADB 连不上")
                        break
                    m_adb(serial, size, ch)
                    note = "adb"
                else:
                    ok = focus(hwnd)
                    if kind == "pynput":
                        m_pynput(ch)
                    elif kind == "keybd":
                        m_keybd_event(ch)
                    elif kind == "scan":
                        m_sendinput_scan(ch)
                    else:
                        m_postmessage(hwnd, ch)
                    note = "前台=%s" % ("是" if ok else "否")
                print("    [%d/%d] 已发 %s   (%s)" % (i, a.times, ch, note))
            except Exception as e:
                print("    [%d/%d] 失败：%r" % (i, a.times, e))
                break
            time.sleep(a.interval)
        time.sleep(1.5)

    print("\n" + "=" * 70)
    print("测完了。哪一种让角色动了？")
    for i, (name, _) in enumerate(METHODS, 1):
        print("   %d = %s" % (i, name))
    print("   0 = 全都没动")
    print("把号码告诉我，我把 6_自动玩.py 的发键方式换成它。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
