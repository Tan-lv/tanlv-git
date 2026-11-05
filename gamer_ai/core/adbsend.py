# -*- coding: utf-8 -*-
r"""ADB 触摸发键通道：不依赖窗口焦点，也不怕模拟器过滤注入的按键。

为什么需要它（2026-10-16 实测结论）：
    · 6_自动玩.py 日志显示"已发键 N 次"，但模拟器里角色一动不动；
    · Python 和雷电**都是管理员权限** -> 不是 UIPI 拦截；
    · 焦点也确认抢到了（启动打印"模拟器已在前台"）；
    · 用户实测：**自己按 W/A/S/D 角色会动，程序注入的就不动**
      -> 雷电的按键映射只认真实硬件按键，忽略 SendInput 注入的事件。
    结论：PC 端无论用 pynput / keybd_event / SendInput / PostMessage 都发不进去
    （详见 tools\测试发键.py 的 5 种方式对照）。
    解法：**跳过 PC 键盘，直接用 ADB 往 Android 里注入触摸事件**。
    地铁跑酷本来就是滑动手游，w/a/s/d 对应 上/下/左/右 滑动，
    语义和模拟器的按键映射完全一致 —— 模型、数据集、标签都不用改。

性能：每次发键都起一个 adb.exe 要 100~200ms，来不及实时操作，
      所以这里用**常驻 adb shell**（Popen 起一次，之后往 stdin 写命令），
      单次发键约 15~30ms，和键盘方式一个量级。
"""
from __future__ import annotations

import os
import subprocess
import time

ADB_CANDIDATES = [
    r"D:\leidian\LDPlayer9\adb.exe",
    r"C:\leidian\LDPlayer9\adb.exe",
    r"D:\leidian\LDPlayer4\adb.exe",
    "adb",
]
CONNECT_PORTS = (5555, 62001, 5554, 21503)

# q 键没有方向语义（你加的 Q 类），默认在屏幕中间点一下。
# 如果 Q 在游戏里是某个具体按钮，把坐标写在这里（Android 像素，例如 1920x1080 下的 (960, 900)）。
Q_TAP_XY = None

# 滑动：长度 = min(宽,高) * SWIPE_RATIO，持续时间（毫秒）。短而快才像"甩"。
SWIPE_RATIO = 0.22
# ★ 2026-11-05（用户："为什么感觉这个模型反应很慢"）：
#   实测 adb 的 input swipe 每次要 **启动一个 Java 进程**，开销约 120ms，
#   再加手势本身的时长：110ms 的手势全程 260~277ms，40ms 的手势也要 197~216ms。
#   所以手势时长从 110 降到 60（手游要的是"甩"不是"拖"，短一点更灵敏也更准）。
SWIPE_MS = 60

# 一条命令从发出到设备执行完的估计总耗时（毫秒）。设备是**串行执行**的，
# 发得比这快，命令就会在 shell 里排队 -> 延迟越滚越大。
# 60ms 手势 + 约 120ms 的 Java 进程启动 ≈ 190ms。
CMD_BUSY_MS = 190

# ---------------------------------------------------------------------------
#  「点击类」按键：这类动作不是滑动，而是要**点到屏幕上的某个按钮**
#  写的是**相对坐标**（0~1，相对 Android 逻辑分辨率），换分辨率也不会失效。
#
#  ★ 2026-11-05（用户："GQ 两个判断出来后没有去触发"）：
#    以前只有 w/s/a/d 有手势，其它键一律"点屏幕中间"兜底 ——
#    而 G（复活弹窗的钥匙按钮）不在正中，Q 根本不是屏幕按钮，
#    于是模型判对了、键也"发了"，游戏里却什么都没发生。
#    G 的坐标是拿你自己的 G 类样本量出来的：
#      "Continue?" 弹窗 + 蓝色钥匙按钮，位于裁剪区 (0.496, 0.530)，
#      换算到 1920x1080 就是 (967, 572)。
TAP_REL = {
    "g": (0.504, 0.530),     # G = 复活：点 "Continue?" 弹窗上的钥匙按钮
    # "q": (?, ?),           # Q = ？你还没告诉我它在游戏里对应什么，
    #                        #     没配就**不会触发**（比乱点安全）
}


class AdbSender:
    """把按键字母翻译成 ADB 触摸滑动。send() 返回 True/False。"""

    def __init__(self, adb: str | None = None, serial: str | None = None,
                 quiet: bool = False):
        self.quiet = quiet
        self.adb = adb or self._find_adb()
        self.serial = serial or self._find_serial()
        self.size = self._screen_size() if self.serial else None
        self.n_sent = 0
        self.n_fail = 0
        self._proc = None
        self._warned_unknown = False
        self._busy_until = 0.0        # 上一条触摸命令预计执行到这个时刻
        self.n_busy = 0               # 因为"设备还忙着"而没发出去的次数

    # ------------------------------------------------------------ 探测
    @staticmethod
    def _find_adb() -> str:
        for c in ADB_CANDIDATES:
            if c == "adb" or os.path.isfile(c):
                return c
        return "adb"

    def _run(self, args, timeout=15):
        try:
            r = subprocess.run([self.adb] + list(args), capture_output=True,
                               text=True, errors="replace", timeout=timeout)
            return r.returncode, (r.stdout or "") + (r.stderr or "")
        except FileNotFoundError:
            return 127, "找不到 adb：%s" % self.adb
        except Exception as e:
            return 1, repr(e)

    def _find_serial(self):
        for attempt in range(3):
            _, out = self._run(["devices"])
            for line in out.splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    return parts[0]
            if attempt < 2:
                for port in CONNECT_PORTS:
                    self._run(["connect", "127.0.0.1:%d" % port], timeout=8)
                time.sleep(1.2)
        return None

    def _screen_size(self):
        _, out = self._run(["-s", self.serial, "shell", "wm", "size"])
        for line in out.splitlines():
            if "size:" in line:
                try:
                    w, h = line.split(":")[-1].strip().lower().split("x")
                    return int(w), int(h)
                except Exception:
                    pass
        return None

    def available(self) -> bool:
        return bool(self.serial and self.size)

    # ------------------------------------------------------------ 常驻 shell
    def _open_shell(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        if not self.serial:
            return False
        try:
            self._proc = subprocess.Popen(
                [self.adb, "-s", self.serial, "shell"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
            return True
        except Exception:
            self._proc = None
            return False

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            p.stdin.close()
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass

    def _shell(self, cmd: str) -> bool:
        """走常驻 shell（快）；shell 断了就退回一次性 adb 调用。"""
        if self._open_shell():
            try:
                self._proc.stdin.write(cmd + "\n")
                self._proc.stdin.flush()
                return True
            except Exception:
                self.close()
        rc, _ = self._run(["-s", self.serial, "shell"] + cmd.split())
        return rc == 0

    # ------------------------------------------------------------ 发键
    def busy(self) -> bool:
        """设备还在执行上一条触摸命令吗？（用于避免命令排队、延迟越滚越大）"""
        return time.time() < self._busy_until

    def send(self, ch: str) -> bool:
        """ch 是按键表里的字符（w/a/s/d/q/n）。返回是否真的发出去了。"""
        ch = (ch or "").lower()[:1]
        if not ch or ch == "n":            # N = 什么都不做，本来就不该发
            return True
        if not self.available():
            self.n_fail += 1
            return False
        w, h = self.size
        cx, cy = w // 2, h // 2
        d = max(120, int(min(w, h) * SWIPE_RATIO))
        moves = {"w": (cx, cy, cx, cy - d),      # 上 = 跳
                 "s": (cx, cy, cx, cy + d),      # 下 = 滚
                 "a": (cx, cy, cx - d, cy),      # 左
                 "d": (cx, cy, cx + d, cy)}      # 右
        if self.busy():
            # 上一条还没执行完 —— 再塞进去只会排队，延迟越滚越大。
            # 由调用方当成"被间隔挡下"计数，下一帧模型还会再判一次。
            self.n_busy += 1
            return False
        self._busy_until = time.time() + CMD_BUSY_MS / 1000.0
        if ch in moves:
            x1, y1, x2, y2 = moves[ch]
            ok = self._shell("input swipe %d %d %d %d %d" % (x1, y1, x2, y2, SWIPE_MS))
        elif ch in TAP_REL and TAP_REL[ch]:
            # 点击类：按相对坐标换算成当前分辨率的像素，点那个按钮
            rx, ry = TAP_REL[ch]
            ok = self._shell("input tap %d %d" % (int(rx * w), int(ry * h)))
        else:
            # ★ 不再"兜底点屏幕中间"：那既点不中按钮，又让人以为已经发键了。
            #   宁可明确失败 + 告警一次，也不要制造"发出去了但没反应"的假象。
            if not self._warned_unknown and not self.quiet:
                print("\n[adb] ⚠ 按键 %r 没有配置手势，**不会发出去**（模型预测了也动不了）\n"
                      "       配法：core/adbsend.py 顶部的 TAP_REL 里加一行 %r: (x比例, y比例)"
                      % (ch, ch))
                self._warned_unknown = True
            self.n_fail += 1
            return False
        if ok:
            self.n_sent += 1
        else:
            self.n_fail += 1
        return ok

    def describe(self) -> str:
        if not self.available():
            return "ADB 不可用（找不到设备或分辨率）"
        taps = [k for k, v in TAP_REL.items() if v]
        return "ADB %s  屏幕 %dx%d  滑动 %dms/%dpx  可点击键 %s" % (
            self.serial, self.size[0], self.size[1], SWIPE_MS,
            int(min(self.size) * SWIPE_RATIO), ("、".join(taps) or "无"))


if __name__ == "__main__":
    import sys
    s = AdbSender()
    print("adb      :", s.adb)
    print("serial   :", s.serial)
    print("size     :", s.size)
    print("available:", s.available())
    print(s.describe())
    if s.available() and "--bench" in sys.argv:
        for _ in range(4):
            t0 = time.time()
            s._shell("true")
            print("  changzhu-shell roundtrip %.1f ms" % ((time.time() - t0) * 1000))
        t0 = time.time()
        s._run(["-s", s.serial, "shell", "true"])
        print("  one-shot adb call      %.1f ms" % ((time.time() - t0) * 1000))
    s.close()
