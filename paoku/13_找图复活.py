# -*- coding: utf-8 -*-
r"""找图复活：角色被捕后，先找「钥匙」点复活，找不到就找「PLAY」重开。

用法（PyCharm 直接 Run，或者命令行）:
    python 13_找图复活.py                 # 正常跑（会真的点击）
    python 13_找图复活.py --dry-run       # 只找不点，用来调阈值/看日志
    python 13_找图复活.py --grab          # 抓一张当前画面 + 打印匹配得分，然后退出
    python 13_找图复活.py --verbose       # 每次搜索都打印最高得分

规则（按你要求的）:
    角色被捕  ──►  循环找【钥匙图】，找到就点
                   5 秒内还没找到  ──►  循环找【PLAY图】，找到就点
                   4 秒内还没找到  ──►  放弃，等下一次被捕，再从头来

图放哪:
    D:\deeplearning\make_dataset\paoku\templates\key.png    钥匙（第二张图）
    D:\deeplearning\make_dataset\paoku\templates\play.png   PLAY（第一张图）

★ 搜索区域（很重要，2026-09-23 实测踩到的坑）:
  训练用的 game_crop = (0, 200, 720, 880) **把最下面 200 像素裁掉了**，
  而 PLAY 按钮正好在 y≈1170（画面最底下那一排）—— 在裁剪图上永远找不到它。
  所以本脚本默认搜**整块安卓画面 720x1280**（--search-area full）；
  训练/标注照旧用 game_crop，两件事互不影响。

★ 关于匹配得分:
  实测：play.png 在整屏上匹配到 **1.000（scale 1.00）**，就是你给的原始尺寸。
  所以最稳的做法是「用 --grab 抓一张 720x1280 的画面，直接从里面裁按钮」。
  如果得分只有 0.4~0.6，说明缩放对不上：换模板，或者调 --scale-min/--scale-max。

★ 怎么点（2026-09-23 实测，别改回去）:
  用 adb shell input tap 帧x 帧y，**不是鼠标**。
  实测：SetCursorPos 把光标精确移到了 PLAY 上（目标屏幕 (2365,1262)，屏幕 2560x1440，
  光标确实到位），连点两次游戏毫无反应；同一位置换成
      adb shell input tap 551 1203
  立刻生效。原因：LDPlayer 不吃 mouse_event 这种合成鼠标事件，窗口不在前台时更没戏。
  adb tap 注入的是真正的安卓触摸事件：不抢前台、不受 DPI 影响、坐标还简单 ——
  我们搜的那张 720x1280 的帧就是安卓画面本身，所以 **帧坐标 = 安卓坐标**。

★ 点完会确认再重试:
  点一下之后等 --verify-wait 秒（默认 2），再看一眼：按钮还在不在 + 有没有回到关卡。
  没点上就重试（--retries，默认 2 次）。日志里会写 [确认] 点击生效 / ⚠ 点了没反应。

怎么知道什么时候被捕的:
  用内存 hook（tools\state_server.py，采集脚本同一套）。
  hook 断了会自动重连（见 collector\hookstate.py），断线期间不会误点。
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2                      # noqa: E402
import numpy as np              # noqa: E402

from collector.capture import WindowCapture          # noqa: E402
from collector.config import Config                  # noqa: E402
from collector.hookstate import open_client          # noqa: E402

TPL_DIR = os.path.join(HERE, "templates")
DEFAULT_KEY_TPL = os.path.join(TPL_DIR, "key.png")
DEFAULT_PLAY_TPL = os.path.join(TPL_DIR, "play.png")
DEBUG_DIR = os.path.join(HERE, "runs", "找图")

# 雷电自带的 adb
ADB = r"D:\leidian\LDPlayer9\adb.exe"

# Windows 鼠标事件
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
SW_RESTORE = 9


# ======================================================================
#  中文路径安全的读写
#  ★ cv2.imread / cv2.imwrite 在 Windows 上走的是 ANSI 接口，路径里带中文
#    （比如 runs\找图\xxx.png、templates\钥匙.png）会**静默失败**：
#    imwrite 只返回 False，读回来是 None —— 排查起来非常费劲。
#    所以统一用 imencode / imdecode + numpy 的字节接口。
# ======================================================================
def imread_u(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_u(path: str, img) -> bool:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    buf.tofile(path)
    return True


# ======================================================================
#  模板匹配
# ======================================================================
def collect_templates(spec: str, prefix: str):
    """收集一个动作要用的模板图。

    spec 写了就用 spec（可逗号分隔多张）；留空就自动用 templates 下 <prefix>*.png
    —— 所以想加一张图，直接丢进 templates 按前缀命名就行：
        key*.png   -> 复活钥匙（可以有多张，任意一张命中就算找到）
        play*.png  -> PLAY（重开）
    """
    if spec and spec.strip():
        return [x.strip() for x in spec.split(",") if x.strip()]
    import glob as _glob
    return sorted(_glob.glob(os.path.join(TPL_DIR, prefix + "*.png")))


class Matcher:
    """**一组**模板图 + 一组缩放比例（缩放后的图预先算好，省得每帧重复缩放）。

    ★ 只挂一张图时，按钮样式/高亮一变就认不出来，所以支持多张，
      **任意一张命中就算找到**。
    """

    def __init__(self, paths, scales):
        if not paths:
            raise FileNotFoundError("没有可用的模板图")
        self.items = []
        self.thr_bonus = {}          # 每张模板的"加严量"（模板越素，加得越多）
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"找不到模板图：{p}")
            img = imread_u(p)
            if img is None:
                raise RuntimeError(f"读不了模板图（不是图片？）：{p}")
            name = os.path.basename(p)
            try:
                # ★ 模板越"平"（标准差越小），越容易在别的画面上也骗到高分
                #   （实测：一块纯蓝的图在随便一个蓝面板上都能得 0.911）。
                #   所以给每张模板单独抬高阈值，最多 +0.25。
                std = float(img.reshape(-1, img.shape[2]).std(axis=0).mean())
            except Exception:
                std = 99.0
            bonus = round(max(0.0, (40.0 - std) / 40.0) * 0.25, 3)
            self.thr_bonus[name] = bonus
            if std < 20.0:
                print(f"[图] ⚠ 模板 {name} 太素了（std={std:.1f}）：实测很容易在别的画面上"
                      f"骗到高分，已把它单独调严（阈值 +{bonus:.2f}）；"
                      "建议换一张带图标/边框/文字的图")
            tpls = []
            for s in scales:
                t = cv2.resize(img, None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
                tpls.append((float(s), t))
            self.items.append((name, img, tpls))
        self.last_name = ""

    @property
    def names(self):
        return [n for n, _, _ in self.items]

    def find(self, frame, base_thr: float):
        """匹配。返回 (是否算命中, (得分, 缩放, (x, y), (w, h)))。

        每张模板有自己的阈值：base_thr + thr_bonus[该模板]（越素的图要求越高）。
        命中哪张模板放在 self.last_name。
        """
        fh, fw = frame.shape[:2]
        best = (-1.0, 0.0, (0, 0), (0, 0))
        for name, _raw, tpls in self.items:
            for s, t in tpls:
                if t.shape[0] > fh or t.shape[1] > fw:
                    continue
                res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(res)
                if mx > best[0]:
                    best = (float(mx), s, ml, (t.shape[1], t.shape[0]))
                    self.last_name = name
        ok = best[0] >= base_thr + self.thr_bonus.get(self.last_name, 0.0)
        return ok, best

    def thr_of(self, name: str, base_thr: float) -> float:
        return base_thr + self.thr_bonus.get(name, 0.0)

    def per_template(self, frame):
        """每张模板各自的最好得分（--grab 用，找出哪张图没用）。"""
        fh, fw = frame.shape[:2]
        out = []
        for name, _raw, tpls in self.items:
            b = (-1.0, 0.0, (0, 0), (0, 0))
            for s, t in tpls:
                if t.shape[0] > fh or t.shape[1] > fw:
                    continue
                res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(res)
                if mx > b[0]:
                    b = (float(mx), s, ml, (t.shape[1], t.shape[0]))
            out.append((name, b))
        return out


# ======================================================================
#  adb 点击（首选）
# ======================================================================
class AdbTap:
    """用 adb shell input tap 点击安卓坐标。

    ★ 为什么不用鼠标（2026-09-23 实测）:
      SetCursorPos + mouse_event 把光标精确移到了 PLAY 上（屏 2560x1440，
      目标 (2365,1262)，GetCursorPos 确认到位），连点两次游戏毫无反应；
      同一位置换成 adb shell input tap 551 1203 立刻生效。
      LDPlayer 不吃合成鼠标事件，窗口不在前台时更没戏。
    """

    def __init__(self, exe: str = ADB):
        self.exe = exe
        self.serial = None

    def _run(self, args, timeout=20):
        try:
            p = subprocess.run([self.exe] + list(args), capture_output=True, timeout=timeout)
            return (p.returncode, p.stdout.decode("utf-8", "replace"),
                    p.stderr.decode("utf-8", "replace"))
        except Exception as e:
            return -1, "", str(e)

    def pick(self, tries: int = 4, gap: float = 2.0):
        """选一个能用的设备。必须带 -s：设备同时以 emulator-5554 和
        127.0.0.1:5555 出现时，不带 -s 的 adb shell 会直接失败。

        ★ 要多试几轮：adb server 刚起来那几秒，emulator-5554 常常先是 offline，
          LDPlayer 要过一会儿才注册好。等急了就会误报「没有设备」。
        """
        last = ""
        for _ in range(tries):
            _, out, _ = self._run(["devices"])
            last = out.strip()
            cands = [l.split()[0] for l in out.splitlines()
                     if l.strip() and "List of devices" not in l and "offline" not in l]
            for s in cands:
                rc, so, _ = self._run(["-s", s, "shell", "echo ok"], timeout=10)
                if rc == 0 and "ok" in so:
                    self.serial = s
                    return s
            self._run(["connect", "127.0.0.1:5555"])
            time.sleep(gap)
        print("      [adb] 没找到可用设备，最后一次 adb devices 输出：")
        for l in last.splitlines():
            print("            " + l)
        print("      [adb] 排查：模拟器开了没？设置里 ADB 调试开了没？")
        return None

    def tap(self, x: float, y: float):
        if self.serial is None and self.pick() is None:
            return False, "找不到可用的 adb 设备（模拟器没开？）"
        for attempt in (0, 1):
            rc, so, se = self._run(["-s", self.serial, "shell", "input", "tap",
                                    str(int(round(x))), str(int(round(y)))], timeout=15)
            if rc == 0:
                return True, (so + se).strip()
            # 设备掉了就重新挑一个再点一次
            self.serial = None
            if attempt == 0 and self.pick() is not None:
                continue
            return False, (so + se).strip()
        return False, "tap 失败"


# ======================================================================
#  点击：帧坐标 -> 安卓坐标（adb） / -> 屏幕坐标（鼠标，兜底）
# ======================================================================
class Clicker:
    def __init__(self, cap: WindowCapture, dry_run: bool = False,
                 backend: str = "adb", adb=None):
        self.cap = cap
        self.dry_run = dry_run
        self.backend = backend
        self.adb = adb

    def focus(self) -> bool:
        """把模拟器窗口拉到前台。

        鼠标点击是**全局**的：窗口不在前台时 SetCursorPos + mouse_event
        点到的是别的窗口，表现就是「点了但游戏没反应」。
        """
        u32 = ctypes.windll.user32
        hwnd = self.cap.hwnd
        if u32.GetForegroundWindow() == hwnd:
            return True
        try:
            u32.ShowWindow(hwnd, SW_RESTORE)
            u32.keybd_event(0x12, 0, 0, 0)          # 按住 Alt 绕过前台窗口限制
            u32.SetForegroundWindow(hwnd)
            u32.keybd_event(0x12, 0, 2, 0)
            time.sleep(0.18)
        except Exception:
            pass
        return u32.GetForegroundWindow() == hwnd

    def click_frame(self, fx: float, fy: float) -> None:
        """点帧坐标 (fx, fy)。

        搜的画面就是安卓画面本身（720x1280，1:1），所以：
            adb  -> 帧坐标 + crop 偏移 = 安卓坐标，直接 input tap
            鼠标 -> 帧坐标换算回客户区，再 ClientToScreen 到屏幕
        """
        cap = self.cap
        ax = cap.crop[0] + fx
        ay = cap.crop[1] + fy

        if self.dry_run:
            print(f"      [dry-run] 本应点击 帧({fx:.0f},{fy:.0f}) -> 安卓({ax:.0f},{ay:.0f})")
            return

        if self.backend == "adb":
            ok, msg = self.adb.tap(ax, ay)
            print(f"      -> adb tap 安卓({ax:.0f},{ay:.0f})  "
                  + ("成功" if ok else f"失败 {msg}"))
            return

        # ---- 鼠标兜底（实测 LDPlayer 点不动，留着备用）----
        cx = int(round(cap.off_x + ax))
        cy = int(round(cap.off_y + ay))
        u32 = ctypes.windll.user32
        pt = wintypes.POINT(0, 0)
        u32.ClientToScreen(cap.hwnd, ctypes.byref(pt))
        sx, sy = pt.x + cx, pt.y + cy
        ok = self.focus()
        if not ok:
            print("      ⚠ 窗口没拉到前台，这一下可能点到别的地方去（点完请看一眼）")
        # 记住原来的鼠标位置，点完放回去，免得你的鼠标被抢走
        old = wintypes.POINT(0, 0)
        u32.GetCursorPos(ctypes.byref(old))
        u32.SetCursorPos(sx, sy)
        time.sleep(0.04)
        u32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.06)
        u32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.05)
        u32.SetCursorPos(old.x, old.y)
        print(f"      -> 鼠标点击 屏幕({sx},{sy})（实测 LDPlayer 可能不响应）")


# ======================================================================
#  调试存图
# ======================================================================
def _save_debug(frame, matcher: Matcher, hit, tag: str) -> str:
    os.makedirs(DEBUG_DIR, exist_ok=True)
    vis = frame.copy()
    score, s, (x, y), (w, h) = hit
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.putText(vis, f"{matcher.last_name} {score:.3f}",
                (max(0, x), max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    p = os.path.join(DEBUG_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.png")
    imwrite_u(p, vis)
    return p


# ======================================================================
#  小工具
# ======================================================================
def in_play(hook) -> bool:
    """现在是不是「真的在关卡里活着」。没有 hook 就返回 False。"""
    st = hook.read() if hook is not None else None
    if not st or not st.get("ok"):
        return False
    return (int(st.get("inPlay", 0)) == 1 and int(st.get("alive", 0)) == 1
            and int(st.get("paused", 0)) == 0)


# ======================================================================
#  在一个时间窗内反复「找 -> 点 -> 确认」，点成返回 True
# ======================================================================
def find_and_click(cap, matcher: Matcher, tag: str, window: float,
                   clicker: Clicker, hook, args) -> str:
    """返回 "hit"（点成）/"abort"（虚惊一场，已经回到关卡）/"miss"（窗口内没点成）。"""
    deadline = time.time() + window
    tries = 0
    n = 0
    while time.time() < deadline:
        if in_play(hook):
            # 撞了一下但没被逮捕（只是招来警察），角色已经站起来了：
            # 别再耗完 5 秒 + 4 秒 —— 那段时间会听不到下一次真正的被捕。
            print(f"      [{tag}] 角色已经回到关卡，这次不是被捕（虚惊），收工")
            return "abort"
        frame = cv2.cvtColor(cap.grab(), cv2.COLOR_RGB2BGR)
        ok, hit = matcher.find(frame, args.thr)
        n += 1
        if args.verbose:
            print(f"      [{tag}] 第{n}次 最高 {hit[0]:.3f} ({matcher.last_name}, "
                  f"scale {hit[1]:.2f}) @{hit[2]}")
        if not ok:
            time.sleep(args.interval)
            continue

        x, y = hit[2]; w, h = hit[3]
        fx, fy = x + w / 2, y + h / 2
        p = _save_debug(frame, matcher, hit, tag)
        print(f"      [命中] {matcher.last_name} {hit[0]:.3f} @帧({fx:.0f},{fy:.0f})  存图 {p}")
        clicker.click_frame(fx, fy)
        tries += 1
        if args.dry_run:
            return True                 # 没真点，没什么可确认的

        # ★ 点完等一会儿，再回头确认到底点没点上
        time.sleep(args.verify_wait)
        f2 = cv2.cvtColor(cap.grab(), cv2.COLOR_RGB2BGR)
        ok2, h2 = matcher.find(f2, args.thr)
        gone = not ok2
        back = in_play(hook)
        if gone or back:
            print(f"      [确认] 点击生效（按钮得分 {h2[0]:.3f}"
                  + ("，已回到关卡" if back else "，按钮已消失") + "）")
            return "hit"
        print(f"      [确认] ⚠ 点了没反应（按钮还在，得分 {h2[0]:.3f}）"
              + ("，重试…" if tries < args.retries else "，不再重试"))
        if tries >= args.retries:
            return "miss"
    return "miss"


# ======================================================================
#  一次「被捕 -> 找图 -> 点击」流程
# ======================================================================
def revive_cycle(cap, key: Matcher, play: Matcher, clicker: Clicker, hook, args) -> str:
    # 阶段①：钥匙（最多 key_window 秒）
    r = find_and_click(cap, key, "钥匙", args.key_window, clicker, hook, args)
    if r == "hit":
        return "key"
    if r == "abort":
        return "none"
    # 阶段②：PLAY（最多 play_window 秒）
    print(f"      [钥匙] {args.key_window:.0f} 秒内没点成 -> 改找 PLAY")
    if find_and_click(cap, play, "PLAY", args.play_window, clicker, hook, args) == "hit":
        return "play"
    frame = cv2.cvtColor(cap.grab(), cv2.COLOR_RGB2BGR)
    _save_debug(frame, play, play.find(frame, args.thr)[1], "miss")
    print(f"      [放弃] {args.key_window:.0f}s 内没点成钥匙、{args.play_window:.0f}s 内没点成 PLAY，"
          "等下一次被捕")
    return "none"


# ======================================================================
#  纯抓图模式（用来裁模板 / 看当前得分）
# ======================================================================
def grab_once(cap, key: Matcher, play: Matcher, args) -> None:
    frame = cv2.cvtColor(cap.grab(), cv2.COLOR_RGB2BGR)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    shot = os.path.join(DEBUG_DIR, f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png")
    imwrite_u(shot, frame)
    print(f"[抓图] {shot}  ({frame.shape[1]}x{frame.shape[0]})")
    print("       建议：直接在这个图上裁按钮，另存到 templates\ 里覆盖，"
          "这样匹配是逐像素对齐的。")
    for m in (key, play):
        for name, hit in m.per_template(frame):
            mark = "命中" if hit[0] >= m.thr_of(name, args.thr) else "    "
            print(f"[得分] {name:<12} 最高 {hit[0]:.3f}  scale {hit[1]:.2f}"
                  f"  位置 {hit[2]}  尺寸 {hit[3]}"
                  f"  该图阈值 {m.thr_of(name, args.thr):.3f}  {mark}")
        ok, hit = m.find(frame, args.thr)
        if ok:
            x, y = hit[2]; w, h = hit[3]
            fx, fy = x + w / 2, y + h / 2
            print(f"       若点击：帧({fx:.0f},{fy:.0f}) -> 安卓"
                  f"({cap.crop[0] + fx:.0f},{cap.crop[1] + fy:.0f})  "
                  f"= adb shell input tap {cap.crop[0] + fx:.0f} {cap.crop[1] + fy:.0f}")
        _save_debug(frame, m, hit, "grab_" + os.path.splitext(m.last_name or m.names[0])[0])
    print(f"[提示] 阈值 {args.thr}；得分明显低于它说明模板缩放/内容对不上。")


# ======================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="找图复活：被捕后找钥匙/PLAY 并点击")
    ap.add_argument("--key-tpl", default="",
                    help="钥匙模板图，可逗号分隔多张；留空=自动用 templates/key*.png")
    ap.add_argument("--play-tpl", default="",
                    help="PLAY 模板图，可逗号分隔多张；留空=自动用 templates/play*.png")
    ap.add_argument("--thr", type=float, default=0.80, help="匹配得分阈值（默认 0.80）")
    ap.add_argument("--scale-min", type=float, default=0.85,
                    help="最小缩放（默认 0.85；模板本来就是原尺寸，附近几档足够）")
    ap.add_argument("--scale-max", type=float, default=1.25,
                    help="最大缩放（默认 1.25；模拟器换了分辨率才需要放宽）")
    ap.add_argument("--scale-step", type=float, default=0.05, help="缩放步长（默认 0.05）")
    ap.add_argument("--search-area", choices=("full", "game"), default="full",
                    help="搜哪块：full=整块安卓画面 720x1280（默认。PLAY 在最底下，只有它能找到）；"
                         "game=训练用的裁剪区 720x880")
    ap.add_argument("--key-window", type=float, default=5.0, help="找钥匙的最长秒数（默认 5）")
    ap.add_argument("--play-window", type=float, default=4.0, help="找 PLAY 的最长秒数（默认 4）")
    ap.add_argument("--interval", type=float, default=0.05, help="两次搜索之间歇多久")
    ap.add_argument("--cooldown", type=float, default=2.0, help="一次流程结束后多久内不再触发")
    ap.add_argument("--click-backend", choices=("adb", "mouse"), default="adb",
                    help="点击方式：adb=adb shell input tap（默认，实测只有它能点动 LDPlayer）；"
                         "mouse=SetCursorPos+mouse_event（实测点不动，留作备用）")
    ap.add_argument("--verify-wait", type=float, default=2.0,
                    help="点完之后等几秒回头确认点没点上（默认 2.0）")
    ap.add_argument("--retries", type=int, default=2, help="点不动时最多重点几次（默认 2）")
    ap.add_argument("--seconds", type=float, default=0.0, help="跑多少秒自动停（0=一直跑，自己 Ctrl+C）")
    ap.add_argument("--dry-run", action="store_true", help="只找不点，用来调阈值")
    ap.add_argument("--grab", action="store_true", help="抓一张画面 + 打印得分，然后退出")
    ap.add_argument("--now", action="store_true",
                    help="不等被捕，启动就立刻跑一轮找图然后退出（配 --dry-run 试参数最好用）")
    ap.add_argument("--verbose", action="store_true", help="每次搜索都打印最高得分")
    args = ap.parse_args()

    scales = list(np.arange(args.scale_min, args.scale_max + 1e-6, args.scale_step))
    key = Matcher(collect_templates(args.key_tpl, "key"), scales)
    play = Matcher(collect_templates(args.play_tpl, "play"), scales)
    print(f"[图] 钥匙模板 {key.names}")
    print(f"[图] PLAY模板 {play.names}")
    print(f"[图] 缩放 {args.scale_min}~{args.scale_max} 步长 {args.scale_step}"
          f" -> 每张 {len(scales)} 个尺度；阈值 {args.thr}")

    cfg = Config()
    if args.search_area == "full":
        # 整块安卓画面 —— PLAY 在 y≈1170，用 game_crop(0,200,720,880) 会把它裁掉
        crop = (0, 0, int(cfg.game_res[0]), int(cfg.game_res[1]))
    else:
        crop = cfg.game_crop
    cap = WindowCapture(cfg.window_keyword, cfg.game_res, crop, backend=cfg.backend)
    print(f"[窗] {cfg.window_keyword} hwnd={cap.hwnd} 客户区 {cap.client_w}x{cap.client_h} "
          f"偏移 ({cap.off_x},{cap.off_y}) 裁剪 {cap.crop}")

    if args.grab:
        grab_once(cap, key, play, args)
        return

    adb = AdbTap() if args.click_backend == "adb" else None
    clicker = Clicker(cap, dry_run=args.dry_run, backend=args.click_backend, adb=adb)
    if args.dry_run:
        print("[模式] dry-run：只找不点")
    print(f"[点] 方式 = {args.click_backend}"
          + ("（adb shell input tap）" if args.click_backend == "adb" else "（鼠标，实测点不动）")
          + f"；点完等 {args.verify_wait:.1f}s 确认，最多重试 {args.retries} 次")
    if args.click_backend == "adb":
        s = adb.pick()
        print(f"[点] adb 设备: {s if s else '★没找到★（模拟器没开？）'}")
    hits = {"key": 0, "play": 0, "none": 0}

    if args.now:
        print("[now] 不等被捕，立刻跑一轮找图…")
        hits[revive_cycle(cap, key, play, clicker, None, args)] += 1
        print(f"[统计] 点钥匙 {hits['key']} 次 / 点 PLAY {hits['play']} 次 / 没找到 {hits['none']} 次")
        cap.close()
        return

    # ---- 被捕检测：内存 hook ----
    hook = open_client(cfg.hook_state_host, cfg.hook_state_port)
    if hook is None:
        print("[错误] 拿不到内存判卷，就不知道什么时候被捕 —— 请先看 runs\state_server.log")
        return
    print("[就绪] 监视「被捕」中…（手动停止请按 Ctrl+C）")

    prev = None
    cooldown_until = 0.0
    t0 = time.time()
    try:
        while True:
            if args.seconds and time.time() - t0 >= args.seconds:
                print(f"\n[停止] 到达 --seconds {args.seconds:.0f}s")
                break
            st = hook.read()
            if not st or not st.get("ok") or "inPlay" not in st:
                time.sleep(0.1)
                continue
            inplay = int(st.get("inPlay", -1))
            alive = int(st.get("alive", -1))
            if prev is None:
                prev = (inplay, alive)
                continue
            p_in, p_al = prev
            prev = (inplay, alive)

            # 被捕的两种边沿：本局结束(inPlay 1->0) 或者 撞车(alive 1->0)
            arrested = (p_in == 1 and inplay == 0) or (p_al == 1 and alive == 0)
            if not arrested or time.time() < cooldown_until:
                time.sleep(0.05)
                continue

            now = time.strftime("%H:%M:%S")
            print(f"\n[{now}] [被捕] 检测到 inPlay {p_in}->{inplay} / alive {p_al}->{alive}，开始找图")
            res = revive_cycle(cap, key, play, clicker, hook, args)
            hits[res] += 1
            # 重新同步状态：流程里游戏可能已经回到关卡，也可能还在弹窗上，
            # 都不要再当成新的「被捕」触发一次
            prev = None
            cooldown_until = time.time() + args.cooldown
    except KeyboardInterrupt:
        print("\n[退出] Ctrl+C")
    finally:
        try:
            hook.close()
        except Exception:
            pass
        cap.close()
    print(f"[统计] 点钥匙 {hits['key']} 次 / 点 PLAY {hits['play']} 次 / 没找到 {hits['none']} 次")


if __name__ == "__main__":
    main()
