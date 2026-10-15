# -*- coding: utf-8 -*-
"""Windows 窗口抓帧后端。

实测结论（本机 LDPlayer9 + RTX4060）:
  * PrintWindow(hwnd, dc, PW_RENDERFULLCONTENT) 在窗口被其它窗口遮挡时
    依然能抓到正确内容, 且与"屏幕像素"逐字节一致（mean|diff| = 0）,
    单帧耗时约 12 ms (~82 fps)。
  * GetDC(hwnd)+BitBlt 更快 (~2 ms, ~500 fps), 但窗口被遮挡时可能抓到
    遮挡窗口的画面, 因此只作为可选后端。

不依赖 ADB —— 本机 LDPlayer 的 adb shell / screencap 会长时间挂起。
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Optional, Tuple

import numpy as np

PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def find_window(keyword: str) -> Optional[int]:
    """按标题关键字查找可见窗口，返回 hwnd。"""
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if keyword in buf.value:
                found.append(hwnd)
        return True

    user32.EnumWindows(_cb, 0)
    if not found:
        return None
    # ★ 2026-10-15：雷电的**控制台窗口**标题里也含"雷电模拟器"（实测只有 237x39），
    #   而且它在 EnumWindows 里可能排在游戏窗口前面 —— 以前直接取 found[0]，
    #   结果抓到的是一块 237x39 的图片，自检报"画面完全没有变化"，很难查。
    #   现在按客户区面积从大到小排序，取**最大的那个**（游戏窗口一定是最大的）。
    def _area(hwnd):
        rc = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rc))
        return max(0, rc.right - rc.left) * max(0, rc.bottom - rc.top)

    ranked = sorted(found, key=_area, reverse=True)
    if len(ranked) > 1:
        sizes = []
        for h in ranked[:4]:
            rc = wintypes.RECT()
            user32.GetClientRect(h, ctypes.byref(rc))
            sizes.append("%dx%d" % (rc.right - rc.left, rc.bottom - rc.top))
        print("[窗口] 标题含 %r 的有 %d 个（%s），取最大的那个"
              % (keyword, len(ranked), "、".join(sizes)))
    return ranked[0]


# 同进程内重复开捕获时不必重复探测（客户区尺寸 + 后端没变就用缓存）
_GEOM_CACHE = {}


def detect_geometry(cfg, geo: dict = None, png: str = None):
    """探测窗口里游戏画面的实际位置和大小。

    返回 (game_res, game_crop, (off_x, off_y), 说明)；探测不出来返回 (None, None, None, 原因)。

    ★ 2026-10-14（用户要求"换分辨率/横竖屏不用改参数"）：
      game_res 用**探测到的游戏画面尺寸**，裁剪框由 cfg.crop_rel（相对比例）换算，
      并把探测到的左上角偏移一起返回 —— 因为实测 LDPlayer 的游戏画面**不是**
      "水平居中"（1322 宽的窗口里它从第 1 列开始，工具栏全在右边），
      以前按居中算会整体偏 20px。

    png 给定时不从窗口抓，而是读一张已保存的整窗截图（离线验证/排查用）。
    """
    from .geometry import (aspect_label, crop_from_rel, detect_game_rect, pick_rel)
    key = None
    if png:
        import cv2
        bgr = cv2.imdecode(np.fromfile(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None, None, "读不到截图 %s" % png
        full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        u32 = ctypes.windll.user32
        try:
            u32.SetProcessDPIAware()
        except Exception:
            pass
        hwnd = find_window(cfg.window_keyword)
        if not hwnd:
            return None, None, None, "找不到标题含 %r 的窗口" % cfg.window_keyword
        rc = wintypes.RECT()
        u32.GetClientRect(hwnd, ctypes.byref(rc))
        cw, ch = rc.right - rc.left, rc.bottom - rc.top
        # ★ 缓存键要带上"用哪个取景比例"：同一个进程里换模型/换取景文件时
        #   不能把上一次的裁剪框复用过来（否则"自动切换区域"会失效）。
        _gkey = tuple(round(float(v), 6) for v in (geo or {}).get("crop_rel") or ())
        _ckey = tuple(round(float(v), 6) for v in (getattr(cfg, "crop_rel", None) or ()))
        key = (cfg.window_keyword, cw, ch, cfg.backend, _gkey, _ckey)
        if key in _GEOM_CACHE:
            return _GEOM_CACHE[key]
        cap = WindowCapture(cfg.window_keyword, (cw, ch), (0, 0, cw, ch), backend=cfg.backend)
        try:
            full = cap.grab()                  # 整个客户区 RGB
        finally:
            cap.close()

    x, y, w, h = detect_game_rect(full)
    if w < 64 or h < 64:
        # ★ 2026-10-15：这个提示以前是"可能画面全黑/均色"，但实际最常见的原因是
        #   抓到了**雷电的控制台窗口**（它的标题里也含"雷电模拟器"，实测客户区只有 237x39，
        #   而且它在 EnumWindows 里可能排在游戏窗口前面）。说清楚，省得来回猜。
        return None, None, None, (
            "探测结果太小（%dx%d）—— 这个窗口的客户区只有这么大，通常是因为抓到了"
            "雷电的**控制台窗口**（标题同样含关键字），或者游戏窗口根本没打开。"
            "请把《地铁跑酷》所在的那个大窗口显示出来再试" % (w, h))
    # ★ geo 是"取景文件/模型自带"的记录；传入时它的比例最优先（见 geometry.pick_rel）
    rel, why = pick_rel(cfg, w, h, geo)
    crop = crop_from_rel(w, h, rel)
    info = "%s，%s，相对框 %s" % (aspect_label(w, h), why,
                                tuple(round(float(v), 4) for v in rel))
    res = ((int(w), int(h)), tuple(int(v) for v in crop), (int(x), int(y)), info)
    if key is not None:
        _GEOM_CACHE[key] = res
    return res


class WindowCapture:
    """从模拟器窗口抓取游戏画面（裁剪后的 RGB uint8 数组）。"""

    def __init__(self, keyword: str, game_res: Tuple[int, int], crop: Tuple[int, int, int, int],
                 backend: str = "printwindow", content_offset: Optional[Tuple[int, int]] = None):
        self.keyword = keyword
        self.game_w, self.game_h = int(game_res[0]), int(game_res[1])
        self.crop = tuple(int(v) for v in crop)
        self.backend = backend

        self._u32 = ctypes.windll.user32
        self._g32 = ctypes.windll.gdi32
        try:
            self._u32.SetProcessDPIAware()
        except Exception:
            pass

        self.hwnd = find_window(keyword)
        if not self.hwnd:
            raise RuntimeError(f"未找到标题包含 '{keyword}' 的窗口，请先启动模拟器")

        self._refresh_geometry(content_offset)
        self._setup_buffers()

        self.frames = 0
        self._t0 = time.time()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg, geo: dict = None, **kw):
        """按配置开捕获，并自动适配窗口里游戏画面的实际位置/大小。

        ★ 2026-10-14：这就是"换分辨率 / 横竖屏后一个参数都不用改"的入口。
          · cfg.auto_geometry=True（默认）：探测游戏画面 -> game_res 用探测值；
            裁剪框用 cfg.crop_rel（没写就按横/竖屏预设）换算；并把探测到的
            左上角偏移传给捕获（连"游戏画面没水平居中"的情况也对得上）。
          · 探测失败 或 auto_geometry=False：退回 cfg.game_res / cfg.game_crop。

        用法: WindowCapture.from_config(cfg, backend=cfg.backend)
        """
        if not getattr(cfg, "auto_geometry", True):
            return cls(cfg.window_keyword, cfg.game_res, cfg.game_crop, **kw)
        res, crop, off, info = detect_geometry(cfg, geo)
        if res is None:
            print("[几何] 自动探测不可用（%s），按配置取景：%s %s"
                  % (info, tuple(cfg.game_res), tuple(cfg.game_crop)))
            return cls(cfg.window_keyword, cfg.game_res, cfg.game_crop, **kw)
        print("[几何] 自动对准：游戏画面 (%d, %d) %dx%d   裁剪 %s   (%s)"
              % (off[0], off[1], res[0], res[1], crop, info))
        kw.setdefault("content_offset", off)
        return cls(cfg.window_keyword, res, crop, **kw)

    @classmethod
    def from_config_full(cls, cfg, geo: dict = None, **kw):
        """按配置开捕获，但取**整块游戏画面**（找图复活找 PLAY/钥匙要用整屏）。

        同样自动适配分辨率/横竖屏；探测失败就退回配置的 game_res + 整屏裁剪。
        """
        res, _crop, off, info = detect_geometry(cfg, geo)
        if res is None:
            res = tuple(int(v) for v in cfg.game_res)
            print("[几何] 自动探测不可用（%s），用配置的整屏 %dx%d" % (info, res[0], res[1]))
        else:
            print("[几何] 自动对准（整屏）：游戏画面 (%d, %d) %dx%d" % (off[0], off[1], res[0], res[1]))
            kw.setdefault("content_offset", off)
        return cls(cfg.window_keyword, res, (0, 0, res[0], res[1]), **kw)

    # ------------------------------------------------------------------
    def _refresh_geometry(self, content_offset=None) -> None:
        u32 = self._u32
        crect = wintypes.RECT()
        u32.GetClientRect(self.hwnd, ctypes.byref(crect))
        self.client_w, self.client_h = crect.right - crect.left, crect.bottom - crect.top
        if self.client_w < self.game_w or self.client_h < self.game_h:
            # ★ 2026-09-26：用户把模拟器调成横屏后撞到这里（客户区 1322x755 < 720x1280），
            #   报错只说"窗口太小"，看不出是横屏。这里把两种情况分开说清楚。
            # ★ 2026-10-15：最省事的排查线索放最前面 —— 抓错窗口（雷电控制台小窗）比
            #   "窗口太小"常见得多，而且报错里原来的话完全看不出来。
            hint = ("最可能的原因：抓到的是**雷电的控制台小窗**（标题也含关键字，实测只有 "
                    "237x39 这种），或者游戏窗口没打开。请把《地铁跑酷》那个大窗口显示出来"
                    "（别最小化）再试；确认不是这种情况后，再把模拟器窗口调大或不要缩放。")
            if self.client_w >= self.game_h and self.client_h >= self.game_w:
                # 窗口和配置的宽高是"转过来的"：一个横屏一个竖屏。
                hint = "\n        ".join([
                    f"窗口宽高和配置里的游戏画面是转过来的（配置 game_res={self.game_w}x{self.game_h}，"
                    f"窗口客户区 {self.client_w}x{self.client_h}）。二选一解决：",
                    "  · 模拟器竖屏 / 配置横屏 -> 把模拟器转成横屏（LDPlayer 工具栏旋转键）",
                    "  · 模拟器横屏 / 配置竖屏 -> 改 core/config.py 的 game_res/game_crop，",
                    "    或命令行临时覆盖：--game-res 1280,720 --game-crop 345,0,590,720",
                    "横屏取景用 tools\\取景标定.py 标定；裁剪框比例务必保持 720:880（= 240:293）。",
                ])
            raise RuntimeError(
                f"窗口客户区 {self.client_w}x{self.client_h} 小于游戏画面 {self.game_w}x{self.game_h}，"
                + hint)

        if content_offset is None:
            # LDPlayer 的安卓画面在客户区内"水平居中、底部对齐"（顶部是模拟器工具条）
            self.off_x = (self.client_w - self.game_w) // 2
            self.off_y = self.client_h - self.game_h
        else:
            self.off_x, self.off_y = int(content_offset[0]), int(content_offset[1])

    def _setup_buffers(self) -> None:
        g32 = self._g32
        self._bi = _BITMAPINFOHEADER()
        self._bi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        self._bi.biWidth = self.client_w
        self._bi.biHeight = -self.client_h      # 负数 = 自上而下
        self._bi.biPlanes = 1
        self._bi.biBitCount = 32
        self._bi.biCompression = 0
        self._buf = ctypes.create_string_buffer(self.client_w * self.client_h * 4)

        self._hdc = self._u32.GetDC(0)
        self._mdc = g32.CreateCompatibleDC(self._hdc)
        self._bmp = g32.CreateCompatibleBitmap(self._hdc, self.client_w, self.client_h)
        g32.SelectObject(self._mdc, self._bmp)
        if self.backend == "bitblt":
            self._wdc = self._u32.GetDC(self.hwnd)

    # ------------------------------------------------------------------
    def grab(self) -> np.ndarray:
        """抓一帧，返回裁剪后的 RGB uint8 数组 (crop_h, crop_w, 3)。"""
        g32 = self._g32
        if self.backend == "printwindow":
            self._u32.PrintWindow(self.hwnd, self._mdc, PW_RENDERFULLCONTENT)
        else:
            g32.BitBlt(self._mdc, 0, 0, self.client_w, self.client_h, self._wdc, 0, 0, SRCCOPY)
        g32.GetDIBits(self._mdc, self._bmp, 0, self.client_h, self._buf,
                      ctypes.byref(self._bi), 0)

        full = np.frombuffer(self._buf, dtype=np.uint8).reshape(
            self.client_h, self.client_w, 4)[:, :, :3][:, :, ::-1]

        ox, oy = self.off_x, self.off_y
        cx, cy, cw, ch = self.crop
        y0, x0 = oy + cy, ox + cx
        out = full[y0:y0 + ch, x0:x0 + cw]
        if out.shape[0] != ch or out.shape[1] != cw:
            raise RuntimeError(
                f"裁剪越界: 期望 {cw}x{ch}, 实际 {out.shape[1]}x{out.shape[0]}；"
                "请检查 game_res / game_crop / 窗口是否被缩放")
        self.frames += 1
        # 返回副本, 否则下一帧会覆盖同一块内存
        return np.ascontiguousarray(out)

    def fps(self) -> float:
        dt = time.time() - self._t0
        return self.frames / dt if dt > 0 else 0.0

    def reset_fps(self) -> None:
        self.frames = 0
        self._t0 = time.time()

    def close(self) -> None:
        try:
            self._g32.DeleteObject(self._bmp)
            self._g32.DeleteDC(self._mdc)
            self._u32.ReleaseDC(0, self._hdc)
            if self.backend == "bitblt":
                self._u32.ReleaseDC(self.hwnd, self._wdc)
        except Exception:
            pass
