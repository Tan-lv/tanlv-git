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
    return found[0] if found else None


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
    def _refresh_geometry(self, content_offset=None) -> None:
        u32 = self._u32
        crect = wintypes.RECT()
        u32.GetClientRect(self.hwnd, ctypes.byref(crect))
        self.client_w, self.client_h = crect.right - crect.left, crect.bottom - crect.top
        if self.client_w < self.game_w or self.client_h < self.game_h:
            # ★ 2026-09-26：用户把模拟器调成横屏后撞到这里（客户区 1322x755 < 720x1280），
            #   报错只说"窗口太小"，看不出是横屏。这里把两种情况分开说清楚。
            hint = "请把模拟器窗口调大或不要缩放。"
            if self.client_w >= self.game_h and self.client_h >= self.game_w:
                # 窗口和配置的宽高是"转过来的"：一个横屏一个竖屏。
                hint = "\n        ".join([
                    f"窗口宽高和配置里的游戏画面是转过来的（配置 game_res={self.game_w}x{self.game_h}，"
                    f"窗口客户区 {self.client_w}x{self.client_h}）。二选一解决：",
                    "  · 模拟器竖屏 / 配置横屏 -> 把模拟器转成横屏（LDPlayer 工具栏旋转键）",
                    "  · 模拟器横屏 / 配置竖屏 -> 改 collector/config.py 的 game_res/game_crop，",
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
