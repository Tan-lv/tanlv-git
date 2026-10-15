# -*- coding: utf-8 -*-
r"""游戏状态检测：自动判断"现在是不是在跑酷"，用来开关录制。

为什么不能用 Frida
------------------------------------------------------------------
雷电用 libhoudini 把 ARM 翻译成 x86 执行，libil2cpp.so 的映射是 r--p（不可执行），
Frida 看不到 ARM 模块、调 ARM 导出函数会挂死、Interceptor 挂钩无效。
（见 C:\Users\Tanlv\Desktop\ds_hr\ld_apk\out\REPORT_subwaysurf_infinite_jump_fix.md）
所以只能从画面判断。

判据
------------------------------------------------------------------
跑酷中整条跑道在向前滚动 —— 相邻帧差异很大（实测中位 17~40）；
菜单、弹窗、"Continue?" 死亡界面、大厅都是静态或只有局部小动画（差异 < 3）。

    状态机（带迟滞，避免抖动）：
        静止 --连续 state_on_ms 都超过 on 阈值-->  跑酷中
        跑酷 --连续 state_off_ms 都低于 off 阈值-->  静止

    一局结束（跑酷 -> 静止）时，游戏实际上已经在几秒前就撞死了
    （撞到障碍 -> 被警察抓住 -> 弹出 Continue?），所以要把这段时间
    采集到的样本**回滚删掉** —— 这正是"警察出现那段不记录"。
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

# 只看画面上部：底部是角色和按键提示，不随速度滚动，会干扰判断
MOTION_ROI = (0.0, 0.0, 1.0, 0.72)


def frame_gray(frame: np.ndarray) -> np.ndarray:
    """降采样灰度图（用于算帧间差异）。"""
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    x0, y0 = int(MOTION_ROI[0] * w), int(MOTION_ROI[1] * h)
    x1, y1 = int(MOTION_ROI[2] * w), int(MOTION_ROI[3] * h)
    return cv2.resize(g[y0:y1, x0:x1], (64, 64), interpolation=cv2.INTER_AREA)


def motion_between(prev: Optional[np.ndarray], cur: np.ndarray) -> float:
    """两张 64x64 灰度图的平均绝对差（0~255）。"""
    if prev is None:
        return 0.0
    return float(np.abs(cur.astype(np.int16) - prev.astype(np.int16)).mean())


class RunStateDetector:
    """把连续帧的"帧间差异"变成一个二值状态：running / idle。"""

    def __init__(self, on_threshold: float = 8.0, off_threshold: float = 3.0,
                 on_ms: int = 500, off_ms: int = 1200, start_state: str = "idle"):
        self.on_th = float(on_threshold)
        self.off_th = float(off_threshold)
        self.on_s = on_ms / 1000.0
        self.off_s = off_ms / 1000.0
        self.state = start_state
        self.motion = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._above_since: Optional[float] = None
        self._below_since: Optional[float] = None

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, now: Optional[float] = None) -> str:
        """喂一帧，返回当前状态（"running" / "idle"）。"""
        now = time.time() if now is None else now
        g = frame_gray(frame)
        self.motion = motion_between(self._prev_gray, g)
        self._prev_gray = g
        return self.update_motion(self.motion, now)

    def update_motion(self, motion: float, now: float) -> str:
        self.motion = motion
        if self.state == "running":
            if motion < self.off_th:
                if self._below_since is None:
                    self._below_since = now
                elif now - self._below_since >= self.off_s:
                    self.state = "idle"
                    self._below_since = None
                    self._above_since = None
            else:
                self._below_since = None
        else:
            if motion > self.on_th:
                if self._above_since is None:
                    self._above_since = now
                elif now - self._above_since >= self.on_s:
                    self.state = "running"
                    self._above_since = None
                    self._below_since = None
            else:
                self._above_since = None
        return self.state

    @property
    def is_running(self) -> bool:
        return self.state == "running"
