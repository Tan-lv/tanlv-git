# -*- coding: utf-8 -*-
"""速度估计：从连续帧估计游戏滚动速率（像素/秒）。

为什么需要
------------------------------------------------------------------
地铁跑酷的速度随存活时间上升，同一个障碍在 1x 和 3x 下画面完全不同：

  * 连续帧输入本身能把速度"编码"进去（窗口内的视差就是速度），
    所以模型原则上能自己看出来 —— 前提是训练数据覆盖了速度区间。
  * 但如果你只在每局前 30 秒采集，模型就永远没见过高速画面，
    上线后一进高速段就废了。这是最容易翻车的地方。

这个模块提供三件事：
  1) 采集时实时显示当前速度（状态栏）
  2) 每个样本落盘时记录 speed_px_s，采完可以统计速度覆盖度
  3) 训练时按速度做重加权，避免慢速段样本把模型带偏

方法：相邻帧的相位相关（phase correlation）。纹理整体往外扩散，
底部中央的位移与前进速度成正比。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# 注意：cv2 只在真正估算速度时才需要，所以**不要**在模块顶层 import ——
# 训练端（collector/trainer.py）会 import 这个模块做速度统计，但训练端不一定装了 opencv。
import numpy as np

# 默认只取画面上部 72%（排除底部的角色和 HUD，它们不随速度移动）
DEFAULT_ROI = (0.0, 0.0, 1.0, 0.72)


def estimate_scroll_rate(frames: Sequence[np.ndarray],
                         roi: Tuple[float, float, float, float] = DEFAULT_ROI,
                         stride: int = 1) -> Tuple[float, float]:
    """估计相邻帧的整体滚动位移。

    frames: RGB 或灰度图序列
    返回 (像素/帧, 相位相关置信度)。置信度长期低于 0.2 说明画面几乎没纹理或没动。
    """
    if len(frames) < 2:
        return 0.0, 0.0
    import cv2          # 惰性导入，见文件顶部说明
    h, w = frames[0].shape[:2]
    x0, y0 = int(roi[0] * w), int(roi[1] * h)
    x1, y1 = int(roi[2] * w), int(roi[3] * h)
    prev: Optional[np.ndarray] = None
    vals: List[float] = []
    resps: List[float] = []
    for k, f in enumerate(frames):
        if k % max(1, stride):
            continue
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
        g = np.ascontiguousarray(g[y0:y1, x0:x1], dtype=np.float32)
        if prev is not None and g.shape == prev.shape:
            try:
                (dx, dy), r = cv2.phaseCorrelate(prev, g)
            except cv2.error:
                prev = g
                continue
            vals.append(float(np.hypot(dx, dy)))
            resps.append(float(r))
        prev = g
    if not vals:
        return 0.0, 0.0
    return float(np.median(vals)), float(np.median(resps))


def speed_px_per_sec(frames: Sequence[np.ndarray], times_ms: Sequence[float],
                     **kw) -> Tuple[float, float]:
    """返回 (像素/秒, 置信度)。times_ms 是每帧相对某基准的毫秒时间戳。"""
    px_per_frame, resp = estimate_scroll_rate(frames, **kw)
    if len(times_ms) >= 2:
        dt = (times_ms[-1] - times_ms[0]) / 1000.0
        n = len(times_ms) - 1
        if dt > 0:
            return px_per_frame * n / dt, resp
    return px_per_frame * 30.0, resp


# ======================================================================
#  覆盖度
# ======================================================================
def speed_histogram(speeds: Sequence[float], bins: int = 10, robust: bool = True):
    """返回 (边界, 每档计数)。

    ★ 2026-09-25：默认**抗离群**。相位相关偶尔会估出 250 px/s 这种离谱值，
      等宽分档会被它整个拉歪（8 档全挤在低速段），看起来就像"高速段是 0%"。
      现在按 [p2, p98] 定范围、超出的夹到边界，分档才反映真实分布。
    """
    s = np.asarray([x for x in speeds if x > 0], dtype=np.float64)
    if s.size == 0:
        return np.array([0.0, 1.0]), np.array([0])
    if robust and s.size >= 20:
        lo, hi = float(np.percentile(s, 2)), float(np.percentile(s, 98))
        if hi > lo:
            s = np.clip(s, lo, hi)
            hist, edges = np.histogram(s, bins=bins, range=(lo, hi))
            return edges, hist
    hist, edges = np.histogram(s, bins=bins)
    return edges, hist


def speed_weights(speeds: Sequence[float], bins: int = 10, strength: float = 1.0
                  ) -> np.ndarray:
    """按速度分布做重加权，让训练时各速度段被采到的概率更均匀。

    strength=0 表示不重加权，1 表示完全拉平。返回每个样本的权重。
    """
    s = np.asarray(speeds, dtype=np.float64)
    w = np.ones_like(s)
    ok = s > 0
    if ok.sum() < 2:
        return w
    hist, edges = np.histogram(s[ok], bins=bins)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(hist) - 1)   # 每个样本落在哪一档
    dens = hist / max(hist.mean(), 1e-9)                            # 每一档的相对密度 (bins,)
    per = dens[idx]                                                 # 展开到每个样本 (n,)
    per = np.where(per > 0, per, 1.0)                               # 空档按 1 处理
    raw = 1.0 / np.maximum(per, 1e-9)
    raw = raw / raw.mean()
    w = np.where(ok, 1.0 + strength * (raw - 1.0), 1.0)
    return np.clip(w, 0.1, 10.0)


def dataset_weights(samples: Sequence[dict], negatives: Sequence[tuple],
                    strength: float, bins: int = 10) -> np.ndarray:
    """给 MultiFrameDataset 用的逐条权重。

    真实样本按自身的速度加权；合成的 N 类负样本跟随它来源的那个按键样本
    （负样本是从该样本的后置窗口切出来的，速度自然一样）。
    """
    sp = [float(m.get("speed_px_s", 0.0) or 0.0) for m in samples]
    w = speed_weights(sp, bins=bins, strength=strength)
    full = list(w)
    for src, _end in negatives:
        full.append(w[src])
    return np.asarray(full, dtype=np.float64)


def describe(speeds: Sequence[float]) -> str:
    s = np.asarray([x for x in speeds if x > 0], dtype=np.float64)
    if s.size == 0:
        return "(没有有效速度数据)"
    return (f"n={s.size}  最小 {s.min():.0f}  5% {np.percentile(s,5):.0f}  "
            f"中位 {np.median(s):.0f}  95% {np.percentile(s,95):.0f}  最大 {s.max():.0f} px/s")
