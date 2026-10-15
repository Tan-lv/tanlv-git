# -*- coding: utf-8 -*-
"""图片读写（中文路径安全）。

★ 为什么需要它：cv2.imread / cv2.imwrite 在 Windows 上走的是 ANSI 接口，
  路径里带中文（例如 `runs\\横屏_v0\\attention\\`、`dataset\\迭代数据\\`）时
  **静默失败** —— imwrite 返回 False、不抛异常、文件也不生成，而调用方照样打印
  "已保存"。2026-10-14 又踩了一次：5_评估.py --cam 出图到 runs\\横屏_v0\\attention
  一张都没写进去（实测：中文路径 imwrite -> False，英文路径 -> True）。

  这里统一用 imencode + ndarray.tofile 写、np.fromfile + imdecode 读，
  并且**返回是否成功**，让调用方能明确报错而不是假装成功。
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def imwrite_u(path: str, img: np.ndarray, quality: int = 95) -> bool:
    """写图片到（可能含中文的）路径。成功返回 True，失败会打印原因并返回 False。"""
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        ext = os.path.splitext(path)[1] or ".png"
        params = []
        if ext.lower() in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            print(f"[存图失败] 编码失败（扩展名 {ext}？）: {path}")
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        print(f"[存图失败] {path}  ({e})")
        return False


def imread_u(path: str):
    """读图片（中文路径安全）。读不到返回 None。"""
    if not os.path.isfile(path):
        return None
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
