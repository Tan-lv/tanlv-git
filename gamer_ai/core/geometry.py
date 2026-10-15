# -*- coding: utf-8 -*-
r"""自动探测游戏画面在窗口里的位置，并把“相对裁剪框”换算成绝对裁剪框。

为什么需要它：
  ① 以前是用“水平居中 + 底部对齐”猜游戏画面的位置（off_x=(客户区宽-game_w)//2）。
     实测 1322x755 的窗口：猜出来 off_x=21，但游戏画面其实从第 1 列开始
     （左边只有 1px 边框，工具栏全在右边）-> 裁剪区实际右移了 20px。
  ② 分辨率/横竖屏一变，写死的 game_res/game_crop 就不对了。

做法：游戏画面是“彩色的”，模拟器工具栏是“灰黑扁平的”，
  所以用 HSV 的（饱和度>60 且 亮度>60）做掩码，找出画面的四条边。
  得到的尺寸就是真实的“安卓屏幕在窗口里占多少像素”
  （= 新的 game_res），再从“相对裁剪框”算出绝对裁剪框。
  这样同一份配置在 720p / 1080p / 任意分辨率下都能取到相同的画面取景
  （落盘帧始终是 240x293，前提是裁剪框比例锁死）。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# 常见安卓屏幕比例（只用来打日志、提醒）
STD_ASPECTS = {
    "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4,
    "1:1": 1.0, "20:9": 20 / 9, "9:20": 9 / 20, "16:10": 16 / 10, "10:16": 10 / 16,
    "3:2": 3 / 2, "2:3": 2 / 3, "21:9": 21 / 9, "9:21": 9 / 21,
}

# 相对裁剪框预设 (x0, y0, x1, y1) —— 分数，相对于游戏画面
#   竖屏：整宽，竖直取中间 68.75%   = 旧的 (0, 200, 720, 880) / 720x1280
PRESET_PORTRAIT = (0.0, 0.15625, 1.0, 0.84375)
#   横屏：整高，水平取中间 46.1%    = 新的 (345, 0, 590, 720) / 1280x720
PRESET_LANDSCAPE = (0.2695, 0.0, 0.7305, 1.0)


def colorful_mask(rgb: np.ndarray) -> np.ndarray:
    """彩色像素掩码：游戏画面是彩色的，模拟器工具栏是灰黑扁平的。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 60)


def _first_sustained(prof: np.ndarray, limit: int, frac: float, thick: int):
    """从 0 往后找第一个“连续 thick 个都 >= frac”的位置（找不到返回 None）。"""
    i = 0
    while i < limit:
        if prof[i:i + thick].size == thick and prof[i:i + thick].min() >= frac:
            return i
        i += 1
    return None


def detect_game_rect(rgb: np.ndarray, min_frac: float = 0.35, min_thick: int = 8,
                     snap: bool = True) -> Tuple[int, int, int, int]:
    """从整窗截图里找出游戏画面矩形，返回 (x0, y0, w, h)。

    实测校验（1322x755 的 LDPlayer 横屏）：得到 (1, 34, 1280, 720)，和真值 1280x720 一致。

    三个经验（都是实测得来的）：
      1) 顶部工具栏很好认：彩色占比从 ~0 突然跳到 >0.9 的那一行就是画面顶部；
      2) 左右边界（右侧工具栏）也可靠；左边通常只有 1px 边框；
      3) 底部不能靠“彩色”判断！游戏自己的底部常常是深色条（菜单的 PLAY 条），
         会把高度算小（实测曾把 720 算成 635）。而 LDPlayer 的
         安卓画面是**底部对齐**的，所以高度直接用“客户区底 - y0”，
         再用标准屏幕比例做吸附校正。
    """
    mask = colorful_mask(rgb).astype(np.float32)
    ch, cw = mask.shape[:2]
    rowf = mask.mean(axis=1)

    y0 = _first_sustained(rowf, ch, min_frac, min_thick)
    if y0 is None:
        return 0, 0, cw, ch                     # 认不出来 -> 整窗

    #   列的阈值要比行宽松：画面最右边几列常常是深色风景
    #   （实测用 0.25 会少算 10px），而模拟器工具栏的彩色占比几乎是 0，所以降到 0.15 很安全。
    sub = mask[y0:, :]
    colf = sub.mean(axis=0)
    x0 = _first_sustained(colf, cw, 0.15, 4)
    x0 = 0 if x0 is None else x0
    x1_rev = _first_sustained(colf[::-1], cw, 0.15, 4)
    x1 = (cw - x1_rev) if x1_rev is not None else cw
    if x1 <= x0 + min_thick:
        x1 = cw
    w = x1 - x0
    h = ch - y0                                 # 底部对齐：安卓画面直接到客户区底

    if snap and w > 0 and h > 0:
        a = nearest_std_aspect(w, h)
        if a:                                   # 用标准比例校正高度（高度是推出来的，宽度是量出来的）
            h2 = int(round(w / a))
            if y0 + h2 <= ch and abs(h2 - h) <= max(6, 0.05 * h):
                h = h2
    return int(x0), int(y0), int(w), int(h)


def nearest_std_aspect(w: int, h: int, tol: float = 0.03):
    """找最接近的标准屏幕比例值；相差太远就返回 None。"""
    if h <= 0:
        return None
    r = w / float(h)
    best, bd = None, 9e9
    for val in STD_ASPECTS.values():
        d = abs(val - r)
        if d < bd:
            best, bd = val, d
    return best if best and (bd / max(r, 1e-6)) <= tol else None


def aspect_label(w: int, h: int) -> str:
    """给尺寸贴一个最接近的屏幕比例名（打日志用）。"""
    if h <= 0:
        return "?"
    r = w / float(h)
    best, bd = "?", 9e9
    for name, val in STD_ASPECTS.items():
        d = abs(val - r)
        if d < bd:
            best, bd = name, d
    return "%s (实测 %.3f%s)" % (best, r, "" if bd < 0.03 else " 偏离较大")


def preset_for(game_w: int, game_h: int):
    """横屏还是竖屏？返回对应的相对裁剪桏预设。"""
    return PRESET_LANDSCAPE if game_w >= game_h else PRESET_PORTRAIT


def crop_from_rel(game_w: int, game_h: int, rel, frame_w: int = 240,
                  frame_h: int = 293) -> Tuple[int, int, int, int]:
    """相对裁剪框 -> 绝对裁剪框 (x, y, w, h)，并把宽高比锁死在 frame_w:frame_h。

    为什么要锁比例：落盘帧长宽比 = 裁剪框长宽比（Config.frame_h 是算出来的），
    而模型前处理是“短边缩到 224 + 中心裁 224”，比例一变有效视野就变了。
    所以不管游戏画面是几百还是几千像素，这里都会把裁剪框强制成 240:293。
    """
    fx0, fy0, fx1, fy1 = [float(v) for v in rel]
    x = int(round(fx0 * game_w))
    x1 = int(round(fx1 * game_w))
    y = int(round(fy0 * game_h))
    y1 = int(round(fy1 * game_h))
    w = max(8, x1 - x)
    hh = max(8, y1 - y)
    target = frame_w / float(frame_h)          # 宽 / 高
    if w / float(hh) > target:                 # 太宽 -> 收窄宽度
        w = max(8, int(round(hh * target)))
        cx = (x + x1) // 2
        x = max(0, min(game_w - w, cx - w // 2))
    else:                                      # 太高 -> 收短高度（保持竖直中心）
        hh = max(8, int(round(w / target)))
        cy = (y + y1) // 2
        y = max(0, min(game_h - hh, cy - hh // 2))
    if x + w > game_w:
        x = max(0, game_w - w)
    if y + hh > game_h:
        y = max(0, game_h - hh)
    return int(x), int(y), int(w), int(hh)


# ======================================================================
#  取景文件：geometry.json —— 区域记在文件里，而不是写死在代码里
# ----------------------------------------------------------------------
#  为什么这么做（用户 2026-10-15 提出）：
#    区域是"跟着模型走"的 —— 用 A 区域训出来的模型，部署时就该用 A 区域。
#    写进 core/config.py 是全局的，训练了不同区域的多个模型时会互相打架。
#
#  文件里记三样东西：
#    game_res  当时采集/标定时的游戏画面尺寸（设备分辨率）
#    game_crop 当时的绝对裁剪框（x, y, w, h）—— 给人看、也用来核对
#    crop_rel  相对比例（x0, y0, x1, y1）—— 真正被后续步骤使用的那一份，
#              因为它换分辨率/横竖屏都能自动换算
#
#  三处会用这个文件：
#    dataset/<会话>/geometry.json   采集时写了什么（溯源用，真实记录）
#    gamer_ai/geometry.json            你自己用自检的弹窗选的（项目默认取景）
#    runs/<模型>/model_meta.json    训练时把取景一起记进模型里
#    部署（6_自动玩.py）优先用**模型自带**的那个 -> 不同区域的模型自动切对
# ======================================================================
GEOMETRY_NAME = "geometry.json"


def project_geometry_path() -> str:
    """项目级取景文件：gamer_ai/geometry.json。"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # -> gamer_ai/
    return os.path.join(here, GEOMETRY_NAME)


def make_geometry(crop_rel, game_res, game_crop=None, offset=None, client=None, note=""):
    """组装一个取景记录（dict），字段就是上面注释里说的那些。"""
    rel = tuple(round(float(v), 6) for v in crop_rel)
    geo = {
        "crop_rel": list(rel),
        "game_res": [int(game_res[0]), int(game_res[1])],
        "note": note,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if game_crop is not None:
        geo["game_crop"] = [int(v) for v in game_crop]
    if offset is not None:
        geo["window_offset"] = [int(offset[0]), int(offset[1])]
    if client is not None:
        geo["window_client"] = [int(client[0]), int(client[1])]
    return geo


def save_geometry(geo: dict, path: str = None) -> str:
    """写取景文件（默认写项目级 gamer_ai/geometry.json）。返回写到的路径。"""
    p = path or project_geometry_path()
    d = os.path.dirname(os.path.abspath(p))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(geo, fh, ensure_ascii=False, indent=2)
    return p


def load_geometry(path: str = None):
    """读取取景文件；没有/读坏返回 None。"""
    p = path or project_geometry_path()
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            geo = json.load(fh)
    except Exception:
        return None
    if not geo.get("crop_rel"):
        return None
    return geo


def deploy_geometry(meta: dict = None, extra: dict = None, extra_src: str = None):
    """部署时要用的取景。优先级：

      ① 模型自带 model_meta.json 里的 geometry —— 训练这块区域就用这块
      ② extra（区域档案 gamer_ai/profiles/<名字>/geometry.json）
      ③ 项目 geometry.json（自检弹窗最近一次选的）
      ④ 都没有 -> None（capture 里退回 config.py 的 crop_rel / 横竖屏预设）

    ★ 放在这里而不是写死在 6_自动玩.py 里，是为了能被单独测试 ——
      也正是这一条实现"训练不同区域的模型，自动切到各自正确的区域"。
    """
    geo = dict((meta or {}).get("geometry") or {})
    if geo.get("crop_rel"):
        geo["_source"] = "模型自带 model_meta.json"
        return geo
    if extra and extra.get("crop_rel"):
        g = dict(extra)
        g["_source"] = extra_src or "档案 geometry.json"
        return g
    g = load_geometry()
    if g:
        g["_source"] = "项目 geometry.json"
        return g
    return None


def describe(geo: dict) -> str:
    """一行字描述这个取景（打日志用）。"""
    if not geo:
        return "(无取景文件)"
    rel = tuple(round(float(v), 4) for v in geo.get("crop_rel", ()))
    gr = geo.get("game_res")
    gc = geo.get("game_crop")
    s = "crop_rel=%s" % (rel,)
    if gr:
        s += "  标定于 %dx%d" % (gr[0], gr[1])
    if gc:
        s += "  绝对框=%s" % (tuple(gc),)
    if geo.get("saved_at"):
        s += "  (%s)" % geo["saved_at"]
    return s


def pick_rel(cfg, game_w: int, game_h: int, geo: dict = None):
    """决定用哪个相对裁剪框。优先级：

      ① 显式传入的 geo（部署时是**模型自带**的取景；采集时是项目 geometry.json）
      ② config.py 里的 crop_rel（手工写死的）
      ③ 按横屏/竖屏用预设

    ★ 这就是"换分辨率/横竖屏不用改参数"的关键：存的是比例，不是像素。
    返回 (相对框, 说明文字)。
    """
    if geo and geo.get("crop_rel"):
        return tuple(float(v) for v in geo["crop_rel"]), "取景文件（%s）" % (
            geo.get("_source") or "geometry.json",)
    rel = getattr(cfg, "crop_rel", None)
    if rel:
        return tuple(float(v) for v in rel), "config.py 里的 crop_rel"
    if game_w >= game_h:
        return PRESET_LANDSCAPE, "横屏预设"
    return PRESET_PORTRAIT, "竖屏预设"


def rel_from_crop(game_w: int, game_h: int, crop) -> Tuple[float, float, float, float]:
    """反过来：绝对裁剪框 -> 相对裁剪框（方便把现在的配置换算成相对写法）。"""
    x, y, w, h = [int(v) for v in crop]
    return (x / float(game_w), y / float(game_h), (x + w) / float(game_w), (y + h) / float(game_h))
