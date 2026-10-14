# -*- coding: utf-8 -*-
"""注意力可视化：Grad-CAM 空间热力图 + 逐帧重要度。

一次反向传播同时拿到三样东西（不引入新依赖，不用 pytorch-grad-cam）：

  1. **Grad-CAM**：最后一级特征图（EfficientNet-B0 是 7x7x1280）的梯度做通道加权，
     再上采样到 224x224。是"证据来自哪里"的可视化，位置准、但分辨率粗。
  2. **输入梯度显著图**：d(类别分数)/d(输入像素)，原生 224x224，更细但更噪。
  3. **逐帧重要度**：6 帧各占多少。这个对本项目最有用 —— 输入是 6 帧早融合，
     如果重要度集中在最后一帧，说明模型其实没在用"运动"信息，只是在看单帧画面。

注意：**必须用 FP32**。9_自动玩.py 为了速度把模型转成 half，反向传播会出 NaN/inf，
所以这里单独加载一份 float32 模型。
"""
from __future__ import annotations

import os

from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
class AttentionProbe:
    """挂在模型上的 Grad-CAM 探针。构造一次，反复 run()。"""

    def __init__(self, model: nn.Module, frames: int = 6, target_layer=None):
        self.model = model
        self.frames = int(frames)
        self.layer = target_layer or self._find_last_conv(model)
        self._feat = None
        self._grad = None
        self._h_f = self.layer.register_forward_hook(self._on_fwd)
        self._h_b = self.layer.register_full_backward_hook(self._on_bwd)

    # ------------------------------------------------------------------
    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        """最后一层 Conv2d 就是 CAM 的目标层（EfficientNet 的最后那个 1x1）。"""
        last = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("模型里找不到 Conv2d，无法做 Grad-CAM")
        return last

    def _on_fwd(self, _m, _i, out):
        self._feat = out.detach()

    def _on_bwd(self, _m, _gi, gout):
        self._grad = gout[0].detach()

    def close(self) -> None:
        self._h_f.remove()
        self._h_b.remove()

    # ------------------------------------------------------------------
    def run(self, x: torch.Tensor, class_idx: Optional[int] = None,
            ablate_frames: bool = False) -> Dict:
        """x: (1, 3K, H, W) float32。

        返回 dict:
            pred, prob(ndarray[C]), cam(ndarray[h,w] 0~1, 可能是 None),
            saliency(ndarray[H,W] 0~1),
            frame_importance(ndarray[K])—— ablate_frames=True 时是**消融掉该帧后
            目标类别置信度的下降点数**（最直观，但要多跑 K 次前向）；
            否则是梯度幅值的占比（免费，但分辨率低、容易一片均匀）
        """
        self.model.zero_grad(set_to_none=True)
        x = x.detach().float().clone().requires_grad_(True)
        out = self.model(x)
        prob = torch.softmax(out.float(), dim=1)[0]
        idx = int(prob.argmax()) if class_idx is None else int(class_idx)
        out[0, idx].backward()

        cam = None
        if self._grad is not None and self._feat is not None:
            w = self._grad.mean(dim=(2, 3), keepdim=True)          # 通道权重
            cam_t = F.relu((w * self._feat).sum(1))[0]             # (h, w)
            cam = cam_t.cpu().numpy().astype(np.float32)
            cam -= cam.min()
            cam /= (cam.max() + 1e-8)

        g = x.grad.detach().abs()                                   # (1, 3K, H, W)
        h, w = g.shape[-2:]
        per_frame = g.view(1, self.frames, 3, h, w).mean(dim=(2, 3, 4))[0]
        per_frame = (per_frame / (per_frame.sum() + 1e-8)).cpu().numpy()
        sal = g[0].mean(0).cpu().numpy()
        sal = cv2.GaussianBlur(sal, (0, 0), 2.0)      # 平滑一下，否则全是噪点
        hi = float(np.percentile(sal, 99))            # 按 99 分位归一，避免被单个亮点拉爆
        sal = np.clip(sal / (hi + 1e-8), 0.0, 1.0)

        imp = per_frame.astype(np.float32)
        is_drop = False
        if ablate_frames:
            imp = self.ablate(x, idx)
            is_drop = True

        return {"pred": idx, "prob": prob.detach().cpu().numpy(),
                "cam": cam, "saliency": sal.astype(np.float32),
                "frame_importance": imp, "frame_is_drop": is_drop}

    def ablate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """逐帧消融：把某一帧的 3 个通道置零，看目标类别置信度掉多少（百分点）。

        这个比梯度幅值直观得多 —— 直接回答"模型到底用没用到这 6 帧"。
        值接近 0（甚至为负）说明这一帧对判断没贡献。
        """
        with torch.no_grad():
            base = float(torch.softmax(self.model(x.float()), dim=1)[0, class_idx])
            out = []
            for k in range(self.frames):
                x2 = x.detach().clone()
                x2[:, 3 * k:3 * k + 3] = 0.0
                p = float(torch.softmax(self.model(x2.float()), dim=1)[0, class_idx])
                out.append((base - p) * 100.0)
        return np.asarray(out, dtype=np.float32)


# ======================================================================
#  坐标映射与叠加
# ======================================================================
def _resize_geom(frame_hw, img_size: int):
    """复现 build_eval_transform 的几何：Resize(短边=img_size) + CenterCrop(img_size)。

    返回 (resized_hw, offset_y, offset_x)。
    """
    h, w = int(frame_hw[0]), int(frame_hw[1])
    if h <= w:
        rh, rw = img_size, int(round(w * img_size / h))
    else:
        rh, rw = int(round(h * img_size / w)), img_size
    return (rh, rw), (rh - img_size) // 2, (rw - img_size) // 2


def cam_to_frame(cam: np.ndarray, frame_hw, img_size: int = 224) -> np.ndarray:
    """把 img_size x img_size 的 CAM 映射回原始帧 (H, W) 的坐标系。

    直接把 224x224 的热力图拉满整帧是**错的** —— 变换里有 Resize + CenterCrop，
    不还原的话热力图会相对画面上下错位。
    """
    h, w = int(frame_hw[0]), int(frame_hw[1])
    (rh, rw), oy, ox = _resize_geom((h, w), img_size)
    full = np.zeros((rh, rw), np.float32)
    full[oy:oy + img_size, ox:ox + img_size] = cv2.resize(
        cam.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(full, (w, h), interpolation=cv2.INTER_LINEAR)


_FONT_CACHE: Dict[int, object] = {}


def _cjk_font(size: int):
    """找一个能显示中文的系统字体（Windows 自带黑体）。"""
    from PIL import ImageFont
    f = _FONT_CACHE.get(size)
    if f is not None:
        return f
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.isfile(p):
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:
                pass
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def put_text_cn(img_bgr: np.ndarray, text: str, org, size: int = 18,
                color=(255, 255, 255), stroke: int = 2) -> None:
    """在 BGR 图上写**中文**（就地修改 img_bgr）。

    cv2.putText 只支持 ASCII —— 中文会变成方块/乱码（实测过）。
    这里借 PIL + 系统黑体来画：BGR->RGB、PIL 画字、再写回原数组。
    单帧开销约 2~3ms（480x586），20fps 下可以接受。
    """
    from PIL import Image, ImageDraw
    rgb = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
    d = ImageDraw.Draw(rgb)
    d.text((int(org[0]), int(org[1])), text, font=_cjk_font(size),
           fill=(int(color[2]), int(color[1]), int(color[0])),
           stroke_width=int(stroke), stroke_fill=(0, 0, 0))
    img_bgr[:, :, :] = np.asarray(rgb)[:, :, ::-1]


def colorize(cam: np.ndarray) -> np.ndarray:
    """0~1 的灰度热力图 -> BGR 彩色图。"""
    c = np.clip(cam, 0.0, 1.0)
    return cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)


def overlay(frame_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45,
            thresh: float = 0.15) -> np.ndarray:
    """把热力图叠到画面上。低响应区域不着色，避免整屏泛蓝。"""
    heat = colorize(cam)
    m = (cam >= thresh).astype(np.float32)[..., None]
    out = frame_bgr.astype(np.float32) * (1 - alpha * m) + heat.astype(np.float32) * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================================================================
#  画面板（离线出图用）
# ======================================================================
def draw_panel(frames_rgb: List[np.ndarray], res: Dict, labels: List[str],
               title: str = "", thumb_w: int = 108) -> np.ndarray:
    """一张样本的面板：左=6 帧输入缩略图，中=画面+热力图，右=纯热力图 / 显著图。

    frames_rgb: K 张 (H, W, 3) RGB 原始帧（按时间顺序）
    """
    import math
    K = len(frames_rgb)
    fh, fw = frames_rgb[0].shape[:2]
    scale = thumb_w / float(fw)
    th, tw = max(1, int(round(fh * scale))), thumb_w
    cols = 3
    rows = int(math.ceil(K / cols))
    gap = 6
    pad = 10
    head = 44

    left_w = cols * tw + (cols - 1) * gap
    left_h = rows * th + (rows - 1) * gap
    mid_h = max(left_h, 200)
    mid_w = int(round(fw * mid_h / float(fh)))

    total_w = pad + left_w + gap + mid_w + gap + mid_w * 2 + gap + pad
    total_h = head + pad + mid_h + pad + 46 + pad
    canvas = np.full((total_h, total_w, 3), 22, np.uint8)

    # --- 左：6 帧输入 ---
    for i, fr in enumerate(frames_rgb):
        r, c = divmod(i, cols)
        y = head + pad + r * (th + gap)
        x = pad + c * (tw + gap)
        small = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        canvas[y:y + th, x:x + tw] = small
        imp = float(res["frame_importance"][i]) if i < len(res["frame_importance"]) else 0.0
        cv2.rectangle(canvas, (x, y), (x + tw - 1, y + th - 1), (70, 70, 70), 1)
        txt = ("第%d帧 %+.0fpt" % (i + 1, imp)) if res.get("frame_is_drop") \
            else ("第%d帧 %.0f%%" % (i + 1, imp * 100))
        put_text_cn(canvas, txt, (x + 3, y + th - 20), size=13,
                    color=(0, 255, 255), stroke=1)

    mid_frame = frames_rgb[-1]
    base = cv2.resize(mid_frame, (mid_w, mid_h), interpolation=cv2.INTER_AREA)[:, :, ::-1]

    # --- 中：画面 + CAM ---
    x0 = pad + left_w + gap
    if res.get("cam") is not None:
        c = cam_to_frame(res["cam"], (fh, fw))
    else:
        c = np.zeros((fh, fw), np.float32)
    c_full = cv2.resize(c, (mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    canvas[head + pad:head + pad + mid_h, x0:x0 + mid_w] = overlay(base, c_full)
    _cap(canvas, "叠加 Grad-CAM", x0, head + pad - 6)

    # --- 右：纯 CAM 与显著图 ---
    x1 = x0 + mid_w + gap
    canvas[head + pad:head + pad + mid_h, x1:x1 + mid_w] = colorize(c_full)
    _cap(canvas, "Grad-CAM", x1, head + pad - 6)
    x2 = x1 + mid_w + gap
    s_full = cv2.resize(res["saliency"], (mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
    canvas[head + pad:head + pad + mid_h, x2:x2 + mid_w] = colorize(s_full)
    _cap(canvas, "输入梯度显著图", x2, head + pad - 6)

    # --- 顶部标题 + 预测 ---
    # 中文必须用 put_text_cn（cv2.putText 只认 ASCII）
    put_text_cn(canvas, title, (pad, 6), size=17, color=(255, 255, 255), stroke=1)
    pred = res["pred"]
    put_text_cn(canvas, "预测 %s   置信度 %.2f" % (labels[pred], res["prob"][pred]),
                (pad, 27), size=17, color=(0, 220, 120), stroke=1)

    # --- 底部概率条 ---
    by = head + pad + mid_h + 14
    bx = pad
    for i, lab in enumerate(labels):
        p = float(res["prob"][i])
        w = int(round(90 * p))
        col = (0, 200, 255) if i == pred else (120, 120, 120)
        cv2.rectangle(canvas, (bx, by), (bx + max(w, 1), by + 12), col, -1)
        cv2.putText(canvas, "%s %.2f" % (lab, p), (bx, by + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        bx += 100
    return canvas


def _cap(img, text, x, y):
    put_text_cn(img, text, (x + 2, y), size=15, color=(210, 210, 210), stroke=1)
