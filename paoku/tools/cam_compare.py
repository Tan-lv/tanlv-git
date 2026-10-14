# -*- coding: utf-8 -*-
r"""对比两个模型的 Grad-CAM「聚焦程度」—— 客观量化"注意力是不是变涣散了"。

用法:
    python tools\cam_compare.py --a runs\20260918_merged_20260918_1822 \
                                --b runs\迭代_20260925_2226 \
                                --session dataset\20260916_170936 --per-class 20

为什么不能只看图：
    collector\attention.py 里 CAM 做了 **(cam - min) / max** 的逐图归一化 ——
    激活范围变小也会被拉满，看起来就"一片糊"。所以要量的是**归一化之前**的
    分布：正向质量有多少集中在少数几格里。
指标（都是"越大越聚焦"或"越小越聚焦"）：
    top10%   正向质量里，最大那 10% 格子占的比例（1.0 = 全挤在一格；均匀=0.10）
    ent      质量分布归一化熵（0 = 全在一格，1 = 完全均匀）
    range    归一化前的动态范围 max-min（反映证据强度）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from collector.dataset import (build_eval_transform, flatten_frames, load_sessions,
                               read_strip, select_window_indices, to_video_tensor)
from collector.model import build_model


def load_model(run_dir, device):
    mp = os.path.join(run_dir, "model_meta.json")
    cp = os.path.join(run_dir, "best.pt")
    if not (os.path.isfile(mp) and os.path.isfile(cp)):
        raise SystemExit(f"[错误] 模型目录不完整：{run_dir}")
    meta = json.load(open(mp, encoding="utf-8"))
    model = build_model(int(meta["in_frames"]), len(meta["labels"]), pretrained=False)
    model.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False)["model_state_dict"])
    return model.to(device).eval().float(), meta


def raw_cam(model, x, target):
    """自己算一遍 Grad-CAM，返回**未归一化**的 (cam_raw, feat_grad_range)。"""
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    store = {}

    def fwd(_m, _i, out):
        store["feat"] = out

    def bwd(_m, _gi, gout):
        store["grad"] = gout[0]

    h1 = last.register_forward_hook(fwd)
    h2 = last.register_full_backward_hook(bwd)
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        out[0, target].backward()
        w = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * store["feat"]).sum(1))[0]
        return cam.detach().cpu().numpy().astype(np.float64)
    finally:
        h1.remove()
        h2.remove()


def focus_metrics(cam: np.ndarray):
    """cam 是 hxw 的非负图（未归一化）。

    返回 (top10份额, 归一化熵, 动态范围, 热点面积@0.5, 热点面积@0.8)

    ★ 2026-09-25：光看"质量集中度"不够 —— 图上"看起来涣不涣"取决于
      **归一化之后有多大的区域被点亮**（overlay 里 cam>=0.15 才着色）。
      所以补两个"面积"指标：归一化后 ≥0.5 / ≥0.8 的像素占比。
    """
    flat = cam.reshape(-1).astype(np.float64)
    rng = float(flat.max() - flat.min())
    mass = np.clip(flat, 0, None)
    tot = mass.sum()
    nrm = (mass - mass.min()) / (mass.max() - mass.min() + 1e-12)
    a50 = float((nrm >= 0.50).mean())
    a80 = float((nrm >= 0.80).mean())
    # ★ 空间分布：质量落在画面哪一带。下半区占比大 = 热区压在画面底部（车/铁轨那条带），
    #   这正是肉眼看图时"亮成一条横杠"的来源。
    h = cam.shape[0]
    colsum = np.clip(cam, 0, None).sum(axis=1)          # 每行的质量
    if colsum.sum() > 1e-12:
        bot50 = float(colsum[h // 2:].sum() / colsum.sum())
        rows_idx = np.arange(h, dtype=np.float64)
        com = float((colsum * rows_idx).sum() / colsum.sum() / max(h - 1, 1))
    else:
        bot50, com = 0.0, 0.5
    if tot <= 1e-12:
        return 0.0, 1.0, rng, a50, a80, bot50, com
    p = mass / tot
    k = max(1, int(round(len(p) * 0.10)))
    top = np.sort(p)[::-1][:k].sum()
    q = p[p > 0]
    ent = float(-(q * np.log(q)).sum() / np.log(len(p)))
    return float(top), ent, rng, a50, a80, bot50, com


def frame_importance(model, x, target, K):
    """逐帧消融：把某帧 3 通道置零，看目标类置信度掉多少百分点（越正 = 越依赖这帧）。"""
    with torch.no_grad():
        base = float(torch.softmax(model(x.float()), dim=1)[0, target])
        out = []
        for k in range(K):
            x2 = x.detach().clone()
            x2[:, 3 * k:3 * k + 3] = 0.0
            p = float(torch.softmax(model(x2.float()), dim=1)[0, target])
            out.append((base - p) * 100.0)
    return np.asarray(out, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description="对比两个模型的 CAM 聚焦程度")
    ap.add_argument("--a", required=True, help="模型 A（比如初代）")
    ap.add_argument("--b", required=True, help="模型 B（比如迭代后）")
    ap.add_argument("--session", default=r"dataset\20260916_170936", help="取样会话（两个模型都没见过的最好）")
    ap.add_argument("--per-class", type=int, default=20, help="每类取多少个样本")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ra = args.a if os.path.isabs(args.a) else os.path.join(HERE, args.a)
    rb = args.b if os.path.isabs(args.b) else os.path.join(HERE, args.b)
    ma, meta_a = load_model(ra, dev)
    mb, meta_b = load_model(rb, dev)
    labels = list(meta_a["labels"])
    print(f"[A] {ra}   end_offset {meta_a.get('input_end_offset_ms')}  帧 {meta_a['in_frames']}")
    print(f"[B] {rb}   end_offset {meta_b.get('input_end_offset_ms')}  帧 {meta_b['in_frames']}")
    if list(meta_b["labels"]) != labels or meta_b["in_frames"] != meta_a["in_frames"]:
        print("[错误] 两个模型的类别/帧数不一致，没法比")
        return 1
    if int(meta_a.get("input_end_offset_ms", 0)) != int(meta_b.get("input_end_offset_ms", 0)):
        print("[警告] 两个模型的输入窗定义不同，对比不公平")

    sess = args.session if os.path.isabs(args.session) else os.path.join(HERE, args.session)
    samples = load_sessions([sess])
    pick = {}
    for s in samples:
        pick.setdefault(s["label"], [])
        if len(pick[s["label"]]) < args.per_class:
            pick[s["label"]].append(s)
    print(f"[data] {sess}  取到 " + "  ".join(f"{k}:{len(v)}" for k, v in sorted(pick.items())))

    K = int(meta_a["in_frames"])
    img = int(meta_a.get("img_size", 224))
    tf = build_eval_transform(img)
    n_idx = {l: i for i, l in enumerate(labels)}
    rows = {"A": [], "B": []}
    fi = {}
    for lab in labels:
        for s in pick.get(lab, []):
            end = -int(meta_a.get("input_end_offset_ms", 0))
            idx = select_window_indices(s, K, int(meta_a.get("input_stride", 1)), end)
            x = flatten_frames(tf(to_video_tensor(read_strip(s, idx)))).unsqueeze(0).to(dev)
            tgt = n_idx[s["label"]]
            for key, mdl in (("A", ma), ("B", mb)):
                cam = raw_cam(mdl, x, tgt)
                rows[key].append((lab,) + focus_metrics(cam))
                fi.setdefault(key, []).append(frame_importance(mdl, x, tgt, K))
    print()
    print("%-6s %8s %8s %9s %9s %9s %9s %8s"
          % ("模型", "top10%", "熵", "热点≥0.5", "热点≥0.8", "动态范围", "下半区质量", "垂直重心"))
    for key, name in (("A", "初代"), ("B", "迭代后")):
        arr = np.array([r[1:] for r in rows[key]], dtype=np.float64)
        m = arr.mean(axis=0)
        print("%-6s %8.3f %8.3f %8.1f%% %8.1f%% %9.2f %9.1f%% %8.3f"
              % (name, m[0], m[1], m[3] * 100, m[4] * 100, m[2], m[5] * 100, m[6]))
    print()
    print("逐帧重要度（消融掉该帧后置信度掉几个百分点，越大=越依赖这帧）：")
    for key, name in (("A", "初代"), ("B", "迭代后")):
        m = np.array(fi.get(key, []), dtype=np.float64).mean(axis=0)
        print("  %-6s %s" % (name, "  ".join("f%d %+5.2f" % (i + 1, v) for i, v in enumerate(m))))
    print()
    print("按类对比（top10% 越大越聚焦 / 熵越小越聚焦）：")
    print("  %-4s %8s %8s %8s %8s" % ("类别", "A top10", "B top10", "A 熵", "B 熵"))
    for lab in labels:
        a = np.array([r[1] for r in rows["A"] if r[0] == lab])
        b = np.array([r[1] for r in rows["B"] if r[0] == lab])
        ae = np.array([r[2] for r in rows["A"] if r[0] == lab])
        be = np.array([r[2] for r in rows["B"] if r[0] == lab])
        if a.size and b.size:
            print("  %-4s %8.3f %8.3f %8.3f %8.3f" % (lab, a.mean(), b.mean(), ae.mean(), be.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
