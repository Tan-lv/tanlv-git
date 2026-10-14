# -*- coding: utf-8 -*-
"""本地评估：加载训练好的 best.pt，给出比 collector/trainer.py 更细的"效果报告"。

用法（PyCharm 里直接运行即可）：
    python 6_评估效果.py --run runs/exp01 --sessions dataset/20260915_211905

和 collector/trainer.py 末尾那段评估的区别：
  * **把"真实样本"和"合成的换道动画负样本"分开统计**
    —— 后者就是"角色已经在换道动画里了，模型却还在输出方向键"（重复触发）
       的直接指标，是判断模型能不能上线的关键数字
  * 给出"危险误判"（该按方向键却输出 N，会直接撞死）的比例
  * 实测单帧推理延迟，判断 9_自动玩.py 能不能跑到 40fps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collector.config import Config
from collector.dataset import (MultiFrameDataset, build_animation_negatives,
                               build_approach_negatives, build_eval_transform,
                               load_sessions, negatives_for_subset, split_indices)
from collector.model import build_model
from collector.runs import pick_run


def load_cfg(sessions, args) -> Config:
    """和 collector/trainer.py 一样：读采集时的 session.json 作为基准配置。"""
    cfg = Config()
    for d in sessions:
        p = os.path.join(d, "session.json")
        if os.path.isfile(p):
            raw = json.load(open(p, encoding="utf-8")).get("config", {})
            valid = {fl.name for fl in fields(Config)}
            for k, v in raw.items():
                if k in valid:
                    if k in ("game_res", "game_crop", "labels", "animation_neg_end_ms",
                              "approach_neg_end_ms"):
                        v = tuple(v)
                    setattr(cfg, k, v)
            break
    if args.img_size:
        pass
    cfg.validate()
    return cfg


def run_batches(model, ds, device, bs, truths, kinds):
    """按顺序跑完整个数据集，返回 (预测数组, 各类概率数组)。"""
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            hi = min(i + bs, len(ds))
            xs = torch.stack([ds[j][0] for j in range(i, hi)])
            out = model(xs.to(device))
            p = torch.softmax(out.float(), dim=1)
            preds.append(out.argmax(1).cpu().numpy())
            probs.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def per_class(y_true, y_pred, labels):
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"label": lab, "n": int(cm[i].sum()), "precision": prec,
                     "recall": rec, "f1": f1})
    return cm, rows


def measure_latency(model, device, in_frames, img_size, n=60):
    x = torch.randn(1, in_frames * 3, img_size, img_size, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
    return dt * 1000.0


def main():
    ap = argparse.ArgumentParser(description="本地评估：模型效果诊断")
    ap.add_argument("--run", default=None, help="训练输出目录；不填就用效果最好的那个")
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="数据集会话目录；不填就用 dataset/ 里最新的一个")
    ap.add_argument("--split", default="test", choices=["test", "val", "all"],
                    help="在哪个集合上评估（默认 test）")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--segment-seconds", type=float, default=15.0)
    ap.add_argument("--split-mode", default="time_balanced")
    ap.add_argument("--boundary-gap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-anim-neg", action="store_true")
    ap.add_argument("--no-approach-neg", action="store_true")
    args = ap.parse_args()

    # PyCharm 里 Working directory 不一定对，相对路径一律按 paoku/ 解析
    HERE = os.path.dirname(os.path.abspath(__file__))
    def _abs(p):
        return p if os.path.isabs(p) else os.path.abspath(os.path.join(HERE, p))
    if not args.run:
        args.run = pick_run()
        print()
    args.run = _abs(args.run)
    if not args.sessions:
        root = os.path.join(HERE, "dataset")
        cands = [os.path.join(root, d) for d in os.listdir(root)
                 if not d.startswith("_")
                 and os.path.isfile(os.path.join(root, d, "meta.jsonl"))]
        if not cands:
            raise SystemExit("[错误] dataset/ 下没有会话，请用 --sessions 指定")
        args.sessions = [max(cands, key=os.path.getmtime)]
        print(f"[data] 没指定 --sessions，自动选用最新会话: {args.sessions[0]}")
    args.sessions = [_abs(s) for s in args.sessions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = os.path.join(args.run, "best.pt")
    if not os.path.isfile(ck_path):
        ck_path = os.path.join(args.run, "best.pt")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    labels = ck["labels"]
    in_frames = ck["in_frames"]
    print("=" * 78)
    print(f"模型: {ck_path}")
    print(f"设备: {device}" + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    print(f"类别: {labels}   输入: 连续 {in_frames} 帧   训练时 val_acc {ck.get('val_acc', 0):.4f}")
    print("=" * 78)

    cfg = load_cfg(args.sessions, args)
    if args.no_anim_neg:
        cfg.animation_neg_enabled = False
    samples = load_sessions(args.sessions)
    print(f"[data] {len(samples)} 个样本   {dict(sorted(Counter(s['label'] for s in samples).items()))}")

    if args.split == "all":
        idx = list(range(len(samples)))
    else:
        sp = split_indices(samples, train_ratio=1 - args.val_ratio - args.test_ratio,
                           val_ratio=args.val_ratio, mode=args.split_mode,
                           boundary_gap=args.boundary_gap, seed=args.seed,
                           segment_seconds=args.segment_seconds)
        idx = sp[args.split]
    if args.no_anim_neg:
        cfg.animation_neg_enabled = False
    if args.no_approach_neg:
        cfg.approach_neg_enabled = False
    sub = [samples[i] for i in idx]
    # 负样本分两类，顺序必须和 negatives_for_subset 一致（先动画、后障碍还远）
    pos = {g: loc for loc, g in enumerate(idx)}
    anim_local = [(pos[g], e) for g, e in build_animation_negatives(samples, cfg, labels)
                  if g in pos]
    appr_local = [(pos[g], e) for g, e in build_approach_negatives(samples, cfg, labels)
                  if g in pos]
    negs = anim_local + appr_local
    print(f"[data] 评估集合 '{args.split}': {len(sub)} 个真实样本 + "
          f"{len(anim_local)} 个换道动画负样本 + {len(appr_local)} 个障碍还远负样本"
          f" = {len(sub) + len(negs)}")

    ds = MultiFrameDataset(sub, cfg, build_eval_transform(args.img_size), args.img_size,
                           labels=labels, negatives=negs)
    n_real = len(sub)
    kinds = np.array(["real"] * n_real + ["anim_neg"] * len(anim_local)
                     + ["approach_neg"] * len(appr_local))
    y_true = np.array([ds.label_to_idx[ds.samples[j]["label"]] if j < n_real
                       else ds.label_to_idx["N"] for j in range(len(ds))])

    model = build_model(in_frames, len(labels), pretrained=False).to(device)
    model.load_state_dict(ck["model_state_dict"])
    y_pred, probs = run_batches(model, ds, device, args.batch_size, y_true, kinds)

    acc = float((y_pred == y_true).mean())
    cm, rows = per_class(y_true, y_pred, labels)
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    print()
    print(f"整体 Accuracy  {acc:.4f}      macro-F1  {macro_f1:.4f}")
    print()
    print(f"  {'类别':<5}{'样本':>6}{'precision':>11}{'recall':>9}{'f1':>8}")
    for r in rows:
        print(f"  {r['label']:<5}{r['n']:>6}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}")
    print()
    print("混淆矩阵 (行=真实, 列=预测):")
    print("        " + "".join(f"{l:>7}" for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l:<5} " + "".join(f"{cm[i, j]:>7d}" for j in range(len(labels))))

    # ---------- 真实样本 vs 换道动画负样本 ----------
    print()
    print("=" * 78)
    print("关键指标：两类合成负样本上的表现")
    print("=" * 78)
    m_real = kinds == "real"
    m_neg = kinds == "anim_neg"
    m_app = kinds == "approach_neg"
    n_idx = labels.index("N")

    if m_real.any():
        print(f"真实样本           n={m_real.sum():4d}  acc={(y_pred[m_real] == y_true[m_real]).mean():.4f}")
        # 危险误判：该按方向键，模型却输出 N（直接撞死）
        danger = m_real & (y_true != n_idx) & (y_pred == n_idx)
        print(f"  其中『该换道却输出 N』(会撞死)  {danger.sum()} / "
              f"{(m_real & (y_true != n_idx)).sum()}  "
              f"= {danger.sum() / max((m_real & (y_true != n_idx)).sum(), 1) * 100:.1f}%")
    if m_neg.any():
        ok = (y_pred[m_neg] == n_idx).mean()
        print(f"换道动画负样本     n={m_neg.sum():4d}  输出 N 的比例 = {ok:.4f}"
              f"   (越高越好；低 = 动画中还会继续按方向键)")
        wrong = y_pred[m_neg][y_pred[m_neg] != n_idx]
        if len(wrong):
            c = Counter(labels[i] for i in wrong)
            print(f"  动画中被误判成的动作: {dict(sorted(c.items()))}")
    if m_app.any():
        ok = (y_pred[m_app] == n_idx).mean()
        print(f"障碍还远负样本     n={m_app.sum():4d}  输出 N 的比例 = {ok:.4f}"
              f"   (越高越好；低 = 障碍还远就忍不住要动，即\"动作偏早\")")
        wrong = y_pred[m_app][y_pred[m_app] != n_idx]
        if len(wrong):
            c = Counter(labels[i] for i in wrong)
            print(f"  提前想动的动作分布: {dict(sorted(c.items()))}")
    else:
        print("障碍还远负样本     n=   0   （数据里没有：save_pre_ms 太短，或开关关了）")
    # 平均 N 概率
    pn = probs[:, n_idx]
    if m_real.any():
        print(f"N 概率均值: 真实 N 样本 {pn[m_real & (y_true == n_idx)].mean():.3f}   "
              f"真实方向样本 {pn[m_real & (y_true != n_idx)].mean():.3f}   "
              f"动画负样本 {pn[m_neg].mean() if m_neg.any() else float('nan'):.3f}   "
              f"障碍还远 {pn[m_app].mean() if m_app.any() else float('nan'):.3f}")

    lat = measure_latency(model, device, in_frames, args.img_size)
    print()
    print(f"单张推理延迟 {lat:.1f} ms  ->  理论上限 {1000 / lat:.0f} fps   "
          f"(9_自动玩.py 需要 >= 40fps，即 <= 25ms)  {'OK' if lat <= 25 else '偏慢'}")

    out = {"run": os.path.abspath(args.run), "split": args.split, "n_real": int(m_real.sum()),
           "n_anim_neg": int(m_neg.sum()), "accuracy": acc, "macro_f1": macro_f1,
           # 下面两个才是决定"能不能拿来玩"的指标（被大量负样本稀释的 macro_f1 不算）
           "real_acc": (float((y_pred[m_real] == y_true[m_real]).mean())
                        if m_real.any() else None),
           "danger_rate": (float(((y_pred == n_idx) & (y_true != n_idx)
                                 & m_real).sum()
                                / max(int(((y_true != n_idx) & m_real).sum()), 1))
                           if m_real.any() else None),
           "per_class": rows, "confusion": cm.tolist(), "labels": labels,
           "anim_neg_n_rate": float((y_pred[m_neg] == n_idx).mean()) if m_neg.any() else None,
           "n_approach_neg": int(m_app.sum()),
           "approach_neg_n_rate": (float((y_pred[m_app] == n_idx).mean())
                                   if m_app.any() else None),
           "latency_ms": lat}
    p = os.path.join(args.run, f"eval_{args.split}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[输出] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
