# -*- coding: utf-8 -*-
r"""上机体检：把整条胶片条按 33ms 逐帧滚动喂给模型，看它「什么时候动手」。

为什么需要它（2026-09-26 用户反馈「数据多的上机判断力反而差」）：
    离线 acc / 宏F1 / 甚至加阈值后的单帧触发率，量的都是「以按键那一刻为锚点」的判断。
    真正上机是**逐帧闭环**：动作早晚、动作频率、空闲时会不会乱动，离线指标全都看不见，
    而跑酷游戏里决定生死的恰恰是这三件事（跳早了落在障碍上 = 死）。

做法：对每条胶片条（37 帧，覆盖按键前 800ms ~ 按键后 400ms），把 6 帧窗口从最早滑到
最晚，每一格预测一次，得到一条「动作时间线」；再以人的真实按键（t=0）为基准统计。
★ 一次推理，多个阈值同时出结果（--sweep），不用重复跑模型。

用法：
    python tools\上机体检.py --runs runs\A runs\B --limit 400
    python tools\上机体检.py --runs runs\A --sweep 0.85,0.9,0.95,0.98
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from collector.dataset import (build_eval_transform, flatten_frames,  # noqa: E402
                               load_sessions, read_strip, split_indices,
                               to_video_tensor)
from collector.model import build_model                                     # noqa: E402

DEFAULT_SESSIONS = [
    r"dataset\20260918_merged",
    r"dataset\20260916_170936",
    r"dataset\20260919_195121",
    r"dataset\20260920_203729",
    r"dataset\迭代补采\正确操作",
    r"dataset\迭代补采\碰撞操作",
]


def build_judge(args):
    sess = [s if os.path.isabs(s) else os.path.join(HERE, s) for s in args.sessions]
    samples = load_sessions(sess)
    sp = split_indices(samples,
                       train_ratio=1 - args.val_ratio - args.test_ratio,
                       val_ratio=args.val_ratio, mode=args.split_mode,
                       boundary_gap=args.boundary_gap, seed=args.seed,
                       segment_seconds=args.segment_seconds)
    excl = [e for e in (args.exclude_sessions or []) if e]
    keep = [i for i in sp["test"]
            if not any(e in samples[i].get("_root", "") for e in excl)]
    print("[裁判集] %d 个真实样本（%d 个会话，已剔除 %s）"
          % (len(keep), len(sess), "、".join(excl) or "无"))
    return samples, keep


def replay(run_dir, samples, idxs, args, device):
    """返回每条时间线的原始预测：[(时间戳数组, 预测类别数组, 置信度数组, 真实类名)]。"""
    meta = json.load(open(os.path.join(run_dir, "model_meta.json"), encoding="utf-8"))
    labels = list(meta["labels"])
    K = int(meta["in_frames"])
    img = int(meta.get("img_size", 224))
    model = build_model(K, len(labels), pretrained=False)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval().float()
    tf = build_eval_transform(img)

    recs = []
    with torch.no_grad():
        for cnt, i in enumerate(idxs):
            s = samples[i]
            times = np.asarray(s["frame_times_ms"], dtype=np.float64)
            nf = len(times)
            frames = read_strip(s, list(range(nf)))          # (nf,H,W,3) 只解码一次
            js = list(range(K - 1, nf))                      # 因果：窗口只用过去的帧
            batch = np.stack([frames[j - K + 1:j + 1] for j in js])
            xs = [flatten_frames(tf(to_video_tensor(batch[q]))) for q in range(len(js))]
            x = torch.stack(xs).to(device)
            p = torch.softmax(model(x).float(), dim=1).cpu().numpy()
            recs.append((times[js], p.argmax(1), p.max(1), s["label"]))
            print("\r    %d/%d" % (cnt + 1, len(idxs)), end="", flush=True)
    print("\r    %d/%d 完成" % (len(idxs), len(idxs)))
    return labels, recs


def timeline_stats(recs, labels, thr):
    """按给定阈值换算时间线：动作率 / 空闲乱动 / 漏动 / 按错方向 / 提前量。"""
    n_idx = {l: i for i, l in enumerate(labels)}
    leads, rates, idle, missed, wrong, act = [], [], [], 0, 0, 0
    for times, pred, conf, lab in recs:
        fire = (conf >= thr) & (pred != n_idx["N"])
        rates.append(float(fire.mean()))
        if lab == "N":
            idle.append(float(fire.mean()))
            continue
        act += 1
        if fire.any():
            q = int(np.argmax(fire))
            leads.append(float(times[q]))
            if labels[int(pred[q])] != lab:
                wrong += 1
        else:
            missed += 1
    return {"thr": thr, "rate": float(np.mean(rates)) if rates else 0.0,
            "idle": float(np.mean(idle)) if idle else float("nan"),
            "leads": np.asarray(leads), "act": act,
            "miss_rate": missed / max(act, 1), "wrong_rate": wrong / max(act, 1)}


def print_sweep(name, labels, recs, thrs):
    print("")
    print("  —— %s ——" % name)
    print("  %-6s %9s %10s %8s %10s %11s"
          % ("阈值", "动作率", "空闲乱动", "漏动", "按错方向", "提前量中位"))
    rows = []
    for t in thrs:
        st = timeline_stats(recs, labels, t)
        med = float(np.median(st["leads"])) if st["leads"].size else float("nan")
        print("  %-6.2f %8.1f%% %9.1f%% %7.1f%% %9.1f%% %9.0fms"
              % (t, st["rate"] * 100, st["idle"] * 100, st["miss_rate"] * 100,
                 st["wrong_rate"] * 100, med))
        rows.append((t, st, med))
    print("  说明：动作率 = 有多少比例的时刻会发键（越高越爱动、越吵）")
    print("        漏动 = 该按键时整条窗口一次都没动（越少越好）")
    print("        提前量中位 = 第一次动手比人早/晚多少（负 = 更早）")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="逐帧滚动回放，体检上机时序行为")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    ap.add_argument("--exclude-sessions", nargs="*", default=["20260918_merged"])
    ap.add_argument("--split-mode", default="time_balanced")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--boundary-gap", type=int, default=5)
    ap.add_argument("--segment-seconds", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", default="0.85,0.90,0.95,0.98",
                    help="要体检的阈值列表（9_自动玩.py 默认 0.85）")
    ap.add_argument("--limit", type=int, default=400, help="用多少个样本（0=全部）")
    ap.add_argument("--write-thr", type=float, default=0.0,
                    help="把这个阈值写进每个模型目录的 thresholds.json（9_自动玩.py 会自动读它）")
    args = ap.parse_args()
    thrs = [float(x) for x in args.sweep.split(",") if x.strip()]

    samples, idxs = build_judge(args)
    if args.limit:
        idxs = idxs[:args.limit]
    if args.write_thr > 0:
        for r in args.runs:
            rd = r if os.path.isabs(r) else os.path.join(HERE, r)
            p = os.path.join(rd, "thresholds.json")
            # 9_自动玩.py 的 load_run() 会读这个文件，按类别覆盖默认阈值
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"default": float(args.write_thr)}, fh, ensure_ascii=False, indent=2)
            print("[阈值] 已写入 %s  default=%.2f" % (p, args.write_thr))
        return 0
    print("[规模] 回放 %d 个样本，每条胶片条逐帧滚窗口" % len(idxs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allrows = []
    for r in args.runs:
        rd = r if os.path.isabs(r) else os.path.join(HERE, r)
        print("\n" + "=" * 78)
        print("回放: %s" % os.path.basename(rd))
        labels, recs = replay(rd, samples, idxs, args, device)
        allrows.append((os.path.basename(rd), print_sweep(os.path.basename(rd),
                                                           labels, recs, thrs)))

    print("\n" + "=" * 78)
    print("【跨模型对比 · 同一个阈值下】")
    for t_i, t in enumerate(thrs):
        print("  阈值 %.2f" % t)
        print("  %-34s %9s %10s %8s %10s" % ("模型", "动作率", "空闲乱动", "漏动", "按错方向"))
        for name, rows in allrows:
            st = rows[t_i][1]
            print("  %-34s %8.1f%% %9.1f%% %7.1f%% %9.1f%%"
                  % (name[:34], st["rate"] * 100, st["idle"] * 100,
                     st["miss_rate"] * 100, st["wrong_rate"] * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
