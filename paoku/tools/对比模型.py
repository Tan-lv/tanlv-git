# -*- coding: utf-8 -*-
r"""公正对比多个模型：在**同一批真实样本**上评一遍，谁好谁坏一目了然。

为什么需要它（2026-09-26 用户踩的坑）：
    两次训练各自划自己的测试集，测试集不一样、印的表也不一样 ——
    旧版 log 只印「全量含合成负样本」的 Accuracy，新版 log 先印「真实样本」那一栏。
    拿 A 的"真实样本 acc 0.9225" 和 B 的"全量 acc 0.9258" 一比，
    就会得出"数据多的反而更差" —— 那是**尺子不一样**，不是模型变差。
    这个脚本把若干模型的 best.pt 放到同一批样本上评，逐类 P/R/F1 + 混淆矩阵都印出来。

裁判集是怎么来的（关键）：
    * 默认用「6 个会话」那批数据 + 和训练时完全相同的划分参数（seed 42 / time_balanced）
      -> 取 test 划分，它本身就是 A 模型没见过的；
    * 再默认剔除 20260918_merged 的样本（初代模型吃过的会话），
      剩下那部分对**两个模型都没见过**，是真正干净的裁判集。
    * 你对比的模型如果还吃过别的会话，用 --exclude-sessions 把它们列出来。

用法：
    python tools\对比模型.py --list-clean
    python tools\对比模型.py --runs runs\20260918_merged_20260918_1822 runs\6个会话_20260926_1400
    python tools\对比模型.py --runs runs\A runs\B --exclude-sessions 20260918_merged
    python tools\对比模型.py --runs runs\A runs\B --limit 400    # 快速看趋势
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from collector.dataset import (build_eval_transform, flatten_frames,  # noqa: E402
                               load_sessions, read_strip,
                               select_window_indices, split_indices,
                               to_video_tensor)
from collector.model import build_model                                     # noqa: E402

# 裁判池 = 那次「6 个会话」训练用过的全部数据（顺序无所谓，划分是按时间段的）
DEFAULT_SESSIONS = [
    r"dataset\20260918_merged",
    r"dataset\20260916_170936",
    r"dataset\20260919_195121",
    r"dataset\20260920_203729",
    r"dataset\迭代补采\正确操作",
    r"dataset\迭代补采\碰撞操作",
]


def build_judge(args):
    """返回 (samples, 裁判样本下标)。划分参数和 trainer 默认值保持一致。"""
    sess = [s if os.path.isabs(s) else os.path.join(HERE, s) for s in args.sessions]
    samples = load_sessions(sess)
    sp = split_indices(samples,
                       train_ratio=1 - args.val_ratio - args.test_ratio,
                       val_ratio=args.val_ratio, mode=args.split_mode,
                       boundary_gap=args.boundary_gap, seed=args.seed,
                       segment_seconds=args.segment_seconds)
    test_idx = list(sp["test"])
    excl = [e for e in (args.exclude_sessions or []) if e]
    keep = [i for i in test_idx
            if not any(e in samples[i].get("_root", "") for e in excl)]
    print("[裁判池] %d 个样本（%d 个会话）" % (len(samples), len(sess)))
    print("[划分]   train %d / val %d / test %d（seed %s, %s）"
          % (len(sp["train"]), len(sp["val"]), len(test_idx), args.seed, args.split_mode))
    if excl:
        print("[剔除]   测试集里属于 %s 的样本 %d 个（有模型训练时吃过）"
              % ("、".join(excl), len(test_idx) - len(keep)))
    print("[裁判集] %d 个真实样本 —— 参与对比的模型都没见过这部分" % len(keep))
    return samples, keep


def eval_run(run_dir: str, samples, idxs, device, batch: int):
    """把某个模型的 best.pt 在这批样本上跑一遍，返回 (y_true, y_pred, labels, 说明)。"""
    mp = os.path.join(run_dir, "model_meta.json")
    cp = os.path.join(run_dir, "best.pt")
    if not (os.path.isfile(mp) and os.path.isfile(cp)):
        return None, None, None, None, "缺 best.pt 或 model_meta.json"
    meta = json.load(open(mp, encoding="utf-8"))
    labels = list(meta["labels"])
    K = int(meta["in_frames"])
    img = int(meta.get("img_size", 224))
    end_off = int(meta.get("input_end_offset_ms", 0))
    stride = int(meta.get("input_stride", 1))
    model = build_model(K, len(labels), pretrained=False)
    ck = torch.load(cp, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval().float()
    tf = build_eval_transform(img)
    n_idx = {l: i for i, l in enumerate(labels)}
    y_true, y_pred, confs = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(idxs), batch):
            xs, ys = [], []
            for i in idxs[s0:s0 + batch]:
                s = samples[i]
                wi = select_window_indices(s, K, stride, -end_off)
                frames = read_strip(s, wi)
                xs.append(flatten_frames(tf(to_video_tensor(frames))))
                ys.append(n_idx[s["label"]])
            x = torch.stack(xs).to(device)
            p = torch.softmax(model(x).float(), dim=1)
            y_pred.extend(p.argmax(1).cpu().numpy().tolist())
            confs.extend(p.max(1).values.cpu().numpy().tolist())
            y_true.extend(ys)
            done = min(s0 + batch, len(idxs))
            print("\r    %d/%d" % (done, len(idxs)), end="", flush=True)
    print("\r    %d/%d 完成" % (len(idxs), len(idxs)))
    return y_true, y_pred, labels, confs, None


def prf(y_true, y_pred, labels, k: int):
    """第 k 类的 precision / recall / f1 / 支持数。"""
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == k and b == k)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != k and b == k)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == k and b != k)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp + fn


def confusion(y_true, y_pred, n: int):
    cm = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    return cm


def deploy_report(y_true, y_pred, confs, labels, thr: float):
    """上线口径（9_自动玩.py 的规则）：conf >= thr 才发键，N 不发键。

    离线算的是 argmax，跟这个阈值完全无关 —— 模型置信度一低，上机就变成"该动不动"。
    返回 (每类部署召回, N 的部署特异度, 低置信丢弃比例)。
    """
    ys = np.asarray(y_true)
    yp = np.asarray(y_pred)
    cf = np.asarray(confs)
    n_idx = {l: i for i, l in enumerate(labels)}
    act = [l for l in labels if l != "N"]
    rec = {}
    for l in act:
        m = ys == n_idx[l]
        if m.sum() == 0:
            rec[l] = float("nan")
            continue
        fired = ((yp == n_idx[l]) & (cf >= thr) & m).sum()
        rec[l] = float(fired) / float(m.sum())
    if "N" in n_idx:
        mn = ys == n_idx["N"]
        idle = float((((yp == n_idx["N"]) | (cf < thr)) & mn).sum()) / float(max(mn.sum(), 1))
    else:
        idle = float("nan")
    lowconf = float((cf < thr).mean())
    return rec, idle, lowconf


# 遮挡体检的区域（比例坐标，作用在「裁剪后的帧」上）
#   右侧 20% = 那排静态 UI（TOP RUN / DAILY HIGH SCORE / ＋ 按钮）所在的位置
REGIONS = [
    ("原始", None),
    ("遮右侧20%(UI)", ("x", 0.80, 1.00)),
    ("遮左侧20%", ("x", 0.00, 0.20)),
    ("遮底部25%", ("y", 0.75, 1.00)),
    ("遮顶部25%", ("y", 0.00, 0.25)),
    ("遮中间(角色/障碍)", ("box", 0.30, 0.70, 0.30, 0.80)),
]


def apply_mask(frames: np.ndarray, region) -> np.ndarray:
    """把 frames (K,H,W,3) 的指定区域涂黑，再喂给模型（遮挡法）。"""
    if region is None:
        return frames
    out = frames.copy()
    h, w = out.shape[1], out.shape[2]
    kind = region[0]
    if kind == "x":
        a, b = int(region[1] * w), int(region[2] * w)
        out[:, :, a:b] = 0
    elif kind == "y":
        a, b = int(region[1] * h), int(region[2] * h)
        out[:, a:b, :] = 0
    else:
        x0, x1, y0, y1 = int(region[1] * w), int(region[2] * w), int(region[3] * h), int(region[4] * h)
        out[:, y0:y1, x0:x1] = 0
    return out


def eval_occlusion(run_dir, samples, idxs, device, batch, regions, thr):
    """一次解码、多区域评估：每个样本只读一次胶片条，6 个区域一起前向。"""
    meta = json.load(open(os.path.join(run_dir, "model_meta.json"), encoding="utf-8"))
    labels = list(meta["labels"])
    K = int(meta["in_frames"])
    img = int(meta.get("img_size", 224))
    end_off = int(meta.get("input_end_offset_ms", 0))
    stride = int(meta.get("input_stride", 1))
    model = build_model(K, len(labels), pretrained=False)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval().float()
    tf = build_eval_transform(img)
    n_idx = {l: i for i, l in enumerate(labels)}
    acc = {nm: [[], []] for nm, _ in regions}          # {名字: [y_true, y_pred]}
    with torch.no_grad():
        for c0, i in enumerate(idxs):
            s = samples[i]
            wi = select_window_indices(s, K, stride, -end_off)
            frames = read_strip(s, wi)
            xs = [flatten_frames(tf(to_video_tensor(apply_mask(frames, rg))))
                  for _, rg in regions]
            x = torch.stack(xs).to(device)                  # (R, C, H, W)
            p = torch.softmax(model(x).float(), dim=1).cpu().numpy()
            for q, (nm, _) in enumerate(regions):
                acc[nm][0].append(n_idx[s["label"]])
                acc[nm][1].append(int(p[q].argmax()))
            if (c0 + 1) % 40 == 0 or c0 + 1 == len(idxs):
                print("\r    %d/%d" % (c0 + 1, len(idxs)), end="", flush=True)
    print("\r    %d/%d 完成" % (len(idxs), len(idxs)))
    out = {}
    for nm, _ in regions:
        yt, yp = acc[nm]
        n = len(labels)
        a = sum(1 for u, v in zip(yt, yp) if u == v) / max(len(yt), 1)
        f1 = float(np.mean([prf(yt, yp, labels, k)[2] for k in range(n)]))
        d_rec, d_idle, _ = deploy_report(yt, yp, [1.0] * len(yt), labels, thr)
        out[nm] = (a, f1, d_idle)
    return labels, out


def main() -> int:
    ap = argparse.ArgumentParser(description="在同一个裁判集上公正对比多个模型")
    ap.add_argument("--runs", nargs="+", default=None, help="要对比的模型目录（含 best.pt）")
    ap.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS, help="裁判池会话")
    ap.add_argument("--exclude-sessions", nargs="*", default=["20260918_merged"],
                    help="剔除这些会话的样本（有模型训练时吃过它们）")
    ap.add_argument("--split-mode", default="time_balanced")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--boundary-gap", type=int, default=5)
    ap.add_argument("--segment-seconds", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个裁判样本（0=全部，快速看趋势用）")
    ap.add_argument("--thr", type=float, default=0.85,
                    help="上线发键阈值（9_自动玩.py 的默认值是 0.85）")
    ap.add_argument("--list-clean", action="store_true", help="只算裁判集，不跑模型")
    ap.add_argument("--occlude", action="store_true",
                    help="遮挡体检：逐个涂黑画面区域，看模型依赖哪一块")
    args = ap.parse_args()

    samples, idxs = build_judge(args)
    dist = Counter(samples[i]["label"] for i in idxs)
    print("[裁判集类别] %s" % dict(sorted(dist.items())))
    if args.limit:
        idxs = idxs[:args.limit]
        print("[限制] 只用了前 %d 个" % len(idxs))
    if args.list_clean or not args.runs:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.occlude:
        for r in args.runs:
            rd = r if os.path.isabs(r) else os.path.join(HERE, r)
            print("\n" + "=" * 78)
            print("遮挡体检: %s" % os.path.basename(rd))
            labels, res = eval_occlusion(rd, samples, idxs, device, args.batch, REGIONS, args.thr)
            base_f1 = res["原始"][1]
            print("  %-22s %9s %10s %11s" % ("遮挡区域", "Acc", "宏F1", "比原始掉"))
            for nm, _ in REGIONS:
                a, f1, _idl = res[nm]
                print("  %-22s %9.4f %10.4f %+11.4f" % (nm, a, f1, f1 - base_f1))
            print("  说明：掉得越多 = 模型越依赖那块；掉 0 附近 = 那块对判断没用。")
        return 0
    results = []
    for r in args.runs:
        rd = r if os.path.isabs(r) else os.path.join(HERE, r)
        print("\n" + "=" * 78)
        print("跑模型: %s" % rd)
        yt, yp, labels, cf, err = eval_run(rd, samples, idxs, device, args.batch)
        if err:
            print("  [跳过] %s" % err)
            continue
        n = len(labels)
        acc = sum(1 for a, b in zip(yt, yp) if a == b) / max(len(yt), 1)
        rows = [prf(yt, yp, labels, k) for k in range(n)]
        macro = float(np.mean([x[2] for x in rows]))
        print("  Accuracy %.4f   macro-F1 %.4f" % (acc, macro))
        print("  %-6s %7s %11s %9s %8s" % ("类别", "样本", "precision", "recall", "f1"))
        for k, l in enumerate(labels):
            p, r_, f, sup = rows[k]
            print("  %-6s %7d %11.4f %9.4f %8.4f" % (l, sup, p, r_, f))

        # ★ 上线口径：9_自动玩.py 是 "argmax 且 conf>=thr 才发键"，
        #   离线 acc/F1 完全没算这个阈值 —— 置信度低的模型上机就会「该动不动」。
        d_rec, d_idle, d_low = deploy_report(yt, yp, cf, labels, args.thr)
        act_all = [l for l in labels if l != "N"]
        n_act = sum(1 for a in yt if labels[a] != "N")
        n_fire = 0
        for l in act_all:
            m = sum(1 for a in yt if labels[a] == l)
            if not np.isnan(d_rec[l]):
                n_fire += d_rec[l] * m
        deploy_rate = n_fire / max(n_act, 1)
        print("")
        print("  —— 上线口径（9_自动玩.py：argmax 且 conf>=%.2f 才发键）——" % args.thr)
        print("  置信度: 中位 %.3f   p10 %.3f   被低置信丢掉 %.1f%%"
              % (float(np.median(cf)), float(np.percentile(cf, 10)), d_low * 100))
        print("  该按键时真的按了: " + "   ".join("%s %.1f%%" % (l, d_rec[l] * 100)
                                                 for l in act_all))
        print("  该空闲时真的没按: N %.1f%%" % (d_idle * 100))
        print("  ==> 总体上只有 %.1f%% 的「该动」样本真的会发出按键" % (deploy_rate * 100))
        results.append({"name": os.path.basename(rd), "acc": acc, "macro": macro,
                        "rows": rows, "labels": labels, "cm": confusion(yt, yp, n),
                        "deploy": deploy_rate, "low": d_low, "rec": d_rec,
                        "idle": d_idle})

    if len(results) < 2:
        return 0
    print("\n" + "=" * 78)
    print("【总结】同一个裁判集（%d 个真实样本）" % len(idxs))
    print("=" * 78)
    hdr = "%-34s %7s %7s %7s" % ("模型", "Acc", "宏F1", "上机触发")
    for l in results[0]["labels"]:
        hdr += " %7s" % (l + "-F1")
    print(hdr)
    for r in results:
        line = "%-34s %7.4f %7.4f %6.1f%%" % (r["name"][:34], r["acc"], r["macro"],
                                              r["deploy"] * 100)
        for x in r["rows"]:
            line += " %7.4f" % x[2]
        print(line)
    print("")
    print("  [上机触发] = 该按键时真的发出按键的比例（按 9_自动玩.py 的 conf>=%.2f 规则算）。"
          % args.thr)
    print("             离线 Acc/宏F1 只看 argmax，不看阈值 —— 置信度低的模型会「该动不动」。")
    for r in results:
        print("             %-32s 低置信丢弃 %.1f%%   空闲误动 %.1f%%"
              % (r["name"][:32], r["low"] * 100, (1 - r["idle"]) * 100))
    best = max(results, key=lambda t: t["macro"])
    best_d = max(results, key=lambda t: t["deploy"])
    print("\n  离线宏F1 最高：%s（%.4f）" % (best["name"], best["macro"]))
    print("  上机触发率最高：%s（%.1f%%）" % (best_d["name"], best_d["deploy"] * 100))
    if best["name"] != best_d["name"]:
        print("  [!] 两个口径给出的最佳模型不一样 —— 这就是「离线好、上机差」的原因。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
