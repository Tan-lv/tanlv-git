# -*- coding: utf-8 -*-
"""时序诊断：模型比人早多久开始动？

    python 7_时序诊断.py

现象：模型"动作过早" —— 障碍还远就换道，等真到时候反而因为动作锁定按不了。
这个脚本用采集数据量化它：

  1. 把输入窗口的结束位置从"按键前 146ms"往后挪到"按键前 21ms"，
     看模型在哪一档就开始输出正确动作。
     * 数据只存了按键前 296ms，6 帧窗口跨度 167ms，所以最早只能测到 -146ms。
       在 -146ms 就动的样本，真实提前量只会更大。
  2. 每个样本"最早在哪一档输出正确动作"的分布。
  3. 阈值扫描：提高阈值能不能把动作压后？代价是多少（错过换道的比例）。

结论通常是：阈值不是好办法（压后一点就开始大量漏动），
真正的补偿手段是部署端的 --delay-ms（让模型看 now-delay 时刻的画面）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import fields

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collector.config import Config
from collector.dataset import (build_eval_transform, flatten_frames, load_sessions,
                               read_strip, select_window_indices, to_video_tensor)
from collector.model import build_model
from collector.runs import pick_run


def load_cfg(sessions) -> Config:
    cfg = Config()
    for d in sessions:
        p = os.path.join(d, "session.json")
        if os.path.isfile(p):
            raw = json.load(open(p, encoding="utf-8")).get("config", {})
            valid = {f.name for f in fields(Config)}
            for k, v in raw.items():
                if k in valid:
                    if k in ("game_res", "game_crop", "labels", "animation_neg_end_ms",
                              "approach_neg_end_ms"):
                        v = tuple(v)
                    setattr(cfg, k, v)
            break
    cfg.validate()
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="时序诊断：模型提前多久开始动")
    ap.add_argument("--run", default=None, help="训练结果目录，不填就用效果最好的那个")
    ap.add_argument("--sessions", nargs="+", default=None, help="数据集会话；不填用 runs/<名>/ 对应不上时手填")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    if not args.run:
        args.run = pick_run()
        print()
    else:
        args.run = args.run if os.path.isabs(args.run) else os.path.join(HERE, args.run)

    if not args.sessions:
        # 默认用 dataset/ 里最新的会话
        root = os.path.join(HERE, "dataset")
        cands = [os.path.join(root, d) for d in os.listdir(root)
                 if os.path.isfile(os.path.join(root, d, "meta.jsonl"))]
        if not cands:
            raise SystemExit("[错误] dataset/ 下没有会话，请用 --sessions 指定")
        args.sessions = [max(cands, key=os.path.getmtime)]
    else:
        args.sessions = [s if os.path.isabs(s) else os.path.join(HERE, s) for s in args.sessions]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(args.run, "best.pt"), map_location=device, weights_only=False)
    labels, K = ck["labels"], ck["in_frames"]
    NIDX = labels.index("N")
    cfg = load_cfg(args.sessions)
    samples = load_sessions(args.sessions)
    model = build_model(K, len(labels), pretrained=False).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    tf = build_eval_transform(args.img_size)

    ts = samples[0]["frame_times_ms"]
    print("=" * 76)
    print(f"模型 {args.run}")
    print(f"数据 {len(samples)} 个样本，{args.sessions}")
    print("=" * 76)
    print(f"保存窗口覆盖 {ts[0]:.0f}ms ~ {ts[-1]:.0f}ms（相对按键）")
    print(f"输入 = 连续 {K} 帧，跨度 {(K-1)*1000.0/cfg.save_fps:.0f}ms")
    print(f"=> 最早能凑齐 {K} 帧的窗口结束位置 = {ts[K-1]:.0f}ms（再往前就没数据了）")
    print()

    def prob(meta, end_ms):
        idx = select_window_indices(meta, K, cfg.input_stride, end_ms)
        x = flatten_frames(tf(to_video_tensor(read_strip(meta, idx)))).unsqueeze(0).to(device)
        with torch.no_grad():
            return torch.softmax(model(x).float(), dim=1)[0].cpu().numpy()

    keys = [s for s in samples if s["label"] != "N"]
    # 从"最早能凑齐 K 帧"一路铺到 0，密集一点才看得出动作是在哪一档开始出现的
    lo = float(ts[K - 1])
    n_lad = 11
    ends = sorted(set([round(lo * (1.0 - i / (n_lad - 1.0)), 1) for i in range(n_lad)]
                      + [-float(cfg.input_end_offset_ms)]))

    print(f"【1】把输入窗口结束位置往后挪，看模型何时开始输出正确动作（{len(keys)} 个按键样本）")
    print()
    print("%-12s %9s %11s %13s %12s" % ("窗口结束于", "正确率", "输出方向键", "正确且conf>.85", "正确动作均conf"))
    print("-" * 62)
    for e in ends:
        corr = act = strong = 0
        confs = []
        for s in keys:
            p = prob(s, e)
            pred, conf = int(p.argmax()), float(p.max())
            if pred == labels.index(s["label"]):
                corr += 1; confs.append(conf)
                if conf >= 0.85:
                    strong += 1
            if pred != NIDX:
                act += 1
        n = len(keys)
        print("%-12s %8.1f%% %10.1f%% %12.1f%% %12.3f"
              % ("%+.0fms" % (e if abs(e) > 1e-9 else 0.0), 100.0*corr/n,
                 100.0*act/n, 100.0*strong/n,
                 np.mean(confs) if confs else 0.0))

    print()
    print("【2】每个样本『最早在哪一档输出正确动作』")
    first = Counter()
    for s in keys:
        t = labels.index(s["label"])
        got = "从未"
        for e in ends:
            if int(prob(s, e).argmax()) == t:
                got = "%.0fms" % e
                break
        first[got] += 1
    for k, v in sorted(first.items(), key=lambda kv: -kv[1]):
        print("     %-10s %4d 个 (%.0f%%)" % (k, v, 100.0*v/len(keys)))
    earliest = "%.0fms" % ends[0]
    print(f"     —— 『{earliest}』这一档占 {100.0*first.get(earliest,0)/len(keys):.0f}%，"
          f"说明模型在能测到的最早时刻就已经决定了，真实提前量更大")

    print()
    print("【3】如果靠提高阈值来压后动作，代价有多大")
    cache = []
    for s in keys:
        lab = labels.index(s["label"])
        cache.append([(int(p.argmax()), float(p.max()), int(p.argmax()) == lab)
                      for p in (prob(s, e) for e in ends)])
    print()
    print("%-8s %16s %16s" % ("阈值", "最早档就动", "整段没动作(漏动)"))
    print("-" * 44)
    for thr in (0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99):
        at0 = never = 0
        for row in cache:
            fired = next((i for i, (p, c, ok) in enumerate(row) if p != NIDX and c >= thr), None)
            if fired is None:
                never += 1
            elif fired == 0:
                at0 += 1
        print("%-8.2f %15.1f%% %16.1f%%" % (thr, 100.0*at0/len(keys), 100.0*never/len(keys)))
    print()
    print("结论：阈值调高能压后一点，但漏动会迅速上升 —— 别指望它。")
    print("      要补偿提前量用部署端的延迟取帧：")
    print("          python 9_自动玩.py --dry-run --delay-ms 150")
    print("      根治办法是让训练时显式学会等待 —— config.py 里已经有：")
    print("          save_pre_ms = 800                        （采集时保存按键前 800ms）")
    print("          approach_neg_enabled = True              （合成「障碍还远 -> N」负样本）")
    print("      重新采集后用 7_时序诊断.py 对比本表，就能看出决策边界有没有被推后。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
