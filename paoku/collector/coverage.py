# -*- coding: utf-8 -*-
"""速度覆盖度报告。

地铁跑酷的速度随存活时间上升。训练数据必须覆盖速度区间，否则模型一进高速段就废。

用法:
    python -m collector.coverage dataset/run01 [run02 ...]
    python -m collector.coverage dataset/run01 --compute     # meta 里没速度就现场算（较慢）
    python -m collector.coverage dataset/run01 --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from . import speed as speed_mod
from .dataset import read_strip, select_window_indices
from .recorder import resolve_path


def load_with_speed(root: str, compute: bool = False):
    p = os.path.join(root, "meta.jsonl")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                m = json.loads(line)
                m["_root"] = root
                out.append(m)
    missing = [m for m in out if m.get("speed_px_s") is None]
    if missing and compute:
        print(f"    正在为 {len(missing)} 个样本补算速度 ...")
        for i, m in enumerate(missing):
            times = m["frame_times_ms"]
            idx = list(range(0, m["n_frames"], 2))
            frames = read_strip(m, idx)
            sp, resp = speed_mod.speed_px_per_sec(
                frames, [times[k] for k in idx], stride=1)
            m["speed_px_s"] = round(sp, 1)
            m["speed_conf"] = round(resp, 3)
            if (i + 1) % 200 == 0:
                print(f"      {i+1}/{len(missing)}")
        with open(p, "w", encoding="utf-8") as f:
            for m in out:
                mm = {k: v for k, v in m.items() if not k.startswith("_")}
                f.write(json.dumps(mm, ensure_ascii=False) + "\n")
        print("    已写回 meta.jsonl")
    return out


def report(root: str, samples, bins: int = 10) -> dict:
    speeds = [float(m.get("speed_px_s") or 0.0) for m in samples]
    valid = [s for s in speeds if s > 0]
    loops = [m for m in samples if m["label"] != "N"]
    print(f"  {os.path.basename(root):<20} 样本 {len(samples):5d}（按键 {len(loops)}）  "
          f"{speed_mod.describe(speeds)}")
    if len(valid) < 10:
        print("      [提示] 有效速度样本太少，无法评估覆盖度")
        return {"root": root, "n": len(samples), "valid": len(valid)}
    edges, hist = speed_mod.speed_histogram(valid, bins=bins)
    total = hist.sum()
    print("      速度分布 (px/s):")
    for i in range(len(hist)):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * int(hist[i] / max(total, 1) * 60)
        pct = hist[i] / max(total, 1) * 100
        print(f"        {lo:6.0f} - {hi:6.0f}  {hist[i]:5d}  {pct:5.1f}%  {bar}")
    top = hist[-2:].sum() / max(total, 1)          # 最高两档占比
    low = hist[:3].sum() / max(total, 1)
    warn = []
    if top < 0.10:
        warn.append(f"最高速段只占 {top*100:.1f}%（建议 >= 10%）—— 高速样本不足，"
                    "模型上线后进高速段会失效")
    if low > 0.60:
        warn.append(f"最低速段占了 {low*100:.1f}% —— 前面几秒采太多，"
                    "建议每局玩久一点再开始记")
    for w in warn:
        print("      [警告] " + w)
    if not warn:
        print("      [OK] 速度覆盖看起来是均匀的")
    return {"root": root, "n": len(samples), "valid": len(valid),
            "p50": float(np.median(valid)), "p95": float(np.percentile(valid, 95)),
            "top2_pct": float(top), "low3_pct": float(low)}


def main() -> None:
    ap = argparse.ArgumentParser(description="速度覆盖度报告")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--compute", action="store_true", help="meta 里没速度就现场补算")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("速度覆盖度报告（游戏速度随存活时间上升，训练数据必须覆盖整个区间）")
    print("=" * 78)
    all_speeds = []
    rows = []
    for s in args.sessions:
        root = s if os.path.isabs(s) else resolve_path(s)
        if not os.path.isdir(root):
            print(f"  [跳过] 找不到 {root}")
            continue
        samples = load_with_speed(root, compute=args.compute)
        rows.append(report(root, samples, bins=args.bins))
        all_speeds += [float(m.get("speed_px_s") or 0.0) for m in samples]
    if len(rows) > 1:
        print("-" * 78)
        print("  合计: " + speed_mod.describe(all_speeds))
    print("-" * 78)
    print("  采集建议：")
    print("    * 每局都玩到速度明显变快之后再退出，不要在开局几秒就结束")
    print("    * 高速段（屏幕上障碍飞得很快）至少要占到 10% 以上的样本")
    print("    * 想补高速数据：多玩几局、每局尽量活久一点，用 --session 分开存")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["session", "n", "valid", "p50", "p95", "top2_pct", "low3_pct"])
            for r in rows:
                w.writerow([os.path.basename(r["root"]), r.get("n"), r.get("valid"),
                            r.get("p50"), r.get("p95"), r.get("top2_pct"), r.get("low3_pct")])
        print(f"  已写出 {args.csv}")


if __name__ == "__main__":
    main()
