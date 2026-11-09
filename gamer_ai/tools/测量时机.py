# -*- coding: utf-8 -*-
r"""量出「模型什么时候出手」vs「人手什么时候按」的时间差，并给出该用的 end_offset。

为什么需要它（要求 4：解决过早/过晚，准确获取按键触发时间差）：
    训练时正样本窗口"结束于按键前 end_offset 毫秒"。这个值决定模型被训练成
    「看到哪个时刻的画面就出手」。设成 0 = 人手按下的那一刻才出手；
    但实际执行链路（抓帧 + 推理 + adb input swipe）要 230~270ms 才让角色动起来，
    所以必须让模型**提前**出手，提前量 = 这个链路延迟。

    这个脚本直接量：对每个真实的按键样本，把窗口结束时间从早到晚扫一遍，
    找出模型第一次输出正确动作的时刻，再和真实按键时刻相减 —— 得到的就是
    「模型比人手早/晚多少」。最后按 建议end_offset = 延迟 + 中位差 给出结论。

用法：
    python tools\测量时机.py                    # 当前档案 + 最新模型，抽 250 个样本
    python tools\测量时机.py --limit 400
    python tools\测量时机.py --run <模型目录>
    python tools\测量时机.py --latency 250      # 执行链路延迟（默认 250ms）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # -> gamer_ai/
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                       # noqa: E402

from core import dataset as D                         # noqa: E402
from core import profiles as P                        # noqa: E402
from core.model import build_model                    # noqa: E402
from core.runs import list_runs, pick_run             # noqa: E402

ACTION = ("A", "D", "S", "W")          # 只对这四个方向/动作类谈"出手时机"


def load_model(run_dir):
    meta = json.load(open(os.path.join(run_dir, "model_meta.json"), encoding="utf-8"))
    labels = list(meta["labels"])
    K, SZ = int(meta["in_frames"]), int(meta.get("img_size", 224))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # ★ 2026-11-05：新模型可能是 3*K+1 通道（带速度标量），必须按 meta 建
    model = build_model(K, len(labels), pretrained=False,
                        in_channels=meta.get("in_channels"))
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.to(dev).eval()
    return meta, labels, K, SZ, dev, model


def main() -> int:
    ap = argparse.ArgumentParser(description="量模型出手时机 vs 人手按键的时间差")
    ap.add_argument("--run", default=None, help="模型目录；不给就用当前档案 runs/ 里最新的")
    ap.add_argument("--profile", default=None, help="区域档案名；不给就用当前档案")
    ap.add_argument("--limit", type=int, default=250, help="抽多少个按键样本（默认 250）")
    ap.add_argument("--latency", type=float, default=316.0,
                    help="从模型出结果到角色真的动，整条链路的延迟（毫秒）。"
                         "实测 316 = 抓帧 35(printwindow) + 推理 31 + adb input swipe 到"
                         "手指按下约 250。换 bitblt 可减 25，改成 291。")
    ap.add_argument("--conf", type=float, default=0.5, help="算不算'出手'的置信度门槛（默认 0.5）")
    a = ap.parse_args()

    prof = a.profile or P.resolve()
    run = a.run or pick_run(quiet=True, root=(P.runs_root(prof) if prof else None))
    if not run:
        print("找不到模型目录，先训练一个")
        return 1
    meta, labels, K, SZ, dev, model = load_model(run)
    print("模型:", os.path.basename(run.rstrip("\\/")))
    print("类别:", labels, " 输入", K, "帧  当前 end_offset =", meta.get("input_end_offset_ms", 0), "ms")

    sessions = []
    if prof:
        sessions = sorted(glob.glob(os.path.join(P.dataset_root(prof), "*")))
    samples = []
    for s in sessions:
        if os.path.isdir(s) and os.path.isfile(os.path.join(s, "meta.jsonl")):
            samples += D.load_session(s)
    if not samples:
        print("没找到数据。用 --profile 指定档案，或先在 2_采集.py 里采几局")
        return 1

    tf = D.build_eval_transform(SZ)
    cand = [m for m in samples if m["label"] in ACTION]
    # 均匀抽样，保证各类都有
    by = {}
    for m in cand:
        by.setdefault(m["label"], []).append(m)
    picked = []
    for lb, ms in sorted(by.items()):
        step = max(1, len(ms) // max(1, a.limit // max(len(by), 1)))
        picked += ms[::step][: max(1, a.limit // max(len(by), 1))]
    picked = picked[: a.limit]
    print("抽样：%s" % {lb: sum(1 for m in picked if m['label'] == lb) for lb in by})
    print()

    # 扫描范围：从按键前 400ms 到按键后 200ms
    res = {lb: [] for lb in ACTION}
    for n, m in enumerate(picked, 1):
        times = m.get("frame_times_ms") or []
        if len(times) < K + 2:
            continue
        step_ms = float(np.median(np.diff(times))) if len(times) > 1 else 33.0
        step_ms = max(10.0, step_ms)
        offs = [o for o in np.arange(-400, 200 + 1, step_ms)]
        xs, used = [], []
        for o in offs:
            if times[-1] < o or times[0] > o:
                continue
            idx = D.select_window_indices(m, K, 1, float(o))
            x1 = D.flatten_frames(tf(D.to_video_tensor(D.read_strip(m, idx))))
            if meta.get("speed_input"):
                # 与 core/dataset.py 完全一致：z = log1p(speed_px_s)/4.6，整图填同一个值
                import math
                _sp = float(m.get("speed_px_s") or 0.0)
                _z = min(math.log1p(max(0.0, _sp)) / 4.6, 2.0)
                x1 = torch.cat([x1, torch.full((1, x1.shape[-2], x1.shape[-1]),
                                               _z, dtype=x1.dtype)], dim=0)
            xs.append(x1)
            used.append(float(o))
        if not xs:
            continue
        batch = torch.stack(xs).to(dev)
        with torch.no_grad():
            prob = F.softmax(model(batch).float(), 1).cpu().numpy()
        gi = labels.index(m["label"])
        fire = None
        for j, o in enumerate(used):            # 从最早的时刻往后找
            if int(prob[j].argmax()) == gi and prob[j, gi] >= a.conf:
                fire = o
                break
        if fire is not None:
            res[m["label"]].append(fire)
        if n % 50 == 0:
            print("  已测 %d/%d" % (n, len(picked)))
        D.clear_strip_cache()
    if any(torch.cuda.is_available() for _ in (0,)):
        torch.cuda.empty_cache()

    print()
    print("=== 结果：模型第一次输出正确动作的时刻（相对人手按键，负数=更早）===")
    allv = []
    for lb in ACTION:
        v = res[lb]
        if not v:
            print("  %s  没有可用样本" % lb)
            continue
        allv += v
        print("  %s  n=%3d  中位 %+6.0f ms  p25 %+6.0f  p75 %+6.0f"
              % (lb, len(v), statistics.median(v),
                 np.percentile(v, 25), np.percentile(v, 75)))
    if not allv:
        print("一个都没测出来 —— 检查 --conf 是不是太高")
        return 1
    med = statistics.median(allv)
    cur = float(meta.get("input_end_offset_ms", 0) or 0)
    # ★ 这个公式必须在**当前 end_offset 的基础上**加差值：
    #   当前出手点 med，希望出手点 = -latency（这样加上链路延迟正好落在人手按键时刻）。
    #   而 end_offset 每 +1ms，出手点大约提前 1ms -> 需要再提前 (med + latency)。
    need = cur + (med + a.latency)
    print()
    print("=== 结论 ===")
    print("  总体中位：模型比人手%s %.0f ms" % ("早" if med < 0 else "晚", abs(med)))
    print("  执行链路延迟（实测常量）：%.0f ms" % a.latency)
    print("  -> 现在动作落地时间 = %.0f(出手点) + %.0f(延迟) = **%+.0f ms**（正数=比人手晚）"
          % (med, a.latency, med + a.latency))
    print("  -> 建议 end_offset = 当前 %.0f + 差值 %+.0f = **%.0f ms**" % (cur, med + a.latency, need))
    if abs(need - cur) >= 40:
        print("  ★ 差得比较多，建议把 4_训练.py 的 TRAIN['end_offset'] 改成 %d 再训一版" % round(need))
    elif abs(need - cur) >= 20:
        print("  差不多，可以先不动（差 %+.0f ms，约 %.1f 帧）" % (need - cur, (need - cur) / 33.3))
    else:
        print("  和当前值基本一致，不用改")
    print()
    print("  注意：end_offset 越大 = 越早出手。太大会让模型在障碍还看不清时就猜，准确性会掉。")
    print("  改完重训后，再跑一次这个脚本，中位差应该落到约 -%.0f ms。" % a.latency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
