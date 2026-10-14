# -*- coding: utf-8 -*-
"""生成一份"假"数据集，用来在不打开模拟器的情况下验证整条流水线。

    python tools/make_synth_dataset.py --out dataset/_synth --samples 120

为什么需要它：采集器、训练、评估、部署四段代码只能靠真机串起来，一旦某一段的接口
对不上（比如 9_自动玩.py 里 RingBuffer 的参数名写错、负样本窗口算错），要等到真跑
才发现。有了合成数据，训练/评估/时序诊断可以随时离线跑通。

生成的会话目录以 _ 开头，**不会被 5_训练模型.py 的"自动挑选最新会话"选中**，
也不会被 9_自动玩.py / 6_评估效果.py 的"自动挑选最新"选中。

生成完可以这样验证：
    python 5_训练模型.py  --session dataset/_synth --out runs/_synth --smoke --workers 0
    python 6_评估效果.py   --run runs/_synth --sessions dataset/_synth
    python 7_时序诊断.py --run runs/_synth --sessions dataset/_synth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from collector.config import Config      # noqa: E402

CLASS_X = {"A": 40, "D": 90, "N": 140, "S": 190, "W": 230}


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成数据集（离线验证流水线用）")
    ap.add_argument("--out", default="dataset/_synth", help="输出目录（建议以 _ 开头）")
    ap.add_argument("--samples", type=int, default=120, help="样本总数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pre-ms", type=int, default=None,
                    help="覆盖 save_pre_ms（想测旧数据兼容就设成 300）")
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(HERE), args.out)
    cfg = Config()
    if args.pre_ms is not None:
        cfg.save_pre_ms = int(args.pre_ms)
    cfg.validate()
    cfg.session_name = os.path.basename(out.rstrip("\\/"))

    for lab in cfg.labels:
        os.makedirs(os.path.join(out, lab), exist_ok=True)

    fw, fh, sep, n = cfg.frame_w, cfg.frame_h, cfg.sep, cfg.saved_frames
    step = (cfg.save_pre_ms + cfg.save_post_ms) / float(n - 1)
    times = [round(-cfg.save_pre_ms + i * step, 2) for i in range(n)]
    print(f"输出 {out}")
    print(f"  save_pre_ms={cfg.save_pre_ms}  帧数={n}  胶片条宽={n*fw+(n-1)*sep}px")

    rng = np.random.default_rng(args.seed)
    counters = {k: 0 for k in cfg.labels}
    rows = []
    t = time.time() - 600
    t0 = time.time()
    labs = [l for l in cfg.labels if l != "N"]
    for i in range(args.samples):
        # 交替制造"紧凑连按"和"长空闲"，让两类合成负样本都有素材
        if i % 11 == 10:
            lab, kind = "N", "auto_n"
        else:
            lab, kind = labs[i % len(labs)], "key"
        t += 0.35 if i % 5 == 0 else (2.4 if i % 3 == 0 else 0.9)

        strip = np.full((fh, n * fw + (n - 1) * sep, 3), 255, dtype=np.uint8)
        base = np.linspace(30, 200, fh, dtype=np.uint8)[:, None]
        cx = CLASS_X.get(lab, 140)
        for k in range(n):
            fr = np.zeros((fh, fw, 3), dtype=np.uint8)
            fr[:, :, 0] = fr[:, :, 1] = fr[:, :, 2] = base
            y = int(fh * (1.0 - (k + 1) / n))          # 越接近按键越靠下
            fr[max(0, y - 6):y + 6, max(0, cx - 6):cx + 6] = (255, 60, 60)
            x0 = k * (fw + sep)
            strip[:, x0:x0 + fw] = fr
        cid = "%s_%06d" % (lab, counters[lab]); counters[lab] += 1
        rel = "%s/%s.jpg" % (lab, cid)
        Image.fromarray(strip).save(os.path.join(out, rel), "JPEG",
                                    quality=cfg.jpeg_quality, subsampling=cfg.jpeg_subsampling)
        rows.append({"id": cid, "label": lab, "kind": kind,
                     "src_key": ("n" if lab == "N" else "?"),
                     "file": rel, "session": cfg.session_name,
                     "t_anchor": round(t, 6), "frame_times_ms": times,
                     "n_frames": n, "frame_w": fw, "frame_h": fh, "sep": sep,
                     "requested_pre_ms": cfg.save_pre_ms,
                     "requested_post_ms": cfg.save_post_ms,
                     "truncated_window": False, "capture_fps": 40.0,
                     "speed_px_s": round(float(rng.uniform(5, 300)), 1), "speed_conf": 0.9})

    with open(os.path.join(out, "meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "session.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "session": cfg.session_name,
                   "started_at": t0, "started_at_str": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False, indent=2)

    sz = sum(os.path.getsize(os.path.join(dp, fn)) for dp, dn, fs in os.walk(out) for fn in fs)
    gaps = np.diff([r["t_anchor"] for r in rows if r["label"] != "N"])
    print(f"  生成 {len(rows)} 个样本  {sz/1e6:.1f} MB  ({sz/len(rows)/1024:.0f} KB/样本)  "
          f"{time.time()-t0:.0f}s")
    print(f"  类别 {dict(counters)}")
    print(f"  按键间隔 中位 {np.median(gaps):.2f}s，<0.6s 的 {int((gaps<0.6).sum())} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
