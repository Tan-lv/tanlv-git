# -*- coding: utf-8 -*-
"""把多个采集会话合并成一个（重新编号 + 复制图片 + 重写索引）。

    python tools/merge_sessions.py --out dataset/20260918_merged \
        dataset/20260917_183851 dataset/20260918_103227

为什么必须重新编号：每个会话都是从 A_000000 开始编的，直接合并会撞号 ——
后拷进去的图片会把前一张**覆盖掉**（之前就踩过这个坑，一条索引指向别人的图）。

合并前会检查各会话的 frame_w/frame_h/sep/labels/game_res/game_crop/
save_pre_ms/save_post_ms/save_fps 是否完全一致，不一致直接拒绝（模型输入会对不上）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PAOKU = os.path.dirname(HERE)

# 这些字段必须完全一致，否则不同会话的画面/时序对不上
MUST_MATCH = ("labels", "frame_w", "game_crop", "game_res",
              "save_pre_ms", "save_post_ms", "save_fps", "sep")


def load_session(d: str):
    mp = os.path.join(d, "meta.jsonl")
    if not os.path.isfile(mp):
        raise SystemExit(f"[错误] {d} 里没有 meta.jsonl")
    rows = [json.loads(l) for l in io.open(mp, encoding="utf-8") if l.strip()]
    sp = os.path.join(d, "session.json")
    cfg = json.load(io.open(sp, encoding="utf-8"))["config"] if os.path.isfile(sp) else {}
    return rows, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="合并多个采集会话")
    ap.add_argument("sessions", nargs="+", help="要合并的会话目录（按此顺序编号）")
    ap.add_argument("--out", required=True, help="输出会话目录")
    ap.add_argument("--force", action="store_true", help="输出目录已存在时清空重建")
    ap.add_argument("--move", action="store_true",
                    help="移动而不是复制（省磁盘，但源会话会被清空）")
    args = ap.parse_args()

    dirs = [s if os.path.isabs(s) else os.path.join(PAOKU, s) for s in args.sessions]
    out = args.out if os.path.isabs(args.out) else os.path.join(PAOKU, args.out)

    # ---------- 1) 读入 + 校验 ----------
    loaded = []
    for d in dirs:
        rows, cfg = load_session(d)
        loaded.append((d, rows, cfg))
        print(f"  {os.path.basename(d):<24} {len(rows)} 条索引")

    base = loaded[0][2]
    for d, _rows, cfg in loaded[1:]:
        for k in MUST_MATCH:
            if k not in base or k not in cfg:
                continue
            if list(base[k]) != list(cfg[k]) if isinstance(base[k], (list, tuple)) else base[k] != cfg[k]:
                raise SystemExit(f"[错误] {os.path.basename(d)} 的 {k} 与第一个会话不一致："
                                 f"{cfg[k]} != {base[k]}\n       画面/时序对不上，不能合并。")
    print(f"  配置一致性检查通过（{', '.join(MUST_MATCH)}）")

    if os.path.isdir(out):
        if not args.force:
            raise SystemExit(f"[错误] {out} 已存在，加 --force 清空重建")
        shutil.rmtree(out)
    for lab in base.get("labels", ["A", "D", "N", "S", "W"]):
        os.makedirs(os.path.join(out, lab), exist_ok=True)
    os.makedirs(os.path.join(out, "debug"), exist_ok=True)

    # ---------- 2) 合并 ----------
    counters = Counter()
    out_rows = []
    seen_per_session = []
    n_dup = n_missing = 0
    t0 = time.time()
    for si, (d, rows, _cfg) in enumerate(loaded):
        seen = set()
        for r in rows:
            f = r["file"]
            if f in seen:            # 同一会话里重复引用同一张图，只留最后一次
                n_dup += 1
                continue
            seen.add(f)
            src = os.path.join(d, f)
            if not os.path.isfile(src):
                n_missing += 1
                continue
            lab = r["label"]
            new_id = "%s_%06d" % (lab, counters[lab])
            counters[lab] += 1
            rel = "%s/%s.jpg" % (lab, new_id)
            dst = os.path.join(out, rel)
            if args.move:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)
            nr = dict(r)
            nr["id"] = new_id
            nr["file"] = rel
            nr["session"] = os.path.basename(out.rstrip("\\/"))
            nr["merged_from"] = os.path.basename(d.rstrip("\\/"))
            out_rows.append(nr)
        seen_per_session.append(len(seen))
        print(f"  [{si+1}/{len(loaded)}] {os.path.basename(d):<24} 取用 {len(seen)} 条"
              f"  累计 {len(out_rows)} 条  {time.time()-t0:.0f}s")

    # ---------- 3) 写出 ----------
    with io.open(os.path.join(out, "meta.jsonl"), "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(base)
    meta["session_name"] = os.path.basename(out.rstrip("\\/"))
    with io.open(os.path.join(out, "session.json"), "w", encoding="utf-8") as f:
        json.dump({"config": meta, "session": meta["session_name"],
                   "started_at": time.time(),
                   "started_at_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "merged_from": [os.path.basename(d.rstrip("\\/")) for d in dirs]},
                  f, ensure_ascii=False, indent=2)

    cnt = Counter(r["label"] for r in out_rows)
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fs in os.walk(out) for f in fs)
    print()
    print(f"合并完成 -> {out}")
    print(f"  {len(out_rows)} 个样本   {sz/1e9:.2f} GB   用时 {time.time()-t0:.0f}s")
    print(f"  类别: {dict(sorted(cnt.items()))}")
    if n_dup or n_missing:
        print(f"  跳过：重复引用 {n_dup} 条，缺图 {n_missing} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
