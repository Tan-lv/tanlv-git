# -*- coding: utf-8 -*-
r"""把已有模型 model_meta.json 里记的取景参数（game_res / game_crop）改成新的。

用途（2026-10-10，用户转横屏）：
    老模型是在竖屏 720x1280 / 裁剪 (0,200,720,880) 下训的，它们的 model_meta.json 里
    也记着这套参数；而 9_自动玩.py 是按 model_meta.json 去抓图的 -> 横屏下直接报错。

    ★ 模型权重和 game_res/game_crop **完全无关**（网络只吃 18x224x224），
      所以改这里不会动到任何权重，只是让模型去看另一块画面。
      换句话说：这是"拿旧模型直接试横屏"的唯一改动，随时可以还原。

注意：改完落盘帧形状必须仍然是 240x293，否则前处理的有效视野会变。
     脚本会检查这一点并警告。

用法：
    python tools\模型取景重指向.py                      # 干跑：列出哪些模型还是竖屏参数
    python tools\模型取景重指向.py --apply               # 只改 --runs 指定的（默认只列不改）
    python tools\模型取景重指向.py --runs runs\A --apply
    python tools\模型取景重指向.py --all-runs --apply    # 改 runs 下所有模型
    python tools\模型取景重指向.py --runs runs\A --restore   # 还原成原来的竖屏参数
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BAK = "model_meta.json.portrait_bak"      # 原始（竖屏）备份，只在第一次改时写


def parse_ints(s: str):
    return [int(v) for v in str(s).replace("x", ",").replace("X", ",")
            .replace(" ", ",").split(",") if v.strip()]


def meta_path(run: str) -> str:
    return os.path.join(run, "model_meta.json")


def find_runs() -> list:
    root = os.path.join(HERE, "runs")
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isfile(meta_path(d)):
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="改模型元数据里的取景参数（可回滚）")
    ap.add_argument("--runs", nargs="+", default=None, help="模型目录；不填=只扫描列出")
    ap.add_argument("--all-runs", action="store_true", help="对 runs 下所有模型生效")
    ap.add_argument("--game-res", default="1280,720")
    ap.add_argument("--game-crop", default="345,0,590,720")
    ap.add_argument("--apply", action="store_true", help="真的写入（不加则只打印）")
    ap.add_argument("--restore", action="store_true", help="从 .portrait_bak 还原")
    args = ap.parse_args()

    gr = parse_ints(args.game_res)
    gc = parse_ints(args.game_crop)
    if len(gr) != 2 or len(gc) != 4:
        raise SystemExit("--game-res 要 宽,高；--game-crop 要 x,y,宽,高")
    if gc[0] + gc[2] > gr[0] or gc[1] + gc[3] > gr[1]:
        raise SystemExit(f"裁剪框 {gc} 超出画面 {gr}")
    frame_h = int(round(240 * gc[3] / gc[2]))
    if frame_h != 293:
        print(f"[警告] 新裁剪框 {gc[2]}x{gc[3]} 会让落盘帧变成 240x{frame_h}（原来是 240x293）")
        print("       前处理是「短边缩到 224 + 中心裁 224」，形状一变模型的有效视野就完全不同，")
        print("       旧模型大概率直接失效。横屏请用 590x720 这类同比例竖窗。")
        if args.apply:
            raise SystemExit("比例不对，已中止（去掉 --apply 只看看也行）")

    if args.all_runs:
        runs = find_runs()
    elif args.runs:
        runs = [p if os.path.isabs(p) else os.path.join(HERE, p) for p in args.runs]
    else:
        runs = find_runs()

    n_changed = 0
    for rd in runs:
        mp = meta_path(rd)
        name = os.path.basename(rd.rstrip(os.sep))
        if not os.path.isfile(mp):
            print(f"  [跳过] 没有 model_meta.json: {name}")
            continue
        bak = os.path.join(rd, BAK)
        if args.restore:
            if not os.path.isfile(bak):
                print(f"  [跳过] 没有备份可还原: {name}")
                continue
            if args.apply:
                shutil.copy2(bak, mp)
            print(f"  [还原] {name}")
            n_changed += 1
            continue
        m = json.load(open(mp, encoding="utf-8"))
        old_res, old_crop = m.get("game_res"), m.get("game_crop")
        if list(old_res or []) == gr and list(old_crop or []) == gc:
            print(f"  [已是横屏] {name}")
            continue
        print(f"  {name}")
        print(f"      game_res  {old_res} -> {gr}")
        print(f"      game_crop {old_crop} -> {gc}")
        if args.apply:
            if not os.path.isfile(bak):
                shutil.copy2(mp, bak)          # 只在第一次改时留原始备份
            m["game_res"] = gr
            m["game_crop"] = gc
            m["geometry_note"] = ("%s 改为横屏取景（原 game_res=%s game_crop=%s）；"
                                  "权重未变，可用 --restore 还原"
                                  % (time.strftime("%Y-%m-%d %H:%M"), old_res, old_crop))
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)
        n_changed += 1

    print("")
    if not args.apply:
        print(f"[干跑] 有 {n_changed} 个模型需要改；确认无误后加 --apply 真写。")
    else:
        print(f"[完成] 处理了 {n_changed} 个模型。原参数备份在各自的 {BAK}；"
              "要还原：--restore --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
