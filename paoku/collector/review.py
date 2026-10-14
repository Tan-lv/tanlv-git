# -*- coding: utf-8 -*-
"""采集后复核工具：逐个查看样本的胶片条，删除坏的 / 改正标错的。

用法（在 PyCharm 里直接运行 3_复核数据.py，或命令行）:
    python -m collector.review dataset/run01
    python -m collector.review dataset/run01 --only A      # 只看某一类
    python -m collector.review dataset/run01 --suspect     # 只从可疑样本开始看

界面说明
    绿色框 = 实际喂给模型的帧（全部在按键之前）
    红线   = 按键时刻 t=0
    每帧上方的数字 = 该帧相对按键时刻的毫秒偏移

按键
    n / 空格 / →   下一个
    p / ←          上一个
    d / Delete     删除这个样本
    1 2 3 4 ...    改标签（对应屏幕上列出的类别顺序）
    f              跳到下一个可疑样本
    q / Esc        保存并退出（改动是即时生效的）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import fields

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .config import Config
from .dataset import select_input_indices
from .recorder import resolve_path

FONT = cv2.FONT_HERSHEY_SIMPLEX
CANVAS_W = 1500
# 窗口内相邻帧最大灰度差低于此值 -> 判定为"画面静止"的废样本
FROZEN_DIFF_THRESHOLD = 3.0


def load_cfg(root: str) -> Config:
    cfg = Config()
    p = os.path.join(root, "session.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            raw = json.load(f).get("config", {})
        valid = {fl.name for fl in fields(Config)}
        for k, v in raw.items():
            if k in valid:
                if k in ("game_res", "game_crop", "labels"):
                    v = tuple(v)
                setattr(cfg, k, v)
    return cfg


def load_meta(root: str):
    p = os.path.join(root, "meta.jsonl")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"找不到 {p}")
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                m = json.loads(line)
                m["_root"] = root
                out.append(m)
    return out


def save_meta(root: str, samples) -> None:
    p = os.path.join(root, "meta.jsonl")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for m in samples:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def mark_suspects(root: str, samples, verbose: bool = True) -> None:
    """逐个样本真正比较帧内容，找出"整个窗口画面都没变"的废样本。

    这类样本来自误采（采到了菜单/结算画面）或模拟器卡死，对训练只有害处。
    判据：窗口内相邻帧的最大灰度差 —— 真实跑酷画面一直在滚动，这个值会明显偏大；
    静止画面则接近 0（剩下的只是 JPEG 块相位带来的压缩噪声）。
    """
    t0 = time.time()
    diffs = []
    for m in samples:
        fp = os.path.join(root, m["file"])
        m["_size"] = os.path.getsize(fp) if os.path.isfile(fp) else 0
        m["_suspect"] = True
        m["_maxdiff"] = 0.0
        if m["_size"] == 0:
            continue
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        fh, fw, sep = m["frame_h"], m["frame_w"], m["sep"]
        prev, mx = None, 0.0
        for k in range(m["n_frames"]):
            x0 = k * (fw + sep)
            tile = img[0:fh:6, x0:x0 + fw:6].astype(np.int16)   # 稀疏采样，够用且快
            if prev is not None:
                d = float(np.abs(tile - prev).mean())
                if d > mx:
                    mx = d
            prev = tile
        m["_maxdiff"] = mx
        m["_suspect"] = mx < FROZEN_DIFF_THRESHOLD
        diffs.append(mx)

    idxs = [i for i, m in enumerate(samples) if m["_suspect"]]
    n = len(idxs)
    if verbose and diffs:
        d = np.array(diffs)
        print(f"    帧间变化量: 中位 {np.median(d):.2f}  5%分位 {np.percentile(d,5):.2f}  "
              f"最小 {d.min():.2f}  最大 {d.max():.2f}")
        print(f"    自动标记 {n} 个可疑样本（窗口内变化量 < {FROZEN_DIFF_THRESHOLD}，画面基本静止）"
              f"，用时 {time.time()-t0:.1f}s")
        if idxs:
            show = "、".join(f"第 {i+1} 个" for i in idxs[:15])
            more = f" 等 {n} 个" if n > 15 else ""
            print(f"    它们的位置: {show}{more}   （共 {len(samples)} 个样本）")
            print("    按 f 键自动跳过去（跳完会自动绕回开头继续循环）")
        else:
            print("    没有发现可疑样本 —— 说明采集时 F8 用得很好")
    return idxs


def render(meta: dict, cfg: Config, labels, idx: int, total: int, status: str = "",
           suspect_n: int = 0) -> np.ndarray:
    path = os.path.join(meta["_root"], meta["file"])
    img = cv2.imread(path)
    top, bottom = 46, 26
    if img is None:
        canvas = np.zeros((200, CANVAS_W, 3), np.uint8)
        cv2.putText(canvas, f"FILE MISSING: {meta['file']}", (20, 60), FONT, 1.0, (0, 0, 255), 2)
        return canvas

    fh, fw, sep = meta["frame_h"], meta["frame_w"], meta["sep"]
    n = meta["n_frames"]
    times = meta["frame_times_ms"]
    sel = set(select_input_indices(meta, cfg))
    anchor_i = int(np.argmin([abs(t) for t in times]))

    scale = CANVAS_W / img.shape[1]
    sh = max(1, int(img.shape[0] * scale))
    canvas = np.zeros((top + sh + bottom, CANVAS_W, 3), np.uint8)
    canvas[:, :] = (28, 28, 28)
    canvas[top:top + sh] = cv2.resize(img, (CANVAS_W, sh), interpolation=cv2.INTER_AREA)

    for i in range(n):
        x0 = int(i * (fw + sep) * scale)
        x1 = int((i * (fw + sep) + fw) * scale)
        color = (0, 255, 0) if i in sel else (90, 90, 90)
        cv2.rectangle(canvas, (x0, top), (x1 - 1, top + sh - 1), color, 2)
        cv2.putText(canvas, "%+.0f" % times[i], (x0 + 3, top - 6), FONT, 0.45,
                    (0, 255, 0) if i in sel else (150, 150, 150), 1)
    ax = int((anchor_i * (fw + sep) + fw // 2) * scale)
    cv2.line(canvas, (ax, top), (ax, top + sh), (0, 0, 255), 2)

    flag = "  <<< SUSPECT (frozen)" if meta.get("_suspect") else ""
    cv2.putText(canvas, f"[{idx+1}/{total}] {meta['id']}  label={meta['label']}  "
                        f"kind={meta['kind']}  {meta.get('_size',0)/1024:.0f}KB  "
                        f"motion={meta.get('_maxdiff',0):.1f}{flag}",
                (8, 20), FONT, 0.6,
                (0, 0, 255) if meta.get("_suspect") else (255, 255, 255), 1)
    cv2.putText(canvas, f"green = model input ({len(sel)} frames, all BEFORE the key press)   "
                        f"red = key press t=0", (8, 40), FONT, 0.5, (0, 255, 255), 1)
    help_txt = (f"[n]ext [p]rev [d]elete [f]ind-suspect({suspect_n})   "
                + "  ".join(f"{i+1}={l}" for i, l in enumerate(labels)) + "   [q]uit")
    cv2.putText(canvas, help_txt, (8, canvas.shape[0] - 8), FONT, 0.5, (200, 200, 200), 1)

    # 状态提示做成画面正中醒目的大横幅 —— 之前写在右上角小字，根本看不见
    if status:
        scale, thick = 1.0, 2
        (tw, th), _ = cv2.getTextSize(status, FONT, scale, thick)
        cx, cy = max(4, (CANVAS_W - tw) // 2), top + sh // 2
        cv2.rectangle(canvas, (cx - 18, cy - th - 16), (cx + tw + 18, cy + 16), (0, 0, 0), -1)
        cv2.rectangle(canvas, (cx - 18, cy - th - 16), (cx + tw + 18, cy + 16), (0, 220, 255), 2)
        cv2.putText(canvas, status, (cx, cy + 2), FONT, scale, (0, 255, 255), thick)
    return canvas


def do_delete(root: str, samples, i: int) -> str:
    m = samples[i]
    fp = os.path.join(root, m["file"])
    if os.path.isfile(fp):
        os.remove(fp)
    del samples[i]
    save_meta(root, samples)
    return f"DELETED {m['id']}"


def do_relabel(root: str, samples, i: int, new_label: str) -> str:
    m = samples[i]
    old_label = m["label"]
    if new_label == old_label:
        return f"already {new_label}"
    old_fp = os.path.join(root, m["file"])
    base = os.path.basename(m["file"])
    new_rel = os.path.join(new_label, base).replace("\\", "/")
    new_fp = os.path.join(root, new_rel)
    os.makedirs(os.path.dirname(new_fp), exist_ok=True)
    if os.path.abspath(old_fp) != os.path.abspath(new_fp):
        if os.path.exists(new_fp):                      # 极少见，避免覆盖
            stem, ext = os.path.splitext(base)
            new_rel = os.path.join(new_label, stem + "_r" + ext).replace("\\", "/")
            new_fp = os.path.join(root, new_rel)
        if os.path.isfile(old_fp):
            shutil.move(old_fp, new_fp)
    m["label"] = new_label
    m["file"] = new_rel
    save_meta(root, samples)
    return f"{m['id']}: {old_label} -> {new_label}"


def main() -> None:
    ap = argparse.ArgumentParser(description="采集后复核：删除坏样本 / 改正错标签")
    ap.add_argument("session", help="会话目录，例如 dataset/run01")
    ap.add_argument("--only", default=None, help="只看某一类，例如 --only A")
    ap.add_argument("--suspect", action="store_true", help="从第一个可疑样本开始")
    ap.add_argument("--no-window", action="store_true", help="只做统计，不开窗口")
    args = ap.parse_args()

    root = args.session if os.path.isabs(args.session) else resolve_path(args.session)
    if not os.path.isdir(root):
        print(f"[FAIL] 找不到会话目录: {root}")
        print("       （相对路径是相对 paoku 目录，例如 dataset/run01）")
        return

    cfg = load_cfg(root)
    samples = load_meta(root)
    if not samples:
        print("这个会话里没有样本")
        return
    labels = list(cfg.labels)
    print("=" * 78)
    print(f"复核会话: {os.path.abspath(root)}")
    print(f"    共 {len(samples)} 个样本，类别: {sorted({m['label'] for m in samples})}")
    print(f"    模型输入: 连续 {cfg.input_frames} 帧, 结束于按键前 {cfg.input_end_offset_ms}ms")
    if args.only:
        samples = [m for m in samples if m["label"] == args.only]
        print(f"    只看类别 {args.only}: {len(samples)} 个")
    if not samples:
        return
    suspects = mark_suspects(root, samples)
    if args.no_window:
        return

    win = "review  (n/p move, d delete, 1-4 relabel, f suspect, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("    提示：窗口打开后，如果按键没反应，先用鼠标点一下这个窗口让它获得焦点")
    print("          （OpenCV 的窗口不获得焦点就收不到按键）")
    i, deleted, relabeled = 0, 0, 0
    status = ""
    while True:
        i = max(0, min(i, len(samples) - 1))
        cv2.imshow(win, render(samples[i], cfg, labels, i, len(samples), status, len(suspects)))
        status = ""
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k in (ord("n"), ord(" "), 83):
            i += 1
            if i >= len(samples):
                print("已经到最后一个了")
                i = len(samples) - 1
        elif k in (ord("p"), 81):
            i -= 1
        elif k in (ord("d"), 255, 8):          # d / Delete
            status = do_delete(root, samples, i)
            deleted += 1
            print(status)
            if not samples:
                print("全部删完了")
                break
        elif ord("1") <= k <= ord("9"):
            idx = k - ord("1")
            if idx < len(labels):
                status = do_relabel(root, samples, i, labels[idx])
                relabeled += 1
                print(status)
        elif k == ord("f"):
            if not suspects:
                status = "没有可疑样本"
            else:
                nxt = next((j for j in range(i + 1, len(samples)) if samples[j].get("_suspect")), None)
                wrapped = False
                if nxt is None:                    # 后面没有了 -> 绕回开头
                    nxt = next((j for j in range(0, i) if samples[j].get("_suspect")), None)
                    wrapped = True
                if nxt is None:
                    status = "当前位置就是唯一的可疑样本"
                else:
                    i = nxt
                    status = f"跳到可疑样本：第 {i+1} / {len(samples)} 个" + ("（已绕回开头）" if wrapped else "")
            print(status)
    cv2.destroyAllWindows()
    print("=" * 78)
    print(f"复核结束: 删除 {deleted} 个，改标签 {relabeled} 个，剩余 {len(samples)} 个")
    from collections import Counter
    print("    类别分布: " + str(dict(sorted(Counter(m['label'] for m in samples).items()))))
    print("=" * 78)


if __name__ == "__main__":
    main()
