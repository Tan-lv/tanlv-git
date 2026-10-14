# -*- coding: utf-8 -*-
r"""标注补数据 —— 把 10_迭代补数据.py 录下来的「错误时刻」变成能训练的标准样本。

一个事件里有撞车前 3 秒的逐帧原图。你要做的是：
    把游标移到「人应该在那一刻按键」的帧，然后按对应的标签键。

为什么是"你选时刻"而不是"程序自动切"：
    撞上的那一帧**不是**训练样本 —— 真正的错误发生在撞车前 200~600ms。
    程序不可能知道人当时该按什么，只有你能判断。

生成的样本和采集器完全同格式：胶片条覆盖 [选中时刻-800ms, +400ms]，
训练时自动取其中 6 帧当输入。所以标完之后 5_训练模型.py 直接就能用。

用法（先跑 10_迭代补数据.py 攒事件，再跑这个）：

    python 11_标注补数据.py                          # 用默认事件目录和默认会话
    python 11_标注补数据.py --events events --session dataset/补采_标注

按键：
    ← →        游标退 / 进一帧          Shift+← →  一次 5 帧
    Home/End   跳到窗口开头 / 结尾（结尾 = 撞车那一刻）
    1 2 3 4 5  在游标处标记 A D N S W，存成样本并跳到下一个事件
    K          用「模型当时的动作」当标签存一个样本（模型这次做对了的话用这个）
    x          丢弃这个事件（画面太乱 / 追击污染）
    q / Esc    退出
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2

from collector.attention import put_text_cn
from collector.config import Config
from collector.recorder import write_sample

LABEL_KEYS = {"1": "A", "2": "D", "3": "N", "4": "S", "5": "W"}
WIN = "11_标注补数据  ——  ←→移动游标  1-5=ADNSW  K=用模型动作  x=丢弃  q=退出"
ZOOM = 2


# ==============================================================================
def init_session(root: str, cfg: Config) -> None:
    """建/补一个普通数据集会话，5_训练模型.py 能直接读。"""
    os.makedirs(root, exist_ok=True)
    sp = os.path.join(root, "session.json")
    if not os.path.isfile(sp):
        from dataclasses import asdict
        try:
            conf = cfg.to_dict() if hasattr(cfg, "to_dict") else asdict(cfg)
        except Exception:
            conf = {}
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump({"session": os.path.basename(root.rstrip("\\/")), "config": conf},
                      fh, ensure_ascii=False, indent=2)
    mp = os.path.join(root, "meta.jsonl")
    if not os.path.isfile(mp):
        open(mp, "w", encoding="utf-8").close()
    for lb in cfg.labels:                      # write_sample 不建子目录
        os.makedirs(os.path.join(root, lb), exist_ok=True)


def load_counters(root: str) -> Dict[str, int]:
    from collections import Counter
    c: Counter = Counter()
    mp = os.path.join(root, "meta.jsonl")
    if os.path.isfile(mp):
        with open(mp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    c[json.loads(line)["label"]] += 1
                except Exception:
                    pass
    # 所有标签预置 0 —— write_sample 直接取 counters[label]，缺键会 KeyError
    from collector.config import Config as _C
    return {lb: int(c.get(lb, 0)) for lb in _C().labels}


def list_events(root: str) -> List[str]:
    """事件目录名列表（跳过已经标过的）。"""
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        ej = os.path.join(d, "event.json")
        if not os.path.isfile(ej):
            continue
        try:
            with open(ej, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        if not meta.get("labeled"):
            out.append(d)
    return out


def load_frames(d: str, n: int) -> List[np.ndarray]:
    frames = []
    for i in range(n):
        p = os.path.join(d, "frames", f"{i:04d}.jpg")
        if not os.path.isfile(p):
            break
        img = cv2.imread(p)
        if img is None:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def model_action_at(meta: dict, t: float) -> Optional[str]:
    """事件里模型在 t 时刻（绝对时间）的动作。"""
    best, bd = None, 1e9
    for item in meta.get("actions", []):
        try:
            ta, lb = float(item[0]), str(item[1])
        except Exception:
            continue
        if abs(ta - t) < bd:
            bd, best = abs(ta - t), lb
    return best if bd < 0.25 else None


# ==============================================================================
def resample_for_anchor(frames: List[np.ndarray], meta: dict, anchor_t: float,
                        cfg: Config) -> Optional[Tuple[List[float], List[np.ndarray], bool]]:
    """从事件帧里重采样出标准采集窗口 [anchor-pre, anchor+post]。"""
    n = len(frames)
    if n < 2:
        return None
    t0, t1 = float(meta["t0"]), float(meta["t1"])
    src_t = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]

    want = cfg.saved_frames
    pre, post = cfg.save_pre_ms / 1000.0, cfg.save_post_ms / 1000.0
    a0, a1 = anchor_t - pre, anchor_t + post
    truncated = (a0 < t0) or (a1 > t1)

    times, out = [], []
    for i in range(want):
        tg = a0 + (a1 - a0) * i / max(want - 1, 1)
        j = min(range(n), key=lambda k: abs(src_t[k] - tg))
        times.append(tg)
        out.append(frames[j])
    return times, out, truncated


def build_strip(frames: List[np.ndarray], cfg: Config) -> np.ndarray:
    fw, fh, sep = cfg.frame_w, cfg.frame_h, cfg.sep
    n = len(frames)
    w = n * fw + (n - 1) * sep
    strip = np.full((fh, w, 3), 40, dtype=np.uint8)
    for i, f in enumerate(frames):
        x = i * (fw + sep)
        ff = cv2.resize(f, (fw, fh), interpolation=cv2.INTER_AREA) if f.shape[:2] != (fh, fw) else f
        strip[:, x:x + fw] = ff
    return strip


def draw(meta: dict, frames: List[np.ndarray], cur: int, anchor_t: float,
         cfg: Config, msg: str = "") -> np.ndarray:
    f = frames[cur]
    big = cv2.resize(f, (f.shape[1] * ZOOM, f.shape[0] * ZOOM), interpolation=cv2.INTER_NEAREST)
    h, w = big.shape[:2]
    panel = np.full((190, w, 3), 28, dtype=np.uint8)
    img = np.vstack([big, panel])
    img = np.ascontiguousarray(img)

    n = len(frames)
    t0, t1 = float(meta["t0"]), float(meta["t1"])
    src_t = t0 + (t1 - t0) * cur / max(n - 1, 1)
    rel = src_t - anchor_t

    put_text_cn(img, f"事件 {meta['name']}   判定: {meta['kind']}"
                     + ("   ⚠ 含追击段" if meta.get("contaminated") else ""), (10, h + 26), 1.0)
    put_text_cn(img, f"游标 {cur+1}/{n}   相对撞车 {rel*1000:+.0f} ms   "
                     f"（撞车后 400ms 是窗口末尾）", (10, h + 56), 0.9)
    act = model_action_at(meta, src_t)
    put_text_cn(img, f"模型此刻的动作: {act if act else '——（低置信/没记）'}", (10, h + 86), 0.9)
    put_text_cn(img, "←→移动  1=A 2=D 3=N 4=S 5=W  K=用模型动作  x=丢弃  q=退出",
                (10, h + 116), 0.85)
    if msg:
        put_text_cn(img, msg, (10, h + 150), 0.95)

    # 游标刻度条
    bar_y = h + 168
    cv2.line(img, (10, bar_y), (w - 10, bar_y), (90, 90, 90), 1)
    if n > 1:
        x = int(10 + (w - 20) * cur / (n - 1))
        cv2.line(img, (x, bar_y - 6), (x, bar_y + 6), (0, 220, 255), 2)
        xa = int(10 + (w - 20) * (anchor_t - t0) / max(t1 - t0, 1e-6))
        cv2.line(img, (xa, bar_y - 10), (xa, bar_y + 10), (0, 0, 255), 2)
    return img


# ==============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="标注补数据：把错误事件变成标准训练样本")
    ap.add_argument("--events", default=os.path.join(HERE, "events"), help="事件目录")
    ap.add_argument("--session", default=os.path.join(HERE, r"dataset\补采_标注"),
                    help="样本写进哪个数据集会话")
    ap.add_argument("--auto-next", action="store_true", default=True, help="标完自动跳下一个")
    ap.add_argument("--list", action="store_true", help="只列出待标注事件，不开窗口")
    args = ap.parse_args()

    if args.list:
        todo = list_events(args.events)
        print(f"[事件目录] {args.events}")
        if not todo:
            print("  （没有待标注事件。先跑 python 10_迭代补数据.py）")
        for d in todo:
            with open(os.path.join(d, "event.json"), encoding="utf-8") as fh:
                m = json.load(fh)
            print(f"  {os.path.basename(d)}   判定={m['kind']}  {m['n_frames']} 帧  "
                  f"{'⚠含追击段' if m.get('contaminated') else '干净'}")
        return

    cfg = Config()
    cfg.validate()
    init_session(args.session, cfg)
    counters = load_counters(args.session)
    print(f"[会话] 样本写到 {args.session}")
    print(f"       胶片条 {cfg.saved_frames} 帧，覆盖 [按键前 {cfg.save_pre_ms}ms, 后 {cfg.save_post_ms}ms]")

    todo = list_events(args.events)
    if not todo:
        print(f"[空] {args.events} 下没有待标注事件。先跑 python 10_迭代补数据.py")
        return
    print(f"[事件] 共 {len(todo)} 个待标注\n")

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    saved, skipped = 0, 0
    qi = 0
    while qi < len(todo):
        d = todo[qi]
        with open(os.path.join(d, "event.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        frames = load_frames(d, int(meta["n_frames"]))
        if len(frames) < 3:
            print(f"[跳过] {meta['name']} 帧不够")
            qi += 1
            continue

        t0, t1 = float(meta["t0"]), float(meta["t1"])
        anchor_t = float(meta["t_anchor"])
        pre, post = cfg.save_pre_ms / 1000.0, cfg.save_post_ms / 1000.0
        # 游标必须让 [T-pre, T+post] 落在事件窗口内
        lo, hi = t0 + pre, t1 - post
        if hi < lo:
            hi = lo = min(max(anchor_t, t0 + pre), t1)
        lo, hi = min(lo, hi), max(lo, hi)
        n = len(frames)
        src_t = [t0 + (t1 - t0) * i / max(n - 1, 1) for i in range(n)]
        cur = min(range(n), key=lambda k: abs(src_t[k] - max(min(anchor_t, hi), lo)))

        msg = ""
        while True:
            img = draw(meta, frames, cur, anchor_t, cfg, msg)
            cv2.imshow(WIN, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(0) & 0xFF

            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"\n[退出] 本次存了 {saved} 个样本，丢弃 {skipped} 个事件")
                return
            if k == 81 or k == ord(","):          # ←
                cur = max(0, cur - 1); msg = ""
            elif k == 83 or k == ord("."):        # →
                cur = min(n - 1, cur + 1); msg = ""
            elif k == ord("a"):
                cur = max(0, cur - 5); msg = ""
            elif k == ord("e"):
                cur = min(n - 1, cur + 5); msg = ""
            elif k == 80 or k == ord("0"):        # Home
                cur = min(range(n), key=lambda i: abs(src_t[i] - lo)); msg = ""
            elif k == 87 or k == ord("9"):        # End
                cur = min(range(n), key=lambda i: abs(src_t[i] - hi)); msg = ""
            elif k == ord("x"):
                meta["labeled"] = True
                meta["discarded"] = True
                with open(os.path.join(d, "event.json"), "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False, indent=2)
                skipped += 1
                break
            elif chr(k) in LABEL_KEYS or chr(k) == "K":
                label = LABEL_KEYS.get(chr(k)) or model_action_at(meta, src_t[cur])
                if not label:
                    msg = "这个时刻没有记录到模型动作，请手动选 1-5"
                    continue
                anchor = src_t[cur]
                rs = resample_for_anchor(frames, meta, anchor, cfg)
                if rs is None:
                    msg = "重采样失败"
                    continue
                times, picked, truncated = rs
                try:
                    m = write_sample(args.session, counters, cfg, label, "relabel", "", anchor,
                                     times, picked, cfg.save_fps, truncated)
                    with open(os.path.join(args.session, "meta.jsonl"), "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(m, ensure_ascii=False) + "\n")
                except Exception as e:
                    msg = f"写样本失败: {e}"
                    continue
                meta["labeled"] = True
                meta["label"] = label
                meta["cursor_t"] = round(anchor, 4)
                meta["used_model_action"] = (chr(k) == "K")
                with open(os.path.join(d, "event.json"), "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False, indent=2)
                saved += 1
                print(f"[样本] {meta['name']}  标签 {label}  游标 {anchor - anchor_t:+.3f}s  "
                      f"{'（用模型动作）' if chr(k) == 'K' else ''}")
                break

        qi += 1

    cv2.destroyAllWindows()
    print(f"\n[完成] 存了 {saved} 个样本到 {args.session}，丢弃 {skipped} 个事件")
    print("       下一步：把这些会话加到 5_训练模型.py 的 DATA_DIRS 里重新训练")


if __name__ == "__main__":
    main()
