# -*- coding: utf-8 -*-
"""看模型"在看哪里" —— 注意力热力图。

★ 2026-10-14：出哪几张图，你现在可以自己挑（以前只能"每类自动挑一张"或看一张）。

    # ① 老行为：每类自动挑一张"最自信且预测正确"的（要扫全部样本，慢，约 1~2 分钟）
    python 5_评估.py --cam

    # ② 自己点菜：按 id 挑（不做全量扫描，约 10 秒）
    python 5_评估.py --cam --ids A_000322 D_000324 N_000051
    python 5_评估.py --cam --ids A_000322,D_000324          # 逗号也行

    # ③ 先看看有哪些 id 可选（不需要模型，秒出）
    python 5_评估.py --cam --list-ids --limit 30

    # ④ ★ 直接填图片路径（最常用）—— 想填几张填几张
    python 5_评估.py --cam --img "D:/deeplearning/make_dataset/gamer_ai/profiles/cs01/dataset/20261011_204728/D/D_000005.jpg"
    python 5_评估.py --cam --img "a.jpg" "b.jpg" "c.jpg"        # 多张
    python 5_评估.py --cam --img "profiles/cs01/dataset/20261011_204728/N"  # 填目录 = 取前 --limit 张
    python 5_评估.py --cam --img-list paths.txt               # 一次填很多：一行一个路径

    说明：路径可以带引号，反斜杠/正斜杠都认；
          不属于当前会话的图片，脚本会**自动把它所在的会话加进来**（不用手填 --sessions）；
          清单文件里 # 开头的行会被当注释跳过。

    # ⑤ 按 id 子串挑 / 随机抽几个（找"奇怪的样本"时用）
    python 5_评估.py --cam --grep 0003
    python 5_评估.py --cam --random 8

    # 实时：模型在玩游戏时，把 Grad-CAM 叠在当前画面上（需要模拟器在关卡里）
    python 5_评估.py --cam --live

常用搭配：
    --run runs/横屏_v0     指定用哪个模型（不填用最近产出的那个）
    --sessions dataset/xxx 从哪个会话里取样本（不填用最近修改的会话）
    --out <目录>           出图到哪里（默认 runs/<模型>/attention）
    --target-true          按真实标签算热力图（默认按模型预测的类别）

离线出图内容（见 core/attention.py 的 draw_panel）：
    左  6 帧输入缩略图，每帧下面标着**逐帧重要度**
    中  当前画面 + Grad-CAM 叠加
    右  纯 Grad-CAM / 输入梯度显著图
    底  各类的概率条

实时窗口按键：q/Esc 退出  空格 暂停  m 切换热力图/显著图  s 存一张 PNG

关于"这算不算注意力"的说明见 gamer_ai/README.md 的「注意力可视化」一节：
Grad-CAM 是**梯度意义上的证据定位**，不是人那种注意力，分辨率也受最后一级
特征图限制（EfficientNet-B0 是 7x7，会有点块状）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

PAOKU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = PAOKU_DIR
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .attention import (AttentionProbe, cam_to_frame, colorize,  # noqa: E402
                                 draw_panel, overlay)
from .config import Config                                        # noqa: E402
from .dataset import (build_eval_transform, flatten_frames, load_sessions,  # noqa: E402
                               read_strip, select_window_indices, to_video_tensor)
from .imgio import imwrite_u                                       # noqa: E402
from .model import build_model                                    # noqa: E402
from .runs import load_run as _load_run, pick_run                 # noqa: E402


def load_run(run_dir: str):
    return _load_run(run_dir)


def build_cfg_from_session(dirs) -> Config:
    from dataclasses import fields
    cfg = Config()
    for d in dirs:
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


# ======================================================================
def run_offline(args, device):
    run_dir = args.run or pick_run()
    meta, ckpt = load_run(run_dir)
    labels = list(meta["labels"])
    K = int(meta["in_frames"])
    img_size = int(meta.get("img_size", 224))
    print(f"[run] {run_dir}   类别 {labels}  输入 {K} 帧  img_size {img_size}")

    # ---------- 先把 --img / --img-list 的路径整理出来 ----------
    #   ★ 2026-10-14（用户要求：想直接填图片路径）：
    #     · 直接填胶片条的绝对路径，想填几张填几张；
    #     · 填【目录】也行 -> 取里面前 --limit 张图；
    #     · 路径属于别的会话时，自动把它所在的会话也加进来（不用再手填 --sessions）。
    img_paths = []
    if getattr(args, "img_list", None):
        for lst in args.img_list:
            fp = lst if os.path.isabs(lst) else os.path.join(HERE, lst)
            if not os.path.isfile(fp):
                print(f"[跳过] 路径清单文件不存在：{fp}")
                continue
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip().strip(chr(34)).strip(chr(39))
                    if line and not line.startswith("#"):
                        img_paths.append(line)
            print(f"[挑图] 已从清单 {os.path.basename(fp)} 读入路径（# 开头的行会跳过）")
    import glob as _glob
    for p in (args.img or []):
        if os.path.isdir(p):
            fs = sorted(_glob.glob(os.path.join(p, "*.jpg"))
                        + _glob.glob(os.path.join(p, "*.jpeg"))
                        + _glob.glob(os.path.join(p, "*.png")))
            if not fs:
                print(f"[跳过] 这个目录里没有图片：{p}")
            else:
                print(f"[挑图] 目录 {p} 取前 {min(len(fs), args.limit)} 张（共 {len(fs)} 张）")
            img_paths.extend(fs[:args.limit])
        else:
            if not os.path.isfile(p):
                print(f"[跳过] 文件不存在：{p}")
            img_paths.append(p)
    _seen, _uniq = set(), []
    for p in img_paths:
        k = os.path.normcase(os.path.abspath(p))
        if k not in _seen:
            _seen.add(k)
            _uniq.append(p)
    img_paths = _uniq

    if args.sessions:
        sessions = [s if os.path.isabs(s) else os.path.join(HERE, s) for s in args.sessions]
    else:
        root = os.path.join(HERE, "dataset")
        cands = [os.path.join(root, d) for d in os.listdir(root)
                 if not d.startswith("_")
                 and os.path.isfile(os.path.join(root, d, "meta.jsonl"))]
        if not cands:
            raise SystemExit("[错误] dataset/ 下没有会话，用 --sessions 指定")
        sessions = [max(cands, key=os.path.getmtime)]

    # 填进来的图片如果不在已加载的会话里，自动把它所在的会话加进来
    for p in img_paths:
        d = os.path.dirname(os.path.abspath(p))
        while d and not os.path.isfile(os.path.join(d, "meta.jsonl")):
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        if d and os.path.isfile(os.path.join(d, "meta.jsonl")):
            if os.path.abspath(d) not in [os.path.abspath(x) for x in sessions]:
                sessions.append(d)
                print(f"[data] 你填的图片在另一个会话里，自动加入：{os.path.relpath(d, HERE)}")
        else:
            print(f"[跳过] 这个路径往上找不到会话（没有 meta.jsonl）：{p}")
    print(f"[data] {sessions}")

    cfg = build_cfg_from_session(sessions)
    samples = load_sessions(sessions)
    print(f"[data] {len(samples)} 个样本  {dict(sorted(__import__('collections').Counter(s['label'] for s in samples).items()))}")

    if args.list_ids:
        by = {}
        for s in samples:
            by.setdefault(s["label"], []).append(s["id"])
        print("")
        print("可选的样本 id（按类别）：")
        for lab in sorted(by):
            ids = by[lab]
            head = ", ".join(ids[:args.limit])
            more = ("  …（共 %d 个）" % len(ids)) if len(ids) > args.limit else ""
            print("  %s: %s%s" % (lab, head, more))
        print("")
        print("挑好之后就这样看：")
        print("  python 5_评估.py --cam --ids A_000322 D_000324")
        print("也可以用图片路径：")
        print("  python 5_评估.py --cam --img <会话目录>/A/A_000322.jpg")
        return 0

    model = build_model(K, len(labels), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model_state_dict"])
    model.eval().float()                      # 必须 float32，half 会导致反向出 NaN
    probe = AttentionProbe(model, frames=K)
    tf = build_eval_transform(img_size)

    n_idx = {lab: i for i, lab in enumerate(labels)}
    os.makedirs(args.out, exist_ok=True)
    panels = []

    # ---------- 挑要出图的样本 ----------
    #   ★ 2026-10-14：以前只能"每类自动挑一张最自信的"或 --id 看一张。现在支持自己挑：
    #     --ids / --img / --grep / --random。自己指定时**不做全量扫描，出图很快**。
    picked = []          # [(类别, 样本), ...] 保持给定顺序

    def _norm_ids(vals):
        """把各种写法拍平：["A_1","D_2"] / ["A_1,D_2"] / ["A_1 D_2"]。"""
        res = []
        for v in vals or []:
            for part in str(v).replace(",", " ").split():
                if part:
                    res.append(part)
        return res

    # ① --img / --img-list：按绝对路径认领（不管分隔符和大小写）
    if img_paths:
        want = {}
        for p in img_paths:
            want[os.path.normcase(os.path.abspath(p))] = p
        for s in samples:
            fp = os.path.abspath(os.path.join(s["_root"],
                                             str(s.get("file", "")).replace("/", os.sep)))
            if os.path.normcase(fp) in want:
                picked.append((s["label"], s))
                want.pop(os.path.normcase(fp), None)
        for miss in want.values():
            print(f"[跳过] 这张图在加载到的会话里没认领到：{miss}")

    # ② --id / --ids：按 id 点菜
    ids = _norm_ids(args.ids) + _norm_ids([args.id] if args.id else [])
    if ids:
        by_id = {s["id"]: s for s in samples}
        for i in ids:
            s = by_id.get(i)
            if s is None:
                print(f"[跳过] 找不到 id={i}")
            else:
                picked.append((s["label"], s))

    # ③ --grep：按 id 子串挑
    if args.grep:
        hit = [s for s in samples if args.grep in str(s["id"])]
        for s in hit[:args.limit]:
            picked.append((s["label"], s))
        print(f"[挑图] --grep {args.grep} 命中 {len(hit)} 个，取前 {min(len(hit), args.limit)} 个")

    # ④ --random：随机抽（想找"奇怪的样本"时用）
    if args.random:
        import random
        rng = random.Random(args.seed)
        for s in rng.sample(samples, min(args.random, len(samples))):
            picked.append((s["label"], s))

    # ⑤ 都没给 -> 老行为：每类挑"最自信且预测正确"的一张（要扫全部样本，慢）
    if not picked:
        print("[挑图] 没指定样本，自动给每类挑一张最自信的（要扫一遍全部样本，稍慢）")
        todo = {}
        with torch.no_grad():
            for i in range(0, len(samples), 32):
                batch = samples[i:i + 32]
                xs, idxs = [], []
                for s in batch:
                    idx = select_window_indices(s, K, cfg.input_stride,
                                                -cfg.input_end_offset_ms)
                    xs.append(flatten_frames(tf(to_video_tensor(read_strip(s, idx)))))
                    idxs.append(s)
                p = torch.softmax(model(torch.stack(xs).to(device)).float(), dim=1).cpu().numpy()
                for j, s in enumerate(idxs):
                    t = n_idx.get(s["label"])
                    if t is None:
                        continue
                    score = float(p[j][t]) if int(p[j].argmax()) == t else -1.0
                    cur = todo.get(s["label"])
                    if cur is None or score > cur[0]:
                        todo[s["label"]] = (score, s)
        for lab in labels:
            if lab in todo:
                picked.append((lab, todo[lab][1]))

    if not picked:
        raise SystemExit("[错误] 没挑到任何样本；用 --list-ids 看看有哪些 id 可选")
    print("[挑图] 本次出图 %d 张：%s" % (len(picked), ", ".join(s["id"] for _l, s in picked)))

    for lab, s in picked:
        idx = select_window_indices(s, K, cfg.input_stride, -cfg.input_end_offset_ms)
        frames = read_strip(s, idx)                      # (K, H, W, 3) RGB
        x = flatten_frames(tf(to_video_tensor(frames))).unsqueeze(0).to(device)
        target = n_idx[s["label"]] if args.target_true else None
        res = probe.run(x, target, ablate_frames=True)   # 离线模式多跑 K 次前向，换可解释的逐帧消融
        panel = draw_panel([f for f in frames], res, labels,
                           title=f"{lab}  id={s['id']}  真实={s['label']}  kind={s.get('kind','')}")
        p = os.path.join(args.out, f"attention_{lab}_{s['id']}.png")
        # ★ 2026-10-14：必须用 unicode 安全写盘。目录名带中文（例如 runs\横屏_v0）时，
        #   cv2.imwrite 会**静默失败**：返回 False、不抛异常、文件不生成，
        #   而这里照样打印 [出图] —— 于是"跑完了但路径里找不到图片"。见 core/imgio.py。
        if not imwrite_u(p, panel):
            print(f"        [错误] 上面这张图没能写盘：{p}")
        print(f"[出图] {p}   预测={labels[res['pred']]} conf={res['prob'][res['pred']]:.2f}   "
              f"逐帧重要度={np.round(res['frame_importance'], 3).tolist()}")
        panels.append(panel)

    if panels and args.grid:
        w = max(p.shape[1] for p in panels)
        gap = np.full((8, w, 3), 60, np.uint8)
        rows = []
        for i, p in enumerate(panels):
            if p.shape[1] < w:
                pad = np.full((p.shape[0], w - p.shape[1], 3), 22, np.uint8)
                p = np.concatenate([p, pad], axis=1)
            rows.append(p)
            if i != len(panels) - 1:
                rows.append(gap)
        grid = np.concatenate(rows, axis=0)
        gp = os.path.join(args.out, "attention_grid.png")
        imwrite_u(gp, grid)
        print(f"[出图] {gp}  ({grid.shape[1]}x{grid.shape[0]})")
    probe.close()
    return 0



# ======================================================================
def run_live(args, device):
    import play as play_mod
    from .capture import WindowCapture
    from .geometry import deploy_geometry
    from .recorder import RingBuffer

    run_dir = args.run or play_mod.pick_run()
    meta, ckpt = load_run(run_dir)
    labels = list(meta["labels"])
    K = int(meta["in_frames"])
    img_size = int(meta.get("img_size", 224))
    cfg = Config()
    for k in ("game_res", "game_crop", "backend"):
        if k in meta:
            cfg.__setattr__(k, tuple(meta[k]) if k in ("game_res", "game_crop") else meta[k])
    # frame_h 是 Config 的只读派生属性（由 frame_w 和 game_crop 算出来），不能直接赋值
    cfg.frame_w = int(meta.get("frame_w", cfg.frame_w))
    cfg.sep = int(meta.get("sep", cfg.sep))
    if int(meta.get("frame_h", cfg.frame_h)) != cfg.frame_h:
        raise SystemExit("[错误] model_meta 的 frame_h=%s 与按 game_crop 算出的 %d 不一致"
                         % (meta.get("frame_h"), cfg.frame_h))
    cfg.input_end_offset_ms = int(meta.get("input_end_offset_ms", cfg.input_end_offset_ms))
    cfg.input_stride = int(meta.get("input_stride", cfg.input_stride))
    cfg.save_pre_ms = int(meta.get("save_pre_ms", cfg.save_pre_ms))
    cfg.save_post_ms = int(meta.get("save_post_ms", cfg.save_post_ms))
    saved = int(meta.get("saved_frames", cfg.saved_frames))
    step_s = ((cfg.save_pre_ms + cfg.save_post_ms) / max(saved - 1, 1)) / 1000.0
    cfg.validate()

    print(f"[run] {run_dir}")
    print(f"      类别 {labels}  输入 {K} 帧  帧间隔 {step_s*1000:.1f}ms  img_size {img_size}")
    model = build_model(K, len(labels), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model_state_dict"])
    model.eval().float()
    probe = AttentionProbe(model, frames=K)
    tf = build_eval_transform(img_size)

    # ★ 用**这个模型自带**的取景（model_meta.json 里的 geometry）：观察到的画面
    #   范围和训练时一致，否则 CAM 的位置会整体偏掉。
    cap = WindowCapture.from_config(cfg, backend=cfg.backend,
                                    geo=deploy_geometry(meta))
    ring = RingBuffer(4.0)
    print(f"[cap] 已绑定窗口 hwnd={cap.hwnd}")
    print("[提示] 模拟器要已经进入关卡；按 q/Esc 退出，空格暂停，m 切换图层，s 存图")

    headless = args.live_frames > 0
    win = "attention"
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    mode, paused, n_saved = "cam", False, 0
    last = None
    t_prev = time.time()
    fps = 0.0
    n_iter = 0

    try:
        while True:
            t = time.time()
            full = cap.grab()                                   # RGB, 裁剪后的游戏画面
            small = cv2.resize(full, (cfg.frame_w, cfg.frame_h), interpolation=cv2.INTER_AREA)
            ring.append(t, small)

            if not paused:
                now = t
                fs = []
                ok = True
                for i in range(K - 1, -1, -1):
                    f = ring.frame_at_or_before(now - i * cfg.input_stride * step_s)
                    if f is None:
                        ok = False
                        break
                    fs.append(f)
                if ok:
                    x = flatten_frames(tf(to_video_tensor(np.stack(fs)))).unsqueeze(0).to(device)
                    res = probe.run(x, None)
                    last = res
            if last is None:
                continue

            base = cv2.resize(full, (cfg.frame_w * 2, cfg.frame_h * 2),
                              interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
            m = last["cam"] if mode == "cam" else last["saliency"]
            if m is None:
                m = last["saliency"]
            m_full = cv2.resize(cam_to_frame(m, (cfg.frame_h, cfg.frame_w)),
                                (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
            view = overlay(base, m_full)

            dt = t - t_prev
            t_prev = t
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 1e-6 else 0.0)
            cv2.putText(view, "%s  %.1ffps%s" % (mode.upper(), fps, "  [暂停]" if paused else ""),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            pred = last["pred"]
            cv2.putText(view, "pred %s %.2f" % (labels[pred], last["prob"][pred]),
                        (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            bar = "  ".join("%s %.2f" % (l, last["prob"][i]) for i, l in enumerate(labels))
            cv2.putText(view, bar, (8, base.shape[0] - 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (220, 220, 220), 1, cv2.LINE_AA)
            fr_imp = "  ".join("f%d %.0f%%" % (i + 1, v * 100)
                               for i, v in enumerate(last["frame_importance"]))
            cv2.putText(view, fr_imp, (8, base.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 255), 1, cv2.LINE_AA)

            n_iter += 1
            if headless:
                if n_iter >= args.live_frames:
                    p = os.path.join(HERE, "attention_live_check.png")
                    imwrite_u(p, view)
                    print(f"[自检] 跑了 {n_iter} 帧，叠加图已存 {p}")
                    break
                continue
            cv2.imshow(win, view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord(" "):
                paused = not paused
            elif k == ord("m"):
                mode = "sal" if mode == "cam" else "cam"
            elif k == ord("s"):
                n_saved += 1
                p = os.path.join(HERE, "attention_live_%02d.png" % n_saved)
                imwrite_u(p, view)
                print(f"[存图] {p}")
    finally:
        cap.close()
        probe.close()
        if not headless:
            cv2.destroyAllWindows()
    return 0


# ======================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="注意力热力图可视化")
    ap.add_argument("--run", default=None, help="训练结果目录；不填用效果最好的")
    ap.add_argument("--sessions", nargs="+", default=None, help="离线模式的数据集会话")
    ap.add_argument("--id", default=None, help="只看某一个样本，例如 A_000123")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="自己挑几个样本：--ids A_000322 D_000324 N_000051"
                         "（也支持逗号分隔，或一次给多个参数）")
    ap.add_argument("--img", nargs="+", default=None,
                    help="直接给胶片条图片的路径（dataset/<会话>/<类>/<id>.jpg），"
                         "脚本会去会话的 meta.jsonl 里认领它")
    ap.add_argument("--img-list", nargs="+", default=None,
                    help="从一个 txt 清单里读图片路径（一行一个，# 开头是注释）—— 一次要填很多张时用这个")
    ap.add_argument("--grep", default=None, help="按 id 子串挑，例如 --grep 0003")
    ap.add_argument("--random", type=int, default=0, help="随机抽 N 个样本")
    ap.add_argument("--seed", type=int, default=42, help="--random 的随机种子")
    ap.add_argument("--limit", type=int, default=20,
                    help="--list-ids / --grep 最多列/选多少个（默认 20）")
    ap.add_argument("--list-ids", action="store_true",
                    help="只列出这个会话里有哪些 id 可选，不出图")
    ap.add_argument("--out", default=None,
                    help="出图目录，默认 <模型>/模型评估/attention（和 5_评估.py 放一起）")
    ap.add_argument("--live", action="store_true", help="实时模式（需要模拟器在关卡里）")
    ap.add_argument("--live-frames", type=int, default=0,
                    help="只跑这么多帧、不开窗口，把最后一张叠加图存成 PNG（自检用）")
    ap.add_argument("--target-true", action="store_true",
                    help="按真实标签算热力图（默认按模型预测的类别）")
    ap.add_argument("--no-grid", dest="grid", action="store_false", default=True,
                    help="不拼总图，每类单独出")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.live:
        return run_live(args, device)
    if args.out is None:
        run_dir = args.run or pick_run()
        rd = run_dir if os.path.isabs(run_dir) else os.path.join(HERE, run_dir)
        # ★ 2026-10-15：和 5_评估.py 一致 —— 评估/可视化的产物都收进
        #   模型目录下的「模型评估」文件夹，模型目录根只留训练产物。
        args.out = os.path.join(rd, "模型评估", "attention")
    return run_offline(args, device)


if __name__ == "__main__":
    sys.exit(main())
