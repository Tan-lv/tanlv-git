# -*- coding: utf-8 -*-
"""看模型"在看哪里" —— 注意力热力图。

两种用法：

    # 离线：从已采数据里每个类别挑一个最有代表性的样本，出图
    python 8_注意力可视化.py

    # 实时：模型在玩游戏时，把 Grad-CAM 叠在当前画面上（需要模拟器在关卡里）
    python 8_注意力可视化.py --live

离线出图内容（见 collector/attention.py 的 draw_panel）：
    左  6 帧输入缩略图，每帧下面标着**逐帧重要度**
    中  当前画面 + Grad-CAM 叠加
    右  纯 Grad-CAM / 输入梯度显著图
    底  5 类的概率条

实时窗口按键：q/Esc 退出  空格 暂停  m 切换热力图/显著图  s 存一张 PNG

关于"这算不算注意力"的说明见 paoku/README.md 的「注意力可视化」一节：
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from collector.attention import (AttentionProbe, cam_to_frame, colorize,  # noqa: E402
                                 draw_panel, overlay)
from collector.config import Config                                        # noqa: E402
from collector.dataset import (build_eval_transform, flatten_frames, load_sessions,  # noqa: E402
                               read_strip, select_window_indices, to_video_tensor)
from collector.model import build_model                                    # noqa: E402
from collector.runs import load_run as _load_run, pick_run                 # noqa: E402


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
    print(f"[data] {sessions}")

    cfg = build_cfg_from_session(sessions)
    samples = load_sessions(sessions)
    print(f"[data] {len(samples)} 个样本  {dict(sorted(__import__('collections').Counter(s['label'] for s in samples).items()))}")

    model = build_model(K, len(labels), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model_state_dict"])
    model.eval().float()                      # 必须 float32，half 会导致反向出 NaN
    probe = AttentionProbe(model, frames=K)
    tf = build_eval_transform(img_size)

    n_idx = {lab: i for i, lab in enumerate(labels)}
    os.makedirs(args.out, exist_ok=True)
    panels = []

    # 先只做前向，挑出每类"最自信且预测正确"的样本
    todo = {}
    if args.id:
        for s in samples:
            if s["id"] == args.id:
                todo[s["label"]] = (1.0, s)
                break
        if not todo:
            raise SystemExit(f"[错误] 找不到 id={args.id}")
    else:
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
    # ★ 2026-09-25：解包这一行原来缩进在 else 分支里，于是 --id 模式下 todo 里
    #   存的是 (score, s) 元组，后面 select_window_indices(s,...) 直接报
    #   "tuple indices must be integers" —— 也就是 --id（同一样本对比）从来没能用。
    todo = {k: (v[1] if isinstance(v, tuple) else v) for k, v in todo.items()}

    for lab in labels:
        s = todo.get(lab)
        if s is None:
            print(f"[跳过] 类别 {lab} 没有样本")
            continue
        idx = select_window_indices(s, K, cfg.input_stride, -cfg.input_end_offset_ms)
        frames = read_strip(s, idx)                      # (K, H, W, 3) RGB
        x = flatten_frames(tf(to_video_tensor(frames))).unsqueeze(0).to(device)
        target = n_idx[s["label"]] if args.target_true else None
        res = probe.run(x, target, ablate_frames=True)   # 离线模式多跑 K 次前向，换可解释的逐帧消融
        panel = draw_panel([f for f in frames], res, labels,
                           title=f"{lab}  id={s['id']}  真实={s['label']}  kind={s.get('kind','')}")
        p = os.path.join(args.out, f"attention_{lab}_{s['id']}.png")
        cv2.imwrite(p, panel)
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
        cv2.imwrite(gp, grid)
        print(f"[出图] {gp}  ({grid.shape[1]}x{grid.shape[0]})")
    probe.close()
    return 0



# ======================================================================
def run_live(args, device):
    import play as play_mod
    from collector.capture import WindowCapture
    from collector.recorder import RingBuffer

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

    cap = WindowCapture(cfg.window_keyword, cfg.game_res, cfg.game_crop, backend=cfg.backend)
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
                    cv2.imwrite(p, view)
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
                cv2.imwrite(p, view)
                print(f"[存图] {p}")
    finally:
        cap.close()
        probe.close()
        if not headless:
            cv2.destroyAllWindows()
    return 0


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="注意力热力图可视化")
    ap.add_argument("--run", default=None, help="训练结果目录；不填用效果最好的")
    ap.add_argument("--sessions", nargs="+", default=None, help="离线模式的数据集会话")
    ap.add_argument("--id", default=None, help="只看某一个样本，例如 A_000123")
    ap.add_argument("--out", default=None, help="出图目录，默认 runs/<模型>/attention")
    ap.add_argument("--live", action="store_true", help="实时模式（需要模拟器在关卡里）")
    ap.add_argument("--live-frames", type=int, default=0,
                    help="只跑这么多帧、不开窗口，把最后一张叠加图存成 PNG（自检用）")
    ap.add_argument("--target-true", action="store_true",
                    help="按真实标签算热力图（默认按模型预测的类别）")
    ap.add_argument("--no-grid", dest="grid", action="store_false", default=True,
                    help="不拼总图，每类单独出")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.live:
        return run_live(args, device)
    if args.out is None:
        run_dir = args.run or pick_run()
        rd = run_dir if os.path.isabs(run_dir) else os.path.join(HERE, run_dir)
        args.out = os.path.join(rd, "attention")
    return run_offline(args, device)


if __name__ == "__main__":
    sys.exit(main())
