# -*- coding: utf-8 -*-
r"""取景标定（横屏/换分辨率后用）：一边看画面一边把裁剪框调准。

为什么需要它：横屏后 game_crop 要重选，而裁剪框对不对只能"看图"判断 ——
    光看整窗看不出来，要看**模型实际吃进去的那张画面**（落盘 240x293 + CNN 吃到的 224x224）。

用法（先把游戏停在一个有赛道和角色的画面，比如 Tap to Play / 关卡里）：
    python tools\取景标定.py --game-res 1280,720 --game-crop 345,0,590,720

按键（先点一下标定窗口让它获得焦点）：
    方向键    移动裁剪框（5 px）
    i/j/k/l   微调 1 px（i 上, j 左, k 下, l 右）
    = / -     加宽 / 收窄（高度按 720:880 自动跟着变，保证比例不变）
    s         存图（整窗+框、裁剪图、模型输入视图）并打印可直接粘贴的参数
    p         只打印当前参数
    q / Esc   退出

★ 比例必须保持 720:880（= 240:293）—— 一旦比例变了，前处理「短边缩到 224 + 中心裁」
  会让模型看到的有效视野完全变样，现有模型全部失效。这个脚本默认锁死比例。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from collector.capture import WindowCapture  # noqa: E402
from collector.config import Config          # noqa: E402
from collector.dataset import (build_eval_transform, to_video_tensor)  # noqa: E402

STD_RATIO = 293 / 240.0        # 项目标准裁剪比例（高:宽）= 1.2208，590 宽 -> 正好 720 高
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def save_png(path: str, bgr: np.ndarray) -> None:
    """用 imencode+tofile 存图（cv2.imwrite 遇到中文路径会静默失败）。"""
    ok, buf = cv2.imencode(".png", bgr)
    if ok:
        buf.tofile(path)


def parse_ints(s: str):
    return [int(v) for v in str(s).replace("x", ",").replace("X", ",")
            .replace(" ", ",").split(",") if v.strip()]


def model_view(bgr_crop: np.ndarray, frame_w: int, frame_h: int, tf) -> np.ndarray:
    """裁剪图 -> 落盘尺寸 -> 模型真正吃到的 224x224（反归一化便于肉眼比）。"""
    stored = cv2.resize(bgr_crop, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(stored[:, :, ::-1])
    x = tf(to_video_tensor(np.stack([rgb])))[0].numpy().transpose(1, 2, 0)
    return (np.clip(x * STD + MEAN, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]


def bar(img: np.ndarray, text: str, h: int = 22) -> np.ndarray:
    b = np.full((h, img.shape[1], 3), 30, np.uint8)
    cv2.putText(b, text, (5, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([b, img], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="实时标定裁剪框（横屏用）")
    ap.add_argument("--game-res", default="1280,720", help="安卓画面尺寸，如 1280,720")
    ap.add_argument("--game-crop", default="345,0,590,720", help="初始裁剪框 x,y,宽,高")
    ap.add_argument("--content-offset", default=None, help="安卓画面在窗口内的偏移 x,y（默认自动算）")
    ap.add_argument("--backend", default=None, choices=["printwindow", "bitblt"])
    ap.add_argument("--out", default=None, help="存图目录，默认 dataset\_selftest")
    ap.add_argument("--lock-aspect", dest="lock", action="store_true", default=True)
    ap.add_argument("--free-aspect", dest="lock", action="store_false")
    ap.add_argument("--once", action="store_true",
                    help="只抓一帧、把三方对比图存下来就退出（自检用，不用交互）")
    args = ap.parse_args()

    cfg = Config()
    gr = parse_ints(args.game_res)
    if len(gr) != 2:
        raise SystemExit("--game-res 要写成 宽,高")
    gw, gh = gr
    gc = parse_ints(args.game_crop)
    if len(gc) != 4:
        raise SystemExit("--game-crop 要写成 x,y,宽,高")
    gx, gy, cw, ch = gc
    if args.lock:
        ch = int(round(cw * STD_RATIO))
    coff = tuple(parse_ints(args.content_offset)) if args.content_offset else None
    if args.backend:
        cfg.backend = args.backend

    # 用「整窗」方式抓图：game_res 传客户区大小、crop 取整窗，这样能自己画框
    probe = WindowCapture(cfg.window_keyword, (1, 1), (0, 0, 1, 1), backend=cfg.backend)
    CW, CH = probe.client_w, probe.client_h
    probe.close()
    off_x = (CW - gw) // 2 if coff is None else coff[0]
    off_y = CH - gh if coff is None else coff[1]
    cap = WindowCapture(cfg.window_keyword, (CW, CH), (0, 0, CW, CH), backend=cfg.backend)
    print(f"客户区 {CW}x{CH}  安卓画面 {gw}x{gh}  偏移 ({off_x},{off_y})")
    print("裁剪框" if not args.lock else "裁剪框（比例锁定 720:880）")

    frame_w = 240
    frame_h = int(round(frame_w * STD_RATIO))
    tf = build_eval_transform(224)
    out_dir = args.out or os.path.join(HERE, "dataset", "_selftest")
    os.makedirs(out_dir, exist_ok=True)
    win = "framing calibration  (arrows move, =/- resize, s save, q quit)"
    if not args.once:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1180, 620)

    def clamp():
        """保证：① 锁定比例时高宽比始终正确；② 框完全落在画面内（x+w / y+h 不越界）。

        ★ 早期版本先把 ch 算成 round(cw*比例)，再单独把 ch 夹到 gh —— 结果
          590 宽会算出 721 高，被夹回 720 后比例就错了，Config.validate 也会
          （越界检查我这次一起补上了）。所以这里改成"放不下就反过来由高定宽"。
        """
        nonlocal gx, gy, cw, ch
        cw = max(80, min(cw, gw))
        if args.lock:
            ch = int(round(cw * STD_RATIO))
            if gy + ch > gh:                       # 高度放不下 -> 由高度反推宽度，保住比例
                ch = max(80, gh - gy)
                cw = max(80, min(int(round(ch / STD_RATIO)), gw))
        ch = max(80, min(ch, gh))
        gx = max(0, min(gx, gw - cw))
        gy = max(0, min(gy, gh - ch))
        if gx + cw > gw or gy + ch > gh:           # 兜底：只可能因为画面太小
            cw, ch = max(80, gw - gx), max(80, gh - gy)

    def report() -> None:
        print(f"  --game-res {gw},{gh} --game-crop {gx},{gy},{cw},{ch}"
              f"   （比例 {cw/ch:.3f}，目标 {1/STD_RATIO:.3f}）")

    last = time.time()
    while True:
        full = cap.grab()                                  # RGB 整窗
        crop = full[off_y + gy:off_y + gy + ch, off_x + gx:off_x + gx + cw]
        if crop.size == 0:
            clamp()
            continue
        mv = model_view(crop, frame_w, frame_h, tf)
        # 左：整窗 + 框；中：落盘画面；右：模型输入 224
        sc = 900.0 / CW
        left = cv2.resize(cv2.cvtColor(full, cv2.COLOR_RGB2BGR), (int(CW * sc), int(CH * sc)))
        cv2.rectangle(left, (int((off_x + gx) * sc), int((off_y + gy) * sc)),
                      (int((off_x + gx + cw) * sc), int((off_y + gy + ch) * sc)), (255, 0, 0), 2)
        left = bar(left, "window + crop box")
        mid = bar(cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_AREA),
                  "stored %dx%d" % (frame_w, frame_h))
        right = bar(cv2.resize(mv, (240, 240), interpolation=cv2.INTER_NEAREST),
                    "CNN input 224x224")
        h = max(left.shape[0], mid.shape[0], right.shape[0])
        def padh(im):
            if im.shape[0] == h:
                return im
            return np.concatenate([im, np.full((h - im.shape[0], im.shape[1], 3), 25, np.uint8)], axis=0)
        view = np.concatenate([padh(left), np.full((h, 8, 3), 60, np.uint8),
                               padh(mid), np.full((h, 8, 3), 60, np.uint8), padh(right)], axis=1)
        cv2.putText(view, "crop=%d,%d,%d,%d" % (gx, gy, cw, ch), (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if args.once:
            p = os.path.join(HERE, "runs", "_framing_calib.png")
            save_png(p, view)
            print("三方对比图已存: " + p)
            report()
            break
        cv2.imshow(win, view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == 81 or k == 2:      # left
            gx -= 5
        elif k == 83 or k == 3:      # right
            gx += 5
        elif k == 82 or k == 0:      # up
            gy -= 5
        elif k == 84 or k == 1:      # down
            gy += 5
        elif k == ord("j"):
            gx -= 1
        elif k == ord("l"):
            gx += 1
        elif k == ord("i"):
            gy -= 1
        elif k == ord("k"):
            gy += 1
        elif k in (ord("="), ord("+")):
            cw += 8
        elif k in (ord("-"), ord("_")):
            cw -= 8
        elif k == ord("p"):
            report()
        elif k == ord("s"):
            save_png(os.path.join(out_dir, "window_with_crop.png"), left)
            save_png(os.path.join(out_dir, "crop.png"), mid[22:])
            save_png(os.path.join(out_dir, "model_view.png"), mv)
            print("已存图到 " + os.path.abspath(out_dir))
            report()
        if time.time() - last > 5:
            last = time.time()
            print(f"\r  当前 crop={gx},{gy},{cw},{ch}   比例 {cw/max(ch,1):.3f}  ", end="")
        clamp()
    cap.close()
    cv2.destroyAllWindows()
    print("\n最终参数：")
    report()
    print("\n拿去跑自检：")
    print(f"  python 1_采集自检.py --game-res {gw},{gh} --game-crop {gx},{gy},{cw},{ch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
