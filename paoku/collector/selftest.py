# -*- coding: utf-8 -*-
"""采集前自检：确认能抓到"活的"游戏画面，并检查参数与变换是否正常。

用法:
    python -m collector.selftest
    python -m collector.selftest --config my.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .capture import WindowCapture, find_window
from .config import Config
from .dataset import build_eval_transform, build_train_transform, flatten_frames, to_video_tensor


def _ints(s: str):
    """把 "1280,720" / "1280x720" / "339 0 590 720" 都解析成整数列表。"""
    return [int(v) for v in str(s).replace("x", ",").replace("X", ",")
            .replace(" ", ",").split(",") if v.strip()]


def check_transform(cfg: Config) -> None:
    print("\n[3/3] 变换管线自检")
    import torch
    fake = np.random.randint(0, 255, (cfg.input_frames, cfg.frame_h, cfg.frame_w, 3), dtype=np.uint8)
    x = to_video_tensor(fake)
    tr = build_train_transform(224)(x)
    ev = build_eval_transform(224)(x)
    print(f"    输入            : {tuple(x.shape)}")
    print(f"    训练增强后      : {tuple(tr.shape)} -> 展平 {tuple(flatten_frames(tr).shape)}")
    print(f"    评估变换后      : {tuple(ev.shape)} -> 展平 {tuple(flatten_frames(ev).shape)}")
    assert flatten_frames(tr).shape == (cfg.input_frames * 3, 224, 224), "变换输出形状不对"
    print("    [OK] 形状正确（同一套随机参数作用于所有帧）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seconds", type=float, default=4.0, help="抓帧测试时长")
    ap.add_argument("--backend", default=None, choices=["printwindow", "bitblt"])
    ap.add_argument("--out", default=None, help="调试图片输出目录")
    # ★ 2026-10-10：取景参数可以在命令行临时覆盖，用来标定横屏/换分辨率后的裁剪框
    ap.add_argument("--game-res", default=None, help="覆盖游戏分辨率，如 1280,720")
    ap.add_argument("--game-crop", default=None, help="覆盖裁剪框 x,y,宽,高，如 339,0,590,720")
    ap.add_argument("--content-offset", default=None,
                    help="安卓画面在窗口内的左上角偏移 x,y（默认自动算：水平居中+底部对齐）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.backend:
        cfg.backend = args.backend
    if args.game_res:
        v = _ints(args.game_res)
        if len(v) != 2:
            raise SystemExit("--game-res 要写成 宽,高，例如 1280,720")
        cfg.game_res = (v[0], v[1])
    if args.game_crop:
        v = _ints(args.game_crop)
        if len(v) != 4:
            raise SystemExit("--game-crop 要写成 x,y,宽,高，例如 339,0,590,720")
        cfg.game_crop = (v[0], v[1], v[2], v[3])
    coff = None
    if args.content_offset:
        v = _ints(args.content_offset)
        if len(v) != 2:
            raise SystemExit("--content-offset 要写成 x,y")
        coff = (v[0], v[1])
    cfg.validate()

    print("=" * 78)
    print("采集自检")
    print("=" * 78)
    print(f"配置: 后端={cfg.backend}  游戏分辨率={cfg.game_res}  裁剪={cfg.game_crop}")
    print(f"      落盘 {cfg.frame_w}x{cfg.frame_h}  窗口 -{cfg.save_pre_ms}ms~+{cfg.save_post_ms}ms "
          f"({cfg.saved_frames} 帧)  模型输入 {cfg.input_frames} 帧")
    # ★ 裁剪比例决定「模型实际看到什么」：前处理是「短边缩到 224 + 中心裁 224」，
    #   比例一变，落盘帧尺寸就变，有效视野跟着变 —— 现有模型全是按 720:880 训的。
    _r_new = cfg.crop_w / float(cfg.crop_h)
    _r_std = 720 / 880.0
    print(f"      裁剪比例 {_r_new:.3f}（项目标准 720:880 = {_r_std:.3f}）")
    if abs(_r_new / _r_std - 1) > 0.01:
        print("      [注意] 比例和项目标准不一致：落盘帧会变成 %dx%d，"
              % (cfg.frame_w, cfg.frame_h))
        print("             模型看到的有效视野会和训练时完全不同（大概率识别失效）。")
        print("             横屏想沿用现有模型，请取 590x720 这类【同比例竖窗】。")

    print("\n[1/3] 查找窗口")
    hwnd = find_window(cfg.window_keyword)
    if not hwnd:
        print(f"    [FAIL] 没找到标题包含 '{cfg.window_keyword}' 的窗口")
        print("           请先启动雷电模拟器并进入游戏")
        return
    print(f"    [OK] hwnd={hwnd}")

    print(f"\n[2/3] 连续抓帧 {args.seconds:.0f} 秒")
    cap = WindowCapture(cfg.window_keyword, cfg.game_res, cfg.game_crop, backend=cfg.backend,
                        content_offset=coff)
    print(f"    客户区 {cap.client_w}x{cap.client_h}  画面偏移 ({cap.off_x},{cap.off_y})")
    from .recorder import resolve_path
    out_dir = args.out or os.path.join(resolve_path(cfg.output_root), "_selftest")
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    prev = None
    changed = 0
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        f = cap.grab()
        frames.append(f)
        if prev is not None and not np.array_equal(f, prev):
            changed += 1
        prev = f
    fps = len(frames) / args.seconds
    arr = np.stack(frames)
    print(f"    抓帧 {len(frames)} 帧  {fps:.1f} fps")
    print(f"    画面 mean={arr.mean():.1f} std={arr.std():.1f}")
    print(f"    帧间发生变化的次数: {changed}/{len(frames)-1}")

    ok = True
    if arr.std() < cfg.black_std_threshold:
        print("    [FAIL] 画面几乎是纯色 —— 抓帧无效")
        ok = False
    if changed == 0:
        print("    [FAIL] 画面完全没有变化 —— 模拟器可能已经卡死，请重启模拟器；")
        print("           如果游戏正停在暂停/菜单界面，这属于正常现象，进入关卡即可。")
        ok = False
    else:
        print("    [OK] 画面是活的")

    cv2.imwrite(os.path.join(out_dir, "crop.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    # ★ 2026-10-10：再存一张「模型真正吃进 CNN 的 224x224」（反归一化回 0~255 便于肉眼对比）
    #   判断"换个裁剪框还能不能识别"，看这张图比看原图准。
    _ev = build_eval_transform(224)
    _v = _ev(to_video_tensor(np.stack([frames[-1]])))[0].numpy().transpose(1, 2, 0)
    _mean = np.array([0.485, 0.456, 0.406], np.float32)
    _std = np.array([0.229, 0.224, 0.225], np.float32)
    _v = (np.clip(_v * _std + _mean, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "model_view.png"), cv2.cvtColor(_v, cv2.COLOR_RGB2BGR))
    print(f"    [存图] 模型实际输入视图 {os.path.join(out_dir, 'model_view.png')}")
    # 整窗 + 裁剪框示意
    full = np.frombuffer(cap._buf, dtype=np.uint8).reshape(cap.client_h, cap.client_w, 4)[:, :, :3][:, :, ::-1].copy()
    x = cap.off_x + cfg.game_crop[0]
    y = cap.off_y + cfg.game_crop[1]
    cv2.rectangle(full, (x, y), (x + cfg.crop_w, y + cfg.crop_h), (0, 0, 255), 3)
    cv2.imwrite(os.path.join(out_dir, "window_with_crop.png"), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    cap.close()
    print(f"    调试图已保存到 {os.path.abspath(out_dir)}")

    check_transform(cfg)
    print("\n" + "=" * 78)
    print("自检结果:", "通过，可以开始采集" if ok else "未通过，请按上面的提示处理后重试")
    print("=" * 78)


if __name__ == "__main__":
    main()
