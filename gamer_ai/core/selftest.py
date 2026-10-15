# -*- coding: utf-8 -*-
"""采集前自检：确认能抓到"活的"游戏画面，并检查参数与变换是否正常。

用法:
    python -m core.selftest
    python -m core.selftest --config my.json
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
from .geometry import aspect_label, crop_from_rel, detect_game_rect, preset_for, rel_from_crop
from .imgio import imwrite_u


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


def auto_detect_geometry(cfg, out_dir, png: str = None) -> None:
    """探测窗口里游戏画面的实际位置和大小，并把 cfg 的取景参数改成适配值。

    ★ 2026-10-14（用户要求：能根据模拟器分辨率自己适应）。
      为什么有必要：以前是用“水平居中 + 底部对齐”猜画面位置，
      实测在 1322x755 的窗口上猜出 off_x=21，而游戏画面其实从第 1 列开始
      （左边只有 1px 边框，工具栏全在右边）-> 裁剪区实际右移 20px。现在直接量。

    　量到的尺寸就是真正的 game_res；裁剪框则由“相对裁剪框”算出（比例锁死 240:293），
    　所以同一份配置在 720p / 1080p / 任意分辨率下都能取到相同的画面取景。
    """
    if png:
        # 调试/离线：直接在一张已保存的整窗截图上探测
        import cv2
        import numpy as np
        bgr = cv2.imdecode(np.fromfile(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            print("[auto] 读不到截图：%s" % png)
            return
        full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ch, cw = full.shape[:2]
        print("[auto] 从截图探测：%s  (%dx%d)" % (png, cw, ch))
    else:
        import ctypes
        from ctypes import wintypes

        u32 = ctypes.windll.user32
        try:
            u32.SetProcessDPIAware()
        except Exception:
            pass
        hwnd = find_window(cfg.window_keyword)
        if not hwnd:
            print("[auto] 没找到模拟器窗口，跳过自动探测")
            return
        rc = wintypes.RECT()
        u32.GetClientRect(hwnd, ctypes.byref(rc))
        cw, ch = rc.right - rc.left, rc.bottom - rc.top

        off = cfg.content_offset if getattr(cfg, "content_offset", None) else None
        cap = WindowCapture(cfg.window_keyword, (cw, ch), (0, 0, cw, ch),
                            backend=cfg.backend, content_offset=off)
        try:
            full = cap.grab()                  # 整个客户区 RGB
        finally:
            cap.close()

    x, y, w, h = detect_game_rect(full)
    print("")
    print("[auto] 客户区 %dx%d  ->  检测到游戏画面 (x=%d, y=%d, %d x %d)  %s"
          % (cw, ch, x, y, w, h, aspect_label(w, h)))
    if w < 64 or h < 64:
        print("[auto] 检测结果太小，不信赖（可能画面全黑/均色），保持原配置")
        return

    rel = preset_for(w, h)
    crop = crop_from_rel(w, h, rel)
    old = (tuple(cfg.game_res), tuple(cfg.game_crop))
    cfg.game_res = (int(w), int(h))
    cfg.game_crop = tuple(int(v) for v in crop)
    cfg.validate()

    print("[auto] 取景参数："
          + ("%s -> %s" % (old[0], cfg.game_res) if old[0] != cfg.game_res else "%s（不变）" % (cfg.game_res,)))
    print("[auto]            "
          + ("%s -> %s" % (old[1], cfg.game_crop) if old[1] != cfg.game_crop else "%s（不变）" % (cfg.game_crop,)))
    print("[auto] 落盘帧 %dx%d（比例锁死）  相对裁剪框 %s"
          % (cfg.frame_w, cfg.frame_h, tuple(round(v, 4) for v in rel_from_crop(w, h, cfg.game_crop))))

    # 存一张标注图，眼睛确认一下框对不对
    try:
        import cv2
        bgr = cv2.cvtColor(full, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy, cw2, ch2 = cfg.game_crop
        cv2.rectangle(bgr, (x + cx, y + cy), (x + cx + cw2, y + cy + ch2), (255, 0, 0), 3)
        cv2.putText(bgr, "game %dx%d" % (w, h), (x + 6, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, "crop %dx%d" % (cw2, ch2), (x + cx + 6, y + cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        p = os.path.join(out_dir, "auto_geometry.png")
        imwrite_u(p, bgr)
        print("[auto] 标注图：%s   （绿框=游戏画面，蓝框=裁剪框）" % p)
    except Exception as e:
        print("[auto] 标注图没存成：%s" % e)

    print("[auto] 想固定下来就把这两行写进 core/config.py：")
    print("           game_res  = %s" % (cfg.game_res,))
    print("           game_crop = %s" % (cfg.game_crop,))


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
    ap.add_argument("--no-pick", dest="pick", action="store_false", default=True,
                    help="自检时**不弹出拉框窗口**（默认会弹：鼠标拖两个对角选采集/部署区域）")
    ap.add_argument("--pick-corners", default=None,
                    help="不开窗口：直接给两个角的窗口坐标 x1,y1,x2,y2 当作拉框结果（脚本/验证用）")
    ap.add_argument("--no-save-geometry", action="store_true",
                    help="拉框后不写取景文件 gamer_ai/geometry.json（只本次生效）")
    ap.add_argument("--save-to-config", action="store_true",
                    help="拉框后**同时**写回 core/config.py 的 crop_rel（老做法，一般不用）")
    ap.add_argument("--game", default=None,
                    help="游戏名（games/<名字>.py）：把它的按键表一起存进新建的档案")
    ap.add_argument("--keys", default=None,
                    help="临时指定按键表并存进档案：w,s,a,d,q,n 或 w=W,n=N")
    ap.add_argument("--profile", default=None,
                    help="区域档案名：拉框后存进 gamer_ai/profiles/<名字>/（含 dataset/、runs/），"
                         "并跳过命名弹窗；不给则弹窗让你起名字")
    ap.add_argument("--auto-crop-png", default=None,
                    help="在一张已保存的整窗截图上做自动探测（调试用，不需要模拟器在跑）")
    ap.add_argument("--auto-crop", action="store_true",
                    help="★ 自动探测游戏画面在窗口里的位置/大小，并把取景参数改成适配值"
                         "（换分辨率/横竖屏后用它，不用再手填 game_res/game_crop）")
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

    # 输出目录先算出来（自动探测要把标注图存进去）
    #   注意：main() 后面还有一处 `from .recorder import resolve_path`，
    #   那样 resolve_path 会成为 main 的局部名字 —— 这里提前用它会 UnboundLocalError，
    #   所以用别名单独导入。
    from .recorder import resolve_path as _resolve_path
    out_dir = args.out or os.path.join(_resolve_path(cfg.output_root), "_selftest")
    if args.auto_crop_png:
        auto_detect_geometry(cfg, out_dir, png=args.auto_crop_png)
        print("")
        print("[auto] 只做探测（--auto-crop-png），不做后续自检。")
        return
    if args.auto_crop:
        auto_detect_geometry(cfg, out_dir)

    # ★ 2026-10-14（用户要求）：自检时弹出窗口，用鼠标拉框选区域，并把选择**持久化** ——
    #   之后 2_采集.py（采集）和 6_自动玩.py（部署）都用这块区域。
    #   实现：选出来的比例写进 core/config.py 的 crop_rel 一行；
    #   采集/部署开捕获时会自动把它换算到当时的分辨率（见 capture.from_config）。
    if args.pick and not args.auto_crop_png:
        try:
            from .geometry import detect_game_rect
            from .picker import (commit_choice, grab_full, pick_crop,
                                 save_crop_rel)
            full = grab_full(cfg)
            if full is None:
                print("[选框] 抓不到画面（窗口没找到？），跳过拉框")
            else:
                corners = None
                if args.pick_corners:
                    v = _ints(args.pick_corners)
                    if len(v) != 4:
                        raise SystemExit("--pick-corners 要写成 x1,y1,x2,y2")
                    corners = (v[0], v[1], v[2], v[3])
                grect = detect_game_rect(full)
                if grect[2] < 200 or grect[3] < 200:
                    # ★ 2026-10-15：以前只警告一句就继续 —— 实测这会让人把一块 237x39 的
                    #   垃圾区域存成正式区域档案，之后采集全废。现在直接不进入拉框。
                    print("[选框] 检测到的游戏画面只有 %dx%d，明显不对（抓错窗口了？）"
                          % (grect[2], grect[3]))
                    print("       已跳过拉框：请把模拟器**游戏画面**显示出来，或检查"
                          " config.py 的 window_keyword 是否只匹配到了控制台窗口。")
                    raise SystemExit(3)
                rel = pick_crop(full, cfg, game_rect=grect, headless_corners=corners,
                                out_png=os.path.join(out_dir, "auto_geometry.png"))
                if rel is None:
                    print("[选框] 没拉框（或按 q 退出）—— 本次仍用原来的取景")
                else:
                    # ★ 2026-10-15（用户要求"设置名字然后确认保存"）：
                    #   鼠标拉完框 -> 弹一个命名小窗 -> 点【确定保存】就建成区域档案
                    #   gamer_ai/profiles/<名字>/（里面自带 geometry.json、dataset/、runs/）。
                    #   --profile NAME 可以跳过弹窗直接建；--no-save-geometry 则完全不存。
                    # ★ 2026-10-15：把当前生效的按键表一起存进档案（用户要求）
                    _gname, _keys = None, None
                    try:
                        from . import gamecfg as _GC
                        _gd = _GC.load(args.game) if args.game else _GC.load()
                        _gname = (_gd or {}).get("_name")
                        if args.keys:
                            _a, _m = _GC.parse_keys(args.keys)
                            _keys = {**_m, **_a}
                        elif _gd:
                            _keys = {**(_gd.get("MARKER_KEYS") or {}),
                                     **(_gd.get("ACTION_KEYS") or {})}
                    except SystemExit as _e:
                        print(_e)
                    if args.no_save_geometry:
                        print("[选框] 加了 --no-save-geometry：这次只在本轮生效，不存文件")
                    else:
                        commit_choice(
                            rel, grect, full.shape,
                            ask=(args.profile is None and not args.pick_corners),
                            profile=args.profile,
                            note="1_自检.py 拉框选的（游戏画面 %dx%d）"
                                 % (grect[2], grect[3]),
                            game=_gname, keys=_keys)
                    if args.save_to_config:
                        save_crop_rel(rel)
                    cfg.crop_rel = tuple(float(x) for x in rel)   # 本次自检立即生效
                    cfg.validate()
        except Exception as e:
            print("[选框] 拉框环节出错，跳过：%s" % e)

    print("=" * 78)
    print("采集自检")
    print("=" * 78)
    # 让"这块数据会进哪个区域档案"在动手前就一目了然
    try:
        from . import profiles as _P
        _cur = _P.resolve(args.profile)
        if _cur:
            print("[档案] 当前区域：%s" % _P.describe(_cur))
            print("       数据 -> %s" % _P.dataset_root(_cur))
            print("       模型 -> %s" % _P.runs_root(_cur))
        else:
            print("[档案] 没有当前区域 -> 采集的数据进 gamer_ai/dataset（拉框保存时会建档案）")
    except Exception as _e:
        print("[档案] 读档案信息失败（不影响自检）：%s" % _e)
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
    # 同样走自动对准；命令行 --content-offset 给了就优先
    cap = WindowCapture.from_config(cfg, backend=cfg.backend, content_offset=coff)
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

    imwrite_u(os.path.join(out_dir, "crop.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    # ★ 2026-10-10：再存一张「模型真正吃进 CNN 的 224x224」（反归一化回 0~255 便于肉眼对比）
    #   判断"换个裁剪框还能不能识别"，看这张图比看原图准。
    _ev = build_eval_transform(224)
    _v = _ev(to_video_tensor(np.stack([frames[-1]])))[0].numpy().transpose(1, 2, 0)
    _mean = np.array([0.485, 0.456, 0.406], np.float32)
    _std = np.array([0.229, 0.224, 0.225], np.float32)
    _v = (np.clip(_v * _std + _mean, 0.0, 1.0) * 255.0).astype(np.uint8)
    imwrite_u(os.path.join(out_dir, "model_view.png"), cv2.cvtColor(_v, cv2.COLOR_RGB2BGR))
    print(f"    [存图] 模型实际输入视图 {os.path.join(out_dir, 'model_view.png')}")
    # 整窗 + 裁剪框示意
    full = np.frombuffer(cap._buf, dtype=np.uint8).reshape(cap.client_h, cap.client_w, 4)[:, :, :3][:, :, ::-1].copy()
    x = cap.off_x + cfg.game_crop[0]
    y = cap.off_y + cfg.game_crop[1]
    cv2.rectangle(full, (x, y), (x + cfg.crop_w, y + cfg.crop_h), (0, 0, 255), 3)
    imwrite_u(os.path.join(out_dir, "window_with_crop.png"), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    cap.close()
    print(f"    调试图已保存到 {os.path.abspath(out_dir)}")

    check_transform(cfg)
    print("\n" + "=" * 78)
    print("自检结果:", "通过，可以开始采集" if ok else "未通过，请按上面的提示处理后重试")
    print("=" * 78)


if __name__ == "__main__":
    main()
