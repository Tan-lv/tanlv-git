# -*- coding: utf-8 -*-
r"""鼠标选裁剪框（独立入口）。

真正的实现在 core/picker.py —— `1_自检.py` 里弹出的是**同一个窗口**、同一套逻辑。
这个脚本只是让你能**单独**调取景，不必跑整套自检。

用法：
    python tools\鼠标选裁剪框.py                    # 在模拟器窗口上鼠标拉框（选完写 gamer_ai/geometry.json）
    python tools\鼠标选裁剪框.py --png 某张截图.png   # 用截图，模拟器不开也行
    python tools\鼠标选裁剪框.py --corners 366,35,956,755    # 不开窗口直接算（脚本/验证用）
    python tools\鼠标选裁剪框.py --no-save           # 只看看，不写文件
    python tools\鼠标选裁剪框.py --save-to-config    # 额外写回 core/config.py（老做法）

★ 2026-10-15：默认存**取景文件** gamer_ai/geometry.json，而不是写死 config.py ——
  区域是跟着模型的：用 A 区域训的模型，部署时就该用 A 区域。存的是比例 + 标定分辨率，
  采集(2_)、训练(5_)、部署(9_)、补数据(10_) 全都读它，换分辨率/横竖屏自动换算。

窗口里：鼠标左键拖两个对角；黄框=你拖的，蓝框=按 240:293 吸附后真正用的，绿框=游戏画面；
        右侧实时显示落盘帧 240x293 和模型真正吃到的 224x224。
按键：  r 重选   l 切换锁定比例   s 存标注图并打印   q/Esc 退出
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from core.config import Config                                    # noqa: E402
from core.geometry import (aspect_label, crop_from_rel,            # noqa: E402
                               detect_game_rect)
from core.picker import (grab_full, pick_crop, save_crop_rel,      # noqa: E402
                              save_geometry_choice)


def main() -> int:
    ap = argparse.ArgumentParser(description="鼠标选裁剪框（拖两个对角）")
    ap.add_argument("--png", default=None, help="用一张已保存的整窗截图，而不是抓实时窗口")
    ap.add_argument("--corners", default=None,
                    help="不开窗口：直接给两个角的窗口坐标 x1,y1,x2,y2")
    ap.add_argument("--out", default=None, help="标注图存哪（默认 runs/_crop_pick.png）")
    ap.add_argument("--no-save", action="store_true",
                    help="不写取景文件（只本次看看）")
    ap.add_argument("--save-to-config", action="store_true",
                    help="额外写回 core/config.py 的 crop_rel（老做法，一般不用）")
    args = ap.parse_args()

    cfg = Config()
    full = grab_full(cfg, args.png)
    if full is None:
        print("抓不到画面：找不到标题含 %r 的窗口 —— 先把模拟器打开（或改用 --png）"
              % cfg.window_keyword)
        return 1
    ch, cw = full.shape[:2]
    grect = detect_game_rect(full)
    print("整窗 %dx%d   检测到游戏画面 (x=%d, y=%d, %dx%d)  %s"
          % (cw, ch, grect[0], grect[1], grect[2], grect[3], aspect_label(grect[2], grect[3])))
    if grect[2] < 64 or grect[3] < 64:
        print("[警告] 检测到的游戏画面太小 —— 可能抓错窗口了（雷电多开器也叫“雷电模拟器”）")

    corners = None
    if args.corners:
        v = [int(t) for t in args.corners.replace(" ", ",").split(",") if t.strip()]
        if len(v) != 4:
            raise SystemExit("--corners 要写成 x1,y1,x2,y2")
        corners = (v[0], v[1], v[2], v[3])
    out_png = args.out or os.path.join(HERE, "runs", "_crop_pick.png")
    rel = pick_crop(full, cfg, game_rect=grect, headless_corners=corners, out_png=out_png)
    if rel is None:
        print("没选（或按 q 退出）")
        return 0
    print("")
    print("  选定相对裁剪框 = %s" % (tuple(round(v, 4) for v in rel),))
    print("  按当前游戏画面 %dx%d 算出来的绝对框 = %s"
          % (grect[2], grect[3], crop_from_rel(grect[2], grect[3], rel)))
    if args.no_save:
        print("  （加了 --no-save，什么都没写）")
        return 0
    # ★ 默认写取景文件：采集/训练/部署/补数据都读它。
    save_geometry_choice(
        rel, (grect[2], grect[3]), crop_from_rel(grect[2], grect[3], rel),
        offset=(grect[0], grect[1]), client=(cw, ch),
        note="tools/鼠标选裁剪框.py 选的（游戏画面 %dx%d%s）"
             % (grect[2], grect[3], "，来自截图" if args.png else ""))
    if args.save_to_config:
        save_crop_rel(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
