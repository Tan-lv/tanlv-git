# -*- coding: utf-8 -*-
r"""鼠标拉框选裁剪区域（自检和独立工具共用这一套）。

为什么要抽出来：1_自检.py 要"跑完自检顺便让你拉框"，tools\鼠标选裁剪框.py 要能单独用。
两边都调这里，行为一致。

选出来的是**相对比例** (x0, y0, x1, y1)（相对游戏画面），再由 save_crop_rel() 写回
core/config.py 的 crop_rel 一行 —— 这样采集和自动玩下次启动就都用这块区域了。
"""
from __future__ import annotations

import os
import re
import shutil
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .dataset import build_eval_transform, to_video_tensor
from .geometry import crop_from_rel, detect_game_rect, rel_from_crop
from .imgio import imwrite_u

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def grab_full(cfg, png: str = None):
    """抓整个窗口客户区（RGB）。png 给定时从截图读。抓不到返回 None。"""
    if png:
        bgr = cv2.imdecode(np.fromfile(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    import ctypes
    from ctypes import wintypes

    from .capture import WindowCapture, find_window
    u32 = ctypes.windll.user32
    try:
        u32.SetProcessDPIAware()
    except Exception:
        pass
    hwnd = find_window(cfg.window_keyword)
    if not hwnd:
        return None
    rc = wintypes.RECT()
    u32.GetClientRect(hwnd, ctypes.byref(rc))
    cw, ch = rc.right - rc.left, rc.bottom - rc.top
    cap = WindowCapture(cfg.window_keyword, (cw, ch), (0, 0, cw, ch), backend=cfg.backend)
    try:
        return cap.grab()
    finally:
        cap.close()


def _cn_font(size: int):
    """找一个能显示中文的字体（微软雅黑/黑体），找不到返回 None。"""
    try:
        from PIL import ImageFont
    except Exception:
        return None
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return None


def put_cn(bgr: np.ndarray, lines, org=(10, 10), size=18, colors=None,
           plate: bool = True) -> np.ndarray:
    """在 BGR 图上写中文（可选先铺一层半透明黑底，保证在花哨画面上也看得清）。

    ★ OpenCV 自带的 cv2.putText 只能画英文/数字（画中文会变成一串问号），
      所以这里借 PIL + 系统字体来画 —— 弹窗里的图例要让人一眼看懂，不能靠猜。
    """
    from PIL import Image, ImageDraw
    out = bgr
    x, y = int(org[0]), int(org[1])
    line_h = size + 6
    if plate and lines:
        # 中文一个字大约占 size 宽，英文按一半估 —— 只用来算底衬大小，不用太准
        wmax = max(sum(size if ord(ch) > 255 else size * 0.55 for ch in t) for t in lines)
        w = int(min(bgr.shape[1] - x - 4, wmax + 16))
        h = int(min(bgr.shape[0] - y - 4, line_h * len(lines) + 10))
        if w > 4 and h > 4:
            roi = out[y - 6:y - 6 + h + 8, x - 8:x - 8 + w + 12]
            if roi.size:
                dark = np.zeros_like(roi)
                out = out.copy()
                out[y - 6:y - 6 + h + 8, x - 8:x - 8 + w + 12] = cv2.addWeighted(
                    roi, 0.32, dark, 0.68, 0)
    img = Image.fromarray(np.ascontiguousarray(out[:, :, ::-1]))   # BGR -> RGB
    d = ImageDraw.Draw(img)
    font = _cn_font(size)
    for i, t in enumerate(lines):
        col = tuple(colors[i]) if colors else (255, 255, 255)
        d.text((x, y), t, font=font, fill=(int(col[0]), int(col[1]), int(col[2])))
        y += line_h
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])         # RGB -> BGR


def model_view(rgb_crop: np.ndarray, frame_w: int, frame_h: int, tf) -> np.ndarray:
    """裁剪图(RGB) -> 落盘尺寸 -> 模型真正吃到的 224x224（返回 BGR 便于显示）。"""
    stored = cv2.resize(rgb_crop, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
    x = tf(to_video_tensor(np.stack([np.ascontiguousarray(stored)])))[0].numpy().transpose(1, 2, 0)
    return (np.clip(x * STD + MEAN, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]


def rel_and_snap(game_rect, box, frame_w, frame_h, lock=True):
    """窗口坐标的框 -> (相对比例, 吸附后的窗口坐标框)。"""
    gx, gy, gw, gh = game_rect
    x0, y0, x1, y1 = box
    cx = max(0, min(gw - 8, min(x0, x1) - gx))
    cy = max(0, min(gh - 8, min(y0, y1) - gy))
    cw = max(8, min(abs(x1 - x0), gw - cx))
    ch = max(8, min(abs(y1 - y0), gh - cy))
    rel = rel_from_crop(gw, gh, (cx, cy, cw, ch))
    if lock:
        sc = crop_from_rel(gw, gh, rel, frame_w, frame_h)
    else:
        sc = (cx, cy, cw, ch)
    return rel, (sc[0] + gx, sc[1] + gy, sc[2], sc[3])


def _compose(full_rgb, game_rect, box, lock, frame_w, frame_h, tf, disp_w):
    """画好显示图：左边整窗+三个框，右边预览条（落盘帧 + CNN 输入）。"""
    gx, gy, gw, gh = game_rect
    ch, cw = full_rgb.shape[:2]
    view = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR).copy()
    cv2.rectangle(view, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 1)
    stored = None
    if box:
        cv2.rectangle(view, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        _rel, snap = rel_and_snap(game_rect, box, frame_w, frame_h, lock)
        cv2.rectangle(view, (snap[0], snap[1]), (snap[0] + snap[2], snap[1] + snap[3]),
                      (255, 0, 0), 2)
        y0 = max(0, snap[1]); x0 = max(0, snap[0])
        sub = full_rgb[y0:y0 + snap[3], x0:x0 + snap[2]]
        if sub.size:
            stored = sub
    scale = min(1.0, disp_w / float(cw))
    left = cv2.resize(view, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    # ★ 2026-10-15（用户问"蓝框黄框是什么意思"）：直接在图上写清楚，
    #   以前只在控制台和窗口标题里写英文，不看就不知道哪个框才是真正生效的。
    _legend = [
        "绿框 = 自动认出的游戏画面",
        "黄框 = 你拖出来的框",
        "蓝框 = 锁定 240:293 后真正会用的框  ← 看这个",
    ]
    _lcol = [(80, 255, 80), (60, 230, 255), (255, 140, 60)]      # RGB
    if box is None:
        _legend.append("按住左键拖两个对角（松手即可）")
        _lcol.append((255, 255, 255))
    else:
        _legend.append("按 s = 命名并保存   r = 重选   l = 解锁比例   q = 退出")
        _lcol.append((255, 255, 255))
    left = put_cn(left, _legend, org=(18, 18), size=20, colors=_lcol)
    panel = np.full((left.shape[0], 260, 3), 24, np.uint8)
    cv2.putText(panel, "stored %dx%d" % (frame_w, frame_h), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if stored is not None:
        pv = cv2.resize(stored[:, :, ::-1], (240, int(240 * frame_h / frame_w)),
                        interpolation=cv2.INTER_AREA)
        h = min(pv.shape[0], left.shape[0] // 2 - 10)
        pv = cv2.resize(pv, (240, h), interpolation=cv2.INTER_AREA)
        panel[24:24 + h, 10:250] = pv
        mv = model_view(stored, frame_w, frame_h, tf)
        mv = cv2.resize(mv, (240, 240), interpolation=cv2.INTER_NEAREST)
        y2 = 24 + h + 26
        if y2 + 240 <= left.shape[0]:
            panel[y2:y2 + 240, 10:250] = mv
            cv2.putText(panel, "CNN 224x224", (10, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "drag 2 corners...", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return np.concatenate([left, panel], axis=1)


def pick_crop(full_rgb, cfg, game_rect=None, headless_corners=None,
              out_png: str = None, disp_w: int = 900) -> Optional[Tuple[float, float, float, float]]:
    """拉框选裁剪区域，返回相对比例 (x0,y0,x1,y1)；取消返回 None。

    headless_corners=(x1,y1,x2,y2) 时不开窗口，直接算（脚本/自检验证用）。
    """
    ch, cw = full_rgb.shape[:2]
    if game_rect is None:
        game_rect = detect_game_rect(full_rgb)
    frame_w, frame_h = int(cfg.frame_w), int(cfg.frame_h)
    tf = build_eval_transform(224)

    if headless_corners is not None:
        x1, y1, x2, y2 = headless_corners
        rel, snap = rel_and_snap(game_rect, (x1, y1, x2, y2), frame_w, frame_h)
        if out_png:
            imwrite_u(out_png, _compose(full_rgb, game_rect, (x1, y1, x2, y2), True,
                                        frame_w, frame_h, tf, disp_w))
        print("[选框] 免窗口：窗口坐标 (%d,%d)-(%d,%d) -> 吸附后 %s" % (x1, y1, x2, y2, snap))
        return rel

    scale = min(1.0, disp_w / float(cw))
    state = {"p0": None, "p1": None, "drag": None, "lock": True, "box": None}

    def on_mouse(event, x, y, _f, _p):
        wx, wy = int(x / scale), int(y / scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["p0"] = (wx, wy)
            state["p1"] = None
            state["drag"] = (wx, wy)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            state["drag"] = (wx, wy)
        elif event == cv2.EVENT_LBUTTONUP:
            state["p1"] = (wx, wy)
            state["drag"] = None

    def clamp(x0, y0, x1, y1):
        gx, gy, gw, gh = game_rect
        x0 = max(gx, min(gx + gw, x0)); x1 = max(gx, min(gx + gw, x1))
        y0 = max(gy, min(gy + gh, y0)); y1 = max(gy, min(gy + gh, y1))
        return int(x0), int(y0), int(x1), int(y1)

    win = "pick crop  |  drag 2 corners   r=reset  l=lock-aspect  s=save+print  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print("")
    print("  在弹出的窗口里用鼠标拖一个矩形（两个对角）——")
    print("    绿框 = 自动认出的游戏画面")
    print("    黄框 = 你拖出来的框")
    print("    蓝框 = 锁定 %d:%d 后**真正会用**的框（比例必须锁死，否则模型看到的视野会变）"
          % (frame_w, frame_h))
    print("  拖完按 s 保存：窗口会关掉，接着弹出「给这块区域起名字」的小窗，")
    print("  里面有【保存】按钮（以前只能按键盘 s，确实不显眼）。")
    print("  右侧实时显示：落盘帧 + 模型真正吃到的 224x224")
    result = None
    while True:
        box = None
        if state["p0"] and state["p1"]:
            box = clamp(state["p0"][0], state["p0"][1], state["p1"][0], state["p1"][1])
        elif state["p0"] and state["drag"]:
            box = clamp(state["p0"][0], state["p0"][1], state["drag"][0], state["drag"][1])
        state["box"] = box
        cv2.imshow(win, _compose(full_rgb, game_rect, box, state["lock"],
                                 frame_w, frame_h, tf, disp_w))
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            state["p0"] = state["p1"] = state["drag"] = None
        elif k == ord("l"):
            state["lock"] = not state["lock"]
            print("  锁定比例 %d:%d -> %s" % (frame_w, frame_h, state["lock"]))
        elif k == ord("s"):
            if not box:
                print("  还没拉框呢")
                continue
            if out_png:
                imwrite_u(out_png, _compose(full_rgb, game_rect, box, state["lock"],
                                            frame_w, frame_h, tf, disp_w))
                print("  标注图 -> %s" % out_png)
            result = rel_and_snap(game_rect, box, frame_w, frame_h, state["lock"])[0]
            print("  选定相对裁剪框 = %s" % (tuple(round(v, 4) for v in result),))
            break        # ★ 关掉这个窗口，接着弹「命名 + 确定保存」的小窗
    cv2.destroyAllWindows()
    if result is None and state["box"] and state["p1"]:
        result = rel_and_snap(game_rect, state["box"], frame_w, frame_h, state["lock"])[0]
    return result


def save_geometry_choice(rel, game_res, game_crop, offset=None, client=None,
                         path: str = None, note: str = "") -> str:
    """把选定的取景写进**取景文件**（默认 gamer_ai/geometry.json）。返回路径。

    ★ 2026-10-15（用户建议）：区域记在文件里，而不是写死在 core/config.py ——
      因为区域是"跟着模型走"的：用 A 区域训的模型，部署时就该用 A 区域。
      文件里同时记了当时的设备分辨率（game_res）和绝对框（game_crop），
      后续步骤读 crop_rel 就能换算到任何分辨率。
    """
    from .geometry import describe, make_geometry, save_geometry
    geo = make_geometry(rel, game_res, game_crop, offset, client,
                        note or "自检弹窗鼠标拉框选出来的区域")
    p = save_geometry(geo, path)
    print("[取景] 已写入 %s" % p)
    print("       %s" % describe(geo))
    print("       采集(2_)、训练(5_)、部署(9_)都会读它；换分辨率/横竖屏自动按比例换算")
    return p


def suggest_profile_name(game_w: int, game_h: int) -> str:
    """给个默认名字：横屏_1280x720 / 竖屏_720x1280 这种，直接回车也能用。"""
    fmt = "横屏" if game_w >= game_h else "竖屏"
    return "%s_%dx%d" % (fmt, int(game_w), int(game_h))


def ask_profile_name(default_name: str = "", default_note: str = "") -> tuple:
    """弹一个小窗：给选好的区域起名字 + 备注，然后点按钮保存。

    返回 (action, name, note)：
        "profile" -> 存成区域档案 gamer_ai/profiles/<名字>/（自带 dataset/ 和 runs/）
        "project" -> 只更新项目级 gamer_ai/geometry.json（老做法，不建档案）
        "cancel"  -> 什么都不做

    ★ 为什么用 tkinter：OpenCV 的窗口（cv2.namedWindow）里**没有输入框和按钮**，
      只能靠键盘；要"设置名字然后确认保存"就必须开一个真窗口。
      tkinter 是 Python 自带的，不用额外装。
    """
    import tkinter as tk
    from . import profiles as P

    res = {"action": "cancel", "name": "", "note": default_note}
    root = tk.Tk()
    root.title("保存这块区域")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    tk.Label(root, text="给这块画面区域起个名字：", font=("Microsoft YaHei", 11)
             ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 2))
    name_var = tk.StringVar(value=default_name)
    e1 = tk.Entry(root, textvariable=name_var, width=36, font=("Microsoft YaHei", 11))
    e1.grid(row=1, column=0, sticky="we", padx=14)
    e1.focus_set()
    e1.select_range(0, "end")

    tk.Label(root, text="备注（可以留空）：", font=("Microsoft YaHei", 10)
             ).grid(row=2, column=0, sticky="w", padx=14, pady=(10, 2))
    note_var = tk.StringVar(value=default_note)
    tk.Entry(root, textvariable=note_var, width=36, font=("Microsoft YaHei", 10)
             ).grid(row=3, column=0, sticky="we", padx=14)

    hint = tk.Label(root, text="", justify="left", anchor="w", wraplength=380,
                    font=("Microsoft YaHei", 10), fg="#333")
    hint.grid(row=4, column=0, sticky="we", padx=14, pady=(10, 4))

    def refresh(*_a):
        nm = P.sanitize(name_var.get())
        if not nm:
            hint.config(text="名字不能为空")
        elif P.exists(nm):
            hint.config(text="已存在同名区域：只更新它的取景，里面的数据/模型不动\n"
                            "目录 gamer_ai\\profiles\\%s\\" % nm)
        else:
            hint.config(text="将新建 gamer_ai\\profiles\\%s\\（已自动建好子目录）\n"
                            "geometry.json 取景 / dataset\\ 采集数据 / runs\\ 训练模型" % nm)

    name_var.trace_add("write", refresh)
    refresh()

    btns = tk.Frame(root)
    btns.grid(row=5, column=0, padx=14, pady=(6, 14), sticky="we")

    def do_profile():
        nm = P.sanitize(name_var.get())
        if not nm:
            hint.config(text="名字不能为空")
            return
        res.update(action="profile", name=nm, note=note_var.get().strip())
        root.destroy()

    def do_project():
        res.update(action="project", name="", note=note_var.get().strip())
        root.destroy()

    def do_cancel():
        res.update(action="cancel")
        root.destroy()

    tk.Button(btns, text="确定保存（建成区域档案）", command=do_profile,
              font=("Microsoft YaHei", 11), bg="#2d7dd2", fg="white", padx=10, pady=4
              ).pack(side="left")
    tk.Button(btns, text="只存项目 geometry.json", command=do_project,
              font=("Microsoft YaHei", 10), padx=8, pady=4).pack(side="left", padx=8)
    tk.Button(btns, text="取消", command=do_cancel,
              font=("Microsoft YaHei", 10), padx=10, pady=4).pack(side="left")

    root.bind("<Return>", lambda _e: do_profile())
    root.bind("<Escape>", lambda _e: do_cancel())
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = root.winfo_width(), root.winfo_height()
    root.geometry("+%d+%d" % (max(0, (sw - w) // 2), max(0, (sh - h) // 2)))
    root.mainloop()
    return res["action"], res["name"], res["note"]


def commit_choice(rel, game_rect, full_shape, ask: bool = True, profile: str = None,
                  note: str = "", game: str = None, keys: dict = None) -> tuple:
    """把选好的区域落盘：起名字 -> 建档案 -> 写 geometry.json。返回 (action, name)。

    ask=False 时直接用 profile 给的名字（脚本/验证用，不弹窗）。
    """
    from .geometry import describe, make_geometry, save_geometry
    from . import profiles as P

    gx, gy, gw, gh = [int(v) for v in game_rect]
    ch, cw = int(full_shape[0]), int(full_shape[1])
    crop = crop_from_rel(gw, gh, rel)
    # ★ 2026-10-16：存档案时**一律记吸附后的框**（rel_from_crop(crop)），而不是你鼠标拖的
    #   原始框。以前如果拖的不是 240:293（或按了 l 解锁比例），档案里就会记下那个"宽框"，
    #   而真正生效的是吸附后的窄框 —— 档案和实际采的数据对不上，看着就像"我框的没生效"。
    from .geometry import rel_from_crop as _rel_from_crop
    rel = _rel_from_crop(gw, gh, crop)
    geo = make_geometry(rel, (gw, gh), game_crop=crop, offset=(gx, gy),
                        client=(cw, ch), note=note or "自检弹窗鼠标拉框选出来的区域")

    if ask:
        print("")
        print("  [保存] 弹出命名窗口 —— OpenCV 窗口里没有按钮，这个才是「确定保存」：")
        action, name, note2 = ask_profile_name(suggest_profile_name(gw, gh), note)
        if note2:
            geo["note"] = note2
    else:
        action, name = ("profile", P.sanitize(profile)) if profile else ("project", "")

    if action == "profile" and name:
        # ★ 2026-10-15：把"这份按键表"也一起存进档案（用户要求：
        #   设计按键时生成的按键要放在分辨率/截图区域那个文件里，后面各步都读它）
        P.create(name, geo, note=geo.get("note", ""), game=game, keys=keys)
        P.set_current(name)
        print("[档案] 已建/更新区域档案：%s" % P.sanitize(name))
        print("       %s" % P.describe(name))
        print("       目录 %s" % P.profile_dir(name))
        print("         ├─ geometry.json  这块区域（分辨率 + 截图区域）")
        if keys:
            print("         ├─ keys.json      按键表 %s" % " ".join(
                "%s=%s" % (k, v) for k, v in keys.items()))
        print("         ├─ dataset\\       采集(2_)的数据落这里")
        print("         └─ runs\\          训练(5_)的模型落这里")
        print("       已设为【当前区域】：2_/5_/6_/9_/10_ 不加 --profile 时就用它")
        return "profile", P.sanitize(name)

    if action == "project":
        p = save_geometry(geo)
        print("[取景] 已写入项目取景文件 %s" % p)
        print("       %s" % describe(geo))
        return "project", ""

    print("[取景] 你点了取消：这次的选择不保存（本次运行仍然会用它）")
    return "cancel", ""


def save_crop_rel(rel, quiet: bool = False) -> Optional[str]:
    """把选定的相对裁剪框写回 core/config.py 的 crop_rel 一行（写前先备份）。

    这样**采集（2_采集.py）和自动玩（6_自动玩.py）下次启动就用这块区域**，
    因为它们的取景都来自 Config.crop_rel + auto_geometry（见 capture.from_config）。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "config.py")
    try:
        src = open(p, encoding="utf-8").read()
    except Exception as e:
        print("[配置] 读不到 config.py（%s），没写。手工填：crop_rel = %s" % (e, rel))
        return None
    lit = "(%s)" % ", ".join("%.4f" % float(v) for v in rel)
    #   注意要有 \s* ：config.py 里这一行是缩进的（4 个空格）
    pat = re.compile(r"(?m)^(\s*crop_rel:.*?=\s*)(.+)$")
    if not pat.search(src):
        print("[配置] config.py 里没找到 crop_rel 那一行，没写。手工填：crop_rel = %s" % lit)
        return None
    new = pat.sub(lambda m: m.group(1) + lit, src, count=1)
    bak = p + ".bak_" + time.strftime("%Y%m%d_%H%M%S")
    try:
        shutil.copy2(p, bak)
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    except Exception as e:
        print("[配置] 写入失败（%s），手工填：crop_rel = %s" % (e, lit))
        return None
    if not quiet:
        print("[配置] 已把取景写进 core/config.py：crop_rel = %s" % lit)
        print("       备份：%s" % os.path.basename(bak))
        print("       以后 2_采集.py / 6_自动玩.py 会自动用这块区域（换分辨率也不需要改）")
    return lit
