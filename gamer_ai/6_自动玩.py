# -*- coding: utf-8 -*-
"""部署端：多帧行为克隆 + 动作锁定。

解决两件事
------------------------------------------------------------------
1. 训练/部署一致：所有参数（裁剪框、帧宽、输入帧数、偏移、归一化）
   全部从训练产出的 model_meta.json 读取，抓帧后端也复用同一份代码。

2. 抑制"移动中重复触发"：
   * 第一道防线（模型侧）：训练数据里包含"换道动画中 -> N"的负样本，
     模型学会在角色已经在动的时候输出 N。
   * 第二道防线（部署侧）：每发出一次按键后，硬锁 --lockout-ms 毫秒内
     不再发出任何按键。换道动画大约 300ms，锁 320ms 正好覆盖。

快捷键：F8 暂停/继续   F10/Esc 退出

用法:
    python 6_自动玩.py                        # 用下面 MODEL_DIR 里写死的模型
    python 6_自动玩.py --run runs/exp01        # 临时换成别的模型
    python 6_自动玩.py --dry-run               # 只打印不发键
    python 6_自动玩.py --calibrate             # 标定阈值（不按键）
    python 6_自动玩.py --list-runs             # 看有哪些模型可选

★ 换模型：直接改下面 MODEL_DIR 那一行，填训练结果目录的绝对路径。
"""


from __future__ import annotations

# ==============================================================================
#  ★★★  换模型就改这一行  ★★★
# ------------------------------------------------------------------------------
#  填「训练结果目录」的**绝对路径** —— 就是里面有 best.pt 和 model_meta.json
#  的那个目录（通常长这样： D:\...\gamer_ai\runs\某个名字 ）
#
#  填 None  =  让它自动挑 runs/ 里「最近产出的」那个（省事，但不确定是哪个）
#
#  你现有的模型（想随时看完整列表： python 6_自动玩.py --list-runs）
#     runs\20260918_merged_20260918_1822        5104 个样本 / 56 轮   <- 数据量最大
#     runs\20260918_103227_20260918_1247        2291 个样本 / 60 轮
#     runs\exp05                                2842 个样本 / 36 轮
#     runs\exp04                                4235 个样本 / 10 轮
#     runs\exp03 / exp01 / exp02_speedbal        更早的，样本少
#
#  注意：这些目录里的 test_acc 是在**各自的划分**上算的，跨模型不能直接比大小，
#        想知道哪个好用只能上机实际跑一跑。
#
#  改完保存直接运行即可；命令行加 --run 仍可临时覆盖，不影响这里。
#
#  ★ 2026-10-15：结构整理后，模型在**区域档案**里：profiles\cs01\runs\横屏_v1。
#     启动时若当前区域档案是"cs01"，不给 --run 也会自动挑档案里最新的模型，
#     所以这一行只是"没有档案时"的兜底。
#
#  ★ 默认用 `横屏_v1` —— 它是横屏数据重训的（6 类 A/D/N/Q/S/W，
#     权重上限 5 + 自动剔除稀有类），真实样本 acc 0.7337 / macro-F1 0.6417。
#     之前默认的 `20261011_204728_20261011_2134` 是训崩的那版（真实 acc 0.1529），别再用了。
#     启动时会打印 [取景] 说明这次用的是模型自带区域还是项目 geometry.json。
# ==============================================================================
#  ★ 2026-11-04：改成 None = **自动挑当前档案 runs/ 里最新产出的模型**。
#    这样你重训完直接 Run，跑的就是刚训出来的那版（启动会打印 [模型] 挑了哪个），
#    不会再因为路径没更新而误跑上一次的旧模型。
#    想固定某个模型：换成那个 run 目录的绝对路径即可（写了路径就它说了算），例如
#      MODEL_DIR = r"D:\deeplearning\make_dataset\gamer_ai\profiles\旧版横屏_1903x1070\runs\旧版横屏_1903x1070_20261104_1908"
MODEL_DIR = None


# ==============================================================================
#  ★★★  自动玩的行为参数：改下面几行（改完直接 Run）  ★★★
# ------------------------------------------------------------------------------
#  置信度阈值：模型预测的置信度低于它，就**不按键**（宁可不动，也别乱动）
#      None                      = 用模型目录里的 thresholds.json（没有就用默认 0.85）
#      0.90                      = 所有动作统一用 0.90
#      {"default": 0.85,          = ★ 每个动作单独设（推荐）：
#       "W": 0.95,                  W(跳) 要更自信才跳；A/D(左右) 稍保守；S(滚) 放宽
#       "A": 0.90, "D": 0.90,
#       "S": 0.80}
#   没写的类别用 "default"。命令行 --conf（全局）/ --conf-class W=0.95,A=0.9（按动作）优先级最高。
CONF_THRESHOLD = None


# ------------------------------------------------------------------------------
#  发键通道（2026-10-16 新增）：模型输出的 A/D/W/S 用哪种方式送进模拟器
#      "key" = pynput 发键盘（老方式：**依赖雷电的「按键映射」**，且窗口必须在前台）
#      "adb" = ADB 直接往 Android 注入触摸滑动（**不抢焦点、不受按键映射影响**）
#
#  为什么加这个：日志里"已发键"在涨、游戏里却一动不动。已排查掉权限（Python 和雷电
#  都是管理员，不是 UIPI 拦截）和焦点（启动打印"模拟器已在前台"），剩下的可能是
#  雷电「按键映射」没开，或雷电**忽略了注入的按键**（只认真实硬件按键）。
#  ADB 通道把整个 PC 键盘层绕过去：地铁跑酷本来就是滑动手游，
#  w/a/s/d = 上/下/左/右滑动，模型和数据集一行都不用改。
#
#  先用 tools\测试发键.py 确认哪种方式能让角色动，再把这里改成对应值。
# ------------------------------------------------------------------------------
#  ★ 2026-10-16 实测：你这台机器上雷电**只认真实硬件按键**（自己按 WASD 会动，
#    程序注入的不动），所以 "key" 通道发不进去。默认改成 "adb"，需要时用 --send key 切回。
SEND_MODE = "adb"


def resolve_thresholds(base_thr: dict, args) -> dict:
    """算出这次真正生效的阈值表 {类别/"default": 阈值}。

    优先级（后者覆盖前者）：
        thresholds.json 里的值  <  文件开头的 CONF_THRESHOLD  <  命令行 --conf / --conf-class
    ★ 2026-10-16 用户问"每个动作的阈值可以自己设置吗" —— 可以，这里就是入口。
    """
    thr = dict(base_thr or {})
    conf = CONF_THRESHOLD
    if conf is not None:
        if isinstance(conf, dict):
            for k, v in conf.items():
                thr["default" if str(k).lower() == "default" else str(k).upper()] = float(v)
        else:
            thr["default"] = float(conf)
    if getattr(args, "conf", None) is not None:
        thr["default"] = float(args.conf)
    spec = getattr(args, "conf_class", None)
    if spec:
        for item in str(spec).replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit("[阈值] --conf-class 要写成 W=0.95,A=0.9 这种")
            k, v = item.split("=", 1)
            thr["default" if k.strip().lower() == "default" else k.strip().upper()] = float(v)
    return thr
#  同方向动作锁定（毫秒）：A 之后按 A 要等这么久；A 之后按 D/W/S 不受影响
LOCKOUT_MS = 400
#  全局最小发键间隔（毫秒）：防止两个不同方向连得太密
MIN_GAP_MS = 80
#  延迟取帧（毫秒）：让模型看"现在-delay"时刻的画面，补偿"动作偏早"。别超过 50
DELAY_MS = 0

# ==============================================================================
#  ★ 热力图（Grad-CAM）开关 —— 默认 **关**
#
#  为什么默认关：热力图不是"顺手画一下"，它是**第二份 FP32 模型 + 一次反向传播**，
#  实测每次约 80ms，而控制模型前向只要 32ms。默认每 4 帧算一次 = 每秒 10 次
#  = 吃掉 GPU 的 80%，和控制循环抢同一块卡 —— 判帧频率会从 ~20 次/秒掉到 ~11 次/秒，
#  表现出来就是"开了热力窗之后识别变慢、动作变迟钝"。
#
#  想看图的时候再开：命令行加 --cam（或把这里改成 True）
#  只想省 GPU、不画图但保留窗口里的状态文字：保持 False，窗口照样开
#  连窗口也不要：保持下面的 SHOW_WINDOW = False（现在就是）
# ==============================================================================
CAM_HEATMAP = False

# ==============================================================================
#  ★ 要不要弹那个"画面 + 状态文字"的窗口 —— 默认 **不弹**
#
#  窗口本身也要钱：它按 20fps 把画面放大两倍、叠中文文字（PIL 渲染，很吃 CPU）、
#  再 cv2.imshow，和抓帧/推理抢同一个 CPU 核。
#
#  不看画面时保持 False（现在就是）：控制台每秒仍会打印
#       [运行] 预测 X 0.93  已发键 N  锁定 N  帧龄 40ms  速度 80px/s
#  想看画面：命令行加 --cam-window（或把这里改成 True）
# ==============================================================================
SHOW_WINDOW = False


import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from core import speed as speed_mod
from core.attention import (AttentionProbe, cam_to_frame, colorize, overlay,
                                 put_text_cn)
from core.capture import WindowCapture
from core.config import Config, key_for_label
from core.geometry import deploy_geometry, rel_from_crop
from core.geometry import describe as geo_describe
from core.dataset import build_eval_transform, flatten_frames, to_video_tensor
from core.model import build_model
from core.recorder import RingBuffer, resolve_path
from core.runs import list_runs, pick_run   # 自动挑选模型用

DEFAULT_THRESHOLDS = {"default": 0.85}


# ======================================================================
def load_run(run_dir: str):
    meta_path = os.path.join(run_dir, "model_meta.json")
    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"找不到 {meta_path}（这个目录应该是训练产出目录，检查路径有没有写错）")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 {ckpt_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    thr = dict(DEFAULT_THRESHOLDS)
    tp = os.path.join(run_dir, "thresholds.json")
    if os.path.isfile(tp):
        with open(tp, encoding="utf-8") as f:
            thr.update(json.load(f))
    return meta, ckpt_path, thr


class Player:
    def __init__(self, args):
        self.args = args
        run_dir = args.run if os.path.isabs(args.run) else resolve_path(args.run)
        self.meta, ckpt, self.thr = load_run(run_dir)
        # ★ 2026-10-16（用户："怎么没有可以修改置信度的地方"）：
        #   优先级 命令行 --conf > 文件开头的 CONF_THRESHOLD > 模型目录的 thresholds.json
        if not args.dry_run and not args.calibrate:
            print("[焦点] 开跑后请**不要用鼠标点别的窗口**（模拟器必须保持前台）；"
                  "程序每次发键前会自检并抢回焦点")
        if getattr(args, "auto_calib", True) and not args.calibrate:
            print("[标定] 平时自动标定：开着（正常玩就行，退出时会用刚才的数据重算阈值；"
                  "数据太少不会覆盖。想关掉加 --no-calibrate）")
        _has = os.path.isfile(os.path.join(run_dir, "thresholds.json"))
        self.thr = resolve_thresholds(self.thr, args)
        _def = float(self.thr.get("default", 0.85))
        _per = {k: v for k, v in self.thr.items() if k != "default"}
        print("[阈值] default=%.2f%s   %s   （来源：%s）"
              % (_def, "  按动作单独设：" if _per else "",
                 "  ".join("%s=%s" % (k, v) for k, v in sorted(_per.items())),
                 "命令行" if (getattr(args, "conf", None) is not None
                              or getattr(args, "conf_class", None))
                 else ("文件开头 CONF_THRESHOLD" if CONF_THRESHOLD is not None
                       else ("模型 thresholds.json" if _has else "默认值（没有 thresholds.json）"))))
        m = self.meta
        self.labels = list(m["labels"])
        # ★ 2026-10-15（用户反馈"不同游戏的模型、按键、数据集会对不上"）：用同一套对齐规则
        #   决定按哪张表发键 —— 模型自带 key_map > 模型那个游戏 > 当前游戏 > config.py。
        #   部署端是"警告但不拦"（你可能就是想拿别的游戏的模型试试），采集端(10_)才硬拦。
        from core import gamecfg as _GC
        _align = _GC.align_with_model(m, getattr(args, "game", None),
                                      getattr(args, "keys", None), allow_mismatch=True)
        self.key_map = dict(_align["key_map"])
        self.game = _align["game"] or m.get("game")
        self._align_notes = list(_align["notes"])
        # ★ 2026-11-04：按键表自检（大写源键永远匹配不上，采集端就是这么丢掉 G 的）
        #   注意方向：这里 self.key_map 是「类别 -> 键字符」，而自检函数要「键字符 -> 类别」，
        #   直接传会全员误报（我第一次就写反了，7 条假警告）。
        for _w in _GC.validate_keymap({str(c): str(lb) for lb, c in self.key_map.items()}):
            print("[按键警告] " + _w)
        # ★ 2026-11-04（用户："为什么不能自己读模型有几个按键，然后自己适配"）：
        #   标定阶段原来硬编码 ch in "wasd" —— 模型有 G/Q 时你按 G/Q **不算人类操作**，
        #   自动标定出来的阈值就偏。现在从模型按键表推：除 N（标记"不动"）外都是动作键。
        self.human_key_chars = {str(c).lower() for lb, c in (self.key_map or {}).items()
                                if str(lb).upper() != "N" and str(c).strip()}
        self.K = int(m["in_frames"])
        self.img_size = int(m.get("img_size", 224))
        # ★ 2026-11-05：模型可能是 3*K 通道（老版），也可能 3*K+1（带速度标量通道）。
        #   一律以 model_meta.json 记的为准，别自己算 —— 否则通道数对不上直接报错。
        #   （注意：必须在下面那行 print 之前定义，否则会 AttributeError）
        self.speed_input = bool(m.get("speed_input", False))
        self.in_channels = int(m.get("in_channels") or (self.K * 3))
        print(f"[run] {os.path.abspath(run_dir)}")
        print(f"      类别 {self.labels}  输入 {self.K} 帧 ({self.in_channels} 通道"
              + ("，含速度标量" if self.speed_input else "") + ")  "
              f"test_acc {m.get('test_acc', float('nan')):.4f}")
        # ★ 2026-10-15：说清楚这次按哪张按键表发键 —— 换游戏/加按键后就靠这几行确认
        for _n in getattr(self, "_align_notes", []):
            print("      " + _n)
        print("      按键表（game=%s）：%s"
              % (self.game or "-",
                 " ".join("%s=%s" % (lb, ch) for lb, ch in sorted(self.key_map.items()))
                 or "（空）"))
        # ★ 2026-10-16（实测：模型 test_acc 0.61，实时置信度只有 0.5~0.75，而默认阈值 0.85
        #   会把 98% 的帧当"低置信"丢掉 —— 表现就是"跑了半天几乎不动"）：
        #   弱模型 + 高阈值 = 什么都不发。这里直接提醒，别等跑完才看出来。
        _acc = float(m.get("test_acc") or 0.0)
        _thr_def = float(self.thr.get("default", 0.85))
        if _acc and _acc < 0.75 and _thr_def >= 0.80:
            print(f"[警告] 这个模型 test_acc 只有 {_acc:.2f}，实时置信度常在 0.5~0.75，"
                  f"默认阈值 {_thr_def:.2f} 会把绝大多数帧丢掉 —— 看着就像「不动」。")
            print(f"       建议 --conf 0.55，或把文件顶部 CONF_THRESHOLD 改成 0.55")

        # 抓帧参数必须与训练一致
        cfg = Config()
        cfg.game_res = tuple(m.get("game_res", cfg.game_res))
        cfg.game_crop = tuple(m.get("game_crop", cfg.game_crop))
        cfg.frame_w = int(m.get("frame_w", cfg.frame_w))
        cfg.backend = m.get("backend", cfg.backend)
        # ★ 2026-11-05（用户："为什么感觉这个模型反应很慢"）：
        #   实测抓帧耗时 printwindow 34.7ms / bitblt 10.3ms。
        #   printwindow 能抓被遮挡的窗口，bitblt 快 3 倍但要求窗口没被别人挡住。
        if getattr(args, "backend", None):
            cfg.backend = args.backend
            print(f"[取景] 抓帧后端被命令行指定为 {cfg.backend}"
                  + ("（快，但模拟器窗口别被挡住）" if cfg.backend == "bitblt" else "（慢，但能抓被遮挡的窗口）"))
        # ★ 2026-10-10（用户转横屏）：允许命令行临时覆盖取景参数。
        #   模型权重和 game_res/game_crop/frame_w 都无关 —— 只影响"喂进去的是哪块画面"，
        #   所以同一个模型可以直接换区域试，不用重训。
        if args.game_res:
            v = _ints(args.game_res)
            if len(v) != 2:
                raise SystemExit("--game-res 要写成 宽,高，例如 1280,720")
            cfg.game_res = (v[0], v[1])
        if args.game_crop:
            v = _ints(args.game_crop)
            if len(v) != 4:
                raise SystemExit("--game-crop 要写成 x,y,宽,高，例如 345,0,590,720")
            cfg.game_crop = (v[0], v[1], v[2], v[3])
        cfg.validate()
        self.content_offset = None
        if args.content_offset:
            v = _ints(args.content_offset)
            if len(v) != 2:
                raise SystemExit("--content-offset 要写成 x,y")
            self.content_offset = (v[0], v[1])

        # ★ 2026-10-14 取景自动切换（用户设计：把选中的区域记成文件，后续只读文件）：
        #   优先级 ① 命令行显式 --game-res/--game-crop（你当场就要试这块，最高）
        #          ② 模型自带 model_meta.json 里的 geometry（训练时用的区域）
        #          ③ 项目级 gamer_ai/geometry.json（自检弹窗选的那次）
        #          ④ config.py 的 crop_rel / 横竖屏预设（capture 里兜底）
        #   存的是**比例 + 标定分辨率**，所以模型换分辨率/换模拟器也能自己对准。
        #   这样"训练不同区域的模型"之间可以直接切换，不用手改任何参数。
        if args.game_res or args.game_crop or getattr(args, "no_auto_geometry", False):
            # ★ 2026-10-16（用户："获取识别区域好像自动改"）：
            #   以前就算显式给了 --game-crop 的**绝对像素**，也会被转成比例再按探测到的
            #   画面尺寸换算回去 —— 结果和你给的像素不一样，看着就像程序自己改了区域。
            #   现在：显式给像素框 / 加 --no-auto-geometry，就**关掉自动探测**，
            #   像素框原样使用（窗口被挪动/缩放导致对不上时，会在下面打印警告）。
            cfg.auto_geometry = False
            self.geo = None
            print("[取景] 固定像素框（已关闭自动探测）：game_res=%s  game_crop=%s"
                  % (tuple(cfg.game_res), tuple(cfg.game_crop)))
            print("       窗口尺寸变了它也不会自己缩放 —— 想让程序自动对准就去掉 "
                  "--game-res/--game-crop/--no-auto-geometry")
        else:
            # 模型自带 -> 区域档案 -> 项目 geometry.json（见 geometry.deploy_geometry）
            _pg = None
            _pn = getattr(args, "profile_name", None)
            if _pn:
                from core import profiles as _P2
                from core.geometry import load_geometry as _lg
                _pg = _lg(_P2.geometry_path(_pn))
            self.geo = deploy_geometry(m, extra=_pg,
                                       extra_src=("区域档案 %s" % _pn) if _pn else None)
        if self.geo:
            print(f"[取景] {self.geo['_source']}：{geo_describe(self.geo)}")
        elif getattr(cfg, "auto_geometry", True):
            print("[取景] 没有取景文件，按 config.py / 横竖屏预设对准")
        # 固定像素框模式下上面已经打印过了，这里不再重复说"没有取景文件"
        # 落盘帧形状和训练时不一致 -> 前处理「短边缩到 224 + 中心裁 224」会让模型
        # 看到完全不同的有效视野，这时识别大概率失效，必须先说清楚。
        _th = int(m.get("frame_h", 0))
        if _th and cfg.frame_h != _th:
            print(f"[警告] 覆盖后落盘帧 {cfg.frame_w}x{cfg.frame_h}，模型训练时是 "
                  f"{m.get('frame_w')}x{_th} —— 有效视野会变，识别很可能失效。")
            print("       想沿用现有模型，裁剪框比例要保持在 720:880（横屏取 590x720 这类竖窗）。")
        self.cfg = cfg

        self.step_s = ((float(m.get("save_pre_ms", 300)) + float(m.get("save_post_ms", 100)))
                       / max(int(m.get("saved_frames", 13)) - 1, 1)) / 1000.0
        self.stride = max(1, int(m.get("input_stride", 1)))
        print(f"      帧间隔 {self.step_s*1000:.1f}ms x{self.stride}  "
              f"输入覆盖 {(self.K-1)*self.stride*self.step_s*1000:.0f}ms")

        # ★ 2026-10-14 自动对准：换分辨率/横竖屏都不用改参数；
        #   命令行 --content-offset 若给了，仍然优先（from_config 里用 setdefault）。
        self.cap = WindowCapture.from_config(cfg, backend=cfg.backend,
                                             content_offset=self.content_offset,
                                             geo=self.geo)
        self.ring = RingBuffer(3.0)
        self.transform = build_eval_transform(self.img_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.K, len(self.labels), pretrained=False,
                                 in_channels=self.in_channels)
        ck = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(ck.get("model_state_dict", ck))
        self.model = self.model.to(self.device).eval()
        if args.half and self.device.type == "cuda":
            self.model = self.model.half()
        print(f"[model] 已加载到 {self.device}"
              + (" (FP16)" if args.half and self.device.type == "cuda" else ""))

        # ---------------- 注意力热力图 + 录像（可选） ----------------
        # 两个坑：
        #   ① 热力图要跑**反向传播**，而主循环为了速度用了 FP16 —— half 反传会出 NaN，
        #      所以必须另加载一份 FP32 副本（4M 参数才 16MB，无所谓）。
        #   ② 每帧都算热力图（~30ms）会把 40fps 的控制循环拖垮，模型就反应变慢。
        #      所以放到**独立线程**里每 N 帧算一次，控制循环完全不受影响。
        self.cam_on = bool(getattr(args, "cam", True))
        self.cam_every = max(1, int(getattr(args, "cam_every", 4)))
        self.cam_window = bool(getattr(args, "cam_window", True))
        # ★ 只在"模型判断发生变化"时重算热力图（外加每 N 秒兜底刷一次）：
        #   大部分时间模型都在输出 N，没必要每秒算 8 次。
        self.cam_on_change = bool(getattr(args, "cam_on_change", False))
        self.cam_force_s = float(getattr(args, "cam_force_s", 2.0))
        self._cam_last_lab = None
        self._cam_last_t = 0.0
        self._cam_label_now = None
        self.cam_probe = None
        self.cam_model = None
        self._cam_lock = threading.Lock()
        self._cam_in = None          # 控制循环最新一次的模型输入
        self._cam_map = None         # 已映射回原始画面坐标的热力图 0~1
        self._cam_info = None        # (预测类别, 置信度, 逐帧重要度)
        self._last_full = None       # 抓帧线程最新一帧完整裁剪画面 (RGB)
        self._cam_done = 0
        self._cam_err = None         # 热力图线程的失败原因（显示在窗口上）
        self._display_done = threading.Event()
        self.video_path = None
        self._video_frames = 0
        if self.cam_on:
            self.cam_model = build_model(self.K, len(self.labels), pretrained=False,
                                         in_channels=self.in_channels)
            self.cam_model.load_state_dict(ck.get("model_state_dict", ck))
            self.cam_model = self.cam_model.to(self.device).eval().float()
            self.cam_probe = AttentionProbe(self.cam_model, frames=self.K)
            # ★ 2026-11-05（用户："开热力窗后识别速度变慢"）：别再说"控制循环不受影响"，
            #   那是错的。热力图是**另一份 FP32 模型 + 反向传播**，实测每次约 80ms
            #   （控制模型前向只要 32ms），而且和控制循环抢同一块 GPU。
            _hz = 40.0 / self.cam_every
            _duty = _hz * 0.080
            print(f"[cam] 热力图开启：另一份 FP32 模型 + 反向传播，每次约 80ms，"
                  f"每 {self.cam_every} 帧一次（约 {_hz:.0f}Hz）"
                  f" -> 约占 GPU {_duty*100:.0f}%，判帧频率会明显下降")
            print(f"       想快一点：--cam-every 20（约 2Hz）或 --no-cam（只看预测不画热力图）"
                  if self.cam_every < 20 else "       当前频率已经很低，影响很小")
        self.delay_s = max(0.0, getattr(args, "delay_ms", 0)) / 1000.0
        if self.delay_s > 0:
            print(f"[delay] 延迟取帧 {self.delay_s*1000:.0f}ms —— 动作会比模型自己的判断晚这么多")
            if self.delay_s > 0.05:
                print("[delay] ⚠ 警告：实测 delay>50ms 会让快连按（WA/WS/AA）识别率从 95.9% 掉到 65%")
                print("        因为连按的第二个键正好在上一个键的换道动画里，画面往前挪就找不到依据了")
                print("        建议先试 <=50ms；要根治\"动作偏早\"应补采数据（save_pre_ms 300->800）")

        self.running = True
        self.playing = True
        self.lockout_until = {}          # {方向: 解禁时刻}，只锁同一个方向
        self.min_gap_s = max(0.0, getattr(args, "min_gap_ms", 80)) / 1000.0
        self.last_label = None
        self.last_send = 0.0
        self.sent = 0
        self.ticks = 0
        self.lockout_hits = 0    # 被"同方向锁定 320ms"挡下的次数
        self.gap_hits = 0        # 被"全局最小间隔 80ms"挡下的次数（和锁定分开算，免得看不懂）
        self.busy_hits = 0       # ADB 设备还在执行上一条手势，这次没发（防止命令排队越滚越慢）
        self.low_conf = 0
        self.n_pred = 0          # 模型预测"不动"(N) 的次数：conf 再高也不会按键，这是设计
        self.last_speed = 0.0
        self.t_calib = []          # 标定用
        self.human_presses = []
        self.human_key_events = []      # [(时刻, 按的哪个键)]，标定时用来量"提前量"
        self.self_sends = []            # 我们自己发键的时刻：用来把"程序发的键"从标定里剔掉

        self.keyboard = None
        self.sender = None
        self.send_mode = (getattr(args, "send", None) or SEND_MODE or "key").lower()
        # dry-run 下也建 ADB sender：构造只做探测（adb devices / wm size），不发任何手势，
        # 于是 python 6_自动玩.py --dry-run --send adb 就能"只验通道、不碰游戏"。
        if self.send_mode == "adb" and not args.calibrate:
            # ★ 2026-10-16：ADB 通道 —— 不抢焦点、不看按键映射，直接把触摸事件注入
            #   Android（w/a/s/d = 上/下/左/右滑动）。为什么加见文件顶部 SEND_MODE。
            from core.adbsend import AdbSender
            self.sender = AdbSender()
            print(f"[发键] 通道：{self.sender.describe()}   —— 不依赖窗口焦点和按键映射")
            if not self.sender.available():
                print("[发键] ⚠ ADB 不可用！先跑 python tools\\测试发键.py --check 看诊断")
        if not args.dry_run and not args.calibrate and self.send_mode != "adb":
            from pynput.keyboard import Controller
            self.keyboard = Controller()
            if not self._focus_window():
                print("\n[焦点] ⚠ 没能把模拟器窗口切到前台 —— 按键会发给当前窗口（多半是 PyCharm），"
                      "\n       游戏里就是一动不动。请**用鼠标点一下模拟器窗口**再开跑，"
                      "或者先最小化/移开挡着的窗口。")
            else:
                print("[焦点] 模拟器已在前台，可以发键")
        if not args.calibrate:
            self._print_send_map()

    # ------------------------------------------------------------------
    def _print_send_map(self) -> None:
        """把「模型每个类别到底怎么发出去」打出来 —— 一眼看出哪类根本动不了。

        ★ 2026-11-05（用户："按键为什么不根据模型提供的按，GQ 两个判断出来后没有去触发"）：
          模型判出 G/Q、置信度也够（实测 P(G) 中位 1.000），但 ADB 通道里只有 w/s/a/d
          有手势，其它键落到了"点屏幕中间"的兜底 —— 复活按钮不在正中，Q 也不是屏幕
          按钮，于是"发了键但游戏没反应"。现在启动就把这张对照表打出来。
        """
        try:
            from core.adbsend import TAP_REL
        except Exception:
            TAP_REL = {}
        rows = []
        for lb in self.labels:
            if str(lb).upper() == "N":
                rows.append("%s=(不发键)" % lb)
                continue
            ch = (self.key_map or {}).get(str(lb).upper())
            if ch is None:
                rows.append("%s=⚠无按键" % lb)
                continue
            c = str(ch).lower()
            if getattr(self, "sender", None) is not None:
                if c in ("w", "s", "a", "d"):
                    rows.append("%s=%s(滑动)" % (lb, ch))
                elif TAP_REL.get(c):
                    rows.append("%s=%s(点击 %.2f,%.2f)" % (lb, ch, TAP_REL[c][0], TAP_REL[c][1]))
                else:
                    rows.append("%s=%s⚠未配手势(不会触发)" % (lb, ch))
            else:
                rows.append("%s=%s(键盘)" % (lb, ch))
        print("[发键] 类别->按键：" + "  ".join(rows))

    # ------------------------------------------------------------------
    def _focus_window(self) -> bool:
        """把模拟器窗口抢到前台。返回 True 表示抢完之后模拟器确实在前台。

        ★ 2026-10-16：以前失败被 except 吞掉（静默 pass），于是"抢焦点失败"和
          "按键被模拟器忽略"看起来一模一样 —— 都是游戏里一动不动。现在返回结果并打印，
          启动时就能看出到底抢没抢到。
        """
        try:
            import ctypes
            u32 = ctypes.windll.user32
            if u32.GetForegroundWindow() == self.cap.hwnd:
                return True
            u32.ShowWindow(self.cap.hwnd, 9)               # SW_RESTORE
            # Windows 有"前台锁定"：后台进程直接 SetForegroundWindow 常被拒绝。
            # 先敲一下 Alt（keybd_event）能骗过它 —— 这是标准绕法。
            u32.keybd_event(0x12, 0, 0, 0)
            u32.SetForegroundWindow(self.cap.hwnd)
            u32.keybd_event(0x12, 0, 2, 0)
            time.sleep(0.15)
            return u32.GetForegroundWindow() == self.cap.hwnd
        except Exception as e:
            print(f"[焦点] 抢焦点出错：{e!r}")
            return False

    def _ensure_focus(self) -> None:
        """发键前确认模拟器在前台；丢焦点了就抢回来（最多每 0.5 秒检查一次）。

        ★ 2026-10-16（用户："模型没去控制角色，一次都没有"）：
          以前只在**启动时**抢一次焦点。你中途用鼠标点一下 PyCharm / 浏览器看日志，
          模拟器就丢焦点了，此后所有按键都发给别的窗口 —— 游戏里一动不动，
          但状态栏的"已发键"照样在涨（因为按键确实发出去了），所以极难发现。
        """
        now = time.time()
        if now - getattr(self, "_focus_t", 0.0) < 0.5:
            return
        self._focus_t = now
        try:
            import ctypes
            u32 = ctypes.windll.user32
            if u32.GetForegroundWindow() != self.cap.hwnd:
                if not getattr(self, "_focus_warned", False):
                    self._focus_warned = True
                    print("\n[焦点] 模拟器不在前台 —— 已自动切回。"
                          "（跑起来之后**别用鼠标点别的窗口**；真要看日志，看控制台别看游戏窗口）")
                self._focus_window()
        except Exception:
            pass

    def _send_key(self, label: str) -> None:
        # ★ 2026-10-15（用户要求"项目要能用于别的游戏"）：**优先用模型自带的按键表**
        #   （model_meta.json 里的 key_map，训练时按 games/<游戏>.py 写进去的）。
        #   于是不同游戏训出来的模型各按各的键；老模型没有这个字段才回退到 config.py。
        ch = (getattr(self, "key_map", None) or {}).get(str(label).upper()) \
            or key_for_label(label)
        if ch is None:
            return
        # ★ 2026-10-16：ADB 通道优先（如果开着）。它不需要焦点，也不需要按键映射。
        if getattr(self, "sender", None) is not None:
            if not self.sender.send(ch):
                print(f"\n[ERR] ADB 发键失败（{ch}）—— 已失败 {self.sender.n_fail} 次，"
                      f"检查 python tools\\测试发键.py --check")
            return
        if self.keyboard is None:
            return
        from pynput.keyboard import KeyCode
        self._ensure_focus()
        try:
            self.self_sends.append(time.time())      # 记下来，免得被当成"人按键"计入标定
            if len(self.self_sends) > 64:
                del self.self_sends[:-32]
            self.keyboard.press(KeyCode.from_char(ch))
            time.sleep(0.03)
            self.keyboard.release(KeyCode.from_char(ch))
        except Exception as e:
            print(f"\n[ERR] 发键失败: {e}")

    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        interval = 1.0 / 40.0
        nxt = time.time()
        while self.running:
            try:
                t = time.time()
                f = self.cap.grab()
                if self.cam_on or self.video_path is not None or self.cam_window:
                    self._last_full = f          # 给热力图/录像线程用（引用赋值，原子）
                import cv2
                self.ring.append(t, cv2.resize(f, (self.cfg.frame_w, self.cfg.frame_h),
                                               interpolation=cv2.INTER_AREA))
            except Exception as e:
                print(f"\n[ERR] 抓帧失败: {e}")
                time.sleep(0.3)
            nxt += interval
            d = nxt - time.time()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.time()

    def _input_tensor(self):
        now = time.time() - self.delay_s
        frames = []
        for i in range(self.K - 1, -1, -1):
            tg = now - i * self.stride * self.step_s
            f = self.ring.frame_at_or_before(tg)
            if f is None:
                return None
            frames.append(f)
        x = to_video_tensor(np.stack(frames))
        x = flatten_frames(self.transform(x))
        if getattr(self, "speed_input", False):
            # ★ 与训练端（core/dataset.py）必须**一字不差**，否则量纲不一致等于喂噪声
            import math
            sp = float(self._live_speed())
            z = min(math.log1p(max(0.0, sp)) / 4.6, 2.0)
            x = torch.cat([x, torch.full((1, x.shape[-2], x.shape[-1]),
                                         z, dtype=x.dtype)], dim=0)
        return x.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    def _human_key(self, key) -> None:
        if not self.running:
            return
        name = getattr(key, "name", None)
        ch = getattr(key, "char", None)
        if name == "f10" or name == "esc":
            self.running = False
        elif name == "f8":
            self.playing = not self.playing
            print(f"\n[切换] {'继续' if self.playing else '暂停'}")
        elif ch and ch.lower() in getattr(self, "human_key_chars", set()):
            # ★ 2026-10-16：程序自己用 Controller 发的键，Windows 的键盘钩子**同样能看到**，
            #   会被当成"人按键"混进标定数据（阈值就围着模型自己的行为转，标不出东西）。
            #   所以刚由我们自己发过键的 150ms 内不记。
            t = time.time()
            if self.self_sends and t - self.self_sends[-1] <= 0.15:
                return
            self.human_presses.append(t)
            self.human_key_events.append((t, ch.lower()))

    def control_loop(self) -> None:
        args = self.args
        thr_default = float(self.thr.get("default", 0.85))
        status_t = time.time()
        calib_t = time.time()
        while self.running:
            x = self._input_tensor()
            if x is None:
                time.sleep(0.005)
                continue
            if self.cam_on:
                with self._cam_lock:
                    self._cam_in = x.detach()     # 只存引用，热力图线程自己算
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=(args.half and self.device.type == "cuda")):
                    out = self.model(x)
                prob = F.softmax(out.float(), dim=1)[0]
            conf, idx = float(prob.max()), int(prob.argmax())
            label = self.labels[idx]
            if self.cam_on:
                self._cam_label_now = label        # 供 --cam-on-change 判断要不要重算
            self.ticks += 1
            now = time.time()

            # ★ 2026-10-16（用户："让程序自己标定作为默认"）：正常玩的时候也记录
            #   （每帧一条 (时刻, 置信度, 预测类别)，40fps 下一分钟才 2400 条，开销可忽略），
            #   退出时用"人是否随后 200ms 内按键"来算最佳阈值并写 thresholds.json。
            # ★ 2026-10-16（用户："模型没有去控制角色移动，一次都没有" —— **真正的原因**）：
            #   上一版把"平时自动标定"默认打开，却在记完一条数据后直接 continue ——
            #   于是**每次正式运行都在纯观察，一个键都不发**。
            #   特征就是四个计数全是 0：发键 0 / 锁定拦截 0 / 低置信 0 / 预测不动(N) 0。
            #   自动标定只是"顺便记数据"，绝不能跳过下面的控制逻辑。
            if args.calibrate or getattr(args, "auto_calib", True):
                self.t_calib.append((now, conf, label))
                if now - calib_t > 2:
                    speed = self._live_speed()
                    print(f"\r[标定] 采样 {self.ticks}  当前 {label} {conf:.2f}  "
                          f"人按键 {len(self.human_presses)} 次  速度 {speed:.0f}px/s  ", end="")
                    calib_t = now
            if args.calibrate:          # 只有 --calibrate（纯标定、不发键）才在这里收工
                time.sleep(0.01)
                continue

            if not self.playing:
                time.sleep(0.005)
                continue
            # ★ 2026-10-16：把「阈值 / N」的判断挪到两道闸门**前面**。
            #   原来"最小间隔(80ms)"和"同方向锁定"检查在最前，40fps 下大量帧还没轮到判断
            #   就被拦掉 —— 低置信 / N 的计数全被吃掉，日志长得像
            #   "低置信 67 次、预测不动 281 次、锁定拦截 183 次"这种看不懂的组合。
            #   顺序换成 阈值 -> N -> 最小间隔 -> 同方向锁定：**发键行为完全不变**，计数才有意义。
            thr = float(self.thr.get(label, thr_default))
            if conf < thr:
                self.low_conf += 1
            elif label == "N":
                # ★ 2026-10-16（用户："为什么置信度超过阈值却没动"）：最常见的原因就是它 ——
                #   模型预测「N = 什么都不做」，而且常常对 N 也很自信（0.9+）。
                #   阈值只决定"敢不敢动"；N 这个类别本身就是"别动"，所以不是故障。
                self.n_pred += 1
            elif now - self.last_send < self.min_gap_s:
                # 全局最小间隔：防止同一瞬间连发（人手最快也就 150ms 左右）
                self.gap_hits += 1
            elif now < self.lockout_until.get(label, 0.0):
                # 同方向锁定：覆盖换道动画，防止同一个键在动画里重复触发；
                # 换个方向说明是新的决策，立刻放行（W->S 这种连按就靠这条）
                self.lockout_hits += 1
            elif (getattr(self, "sender", None) is not None and self.sender.busy()):
                # ★ 2026-11-05（用户："为什么感觉这个模型反应很慢"）：
                #   adb 的 input swipe 每次约 190~280ms（要启动一个 Java 进程），
                #   设备是串行执行的。上一条还没做完就又塞一条 -> 在 shell 里排队，
                #   延迟越积越大、越跑越慢。这里直接跳过，下一帧模型会再判一次。
                self.busy_hits += 1
            else:
                if not args.dry_run:
                    self._send_key(label)
                self.sent += 1          # dry-run 下这是"本来会发"的次数
                self.last_label = label
                self.last_send = now
                self.lockout_until[label] = now + args.lockout_ms / 1000.0   # 只锁这个方向
                if args.verbose:
                    print(f"\n[发键] {label}  conf={conf:.3f}  {label} 方向锁定 {args.lockout_ms}ms")

            if time.time() - status_t > 1.0:
                speed = self._live_speed()
                what = "拟发键" if args.dry_run else "已发键"
                _nx = ("  预测不动(N) %d" % self.n_pred) if self.n_pred else ""
                _bz = ("  忙 %d" % self.busy_hits) if self.busy_hits else ""
                print(f"\r[运行] 预测 {label} {conf:.2f}  {what} {self.sent}  "
                      f"锁定 {self.lockout_hits} 间隔 {self.gap_hits} 低置信 {self.low_conf}{_nx}{_bz}  "
                      f"帧龄 {self._frame_age_ms():.0f}ms  速度 {speed:.0f}px/s  ", end="")
                status_t = time.time()
            time.sleep(0.005)

    def _frame_age_ms(self) -> float:
        """最新一帧的年龄（毫秒）。这是纯延迟：画面已经过去这么久才被模型看到。"""
        rec = self.ring.latest(1)
        if not rec:
            return 0.0
        return max(0.0, (time.time() - rec[-1][0]) * 1000.0)

    def _live_speed(self) -> float:
        rec = self.ring.latest(9)
        if len(rec) < 3:
            return 0.0
        try:
            sp, _ = speed_mod.speed_px_per_sec([f for _, f in rec],
                                               [t * 1000 for t, _ in rec], stride=1)
            self.last_speed = sp
        except Exception:
            pass
        return self.last_speed

    # ------------------------------------------------------------------
    def report_calibration(self) -> None:
        if not self.t_calib or not self.human_presses:
            print("\n[标定] 数据不足（需要你在标定期间正常玩、该按键时按键）")
            return
        import bisect
        hp = sorted(self.human_presses)
        rows = []
        for t, conf, label in self.t_calib:
            k = bisect.bisect_left(hp, t)
            acted = 1 if (k < len(hp) and hp[k] - t <= 0.20) else 0
            rows.append((conf, acted, label))
        conf = np.array([r[0] for r in rows])
        act = np.array([r[1] for r in rows])
        print(f"\n[标定] 采样 {len(rows)} 帧，其中人随后 200ms 内按键的 {act.sum()} 帧")
        best_t, best_j = 0.5, -1
        for t in np.arange(0.30, 0.99, 0.01):
            pred = conf >= t
            tp = ((pred == 1) & (act == 1)).sum()
            fp = ((pred == 1) & (act == 0)).sum()
            fn = ((pred == 0) & (act == 1)).sum()
            tn = ((pred == 0) & (act == 0)).sum()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j = sens + spec - 1
            if j > best_j:
                best_j, best_t = j, float(t)
        print(f"       建议阈值 {best_t:.2f}（Youden J={best_j:.3f}）")
        print(f"       参考：conf 分位 50%={np.percentile(conf,50):.3f} "
              f"90%={np.percentile(conf,90):.3f} 99%={np.percentile(conf,99):.3f}")
        # ★ 2026-10-16：平时自动标定可能只玩了几十秒 —— 数据太少就别覆盖原来那份好的。
        if len(rows) < 200 or act.sum() < 20:
            print("       [跳过] 数据太少（帧 <200 或 人按键 <20 次）—— 不覆盖已有的 thresholds.json")
            print("              想认真标定就跑一轮： python 6_自动玩.py --calibrate")
            return
        # ★ 2026-10-16（用户："模型没有去控制角色移动，一次都没有"）：
        #   标定搜索上限是 0.98、每类还会 +0.05（最多到 0.99）。模型 conf 常年 0.95~0.98，
        #   一旦标定出 0.99，**几乎每一帧都会被"低置信丢弃"** —— 游戏里就是一动不动。
        #   所以写盘前先算一下：按这个阈值到底有多少帧能发出去。太少就拒绝覆盖。
        rate = float((conf >= round(best_t, 2)).mean())
        print(f"       按建议阈值 {best_t:.2f}，标定数据里 {rate*100:.1f}% 的帧会发键")
        if rate < 0.05:
            print("       [拒绝] 这个阈值会把 "
                  f"{(1-rate)*100:.0f}% 的帧全部丢掉 —— 游戏里会一动不动。")
            print("              多半是标定时你不在操作（或没按键就录进来的帧太多）。")
            print("              **不覆盖** thresholds.json；建议先跑一轮正常玩，或手动设")
            print("              6_自动玩.py 开头的 CONF_THRESHOLD = 0.55")
            return
        out = {"default": round(best_t, 2)}
        for lab in self.labels:
            sub = [r[0] for r in rows if r[2] == lab]
            if sub:
                out[lab] = round(min(0.99, max(0.5, best_t + 0.05)), 2)
        p = os.path.join(self.args.run, "thresholds.json")
        old = None
        if os.path.isfile(p):                     # 写前备份，旧值一并打出来对比
            try:
                old = json.load(open(p, encoding="utf-8"))
            except Exception:
                old = None
            import shutil, time as _t
            bak = p + ".bak_" + _t.strftime("%Y%m%d_%H%M%S")
            try:
                shutil.copy2(p, bak)
                print(f"       （旧文件已备份：{os.path.basename(bak)}）")
            except Exception:
                pass
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        if old is not None and old != out:
            print("       旧: " + json.dumps(old, ensure_ascii=False))
        print(f"       已写出 {p}: {out}")
        print("       下次直接 --run 就会自动读它（想关掉自动标定：--no-calibrate）")
        self._report_lead(best_t)

    def _report_lead(self, thr: float) -> None:
        """用真实按键当基准，量模型比人早多久开始动（正=更早，负=更晚）。

        这是判断"动作过早"最直接的证据 —— 离线数据只存到按键前 296ms，
        量不到更早；真实游玩时模型是逐帧判断的，没有这个限制。
        """
        if len(self.human_key_events) < 5:
            print("\n[提前量] 人工按键太少（<5 次），无法统计")
            return
        leads, missed, wrong, sat = [], 0, 0, 0
        for tp, ch in self.human_key_events:
            want = ch.upper()
            act = None
            for t, conf, label in self.t_calib:
                if t < tp - 1.5 or t > tp + 0.2:
                    continue
                if label != "N" and conf >= thr:
                    act = (t, label)
                    break
            if act is None:
                missed += 1
                continue
            if act[0] <= tp - 1.49:
                sat += 1                      # 扫描窗一开头就已经在动了 -> 提前量更大
            leads.append((act[0] - tp) * 1000.0)
            if act[1] != want:
                wrong += 1
        if not leads:
            print("\n[提前量] 模型在自己按键附近从没输出过方向键 —— 阈值可能太高")
            return
        arr = np.array(leads)
        med = float(np.median(arr))
        print("\n[提前量] 以你的真实按键为基准（负 = 模型比你早，正 = 比你晚）：")
        print(f"         中位 {med:+.0f}ms   25% {np.percentile(arr,25):+.0f}ms   "
              f"75% {np.percentile(arr,75):+.0f}ms")
        early = int((arr < -120).sum()); same = int(((arr >= -120) & (arr <= 120)).sum())
        late = int((arr > 120).sum())
        print(f"         早于 120ms: {early} 次   基本同时(±120ms): {same} 次   "
              f"晚于 120ms: {late} 次")
        if sat:
            print(f"         其中 {sat} 次在扫描窗起点(1.5 秒前)就已经在动了 —— 真实提前量比上面更大")
        print(f"         方向按错 {wrong} 次   完全没动 {missed} 次")
        if med < -80:
            print(f"         建议：加 --delay-ms {int(round(-med))} 把动作整体压后")
            print(f"              python 6_自动玩.py --dry-run --delay-ms {int(round(-med))}")
        elif med > 80:
            print('         模型反而偏晚，需要补采「更早决策」的数据')
        else:
            print("         提前/滞后在可接受范围内")

    # ------------------------------------------------------------------
    #  注意力热力图 + 录像（都在独立线程，不拖慢控制循环）
    # ------------------------------------------------------------------
    def _cam_loop(self) -> None:
        """按节流频率算 Grad-CAM，把结果放到共享状态里给显示线程用。

        频率用"每 N 帧"描述（cam_every），换算成时间就是 N/40 秒一次：
        N=4 约 10Hz，N=1 约 40Hz（但那会明显抢 GPU）。
        """
        interval = max(0.03, self.cam_every / 40.0)
        next_t = time.time()
        while self.running:
            now = time.time()
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue
            next_t = now + interval
            # 只在判断变了（或超过兜底间隔）时才重算 —— 稳态下几乎不占 GPU
            if self.cam_on_change:
                _lab = self._cam_label_now
                if (_lab is not None and _lab == self._cam_last_lab
                        and (now - self._cam_last_t) < self.cam_force_s):
                    time.sleep(0.02)
                    continue
            with self._cam_lock:
                x = self._cam_in
            if x is None:
                continue
            try:
                res = self.cam_probe.run(x, None)
                m = res["cam"] if res.get("cam") is not None else res["saliency"]
                fm = cam_to_frame(m, (self.cfg.frame_h, self.cfg.frame_w), self.img_size)
                with self._cam_lock:
                    self._cam_map = fm
                    self._cam_info = (self.labels[res["pred"]],
                                      float(res["prob"][res["pred"]]),
                                      res["frame_importance"])
                self._cam_done += 1
                self._cam_last_lab = self._cam_label_now
                self._cam_last_t = time.time()
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if getattr(self, "_cam_err", None) != msg:
                    print(f"\n[cam] 热力图计算失败: {msg}")
                    self._cam_err = msg          # 同步显示到窗口上，避免"永远计算中"
                time.sleep(0.5)

    def _display_loop(self) -> None:
        """按固定帧率把"画面 + 热力图 + 状态文字"合成出来，一边显示一边录进视频。

        为什么用固定帧率：热力图是 ~10Hz、抓帧是 40Hz，直接录会时快时慢；
        这里固定 20fps 取"当前最新画面 + 当前最新热力图"，视频时长就等于真实时长。
        """
        import cv2
        w, h = self.cfg.frame_w * 2, self.cfg.frame_h * 2
        fps = float(getattr(self.args, "video_fps", 20.0))
        win = "auto-play + Grad-CAM"

        writer = None
        if not getattr(self.args, "no_video", False):
            p = self.args.video
            if not p:
                p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "play_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))
            p = p if os.path.isabs(p) else resolve_path(p)
            writer = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                p2 = os.path.splitext(p)[0] + ".avi"
                writer = cv2.VideoWriter(p2, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
                p = p2
            if writer.isOpened():
                self.video_path = p
                print(f"[video] 录制中 -> {p}   {w}x{h} @ {fps:.0f}fps")
            else:
                print("[video] 打不开视频文件，跳过录制")
                writer = None

        if self.cam_window:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            try:
                cv2.resizeWindow(win, w, h)
            except Exception:
                pass

        period = 1.0 / max(fps, 1.0)
        nxt = time.time()
        while self.running:
            now = time.time()
            if now < nxt:
                time.sleep(min(0.01, nxt - now))
                continue
            nxt = max(nxt + period, now)
            full = self._last_full
            if full is None:
                continue
            # 注意：必须用 cvtColor，不能用 [:, :, ::-1] ——
            # 反转切片得到的是**非连续**数组，OpenCV 5.0 的 putText/VideoWriter 会拒绝写入
            # （报 'Layout of the output array img is incompatible with cv::Mat'）。
            base = cv2.cvtColor(cv2.resize(full, (w, h), interpolation=cv2.INTER_LINEAR),
                                cv2.COLOR_RGB2BGR)
            with self._cam_lock:
                cm = self._cam_map
                info = self._cam_info
            if cm is not None:
                base = overlay(base,
                               cv2.resize(cm, (w, h), interpolation=cv2.INTER_LINEAR))

            # ---- 叠加状态文字 ----
            # 注意：中文必须用 put_text_cn —— cv2.putText 只认 ASCII，中文会变方块
            if cm is not None:
                _title, _color = "热力图", (0, 255, 255)
            elif getattr(self, "_cam_err", None):
                _title, _color = ("热力图失败：" + self._cam_err)[:60], (80, 80, 255)
            else:
                _title, _color = "热力图计算中...", (170, 170, 170)
            put_text_cn(base, _title, (8, 4), size=20, color=_color)
            if info is not None:
                lab, conf, imp = info
                put_text_cn(base, f"模型判断 {lab}  置信度 {conf:.2f}", (8, 30),
                            size=20, color=(0, 255, 0))
                it = "  ".join(f"第{i+1}帧 {v*100:.0f}%" for i, v in enumerate(imp))
                put_text_cn(base, it, (8, h - 24), size=15, color=(0, 220, 255), stroke=1)
            sent = self.last_label or "无"
            put_text_cn(base, f"最近发键 {sent}   已发键 {self.sent} 次", (8, h - 48),
                        size=15, color=(230, 230, 230), stroke=1)
            if not self.playing:
                put_text_cn(base, "已暂停（F8 继续）", (8, 58), size=20, color=(0, 60, 255))

            if writer is not None:
                writer.write(base)
                self._video_frames += 1
            if self.cam_window:
                cv2.imshow(win, base)
                cv2.waitKey(1)

        if writer is not None:
            writer.release()
        if self.cam_window:
            try:
                cv2.destroyWindow(win)
            except Exception:
                pass
        self._display_done.set()

    def run(self) -> None:
        from pynput import keyboard
        print("=" * 78)
        print("快捷键: F8 暂停/继续   F10/Esc 退出")
        print(f"动作锁定: 同一方向 {self.args.lockout_ms}ms（不同方向立刻放行，保证 WA/WS 连按）"
              f"  全局最小间隔 {self.args.min_gap_ms}ms")
        if self.args.calibrate:
            print("标定模式：不会发键。请正常玩，该按键时按键，跑 1~2 分钟后按 F10。")
        elif self.args.dry_run:
            print("dry-run：不会发键，只打印预测。")
        else:
            print("!! 正式模式：马上会开始自动按键 !!")
            print("   3 秒后接管，想取消就在这 3 秒内按 Ctrl+C")
        if self.cam_on:
            print(f"热力图: 开启（每 {self.cam_every} 帧一次，约 {40.0/self.cam_every:.0f}Hz）"
                  f"   窗口 {'开' if self.cam_window else '关'}"
                  f"   录像 {'关' if getattr(self.args, 'no_video', False) else '开'}")
        else:
            print("热力图: 关闭（--cam 可打开）"
                  + ("    窗口: 关（--cam-window 可打开）" if not self.cam_window
                     else "    窗口: 开"))
        print("=" * 78)
        if not self.args.calibrate and not self.args.dry_run:
            for i in (3, 2, 1):
                print(f"   {i} ...", flush=True)
                time.sleep(1.0)
            print("   开始！(F8 暂停 / F10 退出)")
        threading.Thread(target=self._capture_loop, daemon=True).start()
        if self.cam_on:
            threading.Thread(target=self._cam_loop, daemon=True).start()
        want_display = (self.cam_on or self.cam_window
                        or not getattr(self.args, "no_video", False))
        if want_display:
            threading.Thread(target=self._display_loop, daemon=True).start()
        if float(getattr(self.args, "run_seconds", 0) or 0) > 0:
            def _stop_later():
                time.sleep(float(self.args.run_seconds))
                print(f"\n[自检] 已跑满 {self.args.run_seconds:.0f} 秒，自动退出")
                self.running = False
            threading.Thread(target=_stop_later, daemon=True).start()
        listener = keyboard.Listener(on_press=self._human_key)
        listener.start()
        try:
            self.control_loop()
        except KeyboardInterrupt:
            self.running = False
        self.running = False
        if want_display:
            self._display_done.wait(timeout=5.0)   # 等录制线程把文件写完再退出
        listener.stop()
        self.cap.close()
        if self.args.calibrate or getattr(self.args, "auto_calib", True):
            self.report_calibration()
        what = "拟发键" if self.args.dry_run else "发键"
        print(f"\n结束：{what} {self.sent} 次，同方向锁定挡下 {self.lockout_hits} 次，"
              f"最小间隔挡下 {self.gap_hits} 次，设备忙跳过 {self.busy_hits} 次，"
              f"低置信丢弃 {self.low_conf} 次，模型预测不动(N) {self.n_pred} 次")
        if self.sent == 0 and self.ticks > 60 and not self.args.calibrate:
            print("[提示] 一个键都没发出去。按这个顺序查：")
            if self.n_pred > self.low_conf:
                print("       · 模型一直在预测 N（不动）—— 阈值或模型的问题（见下面的提示）")
            elif self.low_conf > 0:
                print(f"       · 全是「低置信丢弃」{self.low_conf} 次 —— 阈值太高：试试 --conf 0.6")
            elif self.lockout_hits > 0 or self.gap_hits > 0:
                print(f"       · 全被闸门拦下了（同方向锁定 {self.lockout_hits} / 最小间隔 "
                      f"{self.gap_hits}）—— 调大 --lockout-ms / --min-gap-ms")
            else:
                print("       · 决策分支根本没跑到 —— 检查 control_loop 里是不是有 continue 把它跳过了")
        if self.n_pred > max(60, self.sent * 3):
            print("[提示] 模型绝大多数时候都在输出 N（不动）—— 它太保守了：")
            print("       · 想让它敢动：阈值调低（--conf 0.6）或重训时 --neg-ratio 0.5")
            print("       · 若是「该动的时候也不动」，那是模型质量问题，补数据重训才是根治")
        if self.cam_on:
            print(f"    热力图共算了 {self._cam_done} 次")
        if self.video_path:
            mb = os.path.getsize(self.video_path) / 1e6 if os.path.isfile(self.video_path) else 0
            print(f"    视频已保存: {self.video_path}")
            print(f"              {self._video_frames} 帧, {mb:.1f} MB")


def _ints(s: str):
    """把 "1280,720" / "1280x720" / "345 0 590 720" 都解析成整数列表。"""
    return [int(v) for v in str(s).replace("x", ",").replace("X", ",")
            .replace(" ", ",").split(",") if v.strip()]


def resolve_model_dir(p) -> str:
    """把 MODEL_DIR / --run 的值解析成真实目录。

    三种写法都支持：
        1) 绝对路径   D:\\...\\gamer_ai\\runs\\exp05
        2) 相对 gamer_ai/  runs/exp05
        3) 只写目录名  exp05          （会自动去 runs/ 下找）

    填 None（或命令行没给 --run 且 MODEL_DIR=None）就自动挑最近产出的那个。
    解析不到时**列出所有可选的模型**再退出，比直接抛 FileNotFoundError 好排查。
    """
    if not p:
        return pick_run()
    cands = [p] if os.path.isabs(p) else [resolve_path(p),
                                          resolve_path(os.path.join("runs", p))]
    for c in cands:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "model_meta.json")):
            return c
    print(f"\n[错误] 找不到可用的模型目录：{p}")
    print("       可用的模型目录里必须有 best.pt 和 model_meta.json。")
    print("\n       现在可选的模型（把路径抄到文件顶部的 MODEL_DIR 即可）：")
    for q, f1, acc, mt, n_tr, ra, dg in list_runs():
        print(f"         {q}")
    print("\n       看完整列表： python 6_自动玩.py --list-runs")
    raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="多帧模型部署（含动作锁定）")
    ap.add_argument("--run", default=MODEL_DIR,
                    help="训练结果目录。默认用本文件顶部 MODEL_DIR 里写死的那个；"
                         "命令行给这个参数可以临时覆盖（不影响代码里的值）")
    ap.add_argument("--dry-run", action="store_true", help="不发键，只打印预测")
    ap.add_argument("--send", choices=["key", "adb"], default=None,
                    help="发键通道：key=键盘注入（默认），adb=ADB 触摸滑动"
                         "（不依赖窗口焦点和雷电按键映射）")
    ap.add_argument("--profile", default=None,
                    help="区域档案名：模型默认挑 gamer_ai/profiles/<名字>/runs/ 里最新的那个，"
                         "取景也用该档案的 geometry.json。不给就用当前档案")
    ap.add_argument("--game", default=None,
                    help="游戏名（games/<名字>.py）：用它的按键表。不给就用模型训练时那个游戏，"
                         "再退回当前游戏（模型说了算，免得按错键）")
    ap.add_argument("--keys", default=None,
                    help="临时指定按键表（最高优先级）：w,s,a,d,q,n 或 w=W,n=N")
    ap.add_argument("--no-auto-geometry", action="store_true",
                    help="关掉自动探测：直接用模型 meta 里的 game_res/game_crop **像素框**，"
                         "不做按比例换算（窗口被挪动/缩放时画面会对不上，但区域绝对不变）")
    ap.add_argument("--game-res", default=None,
                    help="临时覆盖游戏分辨率（横屏用 1280,720），不改 model_meta")
    ap.add_argument("--game-crop", default=None,
                    help="临时覆盖裁剪框 x,y,宽,高（横屏建议 345,0,590,720，比例与 720:880 相同）")
    ap.add_argument("--content-offset", default=None,
                    help="安卓画面在窗口内的左上角偏移 x,y（默认自动算：水平居中+底部对齐）")
    ap.add_argument("--calibrate", action="store_true",
                    help="只标定阈值（不发键，专跑一轮采集置信度数据）")
    ap.add_argument("--no-calibrate", dest="auto_calib", action="store_false", default=True,
                    help="关掉「平时自动标定」（默认开：正常玩时顺便记录，退出时自动写 thresholds.json）")
    ap.add_argument("--conf-class", default=None,
                    help="按动作单独设阈值：--conf-class W=0.95,A=0.9,S=0.8"
                         "（没写的动作用 default）")
    ap.add_argument("--conf", type=float, default=None,
                    help="置信度阈值：低于它就不按键（不填就用模型目录的 thresholds.json / 0.85）；"
                         "想长期固定就改文件开头的 CONF_THRESHOLD")
    ap.add_argument("--lockout-ms", type=int, default=LOCKOUT_MS,
                    help="同一方向的锁定时间，默认 320ms（覆盖换道动画）。"
                         "只锁同一个键：A 之后再按 A 要等，但 A 之后按 D/W/S 立刻放行"
                         "—— 否则 WA / WS / AA 这类连按会被全挡掉")
    ap.add_argument("--min-gap-ms", type=int, default=MIN_GAP_MS,
                    help="任意两次发键之间的全局最小间隔，默认 80ms（防同一瞬间连发）")
    ap.add_argument("--half", action="store_true", default=True, help="FP16 推理")
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--delay-ms", type=int, default=DELAY_MS,
                    help="延迟取帧：模型看到的是 now-delay-ms 的画面，动作整体后移，"
                         "用来补偿模型\"动作偏早\"的系统偏差。"
                         "注意会拖坏连按：实测 delay>50ms 时 WA/WS 这类快连按识别率从 95.9%% 掉到 65%%（delay=81）")
    # ---------------- 注意力热力图 + 录像 ----------------
    ap.add_argument("--backend", choices=["printwindow", "bitblt"], default=None,
                    help="抓帧后端。printwindow 慢（实测 35ms/帧）但能抓被遮挡的窗口；"
                         "bitblt 快（10ms/帧）但要求模拟器窗口没被别的窗口挡住")
    ap.add_argument("--cam", dest="cam", action="store_true", default=CAM_HEATMAP,
                    help="显示 Grad-CAM 实时热力图窗口（默认开启）")
    ap.add_argument("--no-cam", dest="cam", action="store_false",
                    help="关掉热力图，换最高帧率（热力图要跑反向传播，会抢 GPU）")
    ap.add_argument("--cam-every", type=int, default=4,
                    help="每 N 帧算一次热力图，默认 4（40fps 下约 10Hz）。"
                         "调小更流畅但更吃 GPU；控制循环本身不受影响")
    ap.add_argument("--cam-on-change", dest="cam_on_change", action="store_true",
                    default=False,
                    help="只在模型判断发生变化时重算热力图（外加每 --cam-force-s 秒兜底刷新）。"
                         "稳态下几乎不占 GPU，判帧频率基本回到不开热力图时的水平")
    ap.add_argument("--cam-force-s", type=float, default=2.0,
                    help="--cam-on-change 下的兜底刷新间隔（秒，默认 2）")
    ap.add_argument("--cam-window", dest="cam_window", action="store_true",
                    default=SHOW_WINDOW,
                    help="弹出「画面 + 热力图 + 状态文字」窗口。默认不弹（省 CPU，"
                         "控制台每秒仍会打印预测/帧龄/速度）")
    ap.add_argument("--no-cam-window", dest="cam_window", action="store_false",
                    help="不弹窗口（默认行为，写出来只是图个显式）")
    ap.add_argument("--video", default=None,
                    help="录像输出路径；不填自动生成 play_<时间>.mp4（放在 gamer_ai/ 下）")
    # 录像默认**关闭**：只想要实时热力图的话，不会在后台白白写几十 MB 的文件。
    # 想存下来复盘就加 --record。
    ap.add_argument("--no-video", dest="no_video", action="store_true", default=True,
                    help="不录像（默认就是这个，可以不写）")
    ap.add_argument("--record", dest="no_video", action="store_false",
                    help="开启录像，存成 play_<时间>.mp4")
    ap.add_argument("--video-fps", type=float, default=20.0,
                    help="录像帧率，默认 20（跟真实时长一致）")
    ap.add_argument("--list-runs", action="store_true",
                    help="列出所有可用的模型然后退出（不知道 --run 该填什么就用这个）")
    ap.add_argument("--run-seconds", type=float, default=0.0,
                    help="跑满这么多秒后自动优雅退出（自检/录一段素材用；0=一直跑）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.video:
        args.no_video = False        # 明确指定了输出路径，说明就是想录，别被默认值挡住
    # ★ 2026-10-15 区域档案：--profile（或当前档案）生效时，模型在本档案的 runs/ 里找
    from core import profiles as _P
    _prof = _P.resolve(args.profile)
    _runs_root = _P.runs_root(_prof) if _prof else None
    if _prof:
        print("[档案] %s" % _P.describe(_prof))
        print("       模型目录：%s" % _runs_root)

    if getattr(args, "list_runs", False):
        print("[runs] 可用的模型（最近产出的排前面）%s："
              % ("（档案 %s）" % _prof if _prof else ""))
        for p, f1, acc, mt, n_tr, ra, dg in list_runs(_runs_root):
            rel = os.path.relpath(p, os.path.dirname(os.path.abspath(__file__)))
            s0 = f"{ra*100:5.1f}%" if ra >= 0 else "   ?  "
            print(f"   --run {rel:<44} 训练 {n_tr:5d} 个  "
                  f"test_acc {acc:.4f}  macroF1 {f1:.4f}  真实acc {s0}")
        _lst = list_runs(_runs_root)
        if _lst:
            print("\n   例：python 6_自动玩.py --run "
                  + os.path.relpath(_lst[0][0],
                                    os.path.dirname(os.path.abspath(__file__))))
        else:
            print("\n   （这个位置还没有模型：先跑 4_训练.py）")
        return

    # ★ 2026-10-16（用户："为什么我改这里的模型不对路径不对"）：
    #   旧逻辑是「只要档案可用，就**完全无视 MODEL_DIR**，去档案的 runs/ 里挑最新那个」。
    #   于是你把 MODEL_DIR 改成 ..._1022，程序照样加载 ..._1122 —— 看起来就是"路径不生效"。
    #   现在改成：**只要写了路径（顶上 MODEL_DIR 或命令行 --run）就它说了算**；
    #   写 None 才自动挑最近的（有档案就在档案的 runs/ 里挑）。
    if not args.run:
        args.run = pick_run(quiet=True, root=_runs_root) if _prof else pick_run(quiet=True)
        print(f"[模型] MODEL_DIR=None -> 自动挑最近产出的模型：{os.path.basename(args.run)}")
    else:
        print(f"[模型] 用你指定的模型：{args.run}")
    args.profile_name = _prof          # Player 里用它找档案的 geometry.json
    args.run = resolve_model_dir(args.run)
    if _prof and _runs_root:
        try:
            if not os.path.normcase(os.path.abspath(args.run)).startswith(
                    os.path.normcase(os.path.abspath(_runs_root))):
                print(f"[模型] 注意：这个模型不在当前档案 {_prof} 的 runs/ 里 —— "
                      f"取景会优先用模型自带 geometry，画面不对就改 MODEL_DIR 或加 --profile")
        except Exception:
            pass
    Player(args).run()


if __name__ == "__main__":
    main()
