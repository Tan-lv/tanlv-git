# -*- coding: utf-8 -*-
r"""迭代式行为克隆 —— 让模型自己玩，把「它犯错的时刻」自动录下来。

和 9_自动玩.py 的区别：那个只管玩；这个边玩边**判卷**，把可疑片段存成待标注事件。

判卷靠 hook（tools/il2cpp_state.js 暴露的三个信号）：
    inPlay   1=在关卡里
    alive    1=活着；1->0 的那一瞬就是撞到障碍物
    chasing  1=警察正在追；0->1 = 追击开始，1->0 = 被甩掉
    chaseDist 追击者与角色的距离（辅助）

================================ 判卷规则 ================================
撞车有两种后果，靠 chasing 区分：

    撞车 -> chasing 变 1  ->  瑕疵操作（可活，警察追出来）
    撞车 -> inPlay 变 0   ->  失误操作（直接被逮捕，游戏结束）

⚠ 关键设计：**追击期间不开新记录**
    警察追你的时候画面是分布外的（警察在屏幕上、角色趔趄、镜头抖），
    拿它训练会教坏模型。所以：
      * 状态机只在**边沿**触发（chasing 0->1 才算一次事件），
        "正在追"这个持续状态本身不会产生任何记录 ——
        你在追击期间按左键，根本不会进入判定
      * 每个事件还会检查窗口是否和追击区间重叠，重叠的打上
        contaminated 标记，单独放一类，默认不进训练集

============================== 存什么 ==============================
每个事件一个目录 events/<时间>_<类型>/：

    event.json         事件元信息：判卷结果、窗口、状态时间线、模型动作日志
    frames/0000.jpg…   撞车前 pre_ms 到撞车后 post_ms 的**逐帧原图**

逐帧存原图（而不是预先拼好的胶片条）是故意的：标的时候你要自己选
"人在哪一刻该按键"，那一刻决定了 6 帧输入窗的位置，所以必须能任意取窗。

用法（PyCharm 里直接 Run 也行）：

    python 10_迭代补数据.py                      # 用下面 MODEL_DIR 的模型
    python 10_迭代补数据.py --dry-run             # 只判卷不发键（安全预演）
    python 10_迭代补数据.py --pre-ms 3000         # 失误前保留 3 秒（默认就 3000）
    python 10_迭代补数据.py --keep-correct        # 顺便把模型的正确操作也存成 AWSDN

标记工具是 11_标注补数据.py。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as _e:        # 找图复活要用 cv2；没有就自动关掉那个功能
    cv2 = None
    print(f"[找图] 没有 cv2（{_e}）：先 pip install opencv-python 才能用找图复活")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn.functional as F

from collector.capture import WindowCapture
from collector.config import Config, key_for_label
from collector.dataset import (build_eval_transform, flatten_frames,
                               to_video_tensor)
from collector.hookstate import (HookStateClient, ensure_state_server, open_client,
                                kill_state_server, STATE_LOG)
from collector.model import build_model
from collector.recorder import RingBuffer, resolve_path, write_sample
from collector.runs import list_runs, pick_run


# ==============================================================================
#  ★★★  换模型就改这一行  ★★★
# ------------------------------------------------------------------------------
#  填「训练结果目录」的绝对路径（里面有 best.pt 和 model_meta.json）。
#  填 None = 自动挑 runs/ 里最近产出的那个。
#  想先看有哪些模型： python 9_自动玩.py --list-runs
# ==============================================================================
MODEL_DIR = r"D:\deeplearning\make_dataset\paoku\runs\20260918_merged_20260918_1822"

# 待标注事件（原始抓帧 + event.json）存这里（相对 paoku/）
EVENT_DIR = r"events"

# ==============================================================================
#  ★★★  迭代产物的落盘位置  ★★★
# ------------------------------------------------------------------------------
#  迭代收集的数据**全部**放在 dataset\迭代补采\ 下面，和手动采集的会话分开，
#  一眼就能区分哪些是人手玩的、哪些是模型自己玩出来的：
#
#      dataset\
#      ├── 20260918_103227\      ← 手动采集（原有）
#      ├── 20260918_merged\      ← 手动采集（原有）
#      └── 迭代补采\              ← ★ 本脚本 + 12_标注界面.py 的产物
#          ├── 正确操作\          ← 模型做对时自动存的 WASDN 样本（--keep-correct）
#          └── 碰撞操作\          ← 瑕疵/失误事件里人工挑帧标的样本（12_标注界面.py）
#
#  两个会话的目录结构**和手动采集完全一样**
#      <会话>\<标签>\<标签>_NNNNNN.jpg   +   meta.jsonl   +   session.json
#  所以 5_训练模型.py 把它们加进 DATA_DIRS 就能直接读，不需要任何转换。
# ==============================================================================
ITER_ROOT        = r"dataset\迭代补采"
CORRECT_SESSION  = r"dataset\迭代补采\正确操作"    # --keep-correct 写这里
COLLISION_SESSION = r"dataset\迭代补采\碰撞操作"   # 12_标注界面.py 写这里


# ==============================================================================
#  ★★★  找图复活（2026-09-23 合并进本脚本）  ★★★
# ------------------------------------------------------------------------------
#  被捕以后不再用"猜坐标"点，而是**找图**：
#    ① 找 templates\key.png（复活弹窗里那个钥匙按钮）-> 找到就点
#    ② 5 秒找不到 -> 找 templates\play.png（结算页/主菜单的 PLAY）-> 找到就点
#    ③ 4 秒还找不到 -> 放弃，等下一次被捕，再从头来
#
#  ★ 两个实测过的坑，别改回去：
#    1) 必须搜**整块安卓画面 720x1280**（--search-area full 的效果）。
#       训练用的 game_crop=(0,200,720,880) 把最下面 200 像素裁掉了，
#       而 PLAY 就在 y≈1170 —— 在裁剪图上永远找不到它。
#    2) 必须用 **adb shell input tap** 点。实测 SetCursorPos + mouse_event
#       把光标精确移到了 PLAY 上（屏 2560x1440，目标 (2365,1262)，GetCursorPos
#       确认到位），连点两次游戏毫无反应；同一位置换 adb tap 立刻生效。
#       LDPlayer 不吃合成的鼠标事件。而且帧坐标 == 安卓坐标，换算最简单。
#
#  模板命中实测：play.png 1.000（原尺寸）、key.png 0.976~0.995（原尺寸）。
# ==============================================================================
TEMPLATE_DIR = os.path.join(HERE, "templates")
TPL_KEY  = os.path.join(TEMPLATE_DIR, "key.png")     # 钥匙（复活）
TPL_PLAY = os.path.join(TEMPLATE_DIR, "play.png")    # PLAY（重开）
ADB_EXE  = r"D:\leidian\LDPlayer9\adb.exe"           # 雷电自带 adb
FINDIMG_DIR = os.path.join(HERE, "runs", "找图")      # 命中/未命中的证据图


# ==============================================================================
#  找图复活用的小工具
# ==============================================================================
def _imread_u(path: str):
    r"""中文路径安全的读图。

    ★ cv2.imread / imwrite 在 Windows 上走的是 ANSI 接口，路径里带中文
      （runs\找图\xxx.png）会**静默失败**：imwrite 只返回 False，读回来是 None。
      所以统一用 imencode / imdecode + numpy 的字节接口。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _imwrite_u(path: str, img) -> bool:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def _collect_templates(spec: str, prefix: str):
    """收集一个动作要用的模板图。

    spec 写了就用 spec（可逗号分隔多张）；留空就自动用 templates 目录下
    <prefix>*.png —— 所以想加一张图，直接丢进 templates 按名字前缀命名即可：
        key*.png   -> 复活钥匙
        play*.png  -> PLAY（重开）
    """
    if spec and spec.strip():
        return [x.strip() for x in spec.split(",") if x.strip()]
    import glob as _glob
    return sorted(_glob.glob(os.path.join(TEMPLATE_DIR, prefix + "*.png")))


class _Matcher:
    """**一组**模板图 + 一组缩放比例（缩放后的图预先算好，省得每帧重复缩放）。

    ★ 为什么一组（用户 2026-09-23）：只挂一张图，游戏里按钮样式/高亮/分辨率
      一变就认不出来（"只找一张图，有时识别不出来"）。多挂几张，
      **任意一张命中就算找到**。新图丢进 templates\ 按前缀自动加载。
    """

    def __init__(self, paths, scales):
        if not paths:
            raise FileNotFoundError("没有可用的模板图")
        self.items = []            # [(名字, 原图, [(缩放, 模板图), ...]), ...]
        self.thr_bonus = {}        # 每张模板的"加严量"（模板越素加得越多，防止乱匹配）
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"找不到模板图：{p}")
            img = _imread_u(p)
            if img is None:
                raise RuntimeError(f"读不了模板图：{p}")
            name = os.path.basename(p)
            try:
                # ★ 模板越"平"（标准差越小），越容易在别的画面上也骗到高分
                #   （实测：一块纯蓝的图在随便一个蓝面板上都能得 0.911）。
                #   所以给每张模板单独抬高阈值，最多 +0.25。
                std = float(img.reshape(-1, img.shape[2]).std(axis=0).mean())
            except Exception:
                std = 99.0
            bonus = round(max(0.0, (40.0 - std) / 40.0) * 0.25, 3)
            self.thr_bonus[name] = bonus
            if std < 20.0:
                print(f"[找图] ⚠ 模板 {name} 太素了（std={std:.1f}）：实测很容易在别的画面上"
                      f"骗到高分，已把它单独调严（阈值 +{bonus:.2f}）；"
                      "**建议换一张带图标/边框/文字的图**")
            tpls = []
            for s in scales:
                t = cv2.resize(img, None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
                tpls.append((float(s), t))
            self.items.append((name, img, tpls))
        self.last_name = ""

    @property
    def names(self):
        return [n for n, _, _ in self.items]

    def find(self, frame, base_thr: float):
        """匹配。返回 (是否算命中, (得分, 缩放, (x, y), (w, h)))。

        每张模板有自己的阈值：base_thr + thr_bonus[该模板]（越素的图要求越高）。
        命中哪张模板放在 self.last_name。
        """
        fh, fw = frame.shape[:2]
        best = (-1.0, 0.0, (0, 0), (0, 0))
        for name, _raw, tpls in self.items:
            for s, t in tpls:
                if t.shape[0] > fh or t.shape[1] > fw:
                    continue
                res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(res)
                if mx > best[0]:
                    best = (float(mx), s, ml, (t.shape[1], t.shape[0]))
                    self.last_name = name
        ok = best[0] >= base_thr + self.thr_bonus.get(self.last_name, 0.0)
        return ok, best

    def thr_of(self, name: str, base_thr: float) -> float:
        return base_thr + self.thr_bonus.get(name, 0.0)

    def per_template(self, frame):
        """每张模板各自的最好得分 —— --grab 调参时看这个（找出哪张图没用）。"""
        fh, fw = frame.shape[:2]
        out = []
        for name, _raw, tpls in self.items:
            b = (-1.0, 0.0, (0, 0), (0, 0))
            for s, t in tpls:
                if t.shape[0] > fh or t.shape[1] > fw:
                    continue
                res = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, ml = cv2.minMaxLoc(res)
                if mx > b[0]:
                    b = (float(mx), s, ml, (t.shape[1], t.shape[0]))
            out.append((name, b))
        return out


class _AdbTap:
    """用 adb shell input tap 点击安卓坐标。

    ★ 为什么不用鼠标点击（2026-09-23 实测）:
      SetCursorPos + mouse_event 把光标精确移到了 PLAY 上（屏 2560x1440，
      目标 (2365,1262)，GetCursorPos 确认到位），连点两次游戏毫无反应；
      同一位置换成 adb shell input tap 551 1203 立刻生效。
      LDPlayer 不吃合成鼠标事件，窗口不在前台时更没戏。
    """

    def __init__(self, exe: str = ADB_EXE):
        self.exe = exe
        self.serial = None

    def _run(self, args, timeout=20):
        try:
            p = subprocess.run([self.exe] + list(args), capture_output=True, timeout=timeout)
            return (p.returncode, p.stdout.decode("utf-8", "replace"),
                    p.stderr.decode("utf-8", "replace"))
        except Exception as e:
            return -1, "", str(e)

    def pick(self, tries: int = 4, gap: float = 2.0):
        """挑一个能用的设备。必须带 -s：设备同时以 emulator-5554 和
        127.0.0.1:5555 出现时，不带 -s 的 adb shell 会直接失败。
        adb server 刚起来那几秒 emulator-5554 常常还是 offline，所以多试几轮。"""
        last = ""
        for _ in range(tries):
            _, out, _ = self._run(["devices"])
            last = out.strip()
            cands = [l.split()[0] for l in out.splitlines()
                     if l.strip() and "List of devices" not in l and "offline" not in l]
            for s in cands:
                rc, so, _ = self._run(["-s", s, "shell", "echo ok"], timeout=10)
                if rc == 0 and "ok" in so:
                    self.serial = s
                    return s
            self._run(["connect", "127.0.0.1:5555"])
            time.sleep(gap)
        print("      [adb] 没找到可用设备，最后一次 adb devices 输出：")
        for l in last.splitlines():
            print("            " + l)
        return None

    def tap(self, x: float, y: float):
        if self.serial is None and self.pick() is None:
            return False, "找不到可用的 adb 设备（模拟器没开？）"
        for attempt in (0, 1):
            rc, so, se = self._run(["-s", self.serial, "shell", "input", "tap",
                                    str(int(round(x))), str(int(round(y)))], timeout=15)
            if rc == 0:
                return True, (so + se).strip()
            self.serial = None
            if attempt == 0 and self.pick() is not None:
                continue
            return False, (so + se).strip()
        return False, "tap 失败"


# ==============================================================================
#  事件
# ==============================================================================
@dataclass
class Event:
    anchor: float                     # 撞车时刻（判卷的锚点）
    kind: str = "?"                   # flaw(瑕疵) | error(失误)
    decided: bool = False
    open_chase: bool = False          # 撞车后是否正在等追击出现
    chase_started: Optional[float] = None
    chase_ended: Optional[float] = None
    run_ended: Optional[float] = None
    state_log: List[Tuple[float, int, int, int]] = field(default_factory=list)
    actions: List[Tuple[float, str, float, bool]] = field(default_factory=list)
    # 撞车那一刻"正在生效"的那个操作 —— 也就是被判为瑕疵/被捕的那个操作。
    # 人工挑帧时一眼就知道该改哪个动作，不用自己从 actions 里猜。
    culprit: Optional[dict] = None
    # 瑕疵一般是换道时机不准（用户 2026-09-23）：
    # culprit_dt_ms = 撞车时刻 - 那个操作的时刻，负值不可能，越小=按得越晚
    culprit_dt_ms: Optional[float] = None
    culprit_hint: str = ""
    dumped: bool = False


class IterCollector:
    def __init__(self, args):
        self.args = args
        run_dir = args.run if os.path.isabs(args.run) else resolve_path(args.run)
        self.meta, ckpt, self.thr = load_run(run_dir)
        m = self.meta
        self.labels = list(m["labels"])
        self.K = int(m["in_frames"])
        self.img_size = int(m.get("img_size", 224))
        print(f"[run]   {os.path.abspath(run_dir)}")
        print(f"[run]   类别 {self.labels}  输入 {self.K} 帧  "
              f"test_acc {m.get('test_acc', float('nan')):.4f}")

        cfg = Config()
        cfg.game_res = tuple(m.get("game_res", cfg.game_res))
        cfg.game_crop = tuple(m.get("game_crop", cfg.game_crop))
        cfg.frame_w = int(m.get("frame_w", cfg.frame_w))
        cfg.backend = m.get("backend", cfg.backend)
        cfg.validate()
        self.cfg = cfg

        # 推理用的帧间隔必须和训练时一致（否则输入窗口的物理长度就变了）
        self.step_s = ((float(m.get("save_pre_ms", 300)) + float(m.get("save_post_ms", 100)))
                       / max(int(m.get("saved_frames", 13)) - 1, 1)) / 1000.0
        self.stride = max(1, int(m.get("input_stride", 1)))
        self.end_offset_ms = float(m.get("input_end_offset_ms", 40))
        print(f"[run]   帧间隔 {self.step_s*1000:.1f}ms x{self.stride}  "
              f"输入覆盖 {(self.K-1)*self.stride*self.step_s*1000:.0f}ms  "
              f"结束于按键前 {self.end_offset_ms:.0f}ms")

        # 环形缓冲要盖住 pre + post 再加余量
        # 正确操作要等到"下一个动作到达"或兜底时限才落盘，
        # 那时要回头取 t-800ms 的帧 —— 环形缓冲必须盖得住，否则写出来是空窗口。
        ring_s = max((args.pre_ms + args.post_ms) / 1000.0 + 1.5,
                     args.correct_settle_ms / 1000.0 + self.cfg.save_pre_ms / 1000.0 + 1.0)
        self.pre_s = args.pre_ms / 1000.0
        self.post_s = args.post_ms / 1000.0
        self.cap = WindowCapture(cfg.window_keyword, cfg.game_res, cfg.game_crop, backend=cfg.backend)
        self.ring = RingBuffer(ring_s)
        self.transform = build_eval_transform(self.img_size)
        print(f"[ring]  环形缓冲 {ring_s:.1f}s（失误前保留 {args.pre_ms}ms + 后 {args.post_ms}ms；"
              f"正确操作兜底结算 {args.correct_settle_ms:.0f}ms）")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.K, len(self.labels), pretrained=False)
        ck = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(ck.get("model_state_dict", ck))
        self.model = self.model.to(self.device).eval()
        self.use_half = bool(args.half) and self.device.type == "cuda"
        if self.use_half:
            self.model = self.model.half()
        print(f"[model] 已加载到 {self.device}" + (" (FP16)" if self.use_half else ""))

        # ---------------- hook 状态 ----------------
        # open_client()：先确认服务是新协议、能回 JSON，再等它读到 inPlay。
        # 拿不到就返回 None —— 绝不会留下一个已经 close() 的死客户端
        # （那正是 2026-09-22「加了 hook 反而检测不到撞车」的根源）。
        self.hook: Optional[HookStateClient] = None
        self.judge_bad_since: Optional[float] = None
        self.judge_down_s = 0.0
        self.hook_ever_ok = False
        if not args.no_hook:
            self.hook = open_client(cfg.hook_state_host, cfg.hook_state_port)
            if self.hook is not None:
                self.hook_ever_ok = True
            else:
                print("[hook]  ⚠ 本次运行没有内存判卷：撞车/瑕疵/失误 全都会是 0，"
                      "不代表你没撞车！")

        # ---------------- 输出 ----------------
        self.out_root = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
        os.makedirs(self.out_root, exist_ok=True)
        self.index_path = os.path.join(self.out_root, "index.jsonl")
        self.ev_seq = 0
        # --keep-correct 的落盘目标（普通数据集会话）
        def _abs(p):
            return p if os.path.isabs(p) else os.path.join(HERE, p)
        self.correct_root   = _abs(args.correct_dir)
        self.collision_root = _abs(args.collision_dir)
        # 两个会话都先建好，目录结构一眼可见（碰撞操作由 12_标注界面.py 填）
        _init_session(self.correct_root, cfg)
        _init_session(self.collision_root, cfg)
        self.counters: Dict[str, int] = {}
        if args.keep_correct:
            self.counters = _load_counters(self.correct_root)
            print(f"[输出]  正确操作 -> {self.correct_root}")
        else:
            print(f"[输出]  正确操作 -> {self.correct_root}（未开启 --keep-correct，暂时空的）")
        print(f"[输出]  碰撞操作 -> {self.collision_root}（由 12_标注界面.py 里挑帧标注后写入）")

        # ---------------- 运行时状态 ----------------
        self.running = True
        self.playing = True
        self.events: List[Event] = []
        self.pending: List[Event] = []            # 等窗口凑齐再落盘
        self.chase_intervals: List[Tuple[float, float]] = []
        self.cur_chase_start: Optional[float] = None
        self.prev = {"inPlay": None, "alive": None, "chasing": None}
        self.n_collision = 0
        self.n_flaw = 0
        self.n_error = 0
        self.n_correct = 0
        # ---- 逐操作判卷的计数（用户 2026-09-23 定的分类）----
        self.op_seq = 0
        self.n_correct_clean = 0      # 正确操作（其后无碰撞、也不在追击中）
        self.n_correct_chase = 0      # 正常操作（追击中，但之后没再撞车）
        self.n_flaw_ops = 0           # 瑕疵操作（其后撞车并招来警察）
        self.n_arrest_ops = 0         # 撞车被捕操作（其后撞车并且本局结束）
        self.last_action: Optional[Tuple[float, str, float]] = None
        self.last_wasd: Optional[Tuple[float, str, float]] = None   # 最后一次换道(撞车时定位)
        self.pending_n: Optional[dict] = None        # N 的样本流水线（不背瑕疵的锅）
        self.last_op: Optional[Tuple[float, str]] = None   # 最近一次真正记下的操作(N 或 AWSD)
        self.op_seq_n = 0              # N 候选的编号（和换道分开数，日志里一眼能分清）
        self.n_n_dropped = 0           # 被换道/撞车打断而作废的 N 候选个数
        self.n_flaw_unattr = 0         # 撞车时连"最后一次换道"都还没有的次数（开局就撞）
        self._dist_safe = False        # 追击者距离是否已经"甩开"（撞车信号 B 用）
        self.n_stumble = 0             # 趔趄撞车（被追但没被抓）次数
        self.ops_log_path = os.path.join(HERE, "runs", "操作判卷.jsonl")
        self.flaw_by_label: Dict[str, int] = {}    # 瑕疵/被捕操作都是哪个动作造成的
        self._lock = threading.Lock()

        self.lockout_until: Dict[str, float] = {}
        self.min_gap_s = max(0.0, args.min_gap_ms) / 1000.0
        self.last_send = 0.0
        self.sent = 0
        self.lockout_hits = 0
        self.low_conf = 0
        self.ticks = 0
        self.last_correct_kept = 0.0
        self.last_correct_queued = 0.0
        self.last_n_queued = 0.0
        self.n_correct_n = 0
        self.n_correct_dir = 0
        self.pending_correct: Optional[dict] = None
        self._rv_prev = None
        self.chasing = False

        # ---------------- 找图复活 ----------------
        self._reviving = False
        self._revive_cooldown = 0.0
        self.n_revive_key = 0
        self.n_revive_play = 0
        self.cap_full = None          # 整块安卓画面（720x1280），找图专用
        self.m_key = None
        self.m_play = None
        self.adb = None
        self._init_revive()

        self.keyboard = None
        if not args.dry_run:
            from pynput.keyboard import Controller
            self.keyboard = Controller()
            self._focus_window()

    # ------------------------------------------------------------------
    def _focus_window(self) -> None:
        try:
            import ctypes
            u32 = ctypes.windll.user32
            if u32.GetForegroundWindow() != self.cap.hwnd:
                u32.ShowWindow(self.cap.hwnd, 9)
                u32.keybd_event(0x12, 0, 0, 0)
                u32.SetForegroundWindow(self.cap.hwnd)
                u32.keybd_event(0x12, 0, 2, 0)
        except Exception:
            pass

    def _send_key(self, label: str) -> None:
        from pynput.keyboard import KeyCode
        # ★ 2026-10-11：改成查 config.ACTION_KEYS —— 以后再加动作键（Q/E/...）这里不用动
        ch = key_for_label(label)
        if ch is None or self.keyboard is None:
            return
        try:
            self.keyboard.press(KeyCode.from_char(ch))
            time.sleep(0.03)
            self.keyboard.release(KeyCode.from_char(ch))
        except Exception as e:
            print(f"\n[ERR] 发键失败: {e}")

    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        import cv2
        interval = 1.0 / max(1.0, self.args.fps)
        nxt = time.time()
        while self.running:
            try:
                t = time.time()
                f = self.cap.grab()
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
        now = time.time()
        frames = []
        for i in range(self.K - 1, -1, -1):
            tg = now - (self.end_offset_ms / 1000.0) - i * self.stride * self.step_s
            f = self.ring.frame_at_or_before(tg)
            if f is None:
                return None
            frames.append(f)
        x = to_video_tensor(np.stack(frames))
        return flatten_frames(self.transform(x)).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    #  hook 判卷：只看边沿，追击期间不产生新记录
    # ------------------------------------------------------------------
    def _kill_state_server(self) -> None:
        """把状态服务彻底清干净：占着端口的 + 没 bind 成功的残留进程。

        必须杀了再重开 —— 旧进程占着端口时新进程 bind 会失败，
        而旧进程可能只会永远回 "script has been destroyed"（僵尸）。"""
        self._say("[hook] 清理状态服务进程…")
        kill_state_server(self.cfg.hook_state_host, self.cfg.hook_state_port)

    def _hook_loop(self) -> None:
        """轮询内存状态做判卷。

        和旧版最大的区别：**判卷断了必须喊出来，而且要真的能接回去。**
        旧版重连失败时会把已经 close() 的旧客户端留在 self.hook 上，它的
        read() 永远返回缓存里那句 "script has been destroyed" —— 于是每 2 秒
        "发现会话已废" -> 杀端口上的进程（其实是自己刚拉起来的那个）-> 再拉起
        -> 再超时…… 整局判卷都是死的，终端上却只显示「撞车 0」。
        """
        host, port = self.cfg.hook_state_host, self.cfg.hook_state_port
        last_hb = time.time()
        while self.running:
            st = self.hook.read() if self.hook is not None else None
            good = bool(st) and bool(st.get("ok")) and ("inPlay" in st)
            now = time.time()

            if not good:
                err = str((st or {}).get("err") or "状态服务没回话")
                if self.judge_bad_since is None:
                    self.judge_bad_since = now
                    self._say(f"[hook] ⚠ 判卷中断：{err[:70]}")
                elif now - self.judge_bad_since >= 20.0:
                    # 20 秒还回不来：整个重启一遍（换**新**客户端，绝不复用旧连接）
                    self.judge_down_s += now - self.judge_bad_since
                    self.judge_bad_since = None
                    self._say("[hook] ⚠ 判卷中断 20 秒，硬重启状态服务…")
                    try:
                        if self.hook is not None:
                            self.hook.close()
                    except Exception:
                        pass
                    self.hook = None
                    kill_state_server(host, port)
                    t_r = time.time()
                    self.hook = open_client(host, port)
                    if self.hook is not None:
                        self.judge_down_s += time.time() - t_r
                        self.prev = {"inPlay": None, "alive": None, "chasing": None}
                        self._say("[hook] 判卷恢复")
                    else:
                        self._say("[hook] ⚠ 状态服务还是起不来，判卷仍然关闭"
                                  f"（详情见 {STATE_LOG}）")
                time.sleep(0.05)
                continue

            if not self.hook_ever_ok:
                self.hook_ever_ok = True
                self._say("[hook] 判卷已就绪")
            if self.judge_bad_since is not None:
                self.judge_down_s += now - self.judge_bad_since
                self.judge_bad_since = None
                self._say("[hook] 判卷恢复，继续统计")

            # 复活相关值一变就立刻报（心跳 15 秒一次，会漏掉瞬时变化）
            rv = (st.get("rciN"), st.get("gmrN"), st.get("adOffer"), st.get("keyOffer"))
            if self._rv_prev != rv:
                self._rv_prev = rv
                self._say(f"[复活监视] rciN={rv[0]} gmrN={rv[1]} ad={rv[2]} key={rv[3]}"
                          f"   (inPlay={st.get('inPlay')} alive={st.get('alive')} paused={st.get('paused')})")
            try:
                self._on_state(st, now)
            except Exception as e:
                # ★ 判卷线程绝对不能因为一个小异常就静默死掉 ——
                #   以前"撞车 0"就是判卷悄悄停了，查起来非常费劲。
                self._say(f"[hook] ⚠ 判卷异常（已跳过这一帧）：{type(e).__name__}: {e}")
            # ★ 心跳：每 15 秒报一次内存状态，证明 hook 还活着。
            #   没有它，"hook 掉了"和"这一局没撞车"在终端上长得一模一样。
            if now - last_hb > 15.0:
                last_hb = now
                self._say(f"[hook] 心跳 inPlay={st.get('inPlay')} alive={st.get('alive')} "
                          f"chasing={st.get('chasing')} paused={st.get('paused')}  |  "
                          f"撞车 {self.n_collision}  瑕疵 {self.n_flaw}  失误 {self.n_error}"
                          f"  ‖ 判卷中断累计 {self.judge_down_s:.0f}s"
                          f"  ‖ 复活候选 rciN={st.get('rciN')} gmrN={st.get('gmrN')}"
                          f" ad={st.get('adOffer')} key={st.get('keyOffer')}")
            time.sleep(0.02)

    def _stumble_hit(self, dist) -> bool:
        """撞车信号 B：追击者距离**突然掉回来**（趔趄），但角色没被抓。

        ★ 内存实测（tools\collision_trace.py，2026-09-23）：
            正常跑   dist 从 10 一路爬到 30（警察被越甩越远）
            撞一下   dist 在 0.7 秒内从 30 掉回 10，而 alive 一直是 1  ← 以前完全没记
            被抓     alive 1->0（那是信号 A）
        用"甩开过 -> 又掉回来"做迟滞，避免同一次趔趄重复计数。
        """
        if dist is None or float(dist) < 0:
            return False
        d = float(dist)
        if d >= self.args.stumble_safe_dist:
            self._dist_safe = True                      # 甩开了，下一次掉回来就是又撞了
            return False
        if self._dist_safe and d <= self.args.stumble_hit_dist:
            self._dist_safe = False
            return True
        return False

    def _on_state(self, st: dict, now: float) -> None:
        inPlay = int(st.get("inPlay", -1))
        alive = int(st.get("alive", -1))
        chasing = int(st.get("chasing", -1))
        paused = int(st.get("paused", -1))
        dist = st.get("chaseDist")
        with self._lock:
            p = self.prev
            if p["inPlay"] is None:
                p.update(inPlay=inPlay, alive=alive, chasing=chasing)
                return
            if p["inPlay"] == 0 and inPlay == 1:
                self._dist_safe = False        # 新一局开始：距离重新从 10 爬
            # 状态时间线只记变化（缩小 event.json）
            if (inPlay, alive, chasing) != (p["inPlay"], p["alive"], p["chasing"]):
                for ev in self.pending:
                    if not ev.dumped:
                        ev.state_log.append((round(now, 4), inPlay, alive, chasing))

            # ① 撞车：alive 1 -> 0。
            #    注意：追击期间发生碰撞时不新开记录（那是"已在追"状态的延续），
            #    只把它记到当前事件里，用于判定这是不是致命那次。
            # ①b 撞车信号 B：趔趄（追击者距离突然掉回来），但 alive 还是 1
            #     —— 这就是"碰撞被警察追、但没被抓"，以前完全没被记成撞车。
            elif alive == 1 and inPlay == 1 and self._stumble_hit(dist):
                self.n_collision += 1
                self.n_stumble += 1
                self._mark_pending(collision=True)
                culprit = self._culprit()
                hint = self._culprit_hint(culprit, now)
                dt_ms = None if culprit is None else round((now - culprit["t"]) * 1000.0, 1)
                if self.pending_correct is None:
                    if culprit is not None:
                        self._record_backfill_flaw(culprit, hint)
                    else:
                        self.n_flaw_unattr += 1
                self._say(f"[判卷] 检测到撞车（趔趄：警察追上来了但没被抓）dist={dist}  {hint}")
                # 活下来了 -> 直接就是「瑕疵操作」；要不要落盘事件由 --stumble-events 决定
                self.n_flaw += 1
                if self.args.stumble_events:
                    ev = Event(anchor=now, kind="flaw", decided=True)
                    ev.culprit = culprit
                    ev.culprit_dt_ms = dt_ms
                    ev.culprit_hint = hint
                    ev.state_log.append((round(now, 4), inPlay, alive, chasing))
                    self.pending.append(ev)
                    self.events.append(ev)

            # ① 撞车信号 A：alive 1 -> 0（被抓住了）。
            #    注意：追击期间发生碰撞时不新开记录（那是"已在追"状态的延续），
            #    只把它记到当前事件里，用于判定这是不是致命那次。
            if p["alive"] == 1 and alive == 0:
                self.n_collision += 1
                # ★ 撞车这一刻"正在生效的那个操作"就是瑕疵操作（或撞车被捕操作）：
                #   先给它打标记，等下一次操作到来 / 本局结束时再分类结算。
                #   以前这里是直接丢弃，所以「那个操作」既没算正确、也没标瑕疵。
                self._mark_pending(collision=True)
                culprit = self._culprit()
                hint = self._culprit_hint(culprit, now)
                dt_ms = None if culprit is None else round((now - culprit["t"]) * 1000.0, 1)
                # ★ 撞车时手上已经没有挂着的换道操作（上一次换道早被兜底结算走了）：
                #   也要把"最后那次换道"补记成瑕疵操作。否则会出现
                #   「撞车 5 次、瑕疵操作只有 4 次」，看起来像漏检 —— 2026-09-23 实测。
                if self.pending_correct is None:
                    if culprit is not None:
                        self._record_backfill_flaw(culprit, hint)
                    else:
                        self.n_flaw_unattr += 1     # 开局就撞，连换道都还没有
                self._say(f"[判卷] 换道线索：{hint}")
                self._start_revive()              # ★ 找图复活（自己防重入）
                if p["chasing"] == 1:
                    # 追击期间又撞一次 —— 按游戏规则这次必被逮捕。
                    # 事件立即判定，但窗口必然含追击段，会被标成 contaminated。
                    ev = Event(anchor=now, kind="error", decided=True)
                    ev.culprit = culprit          # 这次撞车该算在哪个操作头上
                    ev.culprit_dt_ms = dt_ms
                    ev.culprit_hint = hint
                    self.pending.append(ev)
                    self.events.append(ev)
                    self.n_error += 1
                    self._say(f"[判卷] 失误操作 —— 追击途中再次碰撞，必被逮捕 (t={now:.2f})")
                else:
                    ev = Event(anchor=now, open_chase=True)
                    ev.culprit = culprit          # 这次撞车该算在哪个操作头上
                    ev.culprit_dt_ms = dt_ms
                    ev.culprit_hint = hint
                    # 撞车和追击常常是同一帧发生的（游戏一碰撞就把 _killed 置 1），
                    # 所以起始状态要当场记下来，否则时间线会是空的
                    ev.state_log.append((round(now, 4), inPlay, alive, chasing))
                    self.pending.append(ev)
                    self.events.append(ev)
                    self._say(f"[判卷] 检测到撞车 (t={now:.2f}) 等待判定…")
            # ② 追击开始（边沿）：把最近那个撞车事件判为「瑕疵操作」
            if p["chasing"] == 0 and chasing == 1:
                self.cur_chase_start = now
                # 只认"撞车之后紧接着"的那次追击 —— 开场自带的追击不算
                target = None
                for ev in reversed(self.pending):
                    if not ev.decided and ev.open_chase:
                        target = ev
                        break
                if target is not None:
                    target.decided = True
                    target.chase_started = now
                    if alive == 0:
                        # 追击出现的同时角色已经倒下 = **被抓**（本局结束）：
                        # 这属于「撞车被捕」，不是"撞一下甩开警察"的瑕疵。
                        target.kind = "error"
                        self.n_error += 1
                        self._say(f"[判卷] 撞车被捕 —— 警察追上来抓住了，本局结束 (t={now:.2f})")
                    else:
                        # 撞一下、警察追出来，但角色还活着 -> 瑕疵
                        target.kind = "flaw"
                        self.n_flaw += 1
                        self._say(f"[判卷] 瑕疵操作 —— 撞车后警察追出来了（人还在跑）(t={now:.2f})")
            # ③ 追击结束（被甩掉）
            if p["chasing"] == 1 and chasing == 0:
                if self.cur_chase_start is not None:
                    self.chase_intervals.append((self.cur_chase_start, now))
                    self.cur_chase_start = None
                for ev in self.pending:
                    if ev.chase_started and ev.chase_ended is None:
                        ev.chase_ended = now
                self._say(f"[判卷] 警察被甩掉 (t={now:.2f})")
            # ④ 一局结束：还没被判为瑕疵的撞车 -> 失误操作（直接被逮捕）
            # 被捕的实测表现：**inPlay 由 1 掉到 0**，约 2.2 秒后若点了复活会回到 1。
            # （paused 全程为 0，Continue 弹窗不用它表示 —— 别再拿它当判据）
            if p["inPlay"] == 1 and inPlay == 0:
                # 本局结束（被捕）：锅算在"当时手上那个操作"上
                if (self.last_op is not None and self.last_op[1] == "N"
                        and self.pending_n is not None):
                    # 手里是 N -> 正面撞车被捕：**这是 N 唯一背锅的情况**
                    self.pending_correct = None      # 这一局的锅在 N，换道那份不记
                    self._settle_n(as_arrest=True)
                else:
                    # 手里是换道操作 -> 锅算在换道上
                    self._mark_pending(arrest=True)
                    self._settle_pending("本局结束")
                    self._settle_n()             # N 作废（不能存成"该不动"）
                self._start_revive()              # ★ 本局结束（被捕）也试一次找图复活
                for ev in self.pending:
                    if not ev.decided and ev.open_chase:
                        ev.kind = "error"
                        ev.decided = True
                        ev.run_ended = now
                        self.n_error += 1
                        self._say(f"[判卷] 失误操作 —— 撞车后直接结束，没有追击段 (t={now:.2f})")
                if self.cur_chase_start is not None:
                    self.chase_intervals.append((self.cur_chase_start, now))
                    self.cur_chase_start = None

            # 实测 _killed 一旦置 1 就再也不会回 0（站起来了也还是 1），
            # 所以不能只看它 —— 必须加上"角色已经死了"，否则第一次撞车之后
            # 所有正确操作都会被挡掉。
            self.chasing = (chasing == 1 and alive == 0)
            p.update(inPlay=inPlay, alive=alive, chasing=chasing)
            # 追击开始后开一个"等追击结束"的观察窗，避免把开场追击误判
            for ev in self.pending:
                if not ev.decided and ev.kind == "?" and now - ev.anchor > 3.0:
                    # 撞车 3 秒内既没追击也没结束：很少见，按失误处理
                    ev.kind = "error"
                    ev.decided = True
                    self.n_error += 1

    # ------------------------------------------------------------------
    def _dump_loop(self) -> None:
        """窗口凑齐后把事件落盘。"""
        while self.running:
            time.sleep(0.5)
            now = time.time()
            with self._lock:
                todo = [ev for ev in self.pending
                        if not ev.dumped and ev.decided and now - ev.anchor >= self.post_s]
                for ev in todo:
                    ev.dumped = True
            for ev in todo:
                try:
                    self._dump(ev)
                except Exception as e:
                    print(f"\n[ERR] 落盘失败: {e}")

    def _dump(self, ev: Event) -> None:
        t0, t1 = ev.anchor - self.pre_s, ev.anchor + self.post_s
        items = self.ring.window(t0, t1)
        if not items:
            self._say(f"[跳过] 事件窗口没有帧 (t={ev.anchor:.2f})")
            return
        # 窗口是否和追击区间重叠 —— 重叠的画面是分布外的，必须标出来
        contaminated = False
        intervals = list(self.chase_intervals)
        if self.cur_chase_start is not None:
            intervals.append((self.cur_chase_start, time.time()))
        # 只有「撞车之前警察就已经在追」才算污染。
        # 本次撞车**引起**的那次追击从 anchor 起算，是正常后果，不该判为污染
        # （否则每个 flaw 事件都会被标污染，这个标记就废了）。
        for a, b in intervals:
            if a < ev.anchor - 0.05 and b > t0:
                contaminated = True
                break

        from PIL import Image
        self.ev_seq += 1
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ev.anchor))
        name = f"{stamp}_{ev.kind}_{self.ev_seq:03d}"
        d = os.path.join(self.out_root, name)
        fdir = os.path.join(d, "frames")
        os.makedirs(fdir, exist_ok=True)

        for i, (t, f) in enumerate(items):
            Image.fromarray(f).save(os.path.join(fdir, f"{i:04d}.jpg"),
                                    "JPEG", quality=90, subsampling=2)
        # 再存两张「横排胶片条」，和 2_采集数据.py 输出的一模一样，方便直接肉眼看：
        #   preview_full.jpg   整个 3.4 秒连续排开
        #   preview_anchor.jpg 以撞车点为锚的标准窗口（就是"如果这里生成一个样本"的样子）
        self._write_preview(items, d, "preview_full", None)
        self._write_preview(items, d, "preview_anchor", ev.anchor)

        # 模型在这段时间里做了什么 —— 没有这个就没法判断"它当时该不该动"
        acts = [(round(t, 4), lb, round(cf, 3), bool(sent))
                for (t, lb, cf, sent) in ev.actions
                if t0 <= t <= t1]
        meta = {
            "name": name,
            "kind": ev.kind,                       # flaw | error
            "t_anchor": round(ev.anchor, 4),
            "t0": round(t0, 4), "t1": round(t1, 4),
            "pre_ms": self.args.pre_ms, "post_ms": self.args.post_ms,
            "n_frames": len(items),
            "fps": round(len(items) / max(t1 - t0, 1e-6), 1),
            "contaminated": contaminated,           # True = 窗口里有追击段，默认别用
            "chase_started": ev.chase_started, "chase_ended": ev.chase_ended,
            "state_log": [[round(t, 4), a, b, c] for (t, a, b, c) in ev.state_log],
            "actions": [[t, lb, cf, s] for (t, lb, cf, s) in acts],
            "culprit": ev.culprit,                 # 被判为瑕疵/被捕的那个操作
            "culprit_dt_ms": ev.culprit_dt_ms,     # 撞车时刻 - 那个操作的时刻
            "culprit_hint": ev.culprit_hint,       # 换道太早/太晚/时机不准/很久没换道
            "labels": self.labels,
            "input_frames": self.K,
            "input_stride": self.stride,
            "end_offset_ms": self.end_offset_ms,
            "frame_step_ms": round(self.step_s * 1000, 3),
            "frame_w": self.cfg.frame_w, "frame_h": self.cfg.frame_h, "sep": self.cfg.sep,
            "labeled": False,
        }
        with open(os.path.join(d, "event.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        with open(self.index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: meta[k] for k in
                                 ("name", "kind", "t_anchor", "n_frames", "contaminated", "labeled")},
                                ensure_ascii=False) + "\n")
        tag = "⚠ 含追击段" if contaminated else ""
        who = ""
        if ev.culprit:
            zh = {"flaw": "瑕疵", "error": "失误/被捕"}.get(ev.kind, ev.kind)
            dt = f"（撞车前 {ev.culprit_dt_ms:.0f}ms）" if ev.culprit_dt_ms is not None else ""
            who = f"  判为{zh}的操作：「{ev.culprit['label']}」{dt}"
        self._say(f"[落盘] {name}  {len(items)} 帧  {tag}{who}")

    def _write_preview(self, items, d: str, name: str, anchor: Optional[float]) -> None:
        """把帧横排拼成一张长图 —— 和采集器 write_sample 的胶片条布局完全一致。"""
        from PIL import Image
        fw, fh, sep = self.cfg.frame_w, self.cfg.frame_h, self.cfg.sep
        if anchor is None:
            sel = [fr for _, fr in items]
        else:
            t0 = anchor - self.cfg.save_pre_ms / 1000.0
            t1 = anchor + self.cfg.save_post_ms / 1000.0
            want = self.cfg.saved_frames
            sel = []
            for i in range(want):
                tg = t0 + (t1 - t0) * i / max(want - 1, 1)
                best, bd = items[0][1], 1e9
                for t, fr in items:
                    if abs(t - tg) < bd:
                        bd, best = abs(t - tg), fr
                sel.append(best)
        n = len(sel)
        if n < 1:
            return
        w = n * fw + (n - 1) * sep
        strip = np.full((fh, w, 3), 40, dtype=np.uint8)
        for i, fr in enumerate(sel):
            x = i * (fw + sep)
            strip[:, x:x + fw] = fr
        try:
            Image.fromarray(strip).save(os.path.join(d, name + ".jpg"),
                                        "JPEG", quality=90, subsampling=2)
        except Exception as e:
            self._say(f"[预览] {name} 写失败: {e}")

    # ------------------------------------------------------------------
    #  逐操作判卷：**下一次操作到达时，结算上一次操作**
    # ------------------------------------------------------------------
    #  一个"操作" = 模型做了一次实际动作（按 A/D/S/W，或者决定继续 N）。
    #  它算哪一类，**只看它到下一次操作之间发生了什么**（用户 2026-09-23 定义）：
    #
    #     操作1 --无碰撞、无追击--> 操作2      操作1 = 正确操作   （保存）
    #     操作2 --撞车 + 招来警察--> 操作3     操作2 = 瑕疵操作   （不保存，进事件待标注）
    #     操作3 --追击中但没再撞--> 操作4      操作3 = 正常操作   （保存，kind 标成 chase）
    #     操作4 --还没等到操作5就撞车被逮捕--  操作4 = 撞车被捕操作（不保存，进事件）
    #
    #  * 以前搞错的地方：撞车时把上一个操作直接 _drop_pending_correct() 丢掉了，
    #    所以"那个操作"既没算作正确、也没被标成瑕疵 —— 就是你说的
    #    「没有把那个操作当成瑕疵操作」。现在改成给它打标记，结算时再分类。
    #  * 「追击还在继续」不等于瑕疵操作。只有**撞车那一刻在生效的那个操作**
    #    才是瑕疵操作；追击期间做的操作只要之后没再撞车，就是正常操作。
    #  * 迟迟等不到下一个操作：超过 correct_settle_ms 按同一套规则兜底结算
    #    （不兜底的话锚点对应的帧会被环形缓冲挤掉，写出来是空窗口）。
    #  * 角色倒地(alive=0)期间不记：那是撞车后的趔趄画面，不是一个"操作"。
    #
    #  ★★ 2026-09-23 第二次修正（用户："碰撞被警察追时保存的是 WASD 不是 N"）：
    #      N（此刻不动）**不是"换道操作"**，它不背瑕疵/被捕的锅。
    #      · AWSD 走一条流水线：它才是"操作"，瑕疵/被捕一律算在最后那次 AWSD 头上；
    #      · N 走另一条流水线：只用来教模型"此刻该不动"，撞车/被捕时作废（不存），
    #        但不会给它记一个"瑕疵操作"。
    #      所以 N 不再结束 AWSD 的时间区间 —— 否则撞车时手上挂着的往往是 N，
    #      真正该负责的那次换道反倒被当成"正确操作"存进去了。
    #
    #  ★★ 2026-09-23 第三次修正：
    #      · 上面那 4 个"操作"都指 AWSD 换道，N 不算换道操作。
    #      · 瑕疵（撞车被警察追）一律记在最后那次 AWSD 头上，N 不背这个锅。
    #      · **唯一例外**：如果是 N 导致的正面撞车被捕，N 才被记成「撞车被捕操作」。
    #      · N 的正确样本要求：离前一次换道 ≥ n_mature_s，且之后 n_mature_s 内
    #        没有换道（"保存 N 后 2 秒触发 D，那就不用保存 N"）—— 所以 N 先养
    #        3 秒再存；期间来了换道 / 撞车 / 被捕就作废。
    # ------------------------------------------------------------------
    def _queue_correct(self, label: str, conf: float, now: float) -> None:
        if not self.args.keep_correct:
            return
        if self.chasing:
            return                                   # 倒地期间不记
        if not self._in_play():
            return                                   # 弹窗/菜单里预测出的动作不算"操作"
        # 最近一次真实动作（撞车时用来定位"到底是哪个操作"）
        self.last_action = (now, label, conf)

        if label == "N":
            # ---------- N：不是"换道操作"，走另一条流水线 ----------
            # 规则（用户 2026-09-23）：
            #   · N 只有**离前后两次 AWSD 换道都 ≥ n_mature_s(3s)** 才值得存 ——
            #     紧挨着换道的那几秒"不动"说明不了什么（"如果保存 N 后 2 秒触发 D，
            #     那就不用保存 N"）。
            #   · 所以 N 先挂 3 秒"养着"：期间来了换道 / 撞车 / 被捕就作废；
            #     养够 3 秒还没被打断，才存成"此刻不动是对的"样本。
            #   · N 不背瑕疵的锅；**只有"N 导致的正面撞车被捕"**才记成撞车被捕操作。
            # 另外 N 几乎每帧都在预测，还要限流 + 限比例，否则样本比例会失衡。
            if self.last_wasd is not None and (now - self.last_wasd[0]) < self.args.n_mature_s:
                return                                   # 离上一次换道太近，不值得存
            if self.pending_n is not None:
                if now - self.pending_n["t"] < self.args.n_mature_s:
                    return                               # 上一个 N 还在养着
                self._settle_n()                         # 养够了 -> 存
            if now - self.last_n_queued < self.args.correct_n_every_ms / 1000.0:
                return
            cap = max(8, int(self.n_correct_dir * self.args.correct_n_ratio))
            if self.n_correct_n >= cap:
                return
            self.last_n_queued = now
            self.pending_n = {"t": now, "label": "N", "conf": conf,
                              "collision": False, "arrest": False, "chase": False}
            self.last_op = (now, "N")
            return

        # ---------- AWSD：这才是"换道操作"，瑕疵/被捕的锅算在它头上 ----------
        # 换道来了：还没养够 n_mature_s 的 N 候选直接作废（"保存 N 后 2 秒触发 D
        # 就不用保存 N"）。作废原因要留痕，不然留痕里看不到 N 会以为规则丢了。
        if self.pending_n is not None:
            if now - self.pending_n["t"] < self.args.n_mature_s:
                pn = self.pending_n
                self.pending_n = None
                self.n_n_dropped += 1
                self.op_seq_n += 1
                self._note_op("N未采用", pn, False,
                              note=f"养了 {now - pn['t']:.1f}s 就来了换道「{label}」-> 作废")
            else:
                self._settle_n()
        if now - self.last_correct_queued < self.args.correct_every_ms / 1000.0:
            return
        self.last_correct_queued = now
        self.last_wasd = (now, label, conf)             # 最后一次换道（撞车时用它定位）
        self.last_op = (now, label)
        # ★ 只有**真的记下一个操作**时才结算上一个。
        #   被限流丢掉的那些决定不算"操作" —— 否则上一个操作会在下一帧就被结算掉，
        #   撞车根本来不及落在它头上（2026-09-23 实测：就是这样漏掉全部瑕疵操作的）。
        self._settle_pending()
        self.pending_correct = {
            "t": now, "label": label, "conf": conf,
            "collision": False,                         # 这个操作之后撞车了？
            "arrest": False,                            # 这个操作之后被逮捕了？
            "chase": self.cur_chase_start is not None,  # 做这个操作时警察已经在追？
        }

    def _flush_correct(self, now: float) -> None:
        """兜底结算。

        · 换道操作：等不到下一个换道时，超过 correct_settle_ms 就结算，
          否则锚点对应的帧会被环形缓冲挤掉，写出来是空窗口。
        · N 候选：养够 n_mature_s(3s) 还没被换道/撞车打断，就存成正确样本。
          （环形缓冲 5.8s，晚存 3 秒帧还在）
        """
        limit = self.args.correct_settle_ms / 1000.0
        pc = self.pending_correct
        if pc and now - pc["t"] > limit:
            # ★ 已经撞过车的操作**不要**兜底结算：瑕疵/被捕本来就不存样本，
            #   而"撞车被捕操作"要等本局结束(inPlay 1->0)才能判出来。
            #   一直挂着，等下一个操作或本局结束再结算。
            if not pc.get("collision"):
                self._settle_pending()
        pn = self.pending_n
        if pn and now - pn["t"] >= self.args.n_mature_s:
            self._settle_n()

    def _settle_n(self, as_arrest: bool = False) -> None:
        """结算一个 N 候选。

        · 撞过车 / 被捕过 -> 作废（不存，也**不记瑕疵**）
        · as_arrest=True 只从一处传进来："N 导致的正面撞车被捕" ——
          这是 N **唯一**会被记成「撞车被捕操作」的情况（用户 2026-09-23）
        · 否则（养够 3 秒且没被换道打断）-> 存成"此刻不动是对的"样本
        """
        pn = self.pending_n
        self.pending_n = None
        if pn is None:
            return
        if as_arrest:
            self._record_arrest_n(pn)
            return
        if pn.get("collision") or pn.get("arrest"):
            self.n_n_dropped += 1
            self.op_seq_n += 1
            self._note_op("N未采用", pn, False, note="养够之前撞车/被捕了 -> 作废")
            return
        self._write_correct(pn, "正常操作" if pn.get("chase") else "正确操作")
        self.op_seq_n += 1
        self._note_op("正常操作" if pn.get("chase") else "正确操作", pn, True,
                      note=f"N样本（离前后换道都 ≥{self.args.n_mature_s:.0f}s）")

    def _record_backfill_flaw(self, culprit: dict, hint: str) -> None:
        """撞车时没有挂着的换道操作 —— 把最后那次换道补记成瑕疵操作。

        它的样本窗口早被环形缓冲挤掉了（存不了训练样本），这里只补**记录**：
        保证"撞车次数"和"瑕疵操作次数"对得上，并在日志里说清这是补记的。
        """
        self.op_seq += 1
        self.n_flaw_ops += 1
        self.flaw_by_label[culprit["label"]] = self.flaw_by_label.get(culprit["label"], 0) + 1
        self._say(f"[判卷] 操作#{self.op_seq:05d} 「{culprit['label']}」{culprit['conf']:.2f} "
                  f"-> 瑕疵操作（撞车时它已结算；样本窗口过期，只补记录）"
                  f"  {hint}")
        rec = dict(culprit)
        rec["collision"] = True
        self._note_op("瑕疵操作", rec, False, note="补记：撞车时上次换道已被兜底结算")

    def _record_arrest_n(self, pn: dict) -> None:
        """N 导致的正面撞车被捕 —— N 唯一一次被记成「撞车被捕操作」。"""
        self.op_seq += 1
        self.n_arrest_ops += 1
        self.flaw_by_label["N"] = self.flaw_by_label.get("N", 0) + 1
        self._say(f"[判卷] 操作#{self.op_seq:05d} 「N」{pn['conf']:.2f} -> 撞车被捕操作"
                  "（本局结束时手上是 N：正面撞车、该换道没换）")
        self._note_op("撞车被捕操作", pn, False)

    def _mark_pending(self, **flags) -> None:
        """给"还没结算的操作"打标记（撞车 / 被逮捕）—— 换道操作和 N 都要打。"""
        for pc in (self.pending_correct, self.pending_n):
            if pc:
                pc.update(flags)

    def _culprit(self) -> Optional[dict]:
        """撞车那一刻在生效的**换道操作** —— 这次撞车算在哪个 AWSD 头上。

        ★ 一律返回 AWSD，绝不返回 N（用户 2026-09-23：碰撞被警察追时保存的是
          WASD 不是 N）。万一撞车时手上挂着的正好是 N，就回溯到最后一次换道。
        """
        pc = self.pending_correct
        if pc:
            return {"t": round(pc["t"], 4), "label": pc["label"], "conf": round(pc["conf"], 3)}
        if self.last_wasd:
            t, lb, cf = self.last_wasd
            return {"t": round(t, 4), "label": lb, "conf": round(cf, 3)}
        return None

    def _culprit_hint(self, culprit: Optional[dict], crash_t: float) -> str:
        """瑕疵操作一般是 AWSD 换道时机不准造成的（用户 2026-09-23）。

        换道动画大约 300ms（save_post_ms 就是按它定的），所以：
          按得太晚：< 300ms 就撞上 —— 动作还没做完
          时机不准：300~800ms  —— 早一点晚一点，最值得人工挑帧重新标
          按得太早：> 800ms    —— 换完道了还在这一道/又回到错的道上
        注意：culprit 一定是 AWSD（用户要求瑕疵算在换道头上，不算在 N 上）。
        """
        if not culprit:
            return "没记到撞车那一刻的换道操作（可能刚开局）"
        if not culprit:
            return "没记到撞车那一刻的操作（可能刚开局）"
        dt = (crash_t - culprit["t"]) * 1000.0
        lb = culprit["label"]
        if dt < 0:
            return "记录异常（操作时刻晚于撞车时刻）"
        if dt > 1500:
            return (f"撞车前 {dt / 1000:.1f}s 的最后一次换道「{lb}」-> 换完很久了，"
                    "多半是这一段路上没再换道")
        if dt < 300:
            return f"撞车前 {dt:.0f}ms 才按「{lb}」-> 换道太晚（动作还没做完就撞上）"
        if dt > 800:
            return f"撞车前 {dt:.0f}ms 就按了「{lb}」-> 换道太早（多半又回到错的道上）"
        return f"撞车前 {dt:.0f}ms 按「{lb}」-> 换道时机不准（最值得挑帧重标）"

    def _classify(self, pc: dict) -> str:
        """按用户定的规则给一个操作分类（只看它到下一个操作之间发生了什么）。"""
        if pc.get("arrest"):
            return "撞车被捕操作"
        if pc.get("collision"):
            return "瑕疵操作"
        if pc.get("chase"):
            return "正常操作"          # 追击还在继续，但之后没再撞车
        return "正确操作"

    def _settle_pending(self, why: str = "") -> None:
        """结算上一次操作：分类 -> 计数 -> 正确/正常才保存，瑕疵/被捕进事件。"""
        pc = self.pending_correct
        self.pending_correct = None
        if pc is None:
            return
        kind = self._classify(pc)
        self.op_seq += 1
        span = time.time() - pc["t"]
        # dry-run 是"只判卷不发键"的安全预演：分类照做，但不落盘样本
        save = kind in ("正确操作", "正常操作") and not self.args.dry_run
        what = "撞车" if pc.get("collision") else "无碰撞"
        extra = ("，被逮捕" if pc.get("arrest") else "") + ("，追击中" if pc.get("chase") else "")
        if save:
            tail = "-> 已存成正确样本"
        else:
            # ★ 别写"不保存"：瑕疵/被捕的数据**是存了的**，只是存成 events 下
            #   待标注事件（逐帧原图 + culprit），要靠 12_标注界面.py 挑帧重标后才
            #   变成 dataset迭代补采碰撞操作 里的样本。写"不保存"会让人以为丢了。
            tail = ("-> 不存进「正确操作」；已生成 events 待标注事件"
                    "（跑 12_标注界面.py 挑帧重标后进 碰撞操作）")
        line = (f"[判卷] 操作#{self.op_seq:05d} 「{pc['label']}」{pc['conf']:.2f} -> {kind}"
                f"（之后 {span:.1f}s {what}{extra}）{tail}")
        if kind == "正确操作":
            self.n_correct_clean += 1
        elif kind == "正常操作":
            self.n_correct_chase += 1
        elif kind == "瑕疵操作":
            self.n_flaw_ops += 1
        else:
            self.n_arrest_ops += 1
        if not save:      # 瑕疵/被捕：按"是哪个动作造成的"统计，看是不是某个方向特别差
            self.flaw_by_label[pc["label"]] = self.flaw_by_label.get(pc["label"], 0) + 1
        if save:
            self._write_correct(pc, kind)      # 它自己每 20 个报一次汇总
        else:
            self._say(line + (f"  {why}" if why else ""))
        self._note_op(kind, pc, save)

    def _note_op(self, kind: str, pc: dict, save: bool, note: str = "") -> None:
        """每次判卷记一行，方便事后核对分类对不对（runs 下的 操作判卷.jsonl）。

        ★ N 也要记！以前只记 AWSD，结果留痕里一行 N 都没有，看起来像"规则丢了"。
        """
        try:
            os.makedirs(os.path.dirname(self.ops_log_path), exist_ok=True)
            is_n = pc["label"] == "N"
            rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pc["t"])),
                   "seq": self.op_seq_n if is_n else self.op_seq,
                   "label": pc["label"], "conf": round(pc["conf"], 3),
                   "kind": kind, "collision": bool(pc.get("collision")),
                   "arrest": bool(pc.get("arrest")), "chase": bool(pc.get("chase")),
                   "saved": bool(save), "note": note}
            with open(self.ops_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _write_correct(self, pc: dict, kind: str = "正确操作") -> None:
        if self.n_correct >= self.args.correct_max > 0:
            return
        anchor = pc["t"]
        items = self.ring.window(anchor - self.cfg.save_pre_ms / 1000.0,
                                 anchor + self.cfg.save_post_ms / 1000.0)
        if len(items) < 5:
            return
        sel = _select_like_recorder(items, self.cfg, anchor)
        if sel is None:
            return
        times, frames, truncated = sel
        # tag 用来区分「干净的正确操作」和「追击中的正常操作」，以后训练可以按需过滤
        tag = "auto_correct" if kind == "正确操作" else "auto_correct_chase"
        try:
            meta = write_sample(self.correct_root, self.counters, self.cfg,
                                pc["label"], tag, "", anchor,
                                times, frames, 1.0 / max(self.step_s, 1e-6), truncated)
            with open(os.path.join(self.correct_root, "meta.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            self.n_correct += 1
            if pc["label"] == "N":
                self.n_correct_n += 1
            else:
                self.n_correct_dir += 1
            if self.n_correct % 20 == 0:
                self._say(f"[正确] 已保存 {self.n_correct} 个（方向 {self.n_correct_dir}"
                          f" / N {self.n_correct_n} / 其中追击中 {self.n_correct_chase}）")
        except Exception as e:
            self._say(f"[正确] 写样本失败: {e}")

    # ------------------------------------------------------------------
    def control_loop(self) -> None:
        args = self.args
        thr_default = float(self.thr.get("default", 0.85))
        status_t = time.time()
        while self.running:
            self._flush_correct(time.time())     # 到期的正确操作落盘
            x = self._input_tensor()
            if x is None:
                time.sleep(0.005)
                continue
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_half):
                    out = self.model(x)
                prob = F.softmax(out.float(), dim=1)[0]
            conf, idx = float(prob.max()), int(prob.argmax())
            label = self.labels[idx]
            self.ticks += 1
            now = time.time()

            if not self.playing:
                time.sleep(0.005)
                continue
            if now - self.last_send < self.min_gap_s:
                self.lockout_hits += 1
                time.sleep(0.005)
                continue
            if now < self.lockout_until.get(label, 0.0):
                self.lockout_hits += 1
                time.sleep(0.005)
                continue

            thr = float(self.thr.get(label, thr_default))
            sent = False
            if conf < thr:
                self.low_conf += 1
            elif label == "N":
                # N 也要记 —— 它是"这一刻不该动"的证据
                self._log_action(now, "N", conf, False)
                self._queue_correct("N", conf, now)
            else:
                if not args.dry_run:
                    self._send_key(label)
                sent = True
                self.sent += 1
                self.last_send = now
                self.lockout_until[label] = now + args.lockout_ms / 1000.0
                self._log_action(now, label, conf, True)
                self._queue_correct(label, conf, now)

            if time.time() - status_t > 1.0:
                print(f"\r[运行] 预测 {label} {conf:.2f}  发键 {self.sent}  "
                      f"撞车 {self.n_collision}  瑕疵 {self.n_flaw}  失误 {self.n_error}  "
                      f"正确样本 {self.n_correct}  锁定拦截 {self.lockout_hits}  ", end="")
                status_t = time.time()
            time.sleep(0.005)

    # ------------------------------------------------------------------
    #  状态辅助
    #  ★ 2026-09-23 按要求，把「被捕后自动点击复活 / 点 PLAY 重开」那一套
    #    代码彻底删掉了（_click_frac / _revive_worker / _try_revive /
    #    _on_game_over / _parse_xy / _popup_now，以及重复定义的那个
    #    _focus_window）。理由：
    #      · 全局鼠标点击会抢前台焦点，干扰你自己在模拟器上的操作；
    #      · 撞车/追击的判定完全靠内存边沿（alive 1->0、chasing），不依赖点击；
    #      · 继续留着只会让人以为"撞车没检测到是因为没复活"。
    #    这里只保留"读内存判断现在在不在关卡里"，给正确操作采集用。
    # ------------------------------------------------------------------
    def _in_play(self) -> bool:
        """真的在关卡里活着。不能只看 inPlay —— Continue 弹窗出现时 inPlay 仍然是 1，
        只看它会把"弹窗还开着"误判成"已经复活"，于是反复点击。"""
        st = self.hook.read() if self.hook else None
        if not st:
            return False
        return (int(st.get("inPlay", 0)) == 1
                and int(st.get("alive", 0)) == 1
                and int(st.get("paused", 0)) == 0)

    # ------------------------------------------------------------------
    #  找图复活（被捕后用找图点钥匙/PLAY；详见文件上半部分的说明）
    # ------------------------------------------------------------------
    def _init_revive(self) -> None:
        """准备好模板、整屏抓帧器和 adb。任何一步失败就自动关掉这个功能，不影响判卷。"""
        a = self.args
        if not a.revive:
            return
        if a.dry_run:
            a.revive = False
            print("[找图] dry-run：找图复活关闭（不会点击）")
            return
        if cv2 is None:
            a.revive = False
            print("[找图] 没有 cv2，找图复活关闭（pip install opencv-python）")
            return
        try:
            scales = list(np.arange(a.img_scale_min, a.img_scale_max + 1e-6, a.img_scale_step))
            self.m_key = _Matcher(_collect_templates(a.key_tpl, "key"), scales)
            self.m_play = _Matcher(_collect_templates(a.play_tpl, "play"), scales)
        except Exception as e:
            a.revive = False
            print(f"[找图] 模板加载失败：{e} -> 找图复活关闭")
            return
        cfg = self.cfg
        try:
            # ★ 必须另开一个「整块安卓画面」的抓帧器：
            #   训练用的 game_crop=(0,200,720,880) 把最下面 200 像素裁掉了，
            #   而 PLAY 就在 y≈1170，在裁剪图上永远找不到。
            self.cap_full = WindowCapture(
                cfg.window_keyword, cfg.game_res,
                (0, 0, int(cfg.game_res[0]), int(cfg.game_res[1])), backend=cfg.backend)
        except Exception as e:
            a.revive = False
            print(f"[找图] 开整屏抓帧失败：{e} -> 找图复活关闭")
            return
        self.adb = _AdbTap()
        serial = self.adb.pick()
        print(f"[找图] 钥匙模板 {self.m_key.names}")
        print(f"[找图] PLAY模板 {self.m_play.names}"
              f"   缩放 {a.img_scale_min}~{a.img_scale_max}（{len(scales)} 档）  阈值 {a.img_thr}")
        print(f"[找图] adb 设备 {serial if serial else '★没找到★（模拟器没开？）'}"
              f"   规则：钥匙 {a.key_window:.0f}s -> PLAY {a.play_window:.0f}s"
              f"   点完等 {a.verify_wait:.1f}s 确认，最多重试 {a.img_retries} 次")

    def _start_revive(self) -> None:
        """抓捕后被调用。已经在复活中就跳过（防重入）。"""
        if not self.args.revive or self._reviving:
            return
        if time.time() < self._revive_cooldown:
            return
        self._reviving = True
        threading.Thread(target=self._revive_worker, daemon=True).start()

    def _revive_worker(self) -> None:
        try:
            if self._img_find_click("钥匙", self.m_key, self.args.key_window):
                self.n_revive_key += 1
                return
            self._say("[找图] 钥匙没点成 -> 改找 PLAY")
            if self._img_find_click("PLAY", self.m_play, self.args.play_window):
                self.n_revive_play += 1
                return
            self._say("[找图] 钥匙和 PLAY 都没点成，等下一次被捕")
        except Exception as e:
            self._say(f"[找图] 出错: {e}")
        finally:
            self._reviving = False
            self._revive_cooldown = time.time() + self.args.revive_cooldown

    def _img_find_click(self, tag: str, matcher, window: float) -> bool:
        """在一个时间窗内反复「找 -> adb 点 -> 等 verify_wait 秒确认」。点成返回 True。"""
        a = self.args
        deadline = time.time() + window
        tries = 0
        n = 0
        while time.time() < deadline:
            if self._in_play():
                # 撞了一下但没被逮捕（只是招来警察），角色已经站起来了：
                # 别再耗完 5+4 秒 —— 那段时间会听不到下一次真正的被捕。
                self._say(f"[找图] {tag}: 角色已经回到关卡，这次不是被捕（虚惊）")
                return False
            frame = cv2.cvtColor(self.cap_full.grab(), cv2.COLOR_RGB2BGR)
            ok, hit = matcher.find(frame, a.img_thr)
            n += 1
            if a.img_verbose:
                self._say(f"[找图] {tag} 第{n}次 最高 {hit[0]:.3f} "
                          f"({matcher.last_name}, scale {hit[1]:.2f}) @{hit[2]}")
            if not ok:
                time.sleep(a.img_interval)
                continue

            x, y = hit[2]; w, h = hit[3]
            fx, fy = x + w / 2, y + h / 2
            ax = self.cap_full.crop[0] + fx       # 帧坐标 = 安卓坐标
            ay = self.cap_full.crop[1] + fy
            p = self._save_findimg(frame, matcher, hit, tag)
            self._say(f"[找图] 命中 {matcher.last_name} {hit[0]:.3f} -> adb tap 安卓"
                      f"({ax:.0f},{ay:.0f})   证据 {p}")
            ok, msg = self.adb.tap(ax, ay)
            if not ok:
                self._say(f"[找图] ⚠ adb 点击失败：{msg}")
            tries += 1

            # ★ 点完等一会儿，再回头确认到底点没点上
            time.sleep(a.verify_wait)
            f2 = cv2.cvtColor(self.cap_full.grab(), cv2.COLOR_RGB2BGR)
            ok2, h2 = matcher.find(f2, a.img_thr)
            gone = not ok2
            back = self._in_play()
            if gone or back:
                self._say(f"[找图] ✓ 点击生效（按钮得分 {h2[0]:.3f}"
                          + ("，已回到关卡" if back else "，按钮已消失") + "）")
                return True
            self._say(f"[找图] ⚠ 点了没反应（按钮还在，得分 {h2[0]:.3f}）"
                      + ("，重试…" if tries < a.img_retries else "，不再重试"))
            if tries >= a.img_retries:
                return False
        return False

    def _save_findimg(self, frame, matcher, hit, tag: str) -> str:
        """把命中时刻的截图（带红框）存下来当证据。"""
        try:
            os.makedirs(FINDIMG_DIR, exist_ok=True)
            vis = frame.copy()
            score, s, (x, y), (w, h) = hit
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(vis, f"{matcher.last_name} {score:.3f}", (max(0, x), max(24, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            p = os.path.join(FINDIMG_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.png")
            _imwrite_u(p, vis)
            return p
        except Exception:
            return "(证据图保存失败)"

    def _log_action(self, now: float, label: str, conf: float, sent: bool) -> None:
        """把动作记到所有还"活着"的事件上 —— 事件窗口盖住这一刻就带上它。"""
        with self._lock:
            for ev in self.pending:
                if not ev.dumped and abs(now - ev.anchor) <= self.pre_s + self.post_s + 1.0:
                    ev.actions.append((now, label, conf, sent))

    # ------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        print("\r" + " " * 100 + "\r" + msg)

    def run(self) -> None:
        print("[开始] 想停就按 Ctrl+C（控制台读不到按键，所以没做暂停键）")
        threading.Thread(target=self._capture_loop, daemon=True).start()
        if not self.args.no_hook:
            # 即使没连上也起这个线程 —— 它自己会重试，而不是安静地"判卷 0"
            threading.Thread(target=self._hook_loop, daemon=True).start()
        threading.Thread(target=self._dump_loop, daemon=True).start()
        # ★ 模型循环必须一直跑。旧写法是
        #       if --seconds: 干等 / else: 跑模型
        #   结果只要带了 --seconds，就变成"只录不玩"（模型一发键都不发）。
        #   2026-09-23 修掉：模型放后台线程，主线程只负责等停机条件。
        threading.Thread(target=self.control_loop, daemon=True).start()
        t0 = time.time()
        try:
            while self.running:
                if self.args.seconds and time.time() - t0 >= self.args.seconds:
                    print(f"\n[停止] 到达 --seconds {self.args.seconds:.0f}s")
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        self.running = False
        time.sleep(0.8)
        if self.pending_correct or self.pending_n:    # 最后一个操作 / N 也要结算
            self._settle_pending("收尾")
            self._settle_n()
        # 收尾：把还剩的事件落盘
        with self._lock:
            rest = [ev for ev in self.pending if not ev.dumped and ev.decided]
        for ev in rest:
            ev.dumped = True
            try:
                self._dump(ev)
            except Exception:
                pass
        if self.judge_bad_since is not None:
            self.judge_down_s += time.time() - self.judge_bad_since
            self.judge_bad_since = None
        print(f"\n[结束] 撞车 {self.n_collision}  瑕疵 {self.n_flaw}  失误 {self.n_error}  "
              f"正确样本 {self.n_correct}")
        if self.args.revive:
            print(f"[复活] 点钥匙 {self.n_revive_key} 次 / 点 PLAY {self.n_revive_play} 次"
                  f"（找图命中后 adb tap，点完 2 秒确认）")
        print(f"[判卷] 换道操作分类：正确 {self.n_correct_clean} / 正常(追击中) {self.n_correct_chase}"
              f" / 瑕疵 {self.n_flaw_ops} / 撞车被捕 {self.n_arrest_ops}"
              f"    明细 runs/操作判卷.jsonl")
        print(f"[判卷] 存盘样本：换道 {self.n_correct_dir} / N {self.n_correct_n}"
              f"（N 候写作废 {self.n_n_dropped} 个）"
              f"    N 要求离前后换道都 ≥{self.args.n_mature_s:.1f}s；"
              f"嫌 N 太少就把 --n-mature-s 调小（换道平均间隔只有 ~3s）")
        if self.n_flaw_unattr:
            print(f"[判卷] ⚠ 有 {self.n_flaw_unattr} 次撞车时还没有过任何换道操作（开局就撞），"
                  "没法归属到具体换道")
        if self.flaw_by_label:
            dist = "  ".join(f"{k}={v}" for k, v in
                             sorted(self.flaw_by_label.items(), key=lambda kv: -kv[1]))
            print(f"[判卷] 瑕疵/被捕 都是哪个动作造成的：{dist}"
                  f"    （A/D=左右换道  W=跳  S=滚；瑕疵只算换道动作，N 只在正面撞车被捕时背锅）")
        if not self.args.no_hook and (not self.hook_ever_ok or self.judge_down_s > 1.0):
            print("!" * 74)
            if not self.hook_ever_ok:
                print("⚠ 本次运行【全程没有内存判卷】—— 上面的 撞车/瑕疵/失误=0 不代表没撞车！")
            else:
                print(f"⚠ 判卷中断累计 {self.judge_down_s:.0f} 秒，"
                      f"这段时间的撞车/追击没有被记录。")
            print(f"  状态服务日志：{STATE_LOG}")
            print("!" * 74)
        n_ev = len([e for e in self.events if e.dumped])
        print("")
        print("=" * 74)
        print("数据是三段式的，本脚本只完成第 1 段：")
        try:
            _mp = os.path.join(self.collision_root, "meta.jsonl")
            n_lab = sum(1 for _ in open(_mp, encoding="utf-8")) if os.path.isfile(_mp) else 0
        except Exception:
            n_lab = 0
        print(f"  ① 本脚本          -> {self.out_root}")
        print(f"                       {n_ev} 个待标注事件（逐帧原图 + event.json）"
              f"，瑕疵/被捕都在这里；碰撞操作会话目前 {n_lab} 个样本")
        print("  ② 12_标注界面.py  -> " + self.collision_root)
        print("                       挑帧标 A/D/N/S/W 后，生成碰撞操作的训练样本")
        print("  ③ 5_训练模型.py    -> 把上面两个目录加进 DATA_DIRS 重新训练")
        print("")
        print("迭代产物全部在 dataset\\迭代补采\\ 下，和手动采集的会话分开，便于区分。")
        print("=" * 74)


# ==============================================================================
#  辅助
# ==============================================================================
def _init_session(root: str, cfg: Config) -> None:
    """建一个普通数据集会话（5_训练模型.py 能直接读）。"""
    os.makedirs(root, exist_ok=True)
    sp = os.path.join(root, "session.json")
    if not os.path.isfile(sp):
        from dataclasses import asdict
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump({"session": os.path.basename(root), "config": cfg.to_dict()
                       if hasattr(cfg, "to_dict") else asdict(cfg)}, fh, ensure_ascii=False, indent=2)
    mp = os.path.join(root, "meta.jsonl")
    if not os.path.isfile(mp):
        open(mp, "w", encoding="utf-8").close()
    for lb in cfg.labels:                      # write_sample 不建子目录
        os.makedirs(os.path.join(root, lb), exist_ok=True)


def _load_counters(root: str) -> Dict[str, int]:
    """接着已有会话的编号往后写，避免覆盖。"""
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
    return {lb: int(c.get(lb, 0)) for lb in Config().labels}


def _select_like_recorder(items, cfg: Config, anchor: float):
    """按采集器的规则从环形缓冲里均匀抽 saved_frames 帧（复刻 recorder._select_frames）。"""
    want = cfg.saved_frames
    t0 = anchor - cfg.save_pre_ms / 1000.0
    t1 = anchor + cfg.save_post_ms / 1000.0
    picked: List[Tuple[float, np.ndarray]] = []
    truncated = False
    for i in range(want):
        tg = t0 + (t1 - t0) * i / max(want - 1, 1)
        f = None
        for t, fr in items:
            if t <= tg:
                f = fr
            else:
                break
        if f is None:
            truncated = True
            f = items[0][1]
        picked.append((tg, f))
    return [t for t, _ in picked], [f for _, f in picked], truncated


def load_run(run_dir: str):
    """读 model_meta.json + best.pt + thresholds.json。"""
    meta_path = os.path.join(run_dir, "model_meta.json")
    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"找不到 {meta_path}（这个目录应该是训练产出目录，检查路径）")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 {ckpt_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    thr = {"default": 0.85}
    tp = os.path.join(run_dir, "thresholds.json")
    if os.path.isfile(tp):
        with open(tp, encoding="utf-8") as f:
            thr.update(json.load(f))
    return meta, ckpt_path, thr


def main() -> None:
    ap = argparse.ArgumentParser(description="迭代式行为克隆：让模型玩并把错误时刻录成待标注事件")
    ap.add_argument("--run", default=MODEL_DIR, help="模型目录；默认用文件开头 MODEL_DIR")
    ap.add_argument("--out", default=EVENT_DIR, help="待标注事件的输出目录")
    ap.add_argument("--pre-ms", type=int, default=3000, help="失误前保留多少毫秒（默认 3000）")
    ap.add_argument("--post-ms", type=int, default=400, help="失误后保留多少毫秒")
    ap.add_argument("--fps", type=float, default=40.0, help="抓帧频率")
    ap.add_argument("--dry-run", action="store_true", help="不发按键，只判卷")
    ap.add_argument("--seconds", type=float, default=0.0, help="跑多久自动停（0=一直跑）")
    ap.add_argument("--half", action="store_true", default=True, help="FP16 推理")
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--lockout-ms", type=float, default=320.0, help="同方向动作锁定")
    ap.add_argument("--min-gap-ms", type=float, default=80.0, help="全局最小发键间隔")
    ap.add_argument("--no-hook", action="store_true", help="不用内存判卷（那就只能手动看）")
    # ---------------- 找图复活 ----------------
    ap.add_argument("--no-revive", dest="revive", action="store_false", default=True,
                    help="关掉找图复活（默认开：被捕后找钥匙点复活，找不到就点 PLAY 重开）")
    ap.add_argument("--key-tpl", default="",
                    help="复活钥匙模板图，可逗号分隔多张；留空=自动用 templates/key*.png")
    ap.add_argument("--play-tpl", default="",
                    help="PLAY 模板图，可逗号分隔多张；留空=自动用 templates/play*.png")
    ap.add_argument("--img-thr", type=float, default=0.80, help="找图匹配阈值（默认 0.80）")
    ap.add_argument("--img-scale-min", type=float, default=0.85, help="找图最小缩放（默认 0.85）")
    ap.add_argument("--img-scale-max", type=float, default=1.25, help="找图最大缩放（默认 1.25）")
    ap.add_argument("--img-scale-step", type=float, default=0.05, help="找图缩放步长")
    ap.add_argument("--key-window", type=float, default=5.0, help="被捕后找钥匙的最长秒数（默认 5）")
    ap.add_argument("--play-window", type=float, default=4.0,
                    help="找不到钥匙后找 PLAY 的最长秒数（默认 4）")
    ap.add_argument("--img-interval", type=float, default=0.05, help="两次找图之间歇多久")
    ap.add_argument("--verify-wait", type=float, default=2.0,
                    help="点完等几秒回头确认点没点上（默认 2.0）")
    ap.add_argument("--img-retries", type=int, default=2, help="点不动时最多重点几次（默认 2）")
    ap.add_argument("--revive-cooldown", type=float, default=2.0,
                    help="一轮复活结束后多久内不再触发（默认 2 秒）")
    ap.add_argument("--img-verbose", action="store_true", help="找图时打印每次的得分")
    ap.add_argument("--n-mature-s", type=float, default=3.0,
                    help="N 正确样本要离前后换道都≥这么多秒才存（默认 3 秒）")
    # ---------------- 撞车信号 B：趔趄（被警察追上来但没被抓）----------------
    ap.add_argument("--stumble-safe-dist", type=float, default=20.0,
                    help="追击者距离≥这个值算甩开了，之后掉回来才认作又一次撞击（默认 20）")
    ap.add_argument("--stumble-hit-dist", type=float, default=14.0,
                    help="追击者距离≤这个值且刚从安全值掉下来 -> 判定撞击（默认 14）")
    ap.add_argument("--no-stumble-events", dest="stumble_events", action="store_false",
                    default=True,
                    help="趔趄撞车不落盘事件（只记撞车/瑕疵操作，省磁盘）")
    ap.add_argument("--keep-correct", dest="keep_correct", action="store_true", default=True,
                    help="收集正确操作（默认开）：下一次动作到来前没撞车，才算上一次正确")
    ap.add_argument("--no-keep-correct", dest="keep_correct", action="store_false",
                    help="不收集正确操作")
    ap.add_argument("--correct-settle-ms", type=float, default=4000.0,
                    help="等不到下一个动作时的兜底结算时限（必须小于环形缓冲）")
    ap.add_argument("--correct-max", type=int, default=0,
                    help="本次最多存多少个正确操作（0=不限）")
    ap.add_argument("--correct-dir", default=CORRECT_SESSION, help="正确操作样本写到哪个会话")
    ap.add_argument("--collision-dir", default=COLLISION_SESSION, help="碰撞操作样本会写到哪个会话")
    ap.add_argument("--correct-every-ms", type=float, default=400.0,
                    help="方向键正确样本最少间隔（默认 400ms）")
    ap.add_argument("--correct-n-every-ms", type=float, default=1500.0,
                    help="N 类正确样本最少间隔（默认 1500ms，比方向键慢得多）")
    ap.add_argument("--correct-n-ratio", type=float, default=0.5,
                    help="N 样本数最多是方向样本数的几倍（默认 0.5，即 N 不超过一半）")
    args = ap.parse_args()

    if args.keep_correct:
        print("=" * 74)
        print("⚠ --keep-correct 会把**模型自己认为对的动作**当成标签存进数据集。")
        print("  这是自我蒸馏：模型原有的偏差会被原样固化下来，越训越偏。")
        print("  只在你确认模型已经比较可靠、而且只是想扩量的时候才用。")
        print("  真正有价值的是【瑕疵/失误事件】里你**人工挑帧重新标**的样本。")
        print("=" * 74)

    c = IterCollector(args)
    try:
        c.run()
    finally:
        try:
            if c.hook:
                c.hook.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
