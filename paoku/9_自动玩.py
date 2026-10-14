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
    python 9_自动玩.py                        # 用下面 MODEL_DIR 里写死的模型
    python 9_自动玩.py --run runs/exp01        # 临时换成别的模型
    python 9_自动玩.py --dry-run               # 只打印不发键
    python 9_自动玩.py --calibrate             # 标定阈值（不按键）
    python 9_自动玩.py --list-runs             # 看有哪些模型可选

★ 换模型：直接改下面 MODEL_DIR 那一行，填训练结果目录的绝对路径。
"""


from __future__ import annotations

# ==============================================================================
#  ★★★  换模型就改这一行  ★★★
# ------------------------------------------------------------------------------
#  填「训练结果目录」的**绝对路径** —— 就是里面有 best.pt 和 model_meta.json
#  的那个目录（通常长这样： D:\...\paoku\runs\某个名字 ）
#
#  填 None  =  让它自动挑 runs/ 里「最近产出的」那个（省事，但不确定是哪个）
#
#  你现有的模型（想随时看完整列表： python 9_自动玩.py --list-runs）
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
# ==============================================================================
MODEL_DIR = r"D:\deeplearning\make_dataset\paoku\runs\20261011_204728_20261011_2134"


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

from collector import speed as speed_mod
from collector.attention import (AttentionProbe, cam_to_frame, colorize, overlay,
                                 put_text_cn)
from collector.capture import WindowCapture
from collector.config import Config, key_for_label
from collector.dataset import build_eval_transform, flatten_frames, to_video_tensor
from collector.model import build_model
from collector.recorder import RingBuffer, resolve_path
from collector.runs import list_runs, pick_run   # 自动挑选模型用

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
        m = self.meta
        self.labels = list(m["labels"])
        self.K = int(m["in_frames"])
        self.img_size = int(m.get("img_size", 224))
        print(f"[run] {os.path.abspath(run_dir)}")
        print(f"      类别 {self.labels}  输入 {self.K} 帧 ({self.K*3} 通道)  "
              f"test_acc {m.get('test_acc', float('nan')):.4f}")

        # 抓帧参数必须与训练一致
        cfg = Config()
        cfg.game_res = tuple(m.get("game_res", cfg.game_res))
        cfg.game_crop = tuple(m.get("game_crop", cfg.game_crop))
        cfg.frame_w = int(m.get("frame_w", cfg.frame_w))
        cfg.backend = m.get("backend", cfg.backend)
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

        self.cap = WindowCapture(cfg.window_keyword, cfg.game_res, cfg.game_crop, backend=cfg.backend,
                                 content_offset=self.content_offset)
        self.ring = RingBuffer(3.0)
        self.transform = build_eval_transform(self.img_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.K, len(self.labels), pretrained=False)
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
        self.cam_probe = None
        self.cam_model = None
        self._cam_lock = threading.Lock()
        self._cam_in = None          # 控制循环最新一次的模型输入
        self._cam_map = None         # 已映射回原始画面坐标的热力图 0~1
        self._cam_info = None        # (预测类别, 置信度, 逐帧重要度)
        self._last_full = None       # 抓帧线程最新一帧完整裁剪画面 (RGB)
        self._cam_done = 0
        self._display_done = threading.Event()
        self.video_path = None
        self._video_frames = 0
        if self.cam_on:
            self.cam_model = build_model(self.K, len(self.labels), pretrained=False)
            self.cam_model.load_state_dict(ck.get("model_state_dict", ck))
            self.cam_model = self.cam_model.to(self.device).eval().float()
            self.cam_probe = AttentionProbe(self.cam_model, frames=self.K)
            print(f"[cam] 热力图开启：另一份 FP32 模型，每 {self.cam_every} 帧算一次"
                  f"（约 {40.0/self.cam_every:.0f}Hz）；控制循环不受影响")
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
        self.lockout_hits = 0
        self.low_conf = 0
        self.last_speed = 0.0
        self.t_calib = []          # 标定用
        self.human_presses = []
        self.human_key_events = []      # [(时刻, 按的哪个键)]，标定时用来量"提前量"

        self.keyboard = None
        if not args.dry_run and not args.calibrate:
            from pynput.keyboard import Controller
            self.keyboard = Controller()
            self._focus_window()

    # ------------------------------------------------------------------
    def _focus_window(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes
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
        x = flatten_frames(self.transform(x)).unsqueeze(0)
        return x.to(self.device)

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
        elif ch and ch.lower() in "wasd":
            self.human_presses.append(time.time())
            self.human_key_events.append((time.time(), ch.lower()))

    def control_loop(self) -> None:
        args = self.args
        thr_default = float(self.thr.get("default", 0.85))
        status_t = time.time()
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
            self.ticks += 1
            now = time.time()

            if args.calibrate:
                self.t_calib.append((now, conf, label))
                if now - status_t > 2:
                    speed = self._live_speed()
                    print(f"\r[标定] 采样 {self.ticks}  当前 {label} {conf:.2f}  "
                          f"人按键 {len(self.human_presses)} 次  速度 {speed:.0f}px/s  ", end="")
                    status_t = now
                time.sleep(0.01)
                continue

            if not self.playing:
                time.sleep(0.005)
                continue
            # 全局最小间隔：防止同一瞬间连发（人手最快也就 150ms 左右）
            if now - self.last_send < self.min_gap_s:
                self.lockout_hits += 1
                time.sleep(0.005)
                continue
            # 同方向锁定：覆盖换道动画，防止同一个键在动画里重复触发；
            # 换个方向说明是新的决策，立刻放行（W->S 这种连按就靠这条）
            if now < self.lockout_until.get(label, 0.0):
                self.lockout_hits += 1
                time.sleep(0.005)
                continue

            thr = float(self.thr.get(label, thr_default))
            if conf < thr:
                self.low_conf += 1
            elif label == "N":
                pass                                  # N = 什么都不做
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
                print(f"\r[运行] 预测 {label} {conf:.2f}  {what} {self.sent}  "
                      f"锁定拦截 {self.lockout_hits}  低置信 {self.low_conf}  "
                      f"速度 {speed:.0f}px/s  ", end="")
                status_t = time.time()
            time.sleep(0.005)

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
        out = {"default": round(best_t, 2)}
        for lab in self.labels:
            sub = [r[0] for r in rows if r[2] == lab]
            if sub:
                out[lab] = round(min(0.99, max(0.5, best_t + 0.05)), 2)
        p = os.path.join(self.args.run, "thresholds.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"       已写出 {p}: {out}")
        print("       下次直接 --run 就会自动读它")
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
            print(f"              python 9_自动玩.py --dry-run --delay-ms {int(round(-med))}")
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
            except Exception as e:
                print(f"\n[cam] 热力图计算失败: {e}")
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
            put_text_cn(base, "热力图" if cm is not None else "热力图计算中...", (8, 4),
                        size=20, color=(0, 255, 255) if cm is not None else (170, 170, 170))
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
            print("热力图: 关闭（--cam 可打开）")
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
        if self.args.calibrate:
            self.report_calibration()
        what = "拟发键" if self.args.dry_run else "发键"
        print(f"\n结束：{what} {self.sent} 次，被动作锁定拦下 {self.lockout_hits} 次，"
              f"低置信丢弃 {self.low_conf} 次")
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
        1) 绝对路径   D:\\...\\paoku\\runs\\exp05
        2) 相对 paoku/  runs/exp05
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
    print("\n       看完整列表： python 9_自动玩.py --list-runs")
    raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="多帧模型部署（含动作锁定）")
    ap.add_argument("--run", default=MODEL_DIR,
                    help="训练结果目录。默认用本文件顶部 MODEL_DIR 里写死的那个；"
                         "命令行给这个参数可以临时覆盖（不影响代码里的值）")
    ap.add_argument("--dry-run", action="store_true", help="不发键，只打印预测")
    ap.add_argument("--game-res", default=None,
                    help="临时覆盖游戏分辨率（横屏用 1280,720），不改 model_meta")
    ap.add_argument("--game-crop", default=None,
                    help="临时覆盖裁剪框 x,y,宽,高（横屏建议 345,0,590,720，比例与 720:880 相同）")
    ap.add_argument("--content-offset", default=None,
                    help="安卓画面在窗口内的左上角偏移 x,y（默认自动算：水平居中+底部对齐）")
    ap.add_argument("--calibrate", action="store_true", help="标定阈值（不发键）")
    ap.add_argument("--lockout-ms", type=int, default=320,
                    help="同一方向的锁定时间，默认 320ms（覆盖换道动画）。"
                         "只锁同一个键：A 之后再按 A 要等，但 A 之后按 D/W/S 立刻放行"
                         "—— 否则 WA / WS / AA 这类连按会被全挡掉")
    ap.add_argument("--min-gap-ms", type=int, default=80,
                    help="任意两次发键之间的全局最小间隔，默认 80ms（防同一瞬间连发）")
    ap.add_argument("--half", action="store_true", default=True, help="FP16 推理")
    ap.add_argument("--no-half", dest="half", action="store_false")
    ap.add_argument("--delay-ms", type=int, default=0,
                    help="延迟取帧：模型看到的是 now-delay-ms 的画面，动作整体后移，"
                         "用来补偿模型\"动作偏早\"的系统偏差。"
                         "注意会拖坏连按：实测 delay>50ms 时 WA/WS 这类快连按识别率从 95.9%% 掉到 65%%（delay=81）")
    # ---------------- 注意力热力图 + 录像 ----------------
    ap.add_argument("--cam", dest="cam", action="store_true", default=True,
                    help="显示 Grad-CAM 实时热力图窗口（默认开启）")
    ap.add_argument("--no-cam", dest="cam", action="store_false",
                    help="关掉热力图，换最高帧率（热力图要跑反向传播，会抢 GPU）")
    ap.add_argument("--cam-every", type=int, default=4,
                    help="每 N 帧算一次热力图，默认 4（40fps 下约 10Hz）。"
                         "调小更流畅但更吃 GPU；控制循环本身不受影响")
    ap.add_argument("--no-cam-window", dest="cam_window", action="store_false",
                    default=True, help="不弹热力图窗口，但仍然录制视频")
    ap.add_argument("--video", default=None,
                    help="录像输出路径；不填自动生成 play_<时间>.mp4（放在 paoku/ 下）")
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
    if getattr(args, "list_runs", False):
        print("[runs] 可用的模型（最近产出的排前面）：")
        for p, f1, acc, mt, n_tr, ra, dg in list_runs():
            rel = os.path.relpath(p, os.path.dirname(os.path.abspath(__file__)))
            s0 = f"{ra*100:5.1f}%" if ra >= 0 else "   ?  "
            print(f"   --run {rel:<44} 训练 {n_tr:5d} 个  "
                  f"test_acc {acc:.4f}  macroF1 {f1:.4f}  真实acc {s0}")
        print("\n   例：python 9_自动玩.py --run "
              + os.path.relpath(list_runs()[0][0],
                                os.path.dirname(os.path.abspath(__file__))))
        return

    args.run = resolve_model_dir(args.run)
    Player(args).run()


if __name__ == "__main__":
    main()
