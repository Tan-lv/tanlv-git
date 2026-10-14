# -*- coding: utf-8 -*-
"""多帧数据采集器。

核心思路（解决原来的两个问题）
------------------------------------------------------------------
原来：按下按键 -> 立刻 adb screencap（100~300ms 延迟）-> 存图
      结果存下来的是"动作已经发生之后"的画面，标签语义是错的。

现在：一个独立线程以最高 ~80fps 持续抓帧写入环形缓冲（只留最近 N 秒，内存里）
      按键事件只记录"时间戳"，真正的取帧在事后从缓冲里回溯：
          输入帧 = [t_press - end_offset - (K-1)*dt, ..., t_press - end_offset]
      也就是说，模型看到的一定是"按键之前"的画面，而且是连续多帧。

      同时每个样本会把 [t_press - save_pre, t_press + save_post] 整段
      窗口存成一张横向"胶片条" JPEG，并把每帧的时间戳写进 meta.jsonl。
      这样以后想改 input_frames / end_offset 都不必重新采集，离线重切即可。

快捷键
------------------------------------------------------------------
  F8  : 暂停 / 继续 采集（过场、菜单、死亡时按一下）
  F9  : 开 / 关 自动补 N 类样本（默认关闭；开启时限速 + 避让 WASD 按键）
  F7  : 立刻把最近 3 秒导出一张调试胶片条到 debug/
  F10 : 退出（Esc 也可以）
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:  # Windows 控制台中文输出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .capture import WindowCapture
from .config import Config
from . import speed as speed_mod
from .gamestate import RunStateDetector
from .hookstate import HookStateClient, ensure_state_server

SEP_COLOR = (255, 255, 255)   # 胶片条分隔线颜色

# collector 包的上一级 = 项目根目录（paoku），用它来解析相对路径，
# 这样无论在 PyCharm 里工作目录设成什么，数据都落在 paoku/dataset 下
PAOKU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PAOKU_DIR, p)


# ======================================================================
#  环形缓冲
# ======================================================================
class RingBuffer:
    """保存最近 N 秒的 (时间戳, 帧) 对。"""

    def __init__(self, seconds: float, hard_limit: int = 4000):
        self.seconds = float(seconds)
        self.hard_limit = hard_limit
        self._dq: Deque[Tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()

    def append(self, t: float, frame: np.ndarray) -> None:
        with self._lock:
            self._dq.append((t, frame))
            cutoff = t - self.seconds
            while self._dq and self._dq[0][0] < cutoff:
                self._dq.popleft()
            while len(self._dq) > self.hard_limit:
                self._dq.popleft()

    def window(self, t0: float, t1: float) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            return [(t, f) for (t, f) in self._dq if t0 - 1e-6 <= t <= t1 + 1e-6]

    def frame_at_or_before(self, t: float):
        """取时间戳 <= t 的最近一帧；若都比 t 晚则退化为最早一帧。"""
        with self._lock:
            for ts, f in reversed(self._dq):
                if ts <= t + 1e-6:
                    return f
            return self._dq[0][1] if self._dq else None

    def latest(self, n: int = 1) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            return list(self._dq)[-n:]

    def oldest_time(self) -> Optional[float]:
        with self._lock:
            return self._dq[0][0] if self._dq else None

    def newest_time(self) -> Optional[float]:
        with self._lock:
            return self._dq[-1][0] if self._dq else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)


# ======================================================================
#  待落盘样本
# ======================================================================
@dataclass
class Pending:
    label: str
    anchor: float        # 按键（或自动 N）的时刻
    kind: str            # "key" | "auto_n" | "debug"
    ready_at: float      # 到这个时间之后窗口才完整
    src_key: str = ""


# ======================================================================
#  落盘（抽成独立函数，便于离线测试）
# ======================================================================
def write_sample(root: str, counters: Dict[str, int], cfg: Config,
                 label: str, kind: str, src_key: str, anchor: float,
                 times: List[float], frames: List[np.ndarray],
                 capture_fps: float, truncated: bool) -> dict:
    """把一帧序列拼成横向胶片条存盘，返回该样本的 meta 字典。"""
    n = len(frames)
    fw, fh, sep = cfg.frame_w, cfg.frame_h, cfg.sep
    width = n * fw + (n - 1) * sep
    strip = np.empty((fh, width, 3), dtype=np.uint8)
    strip[:, :, :] = SEP_COLOR
    for i, f in enumerate(frames):
        x = i * (fw + sep)
        strip[:, x:x + fw] = f

    cid = f"{label}_{counters[label]:06d}"
    rel = os.path.join(label, cid + ".jpg")
    path = os.path.join(root, rel)
    Image.fromarray(strip).save(path, "JPEG", quality=cfg.jpeg_quality,
                                subsampling=getattr(cfg, "jpeg_subsampling", 2))
    counters[label] += 1

    return {
        "id": cid, "label": label, "kind": kind, "src_key": src_key,
        "file": rel.replace("\\", "/"), "session": os.path.basename(root.rstrip("\\/")),
        "t_anchor": round(anchor, 6),
        "frame_times_ms": [round((t - anchor) * 1000.0, 2) for t in times],
        "n_frames": n, "frame_w": fw, "frame_h": fh, "sep": sep,
        "requested_pre_ms": cfg.save_pre_ms, "requested_post_ms": cfg.save_post_ms,
        "truncated_window": bool(truncated),
        "capture_fps": round(capture_fps, 1),
    }


# ======================================================================
#  采集器
# ======================================================================
class Recorder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ring = RingBuffer(cfg.ring_seconds)
        self.cap = WindowCapture(cfg.window_keyword, cfg.game_res, cfg.game_crop, backend=cfg.backend)
        print(f"[OK] 已绑定窗口 hwnd={self.cap.hwnd}  客户区={self.cap.client_w}x{self.cap.client_h}"
              f"  画面偏移=({self.cap.off_x},{self.cap.off_y})  裁剪={cfg.game_crop}")

        self.session = cfg.session_name or time.strftime("%Y%m%d_%H%M%S")
        self.root = os.path.join(resolve_path(cfg.output_root), self.session)
        self.counters = {lab: 0 for lab in cfg.labels}
        self.class_dirs = {}
        for lab in cfg.labels:
            d = os.path.join(self.root, lab)
            os.makedirs(d, exist_ok=True)
            self.class_dirs[lab] = d
        os.makedirs(os.path.join(self.root, "debug"), exist_ok=True)
        self.meta_path = os.path.join(self.root, "meta.jsonl")
        self._meta_fh = open(self.meta_path, "a", encoding="utf-8")
        with open(os.path.join(self.root, "session.json"), "w", encoding="utf-8") as f:
            json.dump({"config": cfg.to_dict(), "session": self.session,
                       "started_at": time.time(), "started_at_str": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, ensure_ascii=False, indent=2)

        self.pending: List[Pending] = []
        self.pending_lock = threading.Lock()
        # meta.jsonl 句柄 / self.written / self.counters 被四个线程共用：
        #   写线程(_flush)、键盘线程(_undo)、状态检测线程(_rollback)、退出(_finalize)
        # 必须串行化，否则会 flush 到别的线程已经关掉的句柄上。
        # 加锁顺序固定为 _meta_lock -> pending_lock，不会死锁。
        self._meta_lock = threading.RLock()
        self._closed = False

        self.running = True
        self.recording = True
        # 只有采集集合里包含 N 时才做自动补 N
        self.n_auto = bool(cfg.n_auto_enabled) and ("N" in cfg.labels)

        self._down_keys = set()
        self._last_press = {}          # label -> t
        self._last_dir_time = 0.0
        self._last_any_key_time = 0.0
        self._last_n_anchor = 0.0
        self._auto_n_cand: Optional[float] = None   # 已选中、还在"确认避让"期的自动 N 候选时刻

        self.written: List[dict] = []     # 本次会话已落盘的样本（撤销要用）
        self.undo_used = 0                # 当前这一串连按已经用了几次（撞车自动回滚不占这个额度）
        self._undo_last_time = 0.0        # 上一次成功撤销的时刻，用于"停手后重置额度"
        self.last_speed = 0.0

        # 游戏状态自动检测（开关录制）
        self.detector = None
        if cfg.auto_state:
            self.detector = RunStateDetector(
                on_threshold=cfg.state_motion_on, off_threshold=cfg.state_motion_off,
                on_ms=cfg.state_on_ms, off_ms=cfg.state_off_ms, start_state="idle")
        self.stat_state_changes = 0

        # IL2CPP 内存读取状态（比画面判断精确得多）
        self.hook = None
        self.state_running = False
        self.crash_time = 0.0
        self._last_alive = 1
        if getattr(cfg, "hook_state", False):
            ensure_state_server(cfg.hook_state_host, cfg.hook_state_port)
            self.hook = HookStateClient(cfg.hook_state_host, cfg.hook_state_port)
            if self.hook.wait_ready(6.0):
                st = self.hook.read() or {}
                print(f"[IL2CPP] 状态服务已连接  inst={st.get('inst')}  "
                      f"当前 inPlay={st.get('inPlay')} alive={st.get('alive')}")
                self.state_running = (st.get("inPlay") == 1 and st.get("alive") == 1)
            else:
                print(f"[警告] 连不上状态服务 {cfg.hook_state_host}:{cfg.hook_state_port}，回退到画面判断")
                print("       另开一个窗口跑: D:\pythonx\python.exe -u tools\state_server.py")
                self.hook = None
        self.stat_saved = 0
        self.stat_skipped = 0
        self.stat_truncated = 0
        self.stat_undone = 0
        self.stat_auto_n_dropped = 0     # 因为紧接着又出现按键而被丢掉的自动 N 候选
        self._freeze_run = 0
        self._freeze_warned = 0.0
        self._prev_frame = None

    @property
    def active(self) -> bool:
        """现在该不该记录：手动开关 F8 与自动状态检测都满足才行。"""
        if not self.recording:
            return False
        if self.hook is not None:
            return self.state_running
        if self.detector is None:
            return True
        return self.detector.is_running

    # ---------------- IL2CPP 状态轮询 ----------------
    def _poll_hook_state(self) -> None:
        while self.running:
            if self.hook is None:
                return
            st = self.hook.read()
            if st is None or not st.get("ok"):
                time.sleep(0.05)
                continue
            alive = st.get("alive", 1)
            now = time.time()

            if alive == 0 and self._last_alive == 1:
                self.crash_time = now
                print(f"\n[IL2CPP] 撞车！alive 1->0  (inPlay={st.get('inPlay')})")
            self._last_alive = alive

            running = (st.get("inPlay") == 1 and alive == 1)
            if running != self.state_running:
                self.state_running = running
                self.stat_state_changes += 1
                if running:
                    self.crash_time = 0.0
                    print(f"\n[IL2CPP] 开局 -> 自动开始记录  (paused={st.get('paused')})")
                else:
                    # 从撞车时刻往前多删 0.5 秒：连续失误的那几下也不该要
                    if self.crash_time:
                        cutoff = self.crash_time - 0.5
                    else:
                        cutoff = now - self.cfg.crash_rollback_ms / 1000.0
                    n = self._rollback(cutoff)
                    print(f"\n[IL2CPP] 本局结束 (inPlay={st.get('inPlay')} alive={alive}) -> 自动暂停")
                    if n:
                        print(f"         已删除最后一段 {n} 个样本"
                              f"（{'从撞车时刻往前 0.5s' if self.crash_time else str(self.cfg.crash_rollback_ms)+'ms'}）")
                    print("         重新开始游戏后会自动继续")
            time.sleep(0.02)

    # ---------------- 抓帧线程 ----------------
    def _capture_loop(self) -> None:
        interval = 1.0 / self.cfg.capture_max_fps if self.cfg.capture_max_fps > 0 else 0.0
        next_t = time.time()
        fh, fw = self.cfg.frame_h, self.cfg.frame_w
        while self.running:
            try:
                t = time.time()
                frame = self.cap.grab()
                small = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
                self.ring.append(t, small)
                self._check_quality(small)
                self._update_state(frame, t)
            except Exception as e:
                print(f"[ERR] 抓帧失败: {e}")
                time.sleep(0.5)
            if interval:
                next_t += interval
                sleep = next_t - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.time()

    def _check_quality(self, frame: np.ndarray) -> None:
        """冻结 / 黑屏检测。"""
        # 自动状态检测说"现在不在跑酷"时，画面静止是正常的，不该告警
        if self.detector is not None and not self.detector.is_running:
            self._freeze_run = 0
            self._prev_frame = frame
            return
        if frame.std() < self.cfg.black_std_threshold:
            if time.time() - self._freeze_warned > 10:
                print("[WARN] 画面几乎是纯色/黑屏，请确认游戏已经进入关卡")
                self._freeze_warned = time.time()
            return
        if self._prev_frame is not None and np.array_equal(frame, self._prev_frame):
            self._freeze_run += 1
            if self._freeze_run == self.cfg.freeze_warn_frames and time.time() - self._freeze_warned > 5:
                print(f"[WARN] 画面已经连续 {self._freeze_run} 帧完全相同 —— "
                      "模拟器可能卡死（或游戏停在暂停/菜单）。建议重启模拟器后按 F8 继续。")
                self._freeze_warned = time.time()
        else:
            self._freeze_run = 0
        self._prev_frame = frame

    def _update_state(self, frame, now: float) -> None:
        """跑酷中 <-> 静止 的迁移处理：开局自动记录，结束自动暂停并回滚。"""
        if self.detector is None:
            return
        prev = self.detector.state
        st = self.detector.update(frame, now)
        if st == prev:
            return
        self.stat_state_changes += 1
        m = self.detector.motion
        if st == "running":
            if not self.recording:
                print(f"\n[状态] 检测到开局（帧间差异 {m:.1f}），但你已按 F8 暂停")
            else:
                print(f"\n[状态] 检测到开局（帧间差异 {m:.1f}）→ 自动开始记录")
        else:
            cutoff = now - self.cfg.crash_rollback_ms / 1000.0
            n = self._rollback(cutoff)
            print(f"\n[状态] 本局结束（帧间差异 {m:.1f}）→ 自动暂停")
            if n:
                print(f"       已回滚最后 {self.cfg.crash_rollback_ms}ms 的 {n} 个样本"
                      f"（撞车 + 警察抓住那段，标签没有意义）")
            print("       重新开始游戏后会自动继续采集")

    # ---------------- 落盘线程 ----------------
    def _writer_loop(self) -> None:
        while self.running or self.pending:
            with self.pending_lock:
                item = self.pending[0] if self.pending else None
                if item is not None:
                    self.pending.pop(0)
            if item is None:
                time.sleep(0.01)
                continue
            now = time.time()
            if now < item.ready_at:
                with self.pending_lock:
                    self.pending.insert(0, item)
                time.sleep(min(0.02, item.ready_at - now))
                continue
            try:
                self._flush(item)
            except Exception as e:
                self.stat_skipped += 1
                print(f"[ERR] 写样本失败 ({item.label}): {e}")

    def _select_frames(self, items: List[Tuple[float, np.ndarray]],
                       t_start: float, t_end: float) -> Tuple[List[float], List[np.ndarray]]:
        """把窗口内实际抓到的帧, 重采样成固定数量的目标时刻。"""
        n = self.cfg.saved_frames
        targets = np.linspace(t_start, t_end, n)
        times = [it[0] for it in items]
        picked_t: List[float] = []
        picked_f: List[np.ndarray] = []
        j = 0
        last_idx = -1
        for tg in targets:
            while j + 1 < len(times) and times[j + 1] <= tg:
                j += 1
            idx = j if times[j] <= tg else 0
            if idx == last_idx:          # 抓帧率低于目标率时会出现重复, 去掉
                continue
            picked_t.append(times[idx])
            picked_f.append(items[idx][1])
            last_idx = idx
        return picked_t, picked_f

    def _flush(self, item: Pending) -> None:
        cfg = self.cfg
        t_start = item.anchor - cfg.save_pre_ms / 1000.0
        t_end = item.anchor + cfg.save_post_ms / 1000.0
        items = self.ring.window(t_start, t_end)
        if len(items) < 2:
            self.stat_skipped += 1
            return
        oldest = self.ring.oldest_time()
        truncated = oldest is not None and oldest > t_start + 0.05
        times, frames = self._select_frames(items, t_start, t_end)
        if len(frames) < cfg.input_frames:
            self.stat_skipped += 1
            return

        meta = write_sample(self.root, self.counters, cfg, item.label, item.kind,
                            item.src_key, item.anchor, times, frames,
                            self.cap.fps(), truncated)
        if cfg.record_speed:
            try:
                sp, resp = speed_mod.speed_px_per_sec(
                    frames, [(t - item.anchor) * 1000.0 for t in times], stride=2)
                meta["speed_px_s"] = round(sp, 1)
                meta["speed_conf"] = round(resp, 3)
                self.last_speed = sp
            except Exception:
                pass
        with self._meta_lock:
            if self._closed:
                return
            self.stat_saved += 1
            if truncated:
                self.stat_truncated += 1
            self.written.append(meta)
            self._meta_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            self._meta_fh.flush()

    # ---------------- 撤销 ----------------
    def _rewrite_meta(self) -> None:
        """按 self.written 重写 meta.jsonl（原地截断，不关文件）。

        以前是 close + 重新 open。但撞车自动回滚跑在状态检测线程、手动撤销跑在键盘线程，
        两个线程同时进来时，后一个会 flush 到已经关闭的句柄上，抛
        "ValueError: I/O operation on closed file"，把按键回调整个打崩。
        现在改成原地 seek(0) + truncate()，并且整段由 _meta_lock 串行化。
        """
        with self._meta_lock:
            if self._closed:
                return
            self._meta_fh.flush()
            self._meta_fh.seek(0)
            self._meta_fh.truncate()
            for m in self.written:
                self._meta_fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            self._meta_fh.flush()

    def _recount(self) -> None:
        """按保留下来的样本重算各类别下一个可用编号。（两个调用方都已持有 _meta_lock）"""
        nxt = {lab: 0 for lab in self.cfg.labels}
        for m in self.written:
            try:
                idx = int(m["id"].rsplit("_", 1)[1])
            except Exception:
                continue
            if m["label"] in nxt and idx + 1 > nxt[m["label"]]:
                nxt[m["label"]] = idx + 1
        self.counters = nxt

    def _rollback(self, cutoff: float) -> int:
        """删掉 t_anchor >= cutoff 的所有样本（含还没落盘的）。返回删掉的已落盘样本数。

        会被三个线程调用（撞车自动回滚 = 状态检测线程/抓帧线程，手动 X = 键盘线程），
        所以整个操作必须在 _meta_lock 下串行。
        """
        with self._meta_lock:
            return self._rollback_locked(cutoff)

    def _rollback_locked(self, cutoff: float) -> int:
        """_rollback 的实际实现（调用方必须已持有 _meta_lock）。"""
        with self.pending_lock:
            n_before = len(self.pending)
            self.pending = [p for p in self.pending if p.anchor < cutoff]
            dropped_pending = n_before - len(self.pending)
        self._auto_n_cand = None          # 回滚后旧的候选点已经作废

        victims = [m for m in self.written if m["t_anchor"] >= cutoff]
        if not victims and dropped_pending == 0:
            return 0

        for m in victims:
            fp = os.path.join(self.root, m["file"])
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception as e:
                print(f"   删除失败 {fp}: {e}")

        self.written = [m for m in self.written if m["t_anchor"] < cutoff]
        self._rewrite_meta()
        self._recount()
        self.stat_saved -= len(victims)
        self.stat_undone += len(victims)
        return len(victims)

    def _undo(self) -> None:
        # 和撞车自动回滚共用一把锁：两者都会改 self.written 并重写 meta.jsonl
        with self._meta_lock:
            self._undo_locked()

    def _undo_locked(self) -> None:
        cfg = self.cfg
        limit = cfg.undo_max_times
        now = time.time()

        # 这不是"整场总额度"，而是"当下这一串连按"的上限：
        # 距上次成功撤销超过 undo_reset_after_ms，额度就重新计满。
        cool = getattr(cfg, "undo_reset_after_ms", 5000) / 1000.0
        if self.undo_used and now - self._undo_last_time >= cool:
            if self.undo_used >= limit:   # 只有之前真被拦住过才提示，避免刷屏
                print(f"\n[撤销] 已停手 {now - self._undo_last_time:.0f} 秒，"
                      f"连按次数已重置（重新计满 {limit} 次）")
            self.undo_used = 0

        if self.undo_used >= limit:
            wait = max(0.0, cool - (now - self._undo_last_time))
            print(f"\n[撤销] 已经连按 {limit} 次了，先停一下（防止一口气删光）")
            print(f"       再等约 {wait:.0f} 秒，或者继续正常游玩，次数会自动恢复")
            return

        # 按个数删：取最近落盘的 undo_count 个（按采集时间倒序）。
        # 比这批更晚、还没落盘的 pending 也一起丢掉 —— 它们更"最近"。
        n_del = max(1, int(getattr(cfg, "undo_count", 5)))
        with self.pending_lock:
            if not self.written:
                print("\n[撤销] 已经没有样本可删了（已经退到本次采集的起点）")
                return
            order = sorted(range(len(self.written)),
                           key=lambda i: self.written[i]["t_anchor"])
            kill = set(order[-min(n_del, len(order)):])
            victims = [self.written[i] for i in sorted(kill)]
            oldest = min(m["t_anchor"] for m in victims)
            n_before = len(self.pending)
            self.pending = [p for p in self.pending if p.anchor < oldest]
            dropped_pending = n_before - len(self.pending)

        for m in victims:
            fp = os.path.join(self.root, m["file"])
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception as e:
                print(f"   删除失败 {fp}: {e}")

        self.written = [m for i, m in enumerate(self.written) if i not in kill]
        self._rewrite_meta()
        self._recount()
        self.stat_saved -= len(victims)
        self.stat_undone += len(victims)
        self.undo_used += 1                      # 只有真的删掉了才计数
        self._undo_last_time = now
        left = limit - self.undo_used
        extra = ("（另外丢掉 %d 个还没落盘的）" % dropped_pending) if dropped_pending else ""
        print(f"\n[撤销] 删掉 {len(victims)} 个样本{extra}；"
              f"最近一个是 {time.strftime('%H:%M:%S', time.localtime(oldest))}")
        print("       当前各类别: " + "  ".join(f"{k}:{v}" for k, v in self.counters.items()))
        if left > 0:
            print(f"       再按一次继续往前删 {n_del} 个（这一串还能按 {left} 次）")
        else:
            print(f"       已经连按 {limit} 次，先停手 {cool:.0f} 秒，次数会自动恢复")

    # ---------------- 键盘 ----------------
    def _enqueue(self, label: str, anchor: float, kind: str, src_key: str = "") -> None:
        ready = anchor + self.cfg.save_post_ms / 1000.0 + 0.005
        with self.pending_lock:
            self.pending.append(Pending(label, anchor, kind, ready, src_key))

    def on_press(self, key) -> None:
        k = getattr(key, "char", None)
        name = getattr(key, "name", None)
        keyid = (k.lower() if isinstance(k, str) else (name or str(key)))
        if keyid in self._down_keys:       # 长按重复触发, 忽略
            return
        self._down_keys.add(keyid)

        if name == "f10":
            print("\n[退出] 收到 F10")
            self.running = False
            return
        if name == "esc":
            print("\n[退出] 收到 Esc")
            self.running = False
            return
        if name == "f8":
            self.recording = not self.recording
            print(f"\n[切换] 采集 {'继续' if self.recording else '暂停'}")
            return
        if name == "f9":
            if "N" not in self.cfg.labels:
                print("\n[提示] 本次未采集 N 类，自动补 N 无效（改 config.py 里的 labels 可恢复）")
                return
            self.n_auto = not self.n_auto
            print(f"\n[切换] 自动补 N {'开启' if self.n_auto else '关闭'}")
            return
        if name == "f7":
            self._dump_debug()
            return
        if keyid == "x":
            self._undo()
            return

        if not self.active:
            return
        label = self.cfg.key_map.get(keyid)
        if label is None or label not in self.class_dirs:   # 未采集的类别直接忽略
            return
        now = time.time()
        debounce = self.cfg.key_debounce_ms / 1000.0
        if now - self._last_press.get(label, 0.0) < debounce:
            return
        self._last_press[label] = now
        self._last_dir_time = now
        self._last_any_key_time = now
        self._enqueue(label, now, "key", keyid)

    def on_release(self, key) -> None:
        k = getattr(key, "char", None)
        name = getattr(key, "name", None)
        keyid = (k.lower() if isinstance(k, str) else (name or str(key)))
        self._down_keys.discard(keyid)

    # ---------------- 自动补 N ----------------
    def _maybe_auto_n(self) -> None:
        """空闲时自动补 N 类样本（默认关闭，按 F9 可临时开）。

        两条"避让"规则，保证 N 不会采到 WASD 按键附近的那几帧：
          1) 候选点必须距离上一次按键至少 n_suppress_after_key_ms。
             N 样本实际喂给模型的是"该点往前 207ms ~ 40ms"的 6 帧，
             500ms 的避让能让这 6 帧整体落在换道动画结束之后。
          2) 候选点选中后先不落盘，再等 n_guard_after_key_ms 做确认。
             这期间如果又按了 WASD，说明刚才那一刻其实是"马上要按键"的前奏，
             正确标签是按键而不是"什么都不做"，直接丢掉。
        再加上 n_auto_interval_ms 限速，N 的密度从 2.5 个/秒降到约 0.7 个/秒。
        """
        if "N" not in self.cfg.labels:          # 本次不采集 N 类
            return
        if not (self.active and self.n_auto):   # 暂停 / 未开局 / 没开自动补 -> 候选全部作废
            self._auto_n_cand = None
            return

        now = time.time()

        # 1) 先结算已经攒够确认时间的候选
        cand = self._auto_n_cand
        if cand is not None:
            if now - cand < self.cfg.n_guard_after_key_ms / 1000.0:
                return                          # 还在确认期，等下一轮
            self._auto_n_cand = None
            if self._last_any_key_time > cand:
                # 候选点之后紧接着又按了键 -> 那是"要按键"的前奏，不是"什么都不做"
                self.stat_auto_n_dropped += 1
                return
            self._last_n_anchor = cand
            self._enqueue("N", cand, "auto_n")
            return

        # 2) 条件都满足，才起一个新的候选
        if now - self._last_dir_time < self.cfg.n_idle_after_ms / 1000.0:
            return
        if now - self._last_any_key_time < self.cfg.n_suppress_after_key_ms / 1000.0:
            return
        if now - self._last_n_anchor < self.cfg.n_auto_interval_ms / 1000.0:
            return
        self._auto_n_cand = now

    # ---------------- 调试导出 ----------------
    def _dump_debug(self) -> None:
        """把最近 3 秒导出成一张长胶片条，用来肉眼核对按键与画面的时序。"""
        now = time.time()
        items = self.ring.window(now - 3.0, now)
        if len(items) < 2:
            print("\n[调试] 缓冲里帧数不足")
            return
        frames = [f for _, f in items]
        fw, fh, sep = self.cfg.frame_w, self.cfg.frame_h, self.cfg.sep
        width = len(frames) * fw + (len(frames) - 1) * sep
        strip = np.empty((fh, width, 3), dtype=np.uint8)
        strip[:, :, :] = SEP_COLOR
        for i, f in enumerate(frames):
            x = i * (fw + sep)
            strip[:, x:x + fw] = f
        out = os.path.join(self.root, "debug", f"debug_{time.strftime('%H%M%S')}.jpg")
        Image.fromarray(strip).save(out, "JPEG", quality=self.cfg.jpeg_quality,
                                    subsampling=getattr(self.cfg, "jpeg_subsampling", 2))
        print(f"\n[调试] 已导出最近 3 秒 ({len(frames)} 帧) -> {out}")

    # ---------------- 状态打印 ----------------
    def _status_loop(self) -> None:
        ticks = 0
        live_speed = 0.0
        while self.running:
            time.sleep(1.0)
            if not self.running:
                break
            ticks += 1
            if self.cfg.record_speed and ticks % 2 == 0:      # 每 2 秒测一次实时速度
                recent = self.ring.latest(9)
                if len(recent) >= 3:
                    live_speed, _ = speed_mod.speed_px_per_sec(
                        [f for _, f in recent],
                        [t * 1000.0 for t, _ in recent], stride=1)
            counts = "  ".join(f"{k}:{v}" for k, v in self.counters.items())
            if self.hook is not None:
                hs = self.hook.read() or {}
                st_txt = (f"IL2CPP {'跑酷中' if self.state_running else '未开局'}"
                          f"(play{hs.get('inPlay','?')}/alive{hs.get('alive','?')})")
            elif self.detector is not None:
                st = "跑酷中" if self.detector.is_running else "静止  "
                st_txt = f"画面{st}({self.detector.motion:4.1f})"
            else:
                st_txt = "手动模式    "
            if not self.recording:
                rec_txt = "已按F8暂停"
            else:
                rec_txt = "采集中" if self.active else "等开局中"
            print(f"\r[{time.strftime('%H:%M:%S')}] 抓帧{self.cap.fps():5.1f}fps "
                  f"速度{live_speed:6.0f}px/s 状态{st_txt} {rec_txt}  "
                  f"{counts} 跳过{self.stat_skipped}  ", end="", flush=True)

    # ---------------- 主循环 ----------------
    def run(self) -> None:
        from pynput import keyboard
        cfg = self.cfg
        print("=" * 78)
        print(f"会话: {self.session}   输出目录: {os.path.abspath(self.root)}")
        print(f"抓帧后端: {cfg.backend}   保存窗口: -{cfg.save_pre_ms}ms ~ +{cfg.save_post_ms}ms "
              f"({cfg.saved_frames} 帧 @ {cfg.save_fps}fps, {cfg.frame_w}x{cfg.frame_h})")
        print(f"模型输入: 连续 {cfg.input_frames} 帧, 结束于按键前 {cfg.input_end_offset_ms}ms")
        print(f"采集类别: {list(cfg.labels)}")
        print(f"按键映射: {cfg.key_map}")
        # ★ 2026-10-11：历史累计 + 补采缺口（每个动作类建议 >= 200 个，
        #   低于这个量级模型学不出泛化，还会把类别权重带歪 —— 实测 E 只有 6 个时，
        #   自动类别权重算出 218 倍，模型直接塌成"全预测 E"）。
        TARGET_PER_CLASS = 200
        _acc = accumulate_counts(cfg.output_root, exclude=self.session)
        if _acc:
            print("历史累计（不含本次会话）: "
                  + "  ".join(f"{l}:{_acc.get(l, 0)}" for l in cfg.labels))
            _lack = [f"{l} 还差 {TARGET_PER_CLASS - _acc.get(l, 0)}" for l in cfg.labels
                     if l != "N" and _acc.get(l, 0) < TARGET_PER_CLASS]
            if _lack:
                print(f"补采建议（每个动作类 >= {TARGET_PER_CLASS} 个）: " + "，".join(_lack))
                print("          只想补某几个键： python 2_采集数据.py --labels Q")
            else:
                print(f"每个动作类都 >= {TARGET_PER_CLASS} 个了，可以开始训练")
        else:
            print(f"提示：每个动作类建议采到 >= {TARGET_PER_CLASS} 个再训练"
                  "（Q 这类稀有键可以单独补： python 2_采集数据.py --labels Q）")
        if "N" in cfg.labels:
            n_keys = [k for k, v in cfg.key_map.items() if v == "N"]
            if self.n_auto:
                print(f"N 类采集方式: 自动补（空闲 {cfg.n_idle_after_ms}ms 以上才采，"
                      f"两次间隔 >= {cfg.n_auto_interval_ms}ms；"
                      f"WASD 按键后避让 {cfg.n_suppress_after_key_ms}ms、"
                      f"按键前避让 {cfg.n_guard_after_key_ms}ms）")
            elif n_keys:
                print(f"N 类采集方式: 手动 —— 按 [{' / '.join(n_keys)}] 键标记"
                      f"「此刻该什么都不做」")
            else:
                print("N 类采集方式: key_map 里没有映射 N，N 类将采不到！")
        if cfg.auto_state:
            print(f"状态自动检测: 开启   开局 差异>{cfg.state_motion_on} 持续{cfg.state_on_ms}ms   "
                  f"结束 差异<{cfg.state_motion_off} 持续{cfg.state_off_ms}ms   "
                  f"结束回滚 {cfg.crash_rollback_ms}ms")
            print("              只在跑酷中记录；一局结束会自动暂停并删掉最后几秒（撞车+警察那段）")
        else:
            print("状态自动检测: 关闭（纯手动 F8）")
        print(f"快捷键: F8 暂停/继续   X 撤销最近 {cfg.undo_count} 个样本"
              f"（连按最多 {cfg.undo_max_times} 次，停手 "
              f"{cfg.undo_reset_after_ms // 1000} 秒后重新计满）"
              f"   F7 导出调试条   F9 自动N开关   F10/Esc 退出")
        if not self.n_auto and any(v == "N" for v in cfg.key_map.values()):
            print("        采 N 的时候：按键盘上的 N 键（游戏不会响应，只用来打标记）")
        print("=" * 78)

        threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._writer_loop, daemon=True),
            threading.Thread(target=self._status_loop, daemon=True),
        ]
        for t in threads:
            t.start()

        if self.hook is not None:
            threading.Thread(target=self._poll_hook_state, daemon=True).start()

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        listener_thread = threading.Thread(target=self._auto_loop, daemon=True)
        listener_thread.start()

        try:
            while self.running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[退出] Ctrl+C")
            self.running = False

        listener.stop()
        time.sleep(0.4)          # 给写线程把剩余样本落盘
        self._finalize()

    def _auto_loop(self) -> None:
        while self.running:
            self._maybe_auto_n()
            time.sleep(0.02)

    def _finalize(self) -> None:
        with self._meta_lock:
            self._closed = True      # 之后迟到的 _flush / _rewrite_meta 直接跳过
            self._meta_fh.flush()
            self._meta_fh.close()
        self.cap.close()
        if self.hook is not None:
            self.hook.close()
        summary = {
            "session": self.session,
            "saved": self.stat_saved, "skipped": self.stat_skipped,
            "undone": self.stat_undone,
            "state_changes": self.stat_state_changes,
            "auto_state": self.cfg.auto_state,
            "truncated_window": self.stat_truncated,
            "auto_n_dropped": self.stat_auto_n_dropped,
            "per_class": self.counters,
            "avg_capture_fps": round(self.cap.fps(), 2),
            "finished_at": time.time(),
            "finished_at_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(self.root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 78)
        print(f"采集结束: 保存 {self.stat_saved} 个样本 -> {os.path.abspath(self.root)}")
        print("   各类别: " + "  ".join(f"{k}:{v}" for k, v in self.counters.items()))
        print(f"   跳过: {self.stat_skipped}   窗口被截断: {self.stat_truncated}")
        if self.stat_auto_n_dropped:
            print(f"   自动 N 候选被丢弃（确认期内又按了键）: {self.stat_auto_n_dropped}")
        print("=" * 78)


def accumulate_counts(output_root: str, exclude: str = "") -> dict:
    """\u7edf\u8ba1 output_root \u4e0b\u6240\u6709\u4f1a\u8bdd\uff08\u9664\u5f53\u524d\u4f1a\u8bdd\u3001\u8df3\u8fc7 _ \u5f00\u5934\u7684\uff09\u7684\u7d2f\u8ba1\u7c7b\u522b\u6837\u672c\u6570\u3002

    \u2605 2026-10-11\uff1aE/Q \u8fd9\u7c7b\u7a00\u6709\u52a8\u4f5c\u5728\u81ea\u7136\u6e38\u73a9\u91cc\u672c\u6765\u5c31\u5f88\u5c11\u89e6\u53d1\uff0c\u9760\u611f\u89c9\u91c7\u4f1a\u4e00\u76f4\u592a\u5c11\u3002
    \u542f\u52a8\u65f6\u628a\u300c\u5386\u53f2\u7d2f\u8ba1 + \u8fd8\u5dee\u591a\u5c11\u300d\u6253\u51fa\u6765\uff0c\u8865\u91c7\u65f6\u5fc3\u91cc\u6709\u6570\u3002
    """
    base = resolve_path(output_root)
    out: dict = {}
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        if name.startswith((".", "_")) or name == exclude:
            continue
        mp = os.path.join(base, name, "meta.jsonl")
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        lab = json.loads(line).get("label")
                    except Exception:
                        continue
                    if lab:
                        out[lab] = out.get(lab, 0) + 1
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="地铁跑酷多帧数据采集器")
    ap.add_argument("--config", default=None, help="JSON 配置文件（覆盖默认参数）")
    ap.add_argument("--session", default=None, help="会话名，默认用时间戳")
    ap.add_argument("--root", default=None, help="输出根目录，默认 dataset")
    ap.add_argument("--note", default=None, help="本次采集备注")
    ap.add_argument("--backend", default=None, choices=["printwindow", "bitblt"])
    ap.add_argument("--hook-state", action="store_true",
                    help="用 IL2CPP 内存读取判断状态（需先跑 tools/state_server.py）")
    ap.add_argument("--hook-state-port", type=int, default=None, help="状态服务端口，默认 47800")
    ap.add_argument("--no-auto-state", action="store_true",
                    help="关闭游戏状态自动检测，回到纯手动 F8 模式")
    ap.add_argument("--motion-on", type=float, default=None, help="开局判定的帧间差异阈值（默认 8.0）")
    ap.add_argument("--motion-off", type=float, default=None, help="结束判定的帧间差异阈值（默认 3.0）")
    ap.add_argument("--rollback-ms", type=int, default=None, help="一局结束时回滚多少毫秒（默认 4000）")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="只采集这些类别，例如 --labels A D S W（默认就是 W A S D，不含 N）")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.session:
        cfg.session_name = args.session
    if args.root:
        cfg.output_root = args.root
    if args.note:
        cfg.note = args.note
    if args.backend:
        cfg.backend = args.backend
    if args.labels:
        cfg.labels = tuple(args.labels)
    if args.hook_state:
        cfg.hook_state = True
    if args.hook_state_port is not None:
        cfg.hook_state_port = args.hook_state_port
    if args.no_auto_state:
        cfg.auto_state = False
    if args.motion_on is not None:
        cfg.state_motion_on = args.motion_on
    if args.motion_off is not None:
        cfg.state_motion_off = args.motion_off
    if args.rollback_ms is not None:
        cfg.crash_rollback_ms = args.rollback_ms
    cfg.validate()

    Recorder(cfg).run()


if __name__ == "__main__":
    main()
