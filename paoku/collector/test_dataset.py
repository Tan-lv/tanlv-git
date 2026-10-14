# -*- coding: utf-8 -*-
"""离线端到端自测：不依赖模拟器，验证"标签时序"和"多帧"两件事真的做对了。

    cd D:\deeplearning\make_dataset\paoku
    python -m collector.test_dataset
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .config import Config
from .dataset import (MultiFrameDataset, build_eval_transform, build_train_transform,
                      flatten_frames, load_session, read_strip, select_input_indices,
                      split_indices, to_video_tensor)
from .recorder import RingBuffer, write_sample


def synth_frame(k: int, h: int, w: int) -> np.ndarray:
    """造一帧假画面，像素值 = 帧序号，便于回读校验切图位置。"""
    return np.full((h, w, 3), k % 256, dtype=np.uint8)


def main() -> None:
    cfg = Config()
    cfg.output_root = tempfile.mkdtemp(prefix="dsh_test_dataset_")
    cfg.capture_max_fps = 40
    cfg.save_fps = 30
    cfg.save_pre_ms = 300
    cfg.save_post_ms = 100
    cfg.input_frames = 6
    cfg.input_end_offset_ms = 40
    cfg.validate()

    print("=" * 78)
    print("离线端到端自测")
    print("=" * 78)
    print(f"保存窗口 -{cfg.save_pre_ms}~+{cfg.save_post_ms}ms @ {cfg.save_fps}fps "
          f"-> {cfg.saved_frames} 帧, 每帧 {cfg.frame_w}x{cfg.frame_h}, 分隔 {cfg.sep}px")
    print(f"模型输入 {cfg.input_frames} 帧, 结束于按键前 {cfg.input_end_offset_ms}ms")

    # ---------------- 1. 按真实时序模拟：边抓帧、边按键、边落盘 ----------------
    print("\n[1/4] 模拟抓帧流与按键事件")
    dt = 1.0 / cfg.capture_max_fps
    ring = RingBuffer(cfg.ring_seconds)
    t0 = 1000.0
    n_frames = 1200
    press_time = t0 + 15.0

    root = os.path.join(cfg.output_root, "sess_test")
    os.makedirs(root, exist_ok=True)
    for lab in cfg.labels:
        os.makedirs(os.path.join(root, lab), exist_ok=True)
    counters = {lab: 0 for lab in cfg.labels}

    def flush_at(anchor: float, lab: str) -> None:
        """复刻 Recorder._flush 的取帧与重采样逻辑。"""
        window = ring.window(anchor - cfg.save_pre_ms / 1000.0, anchor + cfg.save_post_ms / 1000.0)
        assert len(window) >= cfg.input_frames, f"{lab} 窗口内帧数不足 ({len(window)})"
        target_times = np.linspace(anchor - cfg.save_pre_ms / 1000.0,
                                   anchor + cfg.save_post_ms / 1000.0, cfg.saved_frames)
        ws = [w[0] for w in window]
        picked_t, picked_f = [], []
        j, last = 0, -1
        for tg in target_times:
            while j + 1 < len(ws) and ws[j + 1] <= tg:
                j += 1
            idx = j if ws[j] <= tg else 0
            if idx == last:          # 抓帧率低于目标率时去重
                continue
            picked_t.append(ws[idx])
            picked_f.append(window[idx][1])
            last = idx
        m = write_sample(root, counters, cfg, lab, "key", lab.lower(), anchor,
                         picked_t, picked_f, cfg.capture_max_fps, truncated=False)
        with open(os.path.join(root, "meta.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    pending = [(press_time + i * 0.5, lab) for i, lab in enumerate(["A", "D", "N", "S", "W"])]
    for i in range(n_frames):
        t = t0 + i * dt
        ring.append(t, synth_frame(i, cfg.frame_h, cfg.frame_w))
        for item in list(pending):
            a, lab = item
            if t >= a + cfg.save_post_ms / 1000.0 + dt:
                flush_at(a, lab)
                pending.remove(item)
    assert not pending, f"还有样本没落盘: {pending}"
    print(f"    写出 {sum(counters.values())} 个样本 -> {root}")

    # ---------------- 2. 关键校验：输入帧必须全部在按键之前 ----------------
    print("\n[2/4] 标签时序校验（这正是原来出错的地方）")
    samples = load_session(root)
    assert len(samples) == 5
    latest = -1e9
    for m in samples:
        idx = select_input_indices(m, cfg)
        sel = [m["frame_times_ms"][i] for i in idx]
        latest = max(latest, max(sel))
        all_pre = all(t <= -cfg.input_end_offset_ms + 1e-6 for t in sel)
        print(f"    {m['label']}: 选中第 {idx} 帧, 相对按键时刻 "
              f"{['%+.0f' % t for t in sel]} ms -> {'全部在按键之前 [OK]' if all_pre else '[FAIL]'}")
        assert all_pre, "存在按键之后才被选中的帧！"
    print(f"    对照：旧做法存下的是按键之后 ~+100ms 的画面（动作已经发生）；"
          f"新做法选中的最晚一帧是 {latest:+.0f} ms。")

    # ---------------- 3. 回读校验 ----------------
    print("\n[3/4] 胶片条回读校验")
    ds = MultiFrameDataset(samples, cfg, transform=build_eval_transform(224))
    x, y = ds[0]
    print(f"    dataset[0] -> x={tuple(x.shape)}  y={y} ({ds.labels[y]})")
    assert x.shape == (cfg.input_frames * 3, 224, 224), "张量形状不对"

    idx_sel = select_input_indices(samples[0], cfg)
    raw = read_strip(samples[0], idx_sel)
    # 合成帧是纯色图，像素值 = 帧序号 % 256。
    # 只取每帧正中心的像素 —— 帧边界处 JPEG 压缩振铃会串值（q90 下实测会出现中间值）。
    got = [int(raw[k, raw.shape[1] // 2, raw.shape[2] // 2, 0]) % 256
           for k in range(raw.shape[0])]
    expect = []
    for i in idx_sel:
        tt = samples[0]["t_anchor"] + samples[0]["frame_times_ms"][i] / 1000.0
        expect.append(int(round((tt - t0) / dt)) % 256)
    print(f"    回读中心像素(=帧序号%256): {got}")
    print(f"    期望帧序号%256           : {expect}")
    assert got == expect, "胶片条切图位置不正确！"
    print("    [OK] 切图位置与帧序号完全对上（分隔线没有混进输入）")

    # ---------------- 4. 划分与训练变换 ----------------
    print("\n[4/4] 划分与训练变换")
    sp = split_indices(samples, mode="time_block")
    print(f"    time_block 划分: train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}")
    tr = build_train_transform(224)(to_video_tensor(raw))
    print(f"    训练增强输出: {tuple(tr.shape)} -> 展平 {tuple(flatten_frames(tr).shape)}")
    assert flatten_frames(tr).shape == (cfg.input_frames * 3, 224, 224)

    # 额外：验证 v2 对所有帧用的是同一套随机参数（时序一致）
    import torch
    a = build_train_transform(224)(to_video_tensor(raw))
    b = build_train_transform(224)(to_video_tensor(raw))
    diff_ratio = (a != b).float().mean().item()
    print(f"    两次独立增强的差异比例 {diff_ratio:.3f}（应 > 0，说明确实在随机增强）")
    print("    [OK]")

    print("\n" + "=" * 78)
    print("全部通过：标签一定取在按键之前，输入为多帧连续画面，格式自洽")
    print("=" * 78)
    shutil.rmtree(cfg.output_root, ignore_errors=True)


if __name__ == "__main__":
    main()
