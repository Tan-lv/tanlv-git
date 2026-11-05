# -*- coding: utf-8 -*-
"""多帧训练实现（本机单机训练，不联网、不依赖任何服务器）。

这是全项目唯一的训练实现，4_训练.py 只是它的中文入口。

与旧 trian.py 的区别
  * 输入是连续多帧（通道维早融合），不是单帧
  * 按采集时间分块划分，避免相邻帧泄漏到不同集合
  * 输出每类 precision/recall/F1 与混淆矩阵（旧版从来没看过这些）
  * AMP 混合精度 + 断点续训 + 完整 checkpoint（含部署所需的全部参数）

通常不用直接运行这个文件 —— 用中文入口更省事（会自动补路径）：
    python 4_训练.py
    python 4_训练.py --epochs 25 --patience 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import fields

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

# 本文件在 core/ 里，但 gamer_ai/ 才是包的根目录（core.xxx 要能 import 到）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import Config
from core.geometry import (GEOMETRY_NAME, crop_from_rel, load_geometry,
                                make_geometry, rel_from_crop)
from core.dataset import (IMAGENET_MEAN, IMAGENET_STD, GroupShuffleSampler,
                               MultiFrameDataset,
                               build_eval_transform, build_train_transform,
                               load_sessions, negative_breakdown,
                               negatives_for_subset, split_indices)
from core import speed as speed_mod


# ======================================================================
class Tee:
    """同时写到控制台和日志文件。"""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def make_amp(enabled: bool):
    """兼容不同 torch 版本的 AMP。返回 (scaler, autocast_ctx_factory)。"""
    if not enabled:
        import contextlib
        return None, (lambda: contextlib.nullcontext())
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        return scaler, (lambda: torch.amp.autocast("cuda", dtype=torch.float16))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        return scaler, (lambda: torch.cuda.amp.autocast())


# ======================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="地铁跑酷 多帧行为克隆训练")
    p.add_argument("--sessions", nargs="+", required=True, help="数据集会话目录（可多个）")
    p.add_argument("--out", required=True, help="输出目录（checkpoint / 日志 / 指标）")
    p.add_argument("--game", default=None,
                   help="游戏名（games/<名字>.py）：用它的按键表，并写进 model_meta.json —— "
                        "部署(9_)按这个发键。不给就用 session.json 里采集时记的那份")
    p.add_argument("--keys", default=None,
                   help="临时指定按键表（覆盖游戏文件/采集记录）：w,s,a,d,q,n 或 w=W,n=N")
    p.add_argument("--pretrained-weights", default=None, help="ImageNet 预训练权重文件（离线加载）")
    p.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--input-frames", type=int, default=None, help="覆盖采集时配置的输入帧数")
    p.add_argument("--end-offset", type=int, default=None, help="覆盖：输入窗口结束于按键前多少 ms")
    p.add_argument("--input-stride", type=int, default=None)
    p.add_argument("--frame-w", type=int, default=None, help="覆盖读取时的帧宽（一般不用改）")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader 进程数。每个 worker 光 import torch 就要约 600MB，"
                        "开太多会把内存吃光（实测 4 和 8 的加载速度几乎一样，别贪多）")
    p.add_argument("--no-group-shuffle", dest="group_shuffle", action="store_false",
                   default=True,
                   help="关闭按胶片条分组打乱（默认开启；开着能省一半 JPEG 解码）")
    p.add_argument("--prefetch", type=int, default=1,
                   help="每个 worker 预取几批。内存占用 ≈ workers x prefetch x batch_size "
                        "x 3.6MB，调太大（比如 12x4x64）会直接把内存打爆")
    p.add_argument("--patience", type=int, default=15, help="val_acc 不提升多少轮后早停")
    p.add_argument("--split-mode", default="by_session",
                   choices=["time_balanced", "time_block", "random"],
                   help="time_balanced=按时间段均衡（推荐）；time_block=纯时间分块；random=旧做法（会泄漏）")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--boundary-gap", type=int, default=5, help="切分边界丢弃的样本数（仅 time_block 用）")
    p.add_argument("--segment-seconds", type=float, default=15.0,
                   help="time_balanced 模式下每个时间段的秒数；段是划分的最小单位")
    p.add_argument("--source-weight", default=None,
                   help='来源加权，形如 "旧会话名=1.0,迭代数据=2.0"：'
                        "按会话路径子串匹配，乘以该样本的损失权重。"
                        "迭代微调时给新数据/人工纠正的数据更高权重，又不用丢掉旧数据")
    p.add_argument("--class-weight", default="auto", choices=["auto", "none"],
                   help="auto = 按训练集频率的倒数加权（N 类偏多时有用）；none = 不加权")
    p.add_argument("--class-weight-max", type=float, default=5.0,
                   help="auto 权重的上限（默认 5.0，设 0 表示不限）。"
                        "★ 2026-10-11 加：E 类只有 3 个训练样本时 auto 算出 218 倍权重，"
                        "损失被这 3 个样本劫持，模型塌成「全预测 E」（那次真实 acc 0.1529）。"
                        "5.0 只保留温和的类别平衡。")
    p.add_argument("--min-train-samples", type=int, default=10,
                   help="训练集里样本少于这个数的类别**直接不参与训练**（默认 10，设 0 关闭）。"
                        "样本量不足以学出泛化特征时，权重再合理也没用；被剔除的类别会在日志里"
                        "单独说明（训练/验证/测试各多少个）。")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--select-metric", default="macro_f1", choices=["macro_f1", "acc"],
                   help="用哪个指标选 best.pt / 早停（默认 macro_f1：五类等权，"
                        "不会被海量合成 N 负样本骗到；acc = 老行为）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mirror-ad", type=float, default=0.5,
                   help="A/D 左右镜像**互换**的概率（默认 0.5 = 每个 A/D 样本有一半概率"
                        "水平翻转并把标签 A<->D 对调）。0 = 关闭。只作用于训练集。")
    p.add_argument("--speed-input", action="store_true", default=True,
                   help="把当前速度作为额外通道拼进输入（默认开）。"
                        "同画面在低速=还远、高速=马上到，光靠像素分不出来，必须给标量")
    p.add_argument("--no-speed-input", dest="speed_input", action="store_false",
                   help="关掉速度标量通道（模型回到 3*K 通道的老结构）")
    p.add_argument("--mirror-keep", type=float, default=0.5,
                   help="W/S/G/Q 左右镜像的概率（默认 0.5 = 水平翻转但**标签不变**）。"
                        "跳/滚是垂直动作、复活/Q 的判据也不分左右，翻转后仍成立，纯增数据。"
                        "0 = 关闭。只作用于训练集。")
    p.add_argument("--mirror-ws", type=float, default=None,
                   help="（旧名，等价于 --mirror-keep）")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--resume", default=None, help="从 checkpoint 继续")
    p.add_argument("--no-anim-neg", action="store_true",
                   help="关闭从换道动画合成的 N 类负样本（默认开启）")
    p.add_argument("--no-approach-neg", action="store_true",
                   help="关闭从「障碍还远」合成的 N 类负样本（默认开启；治动作偏早）")
    p.add_argument("--anim-share", type=float, default=0.6,
                   help="负样本超过上限时，给「换道动画中」这一类的保底比例（默认 0.6）。"
                        "被误判成 AA/DD（换道动画里又按同一个方向）多就调高；"
                        "动作偏早/偏慢就调低，把名额让给「障碍还远」")
    p.add_argument("--approach-neg-ms", default=None,
                   help="覆盖「障碍还远」负样本的结束偏移（毫秒，逗号分隔，填负数，如"
                        " -400,-500,-600）。注意必须比 -(end_offset+167) 更靠前，否则会"
                        "和正样本窗口重叠（程序会自动丢掉重叠的那些）")
    p.add_argument("--anim-neg-ms", default=None,
                   help="覆盖「换道动画中」负样本的结束偏移（毫秒，逗号分隔，如 120,200,300,450）。"
                        "不填就用采集时 session.json 记的值（默认 200,350）")
    p.add_argument("--no-synth-neg-sessions", default=None,
                   help="逗号分隔的会话名：这些会话里的样本不参与合成 N 负样本"
                        "（默认 碰撞操作 —— 人工挑帧标的撞车样本，前后帧本身是瑕疵画面）")
    p.add_argument("--no-synth-neg-kinds", default=None,
                   help="逗号分隔的样本 kind：这些 kind 不参与合成 N 负样本"
                        "（默认 relabel —— 8_标注界面.py 人工挑帧标出来的）")
    p.add_argument("--neg-ratio", type=float, default=1.0,
                   help="每个子集里负样本最多是正样本的几倍（默认 1.0：负样本 ≤ 正样本，"
                        "于是 N 最多占一半；0 = 不限）。"
                        "负样本会随数据集膨胀，太多既拖慢训练、又把模型压得太保守")
    p.add_argument("--speed-balance", type=float, default=0.0,
                   help="按速度重加权的强度 0~1（游戏速度随存活时间上升，"
                        "慢速样本太多时用 0.5 左右拉平；默认 0 不重加权）")
    p.add_argument("--smoke", action="store_true", help="冒烟测试：少量样本跑 1 轮")
    return p.parse_args(argv)


def build_cfg(session_dirs, args) -> Config:
    """读采集时的 session.json，再用命令行覆盖输入窗口相关参数。"""
    cfg = Config()
    for d in session_dirs:
        p = os.path.join(d, "session.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                raw = json.load(f).get("config", {})
            valid = {fl.name for fl in fields(Config)}
            for k, v in raw.items():
                if k in valid:
                    if k in ("game_res", "game_crop", "labels", "animation_neg_end_ms",
                          "approach_neg_end_ms"):
                        v = tuple(v)
                    setattr(cfg, k, v)
            print(f"[config] 已读取采集配置: {p}")
            break
    else:
        print("[config] 没找到 session.json，使用默认 Config")
    # ★ 2026-10-15（用户要求"能区分不同游戏的训练代码"）：按键表按游戏走。
    #   优先级：--keys > --game > session.json 里采集时记的 > config.py 默认表。
    #   写进 model_meta.json 后，部署端就按这个表发键 —— 换个游戏换一份表，互不干扰。
    from core import gamecfg as _G
    _game = args.game or getattr(cfg, "game", None)
    if args.game:
        cfg.game = args.game
    _g = _G.load(_game)
    _km = _G.apply_keys(cfg, _g, args.keys)
    if _km:
        print("[按键] %s  ->  类别 %s"
              % (" ".join("%s=%s" % (k, v) for k, v in _km.items()), list(cfg.labels)))
    if args.input_frames:
        cfg.input_frames = args.input_frames
    if args.end_offset is not None:
        cfg.input_end_offset_ms = args.end_offset
    if args.input_stride:
        cfg.input_stride = args.input_stride
    if args.frame_w:
        cfg.frame_w = args.frame_w
    # ★ 速度标量通道（2026-11-05）：输入通道从 3*K 变成 3*K+1
    cfg.use_speed = bool(getattr(args, "speed_input", True))
    cfg.validate()
    return cfg


def resolve_geometry(session_dirs, cfg):
    """决定写进 model_meta.json 的取景（相对比例），并检查各会话是否混了区域。

    优先级（★ 2026-10-14 用户设计：选一次，采集/训练/部署都只读文件）：
      ① 项目 gamer_ai/geometry.json —— 你最近一次弹窗选的那个，代表"这次要训的区域"
      ② 会话目录里的 geometry.json —— 同一批数据的取景全都一致时用它
      ③ session.json 里的 crop_rel
      ④ 由 game_res/game_crop 反算（老数据没写文件时的兜底）

    返回 (crop_rel, 来源说明, 不一致的会话列表, 来源取景dict或None)。
    """
    sess = []
    for d in session_dirs:
        g = load_geometry(os.path.join(d, GEOMETRY_NAME))
        if g:
            sess.append((os.path.basename(os.path.normpath(d)),
                         tuple(float(v) for v in g["crop_rel"])))
    proj = load_geometry()
    src_geo = None
    if proj:
        rel, src, src_geo = tuple(float(v) for v in proj["crop_rel"]), "项目 geometry.json", proj
    elif sess:
        rels = {tuple(round(v, 4) for v in r) for _, r in sess}
        if len(rels) == 1:
            rel = next(iter(rels))
            src = "会话 geometry.json（%d 个会话一致）" % len(sess)
        else:
            rel = sess[0][1]
            src = "会话 geometry.json（多会话不一致，取 %s）" % sess[0][0]
    elif getattr(cfg, "crop_rel", None):
        rel, src = tuple(float(v) for v in cfg.crop_rel), "session.json 的 crop_rel"
    else:
        rel = rel_from_crop(cfg.game_res[0], cfg.game_res[1], cfg.game_crop)
        src = "由 game_res/game_crop 反算"
    bad = [(n, r) for n, r in sess
           if max(abs(a - b) for a, b in zip(r, rel)) > 0.01]
    return rel, src, bad, src_geo


from core.model import build_model   # 训练与部署共用同一份，避免输入通道对不上


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_report(cm, labels):
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"label": lab, "n": int(cm[i].sum()), "precision": prec,
                     "recall": rec, "f1": f1})
    return rows


def parse_source_weights(spec: str, samples) -> Tuple[List[float], "Counter"]:
    """把 --source-weight "旧=1.0,迭代数据=2.0" 解析成每样本权重。

    按会话路径（_root）**子串**匹配，先匹配到的规则生效。没命中的样本权重 = 1.0。
    """
    rules: List[Tuple[str, float]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            rules.append((k.strip(), float(v)))
        except ValueError:
            continue
    w = [1.0] * len(samples)
    hit: Counter = Counter()
    for i, m in enumerate(samples):
        root = str(m.get("_root", ""))
        for k, v in rules:
            if k and k in root:
                w[i] = v
                hit[k] += 1
                break
    return w, hit


def macro_f1_from_cm(cm) -> float:
    """从混淆矩阵算 macro-F1。

    ★ 为什么不用 accuracy 选模型（2026-09-24 用户反馈"效果差"）：
      val/test 里合成的 N 负样本占了 3/4，accuracy 基本只在量 N 建得好不好；
      而且按 accuracy 选出来的 best.pt 恰好是**最保守（最爱预测 N）**的那个。
      macro-F1 五类等权，能直接反映"每个动作学没学会"。
    """
    cm = np.asarray(cm, dtype=np.float64)
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def evaluate(model, loader, crit, device, n_classes, use_amp):
    model.eval()
    loss_sum, total, correct = 0.0, 0, 0
    preds, gts = [], []
    _, ac = make_amp(use_amp)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with ac():
                out = model(x)
                loss = crit(out, y)
            loss_sum += loss.item() * y.size(0)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
            preds.append(p.cpu().numpy())
            gts.append(y.cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    gts = np.concatenate(gts) if gts else np.array([])
    cm = confusion_matrix(gts, preds, n_classes)
    return loss_sum / max(total, 1), correct / max(total, 1), cm


# ======================================================================
def train(argv=None) -> int:
    r"""训练主流程。**可以当普通函数调用**（2026-10-15 用户要求）：

        from core import trainer
        trainer.train(["--sessions", r"gamer_ai\profiles\cs01\dataset",
                       "--out", r"gamer_ai\profiles\cs01\runs\v2_20261016",
                       "--epochs", "60", "--lr", "1e-3"])

    argv 不传时用命令行参数（python -m core.trainer ... / 4_训练.py ...）。
    """
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    log = Tee(os.path.join(args.out, "log.txt"))
    t_start = time.time()

    log("=" * 78)
    log(f"地铁跑酷 多帧训练   输出: {os.path.abspath(args.out)}")
    log("=" * 78)

    # ---------- 设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log(f"[env] GPU: {torch.cuda.get_device_name(0)}  "
            f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB  "
            f"torch {torch.__version__}  cuda {torch.version.cuda}")
    else:
        log("[env] 警告：没有检测到 CUDA，将在 CPU 上训练")

    # ---------- 数据 ----------
    cfg = build_cfg(args.sessions, args)
    samples = load_sessions(args.sessions)
    log(f"[data] 共 {len(samples)} 个样本，来自 {len(args.sessions)} 个会话")

    # ★ 2026-10-14 取景（区域）跟着数据走：把这次训练用的区域记下来，
    #   写进 model_meta.json -> 部署时（6_自动玩.py）模型自己就能切到正确区域。
    geo_rel, geo_src, geo_bad, geo_src_geo = resolve_geometry(args.sessions, cfg)
    # 记进模型的是**标定当时**的分辨率与绝对框（换设备后部署端只按比例换算），
    # 而不是训练配置里的参考值 —— 两者在换分辨率采集后并不相同。
    geo_res = tuple(int(v) for v in ((geo_src_geo or {}).get("game_res") or cfg.game_res))
    geo_crop = crop_from_rel(geo_res[0], geo_res[1], geo_rel)
    log("[取景] 本次训练区域 crop_rel=%s  来源：%s"
        % (tuple(round(float(v), 4) for v in geo_rel), geo_src))
    if geo_bad:
        log("[取景][警告] 下面这些会话的取景和本次要记录的区域不一致：")
        for _n, _r in geo_bad:
            log("           %s  crop_rel=%s" % (_n, tuple(round(float(v), 4) for v in _r)))
        log("           混区域训练会让模型看到的画面范围不统一，先确认这是你想要的。")
    if not samples:
        log("[data] 没有任何样本，退出")
        return 1
    cnt = Counter(s["label"] for s in samples)
    log(f"[data] 原始类别分布: {dict(sorted(cnt.items()))}")
    labels = sorted(cnt.keys())
    log(f"[data] 类别顺序: {labels}")
    log(f"[data] 输入 = 连续 {cfg.input_frames} 帧, 结束于按键前 {cfg.input_end_offset_ms} ms, "
        f"步长 {cfg.input_stride}  ->  张量通道 "
        f"{cfg.input_frames*3 + (1 if getattr(cfg, 'use_speed', False) else 0)}"
        + ("（含 1 个速度标量通道）" if getattr(cfg, "use_speed", False) else ""))

    sp = split_indices(samples, train_ratio=1 - args.val_ratio - args.test_ratio,
                       val_ratio=args.val_ratio, mode=args.split_mode,
                       boundary_gap=args.boundary_gap, seed=args.seed,
                       segment_seconds=args.segment_seconds)
    if args.smoke:
        sp = {k: v[:200] for k, v in sp.items()}
        args.epochs = 1
        log("[smoke] 冒烟模式：每集只取 200 个样本、跑 1 轮")
    for k in ("train", "val", "test"):
        sub = Counter(samples[i]["label"] for i in sp[k])
        log(f"[split] {k:5s} {len(sp[k]):5d}  {dict(sorted(sub.items()))}")
        missing = [l for l in labels if sub.get(l, 0) == 0]
        if missing:
            log(f"        [警告] {k} 集合缺少类别 {missing} —— 划分不合理，"
                f"建议改用 --split-mode time_balanced 或调小 --segment-seconds")
        elif k == "train":
            rare = [l for l in labels if sub.get(l, 0) < 10]
            if rare:
                log(f"        [警告] {k} 集合中 {rare} 样本不足 10 个，类别权重会被放大到失真")

    # ★ 2026-10-11：样本太少的类别【直接不训练】。
    #   动机（实测）：E 只有 3 个训练样本时，auto 权重给出 218 倍，模型塌成"全预测 E"，
    #   真实样本 acc 0.1529。就算把权重压到 5，3 个样本也学不出泛化 —— 与其让它污染
    #   训练，不如明确剔除，并在日志里写清每个被剔除的类在三个子集里各有多少个。
    _ntr = Counter(samples[i]["label"] for i in sp["train"])
    _rare = [l for l in labels if _ntr.get(l, 0) < args.min_train_samples]
    # ★ 2026-10-16（用户："为什么训练的时候没有 N？不是说 37 帧里除了 6 帧会拿一部分当 N 吗"）：
    #   判"稀有类"时**必须把合成负样本算进去**。N 的手动样本通常很少（按 n 是体力活），
    #   但每个按键样本的胶片条还能合成出"换道动画中""障碍还远"两类 N 负样本，动辄几百上千个。
    #   以前只数手动样本 -> N 被判稀有 -> 从 labels 里删掉 -> 合成负样本也跟着失效
    #   （core/dataset.py：'N' not in labels -> 直接返回空）—— 一步把"别动"的监督全关了。
    if "N" in _rare and (getattr(cfg, "animation_neg_enabled", True)
                         or getattr(cfg, "approach_neg_enabled", True)):
        _rare = [l for l in _rare if l != "N"]
        log("        [rare] N 手动样本只有 %d 个，但会从胶片条合成大量"
            "「换道动画 / 障碍还远」负样本 —— **保留 N 类**"
            "（否则「别动」的监督会被整个关掉）" % _ntr.get("N", 0))
    if _rare:
        log(f"[rare] 训练集样本不足 {args.min_train_samples} 个的类别不参与训练：{_rare}")
        for _l in _rare:
            _nv = sum(1 for i in sp["val"] if samples[i]["label"] == _l)
            _nte = sum(1 for i in sp["test"] if samples[i]["label"] == _l)
            log(f"       {_l}: train {_ntr.get(_l, 0)} / val {_nv} / test {_nte} 个"
                f"  —— 建议先采到每个动作类 100~200 个再训它")
        _keep = set(labels) - set(_rare)
        for _k in ("train", "val", "test"):
            sp[_k] = [i for i in sp[_k] if samples[i]["label"] in _keep]
        labels = [l for l in labels if l in _keep]
        log(f"[rare] 本次实际训练 {len(labels)} 类：{labels}"
            "（想包含被剔除的类就加 --min-train-samples 0）")

    # 两类合成的 N 类负样本：先在全体样本上判定，再映射到各子集
    #   * 换道动画中（按键之后）—— 治"移动中重复触发"
    #   * 障碍还远  （按键之前）—— 治"动作偏早"
    if getattr(args, "no_synth_neg_sessions", None):
        cfg.no_synth_neg_sessions = tuple(
            x.strip() for x in args.no_synth_neg_sessions.split(",") if x.strip())
    if getattr(args, "no_synth_neg_kinds", None):
        cfg.no_synth_neg_kinds = tuple(
            x.strip() for x in args.no_synth_neg_kinds.split(",") if x.strip())
    if args.no_anim_neg:
        cfg.animation_neg_enabled = False
    if args.no_approach_neg:
        cfg.approach_neg_enabled = False
    if getattr(args, "approach_neg_ms", None):
        _am = tuple(float(x) for x in str(args.approach_neg_ms).replace("，", ",").split(",")
                    if x.strip())
        if _am:
            cfg.approach_neg_end_ms = _am
            cfg.approach_neg_enabled = True
            log("[neg] 「障碍还远」偏移被参数覆盖：%s ms" % list(_am))
    # ★ 2026-11-04：偏移超过采集后置窗口时提醒一句（否则那些偏移会静默失效）
    if getattr(cfg, "animation_neg_enabled", True) and cfg.animation_neg_end_ms:
        _maxe = max(cfg.animation_neg_end_ms)
        _post = float(getattr(cfg, "save_post_ms", 0) or 0)
        if _post and _maxe > _post + 1:
            log(f"[neg] [警告] 换道动画偏移最大 {_maxe:.0f}ms 超过采集后置窗口 "
                f"{_post:.0f}ms（session.json 的 save_post_ms）—— 超过的偏移会被逐样本丢掉。"
                f"建议改成 <= {int(_post)}ms，例如 100,200,300,{int(_post)}")
    if getattr(args, "anim_neg_ms", None):
        # ★ 2026-11-04：不重新采集也能加密"换道动画"的覆盖密度。
        #   默认只取按键后 +200/+350ms 两个点；换道动画本身约 250~350ms，
        #   起点（+80~150ms）几乎没有 N 样本，而那一段最容易和"刚按下"混淆。
        _ms = tuple(float(x) for x in str(args.anim_neg_ms).replace("，", ",").split(",")
                    if x.strip())
        if _ms:
            cfg.animation_neg_end_ms = _ms
            cfg.animation_neg_enabled = True
            log("[neg] 「换道动画中」偏移被参数覆盖（TRAIN 里的 anim_neg_ms 或命令行 "
                "--anim-neg-ms）：%s ms" % list(_ms))
    neg_tr = negatives_for_subset(samples, sp["train"], cfg, labels)
    neg_va = negatives_for_subset(samples, sp["val"], cfg, labels)
    neg_te = negatives_for_subset(samples, sp["test"], cfg, labels)
    # 负样本数量随数据集线性膨胀（这次占全部训练条目的 71%）。太多有两个害处：
    #   ① 每轮要读的胶片条翻几倍 -> 训练慢；
    #   ② 模型被大量「什么都不做」压得太保守，该换道时也不敢动。
    # --neg-ratio 给每个子集设上限：负样本数 <= ratio x 正样本数。
    if args.neg_ratio > 0:
        import random as _rnd
        _rng = _rnd.Random(args.seed)
        _capped = []
        for _nm, _cur, _npos in (("train", neg_tr, len(sp["train"])),
                                 ("val", neg_va, len(sp["val"])),
                                 ("test", neg_te, len(sp["test"]))):
            _cap = int(_npos * args.neg_ratio)          # ★ 自动算出来的上限
            if len(_cur) > _cap:
                # ★ 2026-11-04（用户："被误判 AA、DD 的频率太高，角色还在换道动画里就被判成上一步"）：
                #   原来超上限时是**无差别随机抽样** —— "换道动画中"这类负样本
                #   （专门教模型"动画期间别再按同一个方向"）跟着一起被抽掉一大半：
                #   实测 train 有 918 个动画负样本，--neg-ratio 0.3 上限 505 时只剩约 155 个，
                #   结果测试集 N 类的 27 个错误里全是 N->A/N->D。
                #   现在按 --anim-share 先给动画类保底，剩下的名额才给"障碍还远"。
                _anim = [x for x in _cur if x[1] > 0]      # 正偏移 = 按键之后 = 换道动画中
                _appr = [x for x in _cur if x[1] <= 0]     # 负偏移 = 按键之前 = 障碍还远
                _floor = int(_cap * max(0.0, min(1.0, args.anim_share)))
                _take_anim = max(0, min(len(_anim), max(_floor, _cap - len(_appr))))
                _take_appr = max(0, min(len(_appr), _cap - _take_anim))
                if _take_anim + _take_appr < _cap:         # 障碍类不够填，名额还给动画类
                    _take_anim = max(0, min(len(_anim), _cap - _take_appr))
                _pick = (_rng.sample(_anim, _take_anim) if _take_anim < len(_anim)
                         else list(_anim))
                _pick += (_rng.sample(_appr, _take_appr) if _take_appr < len(_appr)
                          else list(_appr))
                log(f"[neg] {_nm} 负样本 {len(_cur)} -> {len(_pick)}"
                    f"（上限 = 正样本 {_npos} x --neg-ratio {args.neg_ratio}；"
                    f"换道动画 {len(_anim)}->{_take_anim}（保底 {int(args.anim_share*100)}%）、"
                    f"障碍还远 {len(_appr)}->{_take_appr}）")
                _cur = _pick
            elif _cur:
                log(f"[neg] {_nm} 负样本 {len(_cur)}（正样本 {_npos}，"
                    f"未超上限 {_cap} = {_npos} x {args.neg_ratio}）")
            _capped.append(_cur)
        neg_tr, neg_va, neg_te = _capped
        _n_tr = len(sp["train"]) + len(neg_tr)
        if _n_tr:
            log(f"[neg] 训练条目 {_n_tr} = 正样本 {len(sp['train'])}"
                f" + 负样本 {len(neg_tr)}；负样本占 {100.0*len(neg_tr)/_n_tr:.0f}%")
    if "N" in labels:
        bd = {k: negative_breakdown(samples, sp[k], cfg, labels)
              for k in ("train", "val", "test")}
        log(f"[neg] 合成 N 类负样本  train {len(neg_tr)} / val {len(neg_va)} "
            f"/ test {len(neg_te)}")
        log(f"      |- 换道动画中: 按键后 {list(cfg.animation_neg_end_ms)}ms 出发，"
            f"要求之后安静 >= {cfg.animation_neg_min_gap_ms}ms"
            f"  -> train {bd['train']['anim']} / val {bd['val']['anim']}")
        from .dataset import synth_neg_allowed as _sna
        _skip = sum(1 for m in samples if m["label"] != "N" and not _sna(m, cfg))
        if _skip:
            log(f"      [排除] {_skip} 个样本不参与负样本合成"
                f"（会话 {list(cfg.no_synth_neg_sessions)} / kind {list(cfg.no_synth_neg_kinds)}："
                "人工挑帧的碰撞样本，前后帧本身是瑕疵画面）")
        _dropped = getattr(cfg, "_approach_dropped", None)
        if _dropped:
            log(f"      [自动丢弃] 这些「障碍还远」偏移会和正样本窗口重叠，已跳过：{_dropped}")
            log(f"                 规则：偏移 <= -(end_offset + 跨度) = "
                f"-{int(cfg.input_end_offset_ms) + 167}ms 才干净（end_offset="
                f"{cfg.input_end_offset_ms}，跨度 167ms）")
        log(rf"      \- 障碍还远  : 按键前 {list(cfg.approach_neg_end_ms)}ms 出发，"
            f"要求之前安静 >= |偏移|+窗口+{cfg.approach_neg_quiet_margin_ms}ms"
            f"  -> train {bd['train']['approach']} / val {bd['val']['approach']}")
        if cfg.approach_neg_enabled and bd["train"]["approach"] == 0:
            span = ((cfg.input_frames - 1) * max(1, cfg.input_stride) * 1000.0
                    / max(float(cfg.save_fps), 1e-6))
            need = (abs(min(cfg.approach_neg_end_ms)) + span) if cfg.approach_neg_end_ms else 0
            log(f"      [提示] 「障碍还远」负样本为 0 —— 多半是数据太旧："
                f"需要按键前约 {need:.0f}ms 的画面（|偏移|+窗口跨度），"
                f"请确认采集时 save_pre_ms >= {need:.0f}（现在是 {cfg.save_pre_ms}）")
        if cfg.animation_neg_end_ms and next(
                (m for m in samples if m["label"] != "N"), None) is not None:
            sample_key = next(m for m in samples if m["label"] != "N")
            if not neg_tr and sample_key["frame_times_ms"][-1] < max(cfg.animation_neg_end_ms):
                log("      [警告] 后置窗口太短，无法合成负样本 —— 请确认 save_post_ms >= "
                    + str(max(cfg.animation_neg_end_ms)))

    # ★ 来源加权（迭代微调用）：给新数据/人工纠正的数据更高权重
    train_src_w = None
    if args.source_weight:
        _w, _hit = parse_source_weights(args.source_weight, samples)
        train_src_w = [_w[i] for i in sp["train"]]
        _nz = sum(1 for x in train_src_w if abs(x - 1.0) > 1e-9)
        log(f"[source] 来源加权 {args.source_weight}  ->  命中 {dict(_hit)}；"
            f"训练集里 {_nz}/{len(train_src_w)} 个样本权重≠1")
        if _nz == 0:
            log("        [警告] 一条都没匹配上，检查会话名写法（按路径子串匹配）")
    train_ds = MultiFrameDataset([samples[i] for i in sp["train"]], cfg,
                                 build_train_transform(args.img_size), args.img_size,
                                 labels=labels, negatives=neg_tr, weights=train_src_w,
                                 mirror_ad=args.mirror_ad,
                                 mirror_keep=(args.mirror_keep if args.mirror_ws is None
                                              else args.mirror_ws))
    _mir_keep = args.mirror_keep if args.mirror_ws is None else args.mirror_ws
    if args.mirror_ad > 0 or _mir_keep > 0:
        _cnt = {}
        for i in sp["train"]:
            lb = samples[i]["label"]
            _cnt[lb] = _cnt.get(lb, 0) + 1
        log(f"[aug] 左右镜像（只作用于训练集，验证/测试不翻，指标才可比）")
        if args.mirror_ad > 0:
            log(f"      |- A/D 互换：概率 {args.mirror_ad:.2f}  "
                f"A {_cnt.get('A', 0)} 个 / D {_cnt.get('D', 0)} 个  "
                f"-> 翻转后各自变成对方那类（A/D 数据量实际翻倍）")
        if _mir_keep > 0:
            log(f"      \- W/S/G/Q 增广：概率 {_mir_keep:.2f}  "
                f"W {_cnt.get('W', 0)} / S {_cnt.get('S', 0)} / "
                f"G {_cnt.get('G', 0)} / Q {_cnt.get('Q', 0)} 个  "
                f"-> 翻转但标签不变（垂直动作 + 复活弹窗，判据都不分左右）")
    val_ds = MultiFrameDataset([samples[i] for i in sp["val"]], cfg,
                               build_eval_transform(args.img_size), args.img_size,
                               labels=labels, negatives=neg_va)
    test_ds = MultiFrameDataset([samples[i] for i in sp["test"]], cfg,
                                build_eval_transform(args.img_size), args.img_size,
                                labels=labels, negatives=neg_te)

    # ---------- 速度覆盖度（游戏速度随存活时间上升）----------
    sp_all = [float(m.get("speed_px_s") or 0.0) for m in samples]
    n_ok = sum(1 for x in sp_all if x > 0)
    if n_ok >= 10:
        if n_ok < len(sp_all):
            log(f"[speed] ⚠ 只有 {n_ok}/{len(sp_all)} 个样本带速度信息"
                f"（{len(sp_all) - n_ok} 个没有 speed_px_s，速度覆盖度看不到它们）"
                f"速度覆盖度看不到它们）")
        log(f"[speed] 全体: {speed_mod.describe(sp_all)}")
        tr_sp = [sp_all[i] for i in sp["train"]]
        log(f"[speed] 训练集: {speed_mod.describe(tr_sp)}")
        edges, hist = speed_mod.speed_histogram([x for x in tr_sp if x > 0], bins=8)
        tot = max(hist.sum(), 1)
        log("        训练集速度分布: " + "  ".join(
            f"{edges[i]:.0f}-{edges[i+1]:.0f}:{hist[i]*100//tot}%" for i in range(len(hist))))
        _sv = np.array([x for x in tr_sp if x > 0])
        _hi = float(np.percentile(_sv, 90)) if _sv.size else 0.0
        n_hi = int((_sv > _hi).sum())
        top2 = hist[-2:].sum() / tot
        log(f"        速度最高的 10% 从 {_hi:.1f} px/s 起（训练集里 {n_hi} 个样本）")
        if top2 < 0.10:
            log(f"        [警告] 最上面两档只占 {top2*100:.1f}%（约 {int(top2*tot)} 个）。"
                f"高速画面见得少，上线后在高速段容易失效。")
            log("               补救：多补一些跑得久的高速局；或加 --speed-balance 0.5 先缓解")
        if args.speed_balance > 0:
            log(f"        [speed] 启用按速度重加权，强度 {args.speed_balance}")
    else:
        log("[speed] 数据里没有速度信息（旧格式或 record_speed=False），跳过覆盖度检查")

    dl_kw = dict(num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        per_batch = args.batch_size * (3 * cfg.input_frames * args.img_size ** 2 * 4) / 1e9
        pf = max(1, int(getattr(args, "prefetch", 1)))
        # 每个 worker 有一份"固定开销"：import torch/torchvision 约 0.6GB，
        # 加上 prefetch 个批次的数据。实测 8 workers 光固定开销就 4.8GB。
        est = args.workers * (0.6 + pf * per_batch)
        log(f"[loader] workers={args.workers} prefetch={pf} batch={args.batch_size}  "
            f"->  预计占内存约 {est:.1f} GB（每 worker 0.6GB 底 + {pf*per_batch:.2f}GB 缓冲）")
        if est > 4.5:
            log(f"        [警告] 这已经不小了。机器上还开着模拟器/PyCharm 的话，"
                f"建议 --workers 2 --batch-size 16")
    if args.workers > 0:
        dl_kw.update(persistent_workers=True,
                     prefetch_factor=max(1, int(getattr(args, "prefetch", 2))))
    if args.speed_balance > 0:
        from torch.utils.data import WeightedRandomSampler
        wts = speed_mod.dataset_weights([samples[i] for i in sp["train"]], neg_tr,
                                        args.speed_balance)
        sampler = WeightedRandomSampler(torch.as_tensor(wts, dtype=torch.double),
                                        num_samples=len(wts), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **dl_kw)
        log(f"[speed] 训练用 WeightedRandomSampler（权重范围 {wts.min():.2f}~{wts.max():.2f}）")
    else:
        # 按「胶片条」分组打乱（默认开）。正/负样本共用同一张胶片条，
        # 分组后一张只解码一次，配合 read_strip 的缓存能省掉一半以上的 JPEG 解码。
        # 随机性没丢：每个 epoch 重新打乱组顺序。
        if getattr(args, "group_shuffle", True):
            gsampler = GroupShuffleSampler(train_ds.strip_groups(), seed=args.seed)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=gsampler, **dl_kw)
            log(f"[loader] 分组打乱已开启：{len(gsampler.groups)} 张胶片条 -> "
                f"{len(train_ds)} 个条目（解码次数约降到 1/2.5）")
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl_kw)
    # ★ 最终评估额外看一眼「只用真实样本」的成绩：合成 N 负样本占了测试集 3/4，
    #   混在一起算 accuracy 会把成绩吹得很高（2026-09-24）。
    real_test_ds = MultiFrameDataset([samples[i] for i in sp["test"]], cfg,
                                     build_eval_transform(args.img_size), args.img_size,
                                     labels=labels, negatives=None)
    real_test_loader = DataLoader(real_test_ds, batch_size=args.batch_size,
                                  shuffle=False, **dl_kw)

    # ---------- 模型 ----------
    _in_ch = cfg.input_frames * 3 + (1 if getattr(cfg, "use_speed", False) else 0)
    model = build_model(cfg.input_frames, len(labels),
                        pretrained=not args.no_pretrained,
                        weights_path=args.pretrained_weights,
                        in_channels=_in_ch).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    log(f"[model] EfficientNet-B0  参数 {n_param/1e6:.2f} M  输入通道 {_in_ch}"
        + ("（含速度标量通道：z=log1p(px/s)/4.6）" if getattr(cfg, "use_speed", False) else ""))

    weights = None
    if args.class_weight == "auto":
        # ★ 2026-09-24：必须按**最终训练集**的分布算权重 —— 合成出来的 N 负样本
        #   也真实地占了一大半条目（这次 3410/4543 = 75%）。以前只数"真实样本"，
        #   N 的权重被算成 9.06，可它又占了 75% 的条目 —— N 被**双重放大**，
        #   于是模型变成"倾向不动"：A/D/S/W 有 20~43% 被预测成 N。
        cnt = Counter(samples[i]["label"] for i in sp["train"])
        for _i, _e in neg_tr:
            cnt["N"] += 1
        freq = np.array([cnt.get(l, 0) for l in labels], dtype=np.float64)
        freq = np.maximum(freq, 1.0)
        w_raw = freq.sum() / (len(labels) * freq)
        # ★ 2026-10-11：上限。没有它的时候，3 个 E 样本算出 218 倍权重 -> 模型全预测 E。
        w = np.minimum(w_raw, args.class_weight_max) if args.class_weight_max > 0 else w_raw
        weights = torch.tensor(w, dtype=torch.float32, device=device)
        log(f"[loss] 类别权重(auto，按含合成负样本的最终训练分布算)"
            f" {dict(zip(labels, [round(float(x),3) for x in w]))}")
        if args.class_weight_max > 0 and (w_raw > args.class_weight_max + 1e-9).any():
            _cl = "，".join(f"{l} {float(a):.1f}->{args.class_weight_max:g}"
                           for l, a in zip(labels, w_raw) if a > args.class_weight_max + 1e-9)
            log(f"[loss] [截断] 超过上限 {args.class_weight_max:g} 的类别权重被压回：{_cl}"
                "（想恢复原始值用 --class-weight-max 0）")
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    # 来源加权用：逐样本不归约，再按 w 加权平均
    crit_none = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing,
                                    reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    scaler, autocast = make_amp(args.amp and device.type == "cuda")

    # 选点指标：macro_f1（默认，五类等权）或 acc（老行为）
    start_epoch, best_acc, best_metric = 0, 0.0, 0.0
    log(f"[select] 用 {args.select_metric} 选 best.pt 和早停"
        + (f"（{len(labels)} 类等权，不会被海量合成 N 负样本骗到）"
           if args.select_metric == "macro_f1"
           else "（注意：测试集里合成负样本多时，acc 会虚高）"))
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            opt.load_state_dict(ck["optimizer_state_dict"])
            # ★ 2026-09-25：优化器状态里带着**上次的 lr**，load 完会把你写的 --lr 盖掉
            #   （实测：--lr 2e-4 被初代的 6.25e-5 顶掉了）。微调要的就是"我指定的
            #   那个小学习率"，所以这里显式设回去。动量/二阶矩照旧继承。
            old_lr = float(opt.param_groups[0]["lr"])
            for g_ in opt.param_groups:
                g_["lr"] = args.lr
            log(f"[resume] 学习率：checkpoint 里是 {old_lr:.2e} -> 按 --lr 设为 {args.lr:.2e}")
        start_epoch = ck.get("epoch", 0) + 1
        best_acc = ck.get("val_acc", 0.0)
        # ★ 关键：老 checkpoint 里只有 val_acc，量纲和 macro_f1 不一样，
        #   直接拿来当 best_metric 会导致"永远打不过它" -> best.pt 一次都不保存。
        #   没有对应指标的记录就从 0 开始（第 1 轮就会存一个 best.pt）。
        if args.select_metric == "macro_f1":
            best_metric = float(ck.get("val_f1", 0.0))
        else:
            best_metric = float(best_acc)
        log(f"[resume] 历史最佳：val_acc {best_acc:.4f} / "
            f"按 {args.select_metric} 重新起算 {best_metric:.4f}")
        log(f"[resume] 从 {args.resume} 继续，epoch={start_epoch} best_acc={best_acc:.4f}")

    # ---------- 训练 ----------
    history = []
    hist_path = os.path.join(args.out, "history.csv")
    if not os.path.isfile(hist_path):
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "sec"])

    def _clear_line(width: int = 130) -> None:
        """用空格盖掉进度行（进度条只在控制台刷，不写进 log.txt，免得把它刷乱）。"""
        print("\r" + " " * width + "\r", end="", flush=True)

    # ★ 微调基线（2026-09-25）：把初代模型在**同一个测试集**上先量一遍。
    #   否则你会拿"初代的旧测试集成绩"和"新测试集成绩"对比 —— 那不是一回事，
    #   会误以为迭代变差了。
    base_real = None
    if args.resume and os.path.isfile(args.resume):
        _bl, _ba, _bcm = evaluate(model, real_test_loader, crit, device, len(labels),
                                  scaler is not None)
        _brows = per_class_report(_bcm, labels)
        base_real = {"acc": float(_ba),
                     "macro_f1": float(np.mean([x["f1"] for x in _brows])),
                     "per_class": _brows}
        log("")
        log("【微调前 · 初代模型在同一测试集上】")
        log(f"  真实样本 {len(sp['test'])} 个   Accuracy {_ba:.4f}   "
            f"macro-F1 {base_real['macro_f1']:.4f}")
        log(f"  {'类别':<6}{'样本':>7}{'precision':>11}{'recall':>9}{'f1':>8}")
        for rr in _brows:
            log(f"  {rr['label']:<6}{rr['n']:>7}{rr['precision']:>11.4f}"
                f"{rr['recall']:>9.4f}{rr['f1']:>8.4f}")
        log("")

    bad = 0
    n_batches = len(train_loader)
    log(f"[train] 每轮 {len(train_ds)} 个条目 = {n_batches} 批，batch={args.batch_size}；"
        f"下面每 5 秒刷一次进度。第一轮前面那段是最慢的，别以为卡死了。")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        prog_t = t0
        for bi, batch in enumerate(train_loader):
            if len(batch) == 3:                     # 来源加权模式
                x, y, w = batch
                w = w.to(device, non_blocking=True)
            else:
                x, y = batch
                w = None
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(x)
                if w is None:
                    loss = crit(out, y)
                else:
                    loss = (crit_none(out, y) * w).sum() / w.sum().clamp(min=1e-6)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

            # ---- 实时进度：每 5 秒刷一次同一行 ----
            nowp = time.time()
            if nowp - prog_t >= 5.0 or bi + 1 == n_batches:
                prog_t = nowp
                frac = (bi + 1) / max(n_batches, 1)
                el = nowp - t0
                eta = (el / frac - el) if frac > 0 else 0.0
                fill = int(24 * frac)
                bar = "#" * fill + "-" * (24 - fill)
                print(f"\r  [{epoch+1:3d}/{args.epochs}] [{bar}] {frac*100:5.1f}%  "
                      f"loss {run_loss/max(total,1):.4f}  acc {correct/max(total,1):.4f}  "
                      f"{total}/{len(train_ds)} 张  本轮剩 {eta/60:4.1f} 分钟",
                      end="", flush=True)
        _clear_line()
        tr_loss, tr_acc = run_loss / total, correct / total
        print("  正在评估验证集 ...", end="", flush=True)
        va_loss, va_acc, va_cm = evaluate(model, val_loader, crit, device, len(labels),
                                          scaler is not None)
        va_f1 = macro_f1_from_cm(va_cm)
        _clear_line()
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        log(f"[{epoch+1:3d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} macroF1 {va_f1:.4f} | "
            f"lr {lr_now:.2e} | {dt:.1f}s")
        history.append((epoch + 1, tr_loss, tr_acc, va_loss, va_acc, lr_now, dt))
        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, tr_loss, tr_acc, va_loss, va_acc, lr_now, round(dt, 2)])

        sel_now = va_f1 if args.select_metric == "macro_f1" else va_acc
        sched.step(sel_now)
        best_acc = max(best_acc, va_acc)
        improved = sel_now > best_metric + 1e-6
        if improved:
            best_metric, bad = sel_now, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "epoch": epoch, "val_acc": va_acc, "val_f1": va_f1,
                        "select_metric": args.select_metric, "labels": labels,
                        "arch": "efficientnet_b0", "in_frames": cfg.input_frames,
                        "config": cfg.to_dict()}, os.path.join(args.out, "best.pt"))
            log(f"        ↑ 新的最佳 {args.select_metric} {sel_now:.4f}"
                f"（val_acc {va_acc:.4f} / macro-F1 {va_f1:.4f}），已保存 best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"[early-stop] {args.select_metric} 连续 {bad} 轮没有提升，停止训练")
                break

    # ★ 2026-10-15 修正：这里原本**又往 best.pt 存了一次**，把上面刚选出的「验证集最优」
    #   权重覆盖成「最后一轮」的权重，而且丢掉了 arch / in_frames / val_f1 字段 ——
    #   5_评估.py 读 best.pt 时正是因此报 KeyError: 'in_frames'，
    #   而 metrics.json 里那些"用 best.pt 评估"的数字其实评的是最后一轮。
    #   现在存成 last.pt：best.pt 名副其实（最优轮），最后一轮另存一份备用。
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch, "val_acc": va_acc, "val_f1": va_f1,
                "select_metric": args.select_metric, "labels": labels,
                "arch": "efficientnet_b0", "in_frames": cfg.input_frames,
                "config": cfg.to_dict()}, os.path.join(args.out, "last.pt"))

    # ---------- 测试集 ----------
    log("\n" + "=" * 78)
    log("测试集评估（用 best.pt）")
    log("=" * 78)
    best_path = os.path.join(args.out, "best.pt")
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
    # ★ 先看「只用真实样本」的成绩 —— 这才是上线时的真实水平
    #   （全量测试集里合成 N 负样本占了 3/4，accuracy 会被吹高）
    r_loss, r_acc, r_cm = evaluate(model, real_test_loader, crit, device, len(labels),
                                   scaler is not None)
    r_rows = per_class_report(r_cm, labels)
    r_f1 = float(np.mean([x["f1"] for x in r_rows]))
    log(f"【真实样本】{len(sp['test'])} 个（不含合成负样本）  "
        f"Accuracy {r_acc:.4f}   macro-F1 {r_f1:.4f}")
    log(f"  {'类别':<6}{'样本':>7}{'precision':>11}{'recall':>9}{'f1':>8}")
    for r in r_rows:
        log(f"  {r['label']:<6}{r['n']:>7}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}")
    if base_real:
        log("")
        log("【微调前后对比 · 同一批真实样本】")
        log(f"  {'指标':<8}{'初代':>10}{'微调后':>10}{'变化':>10}")
        log(f"  {'Accuracy':<8}{base_real['acc']:>10.4f}{r_acc:>10.4f}"
            f"{r_acc - base_real['acc']:>+10.4f}")
        log(f"  {'macro-F1':<8}{base_real['macro_f1']:>10.4f}{r_f1:>10.4f}"
            f"{r_f1 - base_real['macro_f1']:>+10.4f}")
        bmap = {x["label"]: x for x in base_real["per_class"]}
        for rr in r_rows:
            b = bmap.get(rr["label"])
            if b:
                log(f"  {'F1·' + rr['label']:<8}{b['f1']:>10.4f}{rr['f1']:>10.4f}"
                    f"{rr['f1'] - b['f1']:>+10.4f}")
        if r_f1 < base_real["macro_f1"]:
            log("  [提示] 微调后反而更差 —— 多半是学习率太大或数据/标签有问题，"
                "建议调小 --lr 或减少轮数")
    log("")
    log("【全量含合成负样本】")
    te_loss, te_acc, cm = evaluate(model, test_loader, crit, device, len(labels), scaler is not None)
    log(f"测试 Loss {te_loss:.4f}   Accuracy {te_acc:.4f}")
    log("\n每类指标:")
    log(f"  {'类别':<6}{'样本':>7}{'precision':>11}{'recall':>9}{'f1':>8}")
    rows = per_class_report(cm, labels)
    for r in rows:
        log(f"  {r['label']:<6}{r['n']:>7}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}")
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    log(f"  macro-F1 = {macro_f1:.4f}")
    log("\n混淆矩阵 (行=真实, 列=预测):")
    log("        " + "".join(f"{l:>8}" for l in labels))
    for i, l in enumerate(labels):
        log(f"  {l:<6}" + "".join(f"{cm[i, j]:>8d}" for j in range(len(labels))))

    # ★ 2026-09-26：用户拿「本次的真实样本 acc」去比「上次的全量 acc」，得出
    #   「数据多了反而更差」的错觉（真实 0.9225 < 全量 0.9258）。这里把两把尺子
    #   并排说清楚，并指明 model_meta 记的是哪一栏。
    log("")
    log("=" * 78)
    log("【结论 · 两把尺子（跨模型比较必须用同一栏）】")
    log(f"  真实样本   (上线时的真实水平) : Accuracy {r_acc:.4f}   macro-F1 {r_f1:.4f}")
    log(f"  全量含合成 (训练口径, N 占多数): Accuracy {te_acc:.4f}   macro-F1 {macro_f1:.4f}")
    log("  [!] 全量里合成负样本占了约 3/4，accuracy 会被吹高；拿一个模型的")
    log("      「真实样本」去比另一个模型的「全量」，就会误以为「数据多了反而更差」。")
    log("      要公正对比几个模型，用： python tools\\对比模型.py --runs <目录A> <目录B>")
    log(f"  model_meta.json 的 test_acc / macro_f1 记的是【全量】那一栏：{te_acc:.4f} / {macro_f1:.4f}")
    log("      【真实样本】那一栏在 real_only_acc / real_only_macro_f1 字段。")
    log("=" * 78)

    # ★ 2026-10-16（用户："验证 macro-F1 0.72，真实测试 0.56，差距大。建议按会话/局划分"）：
    #   把"验证集比测试集好太多"这件事自动指出来 —— 这基本就是划分泄漏的信号。
    _gap = best_acc - r_acc
    if _gap > 0.10:
        log("")
        log("=" * 78)
        log(f"[!] 验证集 acc {best_acc:.3f} 比真实测试 acc {r_acc:.4f} 高 {_gap:+.3f} —— 差距过大")
        log("    最常见的原因是**划分泄漏**：训练集和验证集来自同一局/同一个会话，")
        log("    画面高度相关（相邻样本的胶片条窗口甚至重叠），验证分自然虚高。")
        log("    怎么办：")
        log("      ① 多采几局/几个会话 —— 这是根治，单会话无论怎么分都不诚实；")
        log("      ② 确认划分方式是 by_session（现在默认）：")
        log("         python 4_训练.py --split-mode by_session")
        log("         它会先把数据按「会话 / 局」分组，整组一起进 train/val/test，不切开；")
        log("      ③ 看开头 [split] 那几行 —— 若提示「只有 1 组…退回 time_balanced」，")
        log("         说明这次数据本身只有一局，验证分只能当参考。")
        log("=" * 78)

    metrics = {"test_loss": te_loss, "test_acc": te_acc, "macro_f1": macro_f1,
               "real_only_acc": r_acc, "real_only_macro_f1": r_f1,
               "real_only_per_class": r_rows, "n_real_test": len(sp["test"]),
               "select_metric": args.select_metric, "best_val_metric": best_metric,
               "per_class": rows, "confusion": cm.tolist(), "labels": labels,
               "n_train": len(sp["train"]), "n_val": len(sp["val"]), "n_test": len(sp["test"]),
               "input_frames": cfg.input_frames, "input_end_offset_ms": cfg.input_end_offset_ms,
               "img_size": args.img_size, "split_mode": args.split_mode,
               "elapsed_sec": round(time.time() - t_start, 1)}
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---------- 部署元数据 ----------
    meta = {
        "arch": "efficientnet_b0", "in_frames": cfg.input_frames, "img_size": args.img_size,
        "labels": labels, "test_acc": te_acc, "macro_f1": macro_f1, "best_val_acc": best_acc,
        # ★ 2026-09-26：把「真实样本」成绩也写进 model_meta，跨模型对比时能一眼看出
        #   两栏数的差别（老目录只有全量，所以显示成 -）。
        "real_only_acc": r_acc, "real_only_macro_f1": r_f1,
        "frame_w": cfg.frame_w, "frame_h": cfg.frame_h, "sep": cfg.sep,
        "input_end_offset_ms": cfg.input_end_offset_ms, "input_stride": cfg.input_stride,
        # ★ 速度标量通道：部署端（6_自动玩.py）必须照这两个字段准备输入，否则通道数对不上
        "speed_input": bool(getattr(cfg, "use_speed", False)),
        "in_channels": cfg.input_frames * 3 + (1 if getattr(cfg, "use_speed", False) else 0),
        # 负样本的取样偏移也记下来：评估时复现同一套，指标才和训练日志对得上
        "animation_neg_end_ms": list(cfg.animation_neg_end_ms or ()),
        "approach_neg_end_ms": list(cfg.approach_neg_end_ms or ()),
        "save_pre_ms": cfg.save_pre_ms, "save_post_ms": cfg.save_post_ms,
        "save_fps": cfg.save_fps, "saved_frames": cfg.saved_frames,
        "animation_neg_enabled": cfg.animation_neg_enabled,
        "game_res": list(cfg.game_res), "game_crop": list(cfg.game_crop),
        # ★ 2026-10-14：模型自带取景（相对比例）。部署端优先读它，
        #   于是"训练不同区域的模型"可以各自自动切到正确区域。
        "geometry": make_geometry(geo_rel, geo_res, game_crop=geo_crop,
                                  note="训练时记录（来源：%s）" % geo_src),
        "backend": cfg.backend,
        # ★ 2026-10-15：这个模型属于哪个游戏、每个类别按哪个键 —— 部署(9_)照它发键。
        #   换游戏/加按键都靠这两个字段，不用改部署代码。
        "game": getattr(cfg, "game", None),
        #   只保留真正进了模型的类别，顺序无关（字典：类别 -> 键盘字符）
        "key_map": {lb: ch for ch, lb in (getattr(cfg, "key_map", None) or {}).items()
                    if lb in labels},
        "norm_mean": IMAGENET_MEAN, "norm_std": IMAGENET_STD,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"\n[输出] {os.path.abspath(args.out)}")
    log("       best.pt（验证集最优轮）/ last.pt（最后一轮）/ model_meta.json"
        " / metrics.json / history.csv / log.txt")

    # ---------- 曲线（matplotlib 可选）----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = np.array(history)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(h[:, 0], h[:, 1], label="train"); ax[0].plot(h[:, 0], h[:, 3], label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(h[:, 0], h[:, 2], label="train"); ax[1].plot(h[:, 0], h[:, 4], label="val")
        ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "curves.png"), dpi=150)
        log("[输出] curves.png")
    except ImportError:
        log("[提示] 没装 matplotlib，跳过曲线图（history.csv 里有全部数据）")

    log(f"\n总耗时 {(time.time()-t_start)/60:.1f} 分钟")
    return 0


def main(argv=None) -> int:
    """兼容入口：python -m core.trainer ... （内部就是 train()）"""
    return train(argv)


if __name__ == "__main__":
    sys.exit(main())
