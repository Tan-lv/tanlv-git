# -*- coding: utf-8 -*-
"""本地评估：加载训练好的 best.pt，给出比 core/trainer.py 更细的"效果报告"。

用法（PyCharm 里直接运行即可）：
    python 5_评估.py --run runs/exp01 --sessions dataset/20260915_211905

和 core/trainer.py 末尾那段评估的区别：
  * **把"真实样本"和"合成的换道动画负样本"分开统计**
    —— 后者就是"角色已经在换道动画里了，模型却还在输出方向键"（重复触发）
       的直接指标，是判断模型能不能上线的关键数字
  * 给出"危险误判"（该按方向键却输出 N，会直接撞死）的比例
  * 实测单帧推理延迟，判断 6_自动玩.py 能不能跑到 40fps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.dataset import (MultiFrameDataset, build_animation_negatives,
                               build_approach_negatives, build_eval_transform,
                               load_sessions, negatives_for_subset, split_indices)
from core.model import build_model
from core.runs import pick_run


# ==============================================================================
#  ★★★  要评估哪个模型 / 用哪份数据：改下面这两行就行  ★★★
# ------------------------------------------------------------------------------
#  MODEL_DIR —— 评估哪个模型，三种写法随便挑：
#      None                                      = 自动用【当前区域档案】runs/ 里最新的那个
#      r"D:\...\gamer_ai\profiles\cs01\runs\横屏_v1"   = 绝对路径（最直白）
#      r"profiles\cs01\runs\横屏_v1"                    = 相对 gamer_ai/ 的路径
#      "横屏_v1"                                  = 只写目录名，自动去各档案的 runs/ 里找
#    （命令行 --run xxx 优先级最高，临时换一个用不着改文件）
#
#  SESSIONS —— 用哪份数据评估；
#      None                                       = 当前区域档案 dataset/ 里全部会话
#      [r"profiles\cs01\dataset\20261011_204728"]   = 指定会话（可给多个）
#
#  想先看有哪些模型：python 5_评估.py --list-runs
# ==============================================================================
MODEL_DIR = None
SESSIONS = None


def resolve_model_path(p, runs_root_hint=None) -> str:
    """把 MODEL_DIR 的写法解析成真实模型目录（绝对路径 / 相对 gamer_ai / 只写目录名）。"""
    HERE = os.path.dirname(os.path.abspath(__file__))
    cands = []
    if os.path.isabs(p):
        cands.append(p)
    else:
        cands.append(os.path.join(HERE, p))
        if runs_root_hint:
            cands.append(os.path.join(runs_root_hint, p))
        cands.append(os.path.join(HERE, "runs", p))
        profs = os.path.join(HERE, "profiles")
        if os.path.isdir(profs):                    # 各区域档案的 runs/ 里找同名目录
            for d in sorted(os.listdir(profs)):
                cands.append(os.path.join(profs, d, "runs", p))
    for c in cands:
        if os.path.isfile(os.path.join(c, "model_meta.json")):
            return c
    print("[错误] 找不到模型目录：%s" % p)
    print("       可用的（把路径抄到文件开头的 MODEL_DIR）：")
    for c in cands[:12]:
        print("         %s%s" % (c, "" if os.path.isdir(c) else "   （目录不存在）"))
    raise SystemExit(2)


def load_cfg(sessions, args) -> Config:
    """和 core/trainer.py 一样：读采集时的 session.json 作为基准配置。"""
    cfg = Config()
    for d in sessions:
        p = os.path.join(d, "session.json")
        if os.path.isfile(p):
            raw = json.load(open(p, encoding="utf-8")).get("config", {})
            valid = {fl.name for fl in fields(Config)}
            for k, v in raw.items():
                if k in valid:
                    if v is not None and k in ("game_res", "game_crop", "labels", "crop_rel",
                                               "animation_neg_end_ms", "approach_neg_end_ms"):
                        v = tuple(v)
                    setattr(cfg, k, v)
            break
    if args.img_size:
        pass
    cfg.validate()
    return cfg


def run_batches(model, ds, device, bs, truths, kinds):
    """按顺序跑完整个数据集，返回 (预测数组, 各类概率数组)。"""
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            hi = min(i + bs, len(ds))
            xs = torch.stack([ds[j][0] for j in range(i, hi)])
            out = model(xs.to(device))
            p = torch.softmax(out.float(), dim=1)
            preds.append(out.argmax(1).cpu().numpy())
            probs.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def per_class(y_true, y_pred, labels):
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
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
    return cm, rows


def measure_latency(model, device, in_frames, img_size, n=60):
    x = torch.randn(1, in_frames * 3, img_size, img_size, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
    return dt * 1000.0


def recommend_thresholds(probs, y_true, y_pred, labels, m_real, run_dir, args) -> None:
    """用这次评估的数据**自动推荐置信度阈值**，并写 模型评估/thresholds_建议.json。

    ★ 2026-10-16（用户："thresholds.json 是怎么生成的手动还是评估自动？手动调阈值不准"）：
      评估手里正好有"每个样本的置信度 + 真实标签"，直接扫一遍阈值就行 —— 不用手动试。

    规则（和 6_自动玩.py 部署时完全一致）：置信度 < 阈值 = 不按键（等于输出不动）。
        · 真实是动作类：按了且预测对 = 有效动作；按了但预测错 = 动作错误；没按 = 漏动
        · 真实是不动（N）：没按 = 正确不动；按了 = 误触发
    选总正确率最高的阈值；同时把漏动率一起打出来（那是最容易撞死的错）。

    注意：这是在**同一份评估数据**上挑阈值，天然偏乐观。正式上线建议在 val 上挑、在 test 上看。
    """
    import numpy as _np
    labels = list(labels)
    n_idx = labels.index("N") if "N" in labels else -1
    real = _np.asarray(m_real)[:len(y_true)]
    yt, yp = _np.asarray(y_true)[real], _np.asarray(y_pred)[real]
    cf = _np.asarray(probs)[real].max(axis=1)
    if not len(yt):
        return
    should = (yt != n_idx) if n_idx >= 0 else _np.ones(len(yt), bool)
    rows = []
    for t in _np.arange(0.50, 0.996, 0.01):
        acts = (yp != n_idx if n_idx >= 0 else _np.ones(len(yp), bool)) & (cf >= t)
        ok = (acts & (yp == yt)) | (~acts & ~should)
        miss = should & ~acts
        rows.append((float(t), float(ok.mean()), float(miss.mean())))
    best = max(rows, key=lambda r: (r[1], -r[2]))
    print()
    print("=" * 78)
    print("【阈值推荐 · 用这次评估的置信度分布自动算】")
    print("=" * 78)
    print("  %-8s %-12s %s" % ("阈值", "总正确率", "该动却没动(漏动)"))
    for t, ok, miss in rows:
        if abs((t * 100) % 10) < 1e-6:            # 每 0.1 打一行，够看趋势
            mark = "  <- 推荐" if abs(t - best[0]) < 1e-9 else ""
            print("  %-8.2f %-12.4f %.1f%%%s" % (t, ok, miss * 100, mark))
    per = {}
    for _lab in labels:
        sub = [r[0] for r in zip(cf, yp, yt) if False]
    # 按动作给"建议值"：动作类稍微保守一点（+0.05），不定则用总体值
    sug = {"default": round(best[0], 2)}
    for _lab in labels:
        if _lab == "N":
            continue
        m = (yt == labels.index(_lab))
        if m.sum() >= 5:
            sub = rows
            sug[_lab] = round(min(0.99, best[0] + 0.05), 2)
    print("  推荐阈值 default=%.2f（总正确率 %.4f，漏动 %.1f%%）"
          % (best[0], best[1], best[2] * 100))
    print("  对比：当前 default=0.85 时 总正确率 %.4f，漏动 %.1f%%"
          % next(((o, m) for t, o, m in rows if abs(t - 0.85) < 1e-9), (float("nan"), float("nan"))))
    _ed = os.path.join(run_dir, "模型评估")
    os.makedirs(_ed, exist_ok=True)
    _p = os.path.join(_ed, "thresholds_建议.json")
    json.dump(sug, open(_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  已写出建议：%s" % _p)
    if getattr(args, "write_thresholds", False):
        _rp = os.path.join(run_dir, "thresholds.json")
        if os.path.isfile(_rp):
            import shutil, time as _t
            shutil.copy2(_rp, _rp + ".bak_" + _t.strftime("%Y%m%d_%H%M%S"))
        json.dump(sug, open(_rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("  已应用（--write-thresholds）：%s" % _rp)
    else:
        print("  想直接应用： python 5_评估.py --write-thresholds")
    print("=" * 78)


def run_auto_cam(args) -> None:
    """评估结束自动出"每类抽样"注意力热力图（默认开，--no-cam 关）。

    ★ 2026-10-16：以前这段直接写在 main() 末尾，结果"模型没有 N 类"那条早退分支
      会先 return，热力图就不生成了（用户："热力图片怎么没有生成"）。抽成函数后
      两条路径都调用它。
    """
    if not getattr(args, "cam_auto", True) or getattr(args, "cam", False):
        return
    try:
        from core import attention_report as _AR
        _adir = os.path.join(args.run, "模型评估", "attention")
        print("\n[注意力] 自动出「每类抽样」热力图（每类一张最自信的样本 -> 拼成一张）…")
        print("         想跳过加 --no-cam；想自己挑样本用 --cam --ids …")
        _rc = _AR.main(["--run", args.run, "--sessions", *list(args.sessions),
                        "--out", _adir])
        if _rc:
            print("[注意力] 出图失败（返回码 %s），不影响上面的评估结果" % _rc)
    except SystemExit as _e:
        print("[注意力] 跳过：%s" % _e)
    except Exception as _e:
        print("[注意力] 出图出错（不影响评估结果）：%s" % _e)


def main():
    ap = argparse.ArgumentParser(description="本地评估：模型效果诊断")
    ap.add_argument("--run", default=None,
                    help="要评估的模型目录（临时换用；不填就看文件开头的 MODEL_DIR，"
                         "MODEL_DIR 也是 None 就用当前区域档案 runs/ 里最新的）")
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="数据集会话目录；不填就用 dataset/ 里最新的一个")
    ap.add_argument("--split", default="test", choices=["test", "val", "all"],
                    help="在哪个集合上评估（默认 test）")
    # ★ 2026-10-15：这两个诊断原来各占一个入口脚本（4_速度覆盖度.py / 7_时序诊断.py）。
    #   用户反馈"不怎么用、能不能合到一起"，于是合并到这里：功能一模一样，少两个编号脚本。
    ap.add_argument("--coverage", action="store_true",
                    help="只看【速度覆盖度】（原来的 4_速度覆盖度.py）：高速段采够了没有")
    ap.add_argument("--timing", action="store_true",
                    help="只看【时序诊断】（原来的 7_时序诊断.py）：模型比人早多久开始动")
    # ★ 2026-10-15：原来的 8_注意力可视化.py 合并进来了 ——
    #   --cam  离线出 Grad-CAM 热力图（挑样本看图/批量出图）
    #   --live 实时可视化（开着模拟器进关卡，边跑边看模型盯哪儿）
    ap.add_argument("--write-thresholds", action="store_true",
                    help="把评估算出的阈值建议直接写进模型目录的 thresholds.json（旧文件自动备份）")
    ap.add_argument("--no-cam", dest="cam_auto", action="store_false", default=True,
                    help="评估完**不**自动出每类抽样热力图（默认会出，放在 <模型>/模型评估/attention/）")
    ap.add_argument("--cam", action="store_true",
                    help="看模型在看画面哪里（Grad-CAM 热力图，原 8_注意力可视化.py）")
    ap.add_argument("--live", action="store_true",
                    help="实时注意力可视化（需要模拟器在关卡里；按 q 退出）")
    ap.add_argument("--out", default=None, help="配合 --cam：出图目录（默认 <模型>/模型评估/attention）")
    ap.add_argument("--limit", type=int, default=0,
                    help="配合 --timing：只抽这么多个样本（默认 0=全部，全量要跑 10~20 分钟，"
                         "先用 --limit 300 看趋势）")
    ap.add_argument("--profile", default=None,
                    help="区域档案名：默认评估该档案 runs/ 里最新的模型、用该档案 dataset/ 的数据")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--segment-seconds", type=float, default=15.0)
    ap.add_argument("--split-mode", default="by_session",
                    help="划分方式：by_session（默认，按会话/局整段划分，最诚实）、"
                         "time_balanced（会话内按时间段分，val 会虚高）、time_block、random")
    ap.add_argument("--boundary-gap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-anim-neg", action="store_true")
    ap.add_argument("--no-approach-neg", action="store_true")
    # ★ 2026-10-15：--cam/--live 的专属参数（--ids/--img/--grep/--random…）不在这里重列一遍，
    #   改成"已知参数自己用，剩下的原样透传给注意力模块"。
    args, rest = ap.parse_known_args()

    # PyCharm 里 Working directory 不一定对，相对路径一律按 gamer_ai/ 解析
    HERE = os.path.dirname(os.path.abspath(__file__))
    def _abs(p):
        return p if os.path.isabs(p) else os.path.abspath(os.path.join(HERE, p))
    # ★ 2026-10-15 区域档案：模型和数据都默认取这个档案里的
    from core import profiles as P
    from core.sessions import expand_sessions, resolve_data_path
    prof = P.resolve(args.profile)
    _rroot = P.runs_root(prof) if prof else None
    if prof:
        print("[档案] %s" % P.describe(prof))
        print("       模型目录：%s" % _rroot)
        if not args.sessions:
            args.sessions = [P.dataset_root(prof)]
            print("       数据目录：%s（自动展开成里面的会话）" % args.sessions[0])
    # ---------- 子功能①：注意力可视化（原 8_注意力可视化.py）----------
    if args.cam or args.live:
        from core import attention_report as AR
        argv3 = list(rest)
        if args.live:
            argv3.append("--live")
        # 没给 --run 就按当前区域档案挑一个模型（runs 现在在 profiles/<档案>/runs 里，
        # 注意力模块自己不知道档案，所以这里解析好再传给它）
        if not args.run and _rroot:
            from core.runs import pick_run as _pick
            try:
                args.run = _pick(quiet=True, root=_rroot)
                print("[模型] %s" % args.run)
            except SystemExit as _e:
                print(_e)
        if args.run:
            argv3 += ["--run", args.run]
        # 档案给的是 dataset 根目录，注意力模块要的是**会话目录**，先展开（同 --timing）
        _ses = list(args.sessions or [])
        if _ses:
            _ses = [resolve_data_path(s) if not os.path.isabs(s) else s for s in _ses]
            _ses, _bad = expand_sessions(_ses)
        if _ses:
            argv3 += ["--sessions"] + _ses
        if args.out:
            argv3 += ["--out", args.out]
        if args.limit:
            argv3 += ["--limit", str(args.limit)]
        print()
        return AR.main(argv3)

    if rest:
        raise SystemExit("[参数] 不认识这些参数：%s\n       看用法：python 5_评估.py --help"
                         % " ".join(rest))

    # ---------- 两个"子功能"：直接转交给对应模块，不动模型评估那套流程 ----------
    if args.coverage:
        from core import coverage
        # 档案给的是 dataset 根目录，coverage 要的是**会话目录**，先展开（同 --timing）
        _ses = list(args.sessions or ([P.dataset_root(prof)] if prof else []))
        _ses = [resolve_data_path(s) if not os.path.isabs(s) else s for s in _ses]
        _ses, _bad = expand_sessions(_ses)
        if not _ses:
            raise SystemExit("[错误] 速度覆盖度要看哪个会话？加 --sessions dataset/xxx"
                             "（或先用自检建一个区域档案）")
        print()
        return coverage.main(_ses)

    if args.timing:
        from core import timing
        argv2 = ["--run", args.run] if args.run else []
        if args.limit:
            argv2 += ["--limit", str(args.limit)]
        if args.sessions:
            argv2 += ["--sessions"] + list(args.sessions)
        if prof:
            argv2 += ["--profile", prof]
        print()
        return timing.main(argv2)

    if not args.run and MODEL_DIR:
        print("[模型] 用文件开头 MODEL_DIR 指定的：%s" % MODEL_DIR)
        args.run = resolve_model_path(MODEL_DIR, _rroot)
    if not args.run:
        args.run = pick_run(root=_rroot)
        print()
    args.run = _abs(args.run)
    if not args.sessions and SESSIONS:
        args.sessions = list(SESSIONS) if isinstance(SESSIONS, (list, tuple)) else [SESSIONS]
        print("[数据] 用文件开头 SESSIONS 指定的：%s" % args.sessions)
    if not args.sessions:
        root = os.path.join(HERE, "dataset")
        cands = [os.path.join(root, d) for d in os.listdir(root)
                 if not d.startswith("_")
                 and os.path.isfile(os.path.join(root, d, "meta.jsonl"))]
        if not cands:
            raise SystemExit("[错误] dataset/ 下没有会话，请用 --sessions 指定")
        args.sessions = [max(cands, key=os.path.getmtime)]
        print(f"[data] 没指定 --sessions，自动选用最新会话: {args.sessions[0]}")
    # 会话路径解析：支持绝对路径 / 相对 gamer_ai\ / dataset\xxx / 档案的 dataset 根目录
    #   ★ 以前这里只做 os.path.abspath(os.path.join(HERE, p))，所以写 "dataset/xxx"
    #     会解析成 gamer_ai/dataset/xxx（对），但写档案根目录就找不到 meta.jsonl 直接报错。
    raw = [resolve_data_path(s) if not os.path.isabs(s) else s for s in args.sessions]
    args.sessions, _bad = expand_sessions(raw)
    if _bad:
        print("[错误] 这些路径里没找到会话（没有 meta.jsonl）：")
        for b in _bad:
            print("       %s" % b)
        raise SystemExit(2)
    if not args.sessions:
        raise SystemExit("[错误] 没有可评估的会话")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = os.path.join(args.run, "best.pt")
    if not os.path.isfile(ck_path):
        ck_path = os.path.join(args.run, "last.pt")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    labels = ck["labels"]
    in_frames = ck["in_frames"]
    print("=" * 78)
    print(f"模型: {ck_path}")
    print(f"设备: {device}" + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    print(f"类别: {labels}   输入: 连续 {in_frames} 帧   训练时 val_acc {ck.get('val_acc', 0):.4f}")
    print("=" * 78)

    cfg = load_cfg(args.sessions, args)
    # ★ 2026-11-05：速度标量通道 —— 必须**在构建数据集之前**决定，
    #   否则输入少一个通道，模型直接报 shape 不匹配。
    _meta = {}
    _mf = os.path.join(args.run, "model_meta.json")
    if os.path.isfile(_mf):
        try:
            _meta = json.load(open(_mf, encoding="utf-8"))
        except Exception:
            _meta = {}
    cfg.use_speed = bool(_meta.get("speed_input", False))
    # ★ 2026-11-05：**必须**用模型训练时的窗口定义，不能用采集时 session.json 里的。
    #   训练用了 end_offset=102（正样本窗口提前 102ms），评估还按 0 取帧的话，
    #   等于换了考卷 —— 实测同一份权重能测出 0.78 和 0.71 两个差 7 个点的结果。
    if _meta.get("input_end_offset_ms") is not None:
        cfg.input_end_offset_ms = int(_meta["input_end_offset_ms"])
    if _meta.get("animation_neg_end_ms"):
        cfg.animation_neg_end_ms = tuple(float(x) for x in _meta["animation_neg_end_ms"])
    if _meta.get("approach_neg_end_ms"):
        cfg.approach_neg_end_ms = tuple(float(x) for x in _meta["approach_neg_end_ms"])
    if args.no_anim_neg:
        cfg.animation_neg_enabled = False
    samples = load_sessions(args.sessions)
    print(f"[data] {len(samples)} 个样本   {dict(sorted(Counter(s['label'] for s in samples).items()))}")
    # ★ 2026-10-15：模型训练时会剔除样本太少的类（例如只有 6 个样本的 E），那份类别表
    #   存在 checkpoint 的 labels 里。**过滤必须放在"划分之后"**（和 trainer 一样）：
    #   如果先删样本再划分，"按时间分段"的段边界会整体重排 —— 实测先删 6 个 E 之后，
    #   test 集合和训练时的 test 只重叠 12%，等于换了张考卷（同一份权重能测出
    #   4.9% 和 21% 两个完全不同的"该换道却输出 N"）。

    if args.split == "all":
        idx = list(range(len(samples)))
    else:
        sp = split_indices(samples, train_ratio=1 - args.val_ratio - args.test_ratio,
                           val_ratio=args.val_ratio, mode=args.split_mode,
                           boundary_gap=args.boundary_gap, seed=args.seed,
                           segment_seconds=args.segment_seconds)
        idx = sp[args.split]
    if args.no_anim_neg:
        cfg.animation_neg_enabled = False
    if args.no_approach_neg:
        cfg.approach_neg_enabled = False
    # 模型没训过的类在这里剔除（下标映射要一起维护，负样本锚点仍允许用全部样本）
    kept = [i for i in idx if samples[i]["label"] in labels]
    if len(kept) != len(idx):
        _cnt = Counter(samples[i]["label"] for i in idx if samples[i]["label"] not in labels)
        print(f"[data] 跳过模型未训练的类别 {dict(sorted(_cnt.items()))}，共 {len(idx) - len(kept)} 个样本"
              f"（这些类没进模型；想评它们要在训练时加 --min-train-samples 0）")
    sub = [samples[i] for i in kept]
    # 负样本分两类，顺序必须和 negatives_for_subset 一致（先动画、后障碍还远）
    pos = {g: loc for loc, g in enumerate(kept)}
    anim_local = [(pos[g], e) for g, e in build_animation_negatives(samples, cfg, labels)
                  if g in pos]
    appr_local = [(pos[g], e) for g, e in build_approach_negatives(samples, cfg, labels)
                  if g in pos]
    negs = anim_local + appr_local
    print(f"[data] 评估集合 '{args.split}': {len(sub)} 个真实样本 + "
          f"{len(anim_local)} 个换道动画负样本 + {len(appr_local)} 个障碍还远负样本"
          f" = {len(sub) + len(negs)}")

    ds = MultiFrameDataset(sub, cfg, build_eval_transform(args.img_size), args.img_size,
                           labels=labels, negatives=negs)
    n_real = len(sub)
    kinds = np.array(["real"] * n_real + ["anim_neg"] * len(anim_local)
                     + ["approach_neg"] * len(appr_local))
    y_true = np.array([ds.label_to_idx[ds.samples[j]["label"]] if j < n_real
                       else ds.label_to_idx["N"] for j in range(len(ds))])

    # ★ 2026-11-05：按模型 meta 决定通道数（3*K 或 3*K+1 带速度标量通道）
    _in_ch = int(_meta.get("in_channels") or (in_frames * 3))
    model = build_model(in_frames, len(labels), pretrained=False,
                        in_channels=_in_ch).to(device)
    model.load_state_dict(ck["model_state_dict"])
    y_pred, probs = run_batches(model, ds, device, args.batch_size, y_true, kinds)

    acc = float((y_pred == y_true).mean())
    cm, rows = per_class(y_true, y_pred, labels)
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    print()
    print(f"整体 Accuracy  {acc:.4f}      macro-F1  {macro_f1:.4f}")
    print()
    print(f"  {'类别':<5}{'样本':>6}{'precision':>11}{'recall':>9}{'f1':>8}")
    for r in rows:
        print(f"  {r['label']:<5}{r['n']:>6}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}")
    print()
    print("混淆矩阵 (行=真实, 列=预测):")
    print("        " + "".join(f"{l:>7}" for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l:<5} " + "".join(f"{cm[i, j]:>7d}" for j in range(len(labels))))

    m_real = kinds == "real"
    m_neg = kinds == "anim_neg"
    m_app = kinds == "approach_neg"
    # ★ 2026-10-16：模型可能**没有 N 类**（N 样本太少被 min-train-samples 当稀有类剔除了）。
    #   以前这里直接 labels.index("N") 会 ValueError 崩掉；而且没有 N 时，两类合成负样本
    #   根本不会被生成（core/dataset.py 里 "N" not in labels -> return []），
    #   所以"N 相关指标"没有任何意义 —— 打印一份说明，把能给的数字给完就收工。
    if "N" not in labels:
        print()
        print("=" * 78)
        print("[警告] 这个模型**没有 N 类**（'别动'）")
        print("=" * 78)
        print("  原因：训练时 N 样本太少（< --min-train-samples，默认 10），被当稀有类剔除了。")
        print("  连带后果：两类合成 N 负样本也一个都不会生成（core/dataset.py：")
        print("            'N' not in labels -> 负样本生成器直接返回空）。")
        print("  实际影响：模型对每一帧都**必须**输出 A/D/S/W 里的一个 —— 学不会'别动'，")
        print("            换道动画里会重复触发、障碍还远就乱动，只能靠部署端锁定硬压。")
        print("  怎么办：补采 n（正常玩时每 1~2 秒按一下，或用 2_采集.py 的 F9 自动补 N）到 100+ 后重训；")
        print("          只做验证可以加 --min-train-samples 0 把 N 保留进来（9 个样本权重会爆，别上线）。")
        print("  因此：『该换道却输出 N』『换道动画负样本』『障碍还远负样本』这几项无法评估，已跳过。")
        print("=" * 78)
        # 延迟这里单独算一次（正常的计算在后面那段，这里提前返回了）
        lat = measure_latency(model, device, in_frames, args.img_size)
        print("单张推理延迟 %.1f ms  ->  理论上限 %.0f fps   (6_自动玩.py 需要 >= 40fps)"
              % (lat, 1000.0 / max(lat, 1e-6)))
        _ed = os.path.join(args.run, "模型评估")
        os.makedirs(_ed, exist_ok=True)
        json.dump({"run": os.path.abspath(args.run), "split": args.split,
                   "n_real": int(m_real.sum()), "accuracy": acc, "macro_f1": macro_f1,
                   "real_acc": (float((y_pred[m_real] == y_true[m_real]).mean())
                                if m_real.any() else None),
                   "danger_rate": None, "no_N": True,
                   "per_class": rows, "confusion": cm.tolist(), "labels": labels,
                   "latency_ms": lat},
                  open(os.path.join(_ed, f"eval_{args.split}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print()
        print("[输出] %s" % os.path.join(_ed, f"eval_{args.split}.json"))
        recommend_thresholds(probs, y_true, y_pred, labels, m_real, args.run, args)
        run_auto_cam(args)        # ★ 没有 N 类也要出热力图（默认就出）
        return 0

    print()
    print("=" * 78)
    print("关键指标：两类合成负样本上的表现")
    print("=" * 78)
    n_idx = labels.index("N")

    if m_real.any():
        print(f"真实样本           n={m_real.sum():4d}  acc={(y_pred[m_real] == y_true[m_real]).mean():.4f}")
        # 危险误判：该按方向键，模型却输出 N（直接撞死）
        danger = m_real & (y_true != n_idx) & (y_pred == n_idx)
        print(f"  其中『该换道却输出 N』(会撞死)  {danger.sum()} / "
              f"{(m_real & (y_true != n_idx)).sum()}  "
              f"= {danger.sum() / max((m_real & (y_true != n_idx)).sum(), 1) * 100:.1f}%")
    if m_neg.any():
        ok = (y_pred[m_neg] == n_idx).mean()
        print(f"换道动画负样本     n={m_neg.sum():4d}  输出 N 的比例 = {ok:.4f}"
              f"   (越高越好；低 = 动画中还会继续按方向键)")
        wrong = y_pred[m_neg][y_pred[m_neg] != n_idx]
        if len(wrong):
            c = Counter(labels[i] for i in wrong)
            print(f"  动画中被误判成的动作: {dict(sorted(c.items()))}")
    if m_app.any():
        ok = (y_pred[m_app] == n_idx).mean()
        print(f"障碍还远负样本     n={m_app.sum():4d}  输出 N 的比例 = {ok:.4f}"
              f"   (越高越好；低 = 障碍还远就忍不住要动，即\"动作偏早\")")
        wrong = y_pred[m_app][y_pred[m_app] != n_idx]
        if len(wrong):
            c = Counter(labels[i] for i in wrong)
            print(f"  提前想动的动作分布: {dict(sorted(c.items()))}")
    else:
        print("障碍还远负样本     n=   0   （数据里没有：save_pre_ms 太短，或开关关了）")
    # 平均 N 概率
    pn = probs[:, n_idx]
    if m_real.any():
        print(f"N 概率均值: 真实 N 样本 {pn[m_real & (y_true == n_idx)].mean():.3f}   "
              f"真实方向样本 {pn[m_real & (y_true != n_idx)].mean():.3f}   "
              f"动画负样本 {pn[m_neg].mean() if m_neg.any() else float('nan'):.3f}   "
              f"障碍还远 {pn[m_app].mean() if m_app.any() else float('nan'):.3f}")

    # ---------- 按游戏速度分桶（只算真实样本）----------
    # ★ 2026-10-15 补：文档一直说会按速度分桶，代码里其实没有。游戏越跑越快，
    #   "平均还行、高速段必死"是这类模型最典型的失败方式，只报一个总数看不出来。
    #   桶边界取真实样本速度的 20/40/60/80 分位点（自适应，不用手填阈值）。
    if n_real:
        sp_arr = np.array([float(s.get("speed_px_s") or 0.0) for s in sub], dtype=float)
        nz = sp_arr > 0
        if int(nz.sum()) >= 20:
            qs = np.percentile(sp_arr[nz], [20, 40, 60, 80])
            edges = [-1.0] + [float(q) for q in qs] + [float("inf")]
            yt, yp = y_true[:n_real], y_pred[:n_real]
            print()
            print("按游戏速度分桶（真实样本；分位点 %s px/s）"
                  % "/".join("%.0f" % q for q in qs))
            print("  %-18s%8s%11s%14s" % ("速度区间 px/s", "样本", "Accuracy", "该换道却输出N"))
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                msk = nz & (sp_arr > lo) & (sp_arr <= hi)
                if not msk.any():
                    continue
                a_b = float((yp[msk] == yt[msk]).mean())
                d_b = (yt[msk] != n_idx)
                dg_b = float(((yp[msk] == n_idx) & d_b).sum() / max(int(d_b.sum()), 1))
                lab = ("%.0f~%.0f" % (max(lo, 0.0), hi)) if np.isfinite(hi) else ("%.0f+" % max(lo, 0.0))
                print("  %-18s%8d%11.4f%13.1f%%" % (lab, int(msk.sum()), a_b, dg_b * 100))

    lat = measure_latency(model, device, in_frames, args.img_size)
    print()
    print(f"单张推理延迟 {lat:.1f} ms  ->  理论上限 {1000 / lat:.0f} fps   "
          f"(6_自动玩.py 需要 >= 40fps，即 <= 25ms)  {'OK' if lat <= 25 else '偏慢'}")

    out = {"run": os.path.abspath(args.run), "split": args.split, "n_real": int(m_real.sum()),
           "n_anim_neg": int(m_neg.sum()), "accuracy": acc, "macro_f1": macro_f1,
           # 下面两个才是决定"能不能拿来玩"的指标（被大量负样本稀释的 macro_f1 不算）
           "real_acc": (float((y_pred[m_real] == y_true[m_real]).mean())
                        if m_real.any() else None),
           "danger_rate": (float(((y_pred == n_idx) & (y_true != n_idx)
                                 & m_real).sum()
                                / max(int(((y_true != n_idx) & m_real).sum()), 1))
                           if m_real.any() else None),
           "per_class": rows, "confusion": cm.tolist(), "labels": labels,
           "anim_neg_n_rate": float((y_pred[m_neg] == n_idx).mean()) if m_neg.any() else None,
           "n_approach_neg": int(m_app.sum()),
           "approach_neg_n_rate": (float((y_pred[m_app] == n_idx).mean())
                                   if m_app.any() else None),
           "latency_ms": lat}
    # ★ 2026-10-15（用户要求）：评估产物统一收进模型目录下的「模型评估」文件夹，
    #   模型目录根只留 best.pt / last.pt / model_meta.json / metrics.json 这些训练产物。
    _eval_dir = os.path.join(args.run, "模型评估")
    os.makedirs(_eval_dir, exist_ok=True)
    p = os.path.join(_eval_dir, f"eval_{args.split}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[输出] {p}")

    recommend_thresholds(probs, y_true, y_pred, labels, m_real, args.run, args)
    # ★ 评估结束自动出"每类抽样"热力图（默认开；--no-cam 关）—— 见 run_auto_cam()
    run_auto_cam(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
