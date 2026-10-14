# -*- coding: utf-8 -*-
"""训练结果的定位：列出 runs/ 下可用的模型、自动挑效果最好的那个、加载它。

抽成模块而不是留在自动玩脚本里，是为了让"评估 / 时序诊断 / 注意力可视化"这些
入口都能共用一个实现 —— 而且都是 ASCII 模块名，顶层脚本改中文名也不会牵连 import。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from .recorder import resolve_path


def runs_root() -> str:
    return resolve_path("runs")


def list_runs() -> List[Tuple[str, float, float, float]]:
    """列出 runs/ 下所有可用的训练结果，按效果（macro-F1, test acc）从好到差排序。

    返回 [(目录, macro_f1, test_acc, mtime), ...]。
    下划线开头的目录是临时/对照产物，不参与自动挑选。
    """
    base = runs_root()
    if not os.path.isdir(base):
        return []
    out = []
    for d in sorted(os.listdir(base)):
        if d.startswith("_"):
            continue
        p = os.path.join(base, d)
        if not (os.path.isfile(os.path.join(p, "best.pt"))
                and os.path.isfile(os.path.join(p, "model_meta.json"))):
            continue
        f1 = acc = -1.0
        n_train = 0
        mp = os.path.join(p, "metrics.json")
        if os.path.isfile(mp):
            try:
                m = json.load(open(mp, encoding="utf-8"))
                f1 = float(m.get("macro_f1") or -1.0)
                acc = float(m.get("test_acc") or -1.0)
                n_train = int(m.get("n_train") or 0)
            except Exception:
                pass
        # 真实样本准确率 / 危险误判率 —— 来自 <run>/eval_test.json（6_评估效果.py 生成）
        real_acc = danger = -1.0
        ep = os.path.join(p, "eval_test.json")
        if os.path.isfile(ep):
            try:
                e = json.load(open(ep, encoding="utf-8"))
                real_acc = float(e["real_acc"]) if e.get("real_acc") is not None else -1.0
                danger = float(e["danger_rate"]) if e.get("danger_rate") is not None else -1.0
            except Exception:
                pass
        out.append((p, f1, acc, os.path.getmtime(p), n_train, real_acc, danger))
    # 排序：**最近训练/拉取的在最前面**。
    #
    # 为什么不按分数排：每个 run 的 accuracy / macro-F1 / eval_test.json 里的数字，
    # 都是在**它自己那套划分**上算出来的。数据集一变（加会话、重采、修复索引），
    # 划分就变，测试集之间会互相重叠 —— 实测过两个 run 的测试集有 73.6% 重合，
    # 谁见过谁就虚高，跨 run 比分数完全是误导。
    #
    # 所以默认给"你刚做出来的那个"，要换就用 --run 明确指定。
    # 表里的数字只是让你有个大概印象，别拿来跨 run 比大小。
    out.sort(key=lambda r: r[3], reverse=True)
    return out


def pick_run(quiet: bool = False) -> str:
    """没给 --run 时自动挑效果最好的那次训练结果，并把候选打印出来。"""
    runs = list_runs()
    if not runs:
        raise SystemExit("[错误] " + runs_root() + " 里没有可用的训练结果\n"
                         "       （需要 best.pt + model_meta.json）\n"
                         "       先训练一版： python 5_训练模型.py")
    if len(runs) > 1 and not quiet:
        print(f"[runs] 找到 {len(runs)} 个训练结果，最近产出的排前面：")
        for i, (p, f1, acc, _, n_tr, ra, dg) in enumerate(runs):
            tag = "   <- 选用" if i == 0 else ""
            s0 = f"{ra*100:5.1f}%" if ra >= 0 else "   ?  "
            sd = f"{dg*100:5.1f}%" if dg >= 0 else "   ?  "
            print(f"         {os.path.basename(p):<16} 真实 acc {s0}  "
                  f"该换道却输出N {sd}  训练 {n_tr:5d} 个{tag}")
        print("       ⚠ 表里的分数是各 run 在**自己那套划分**上算的，数据集一变划分就变，"
              "测试集之间会重叠，所以**不要跨 run 比大小**。")
        print("       ⚠ 实测两个 run 的测试集重合过 73.6%，谁见过谁的样本谁就虚高。")
        print("       想用别的： --run runs/xxx")
    return runs[0][0]


def load_run(run_dir: str) -> Tuple[dict, str]:
    """读一个训练结果目录，返回 (model_meta 字典, 权重文件路径)。"""
    meta_p = os.path.join(run_dir, "model_meta.json")
    if not os.path.isfile(meta_p):
        raise SystemExit(f"[错误] 找不到 {meta_p}")
    meta = json.load(open(meta_p, encoding="utf-8"))
    for name in ("best.pt", "best.pt"):
        ck = os.path.join(run_dir, name)
        if os.path.isfile(ck):
            return meta, ck
    raise SystemExit(f"[错误] {run_dir} 里没有 best.pt / best.pt")
