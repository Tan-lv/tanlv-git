# -*- coding: utf-8 -*-
r"""会话目录的解析与展开（4_训练.py 和 5_评估.py 共用）。

★ 2026-10-15 从 4_训练.py 里抽出来：数据集搬进区域档案之后，
  脚本拿到的往往是**档案的 dataset 根目录**（里面才是各个会话），
  两处各写一份展开逻辑迟早会不一致，所以统一放这里。

一个"会话目录" = 自己带 meta.jsonl 的目录（例如
    gamer_ai/profiles/cs01/dataset/20261011_204728/）。
用户写上层目录（档案的 dataset\、dataset\迭代数据）也能自动往下展开。
"""
from __future__ import annotations

import os
from typing import List, Tuple

PAOKU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def as_data_list(v):
    """把 DATA_DIRS 归一成 None / "auto" / [路径...]。

    ★ 2026-10-11 踩过的坑：DATA_DIRS 是字符串时**绝不能**直接 list(DATA_DIRS)，
      那会把 "20261011_204728" 拆成 15 个字符 -> gamer_ai\2、gamer_ai\0 … 全报错。
      字符串一律当成【一个路径】（除非它就是 "auto"）。
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.lower() == "auto":
            return "auto"
        return [s]
    return list(v)


def resolve_data_path(p: str, root: str = "dataset") -> str:
    """把一项数据路径解析成绝对路径。

    支持：绝对路径 / 相对 gamer_ai\ 的路径（profiles\cs01\dataset）/
    只写会话名（自动去 dataset\ 下找）。
    """
    if os.path.isabs(p):
        return p
    under_here = os.path.join(PAOKU_DIR, p)
    if os.path.exists(under_here):
        return under_here
    under_root = os.path.join(PAOKU_DIR, root, p)
    if os.path.exists(under_root):
        return under_root
    return under_here                   # 都不存在 -> 交给后面的报错逻辑


def count_samples(s: str) -> int:
    """一个会话里有几个样本。"""
    try:
        with open(os.path.join(s, "meta.jsonl"), encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def expand_sessions(paths, max_depth: int = 2) -> Tuple[List[str], List[str]]:
    """把用户写的路径展开成「会话目录」列表。返回 (会话列表, 找不到的路径列表)。

    会话目录 = 自己带 meta.jsonl 的目录；给上层目录（档案的 dataset\、
    dataset\迭代数据）也没关系，自动往下找最多 max_depth 层。
    """
    out, bad = [], []

    def walk(d: str, depth: int) -> bool:
        if os.path.isfile(os.path.join(d, "meta.jsonl")):
            if d not in out:
                out.append(d)
            return True
        if depth <= 0 or not os.path.isdir(d):
            return False
        found = False
        try:
            names = sorted(os.listdir(d))
        except Exception:
            return False
        for name in names:
            if name.startswith((".", "_")):
                continue                      # .cache / _selftest 之类跳过
            sub = os.path.join(d, name)
            if os.path.isdir(sub) and walk(sub, depth - 1):
                found = True
        return found

    for p in paths:
        if not walk(p, max_depth):
            bad.append(p)
    return out, bad


ITER_DIR_NAME = "迭代数据"          # AI 自己跑出来的正确/错误数据都放这个目录下
ITER_DIR_OLD = "迭代补采"           # 2026-10-15 之前的名字（见到就自动改名）


def ensure_iter_dir(dataset_root: str) -> str:
    """拿到（必要时创建）"迭代数据"目录，并兼容改名前的老目录。

    ★ 2026-10-15 用户要求：AI 自己跑出来的数据要和"同一分辨率的数据"放在一起，
      目录名统一叫 **迭代数据**（以前叫 迭代补采）。

    结构（在区域档案里就是 profiles/<档案>/dataset/ 下）：
        dataset/
            20261011_204728/     手动采集的会话
            迭代数据/
                正确操作/        7_迭代采集.py --keep-correct 写的
                碰撞操作/        8_标注界面.py 人工标完写的
    """
    new = os.path.join(dataset_root, ITER_DIR_NAME)
    old = os.path.join(dataset_root, ITER_DIR_OLD)
    if not os.path.isdir(new) and os.path.isdir(old):
        try:
            os.rename(old, new)
            print("[数据] 老目录名 %s 已自动改名为 %s" % (ITER_DIR_OLD, ITER_DIR_NAME))
        except OSError as e:
            print("[数据] 改名失败（%s），继续用老目录 %s" % (e, old))
            return old
    os.makedirs(new, exist_ok=True)
    return new

def list_sessions(root: str = "dataset") -> List[str]:
    """root 下所有会话目录（含 meta.jsonl），新的排前面。"""
    base = root if os.path.isabs(root) else os.path.join(PAOKU_DIR, root)
    if not os.path.isdir(base):
        return []
    ds = [os.path.join(base, d) for d in os.listdir(base)
          if os.path.isfile(os.path.join(base, d, "meta.jsonl"))]
    return sorted(ds, key=os.path.getmtime, reverse=True)


def newest_session(root: str = "dataset") -> str:
    """root 下修改时间最新的一个会话目录。"""
    base = root if os.path.isabs(root) else os.path.join(PAOKU_DIR, root)
    if not os.path.isdir(base):
        raise SystemExit(f"[错误] 找不到数据目录 {base}，请先用 2_采集.py 采集数据")
    cands = [os.path.join(base, d) for d in os.listdir(base)
             if os.path.isfile(os.path.join(base, d, "meta.jsonl"))]
    if not cands:
        raise SystemExit(f"[错误] {base} 下没有任何会话（没有 meta.jsonl），请先采集数据")
    return max(cands, key=os.path.getmtime)
