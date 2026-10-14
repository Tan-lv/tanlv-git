# -*- coding: utf-8 -*-
"""本地训练入口 —— 在 PyCharm 里直接运行（Run）就能训练，不用配 PYTHONPATH。

    python 5_训练模型.py --dry-run                # 只看这次会用到哪些会话、输出到哪，不训练
    python 5_训练模型.py                          # 默认：dataset/ 下所有会话
    python 5_训练模型.py --session dataset/xxx    # 只练指定的会话（可以给多个）
    python 5_训练模型.py --out runs/exp03 --epochs 80 --speed-balance 0.5

★ 换训练数据：改下面 DATA_DIRS。默认 "auto" = dataset/ 下所有会话（新采的数据自动进来，
  不用每次手改路径）。想只练某几个会话就写成路径列表，或命令行 --session。

★ 类别（A/D/N/Q/S/W）**不在这里配**：trainer 直接从数据里读，数据里有哪几类就训哪几类。
  所以以后再加动作键（比如这次的 Q）也只要采集时按到就行，这里一个字都不用改。

真正的训练实现在 collector/trainer.py，
这里只负责：① 把 paoku/ 加进 sys.path；② 给 --sessions/--out 补默认值。

训练完的产物在 --out 目录下：
    best.pt / best.pt            模型权重
    model_meta.json              部署所需参数（9_自动玩.py 读这个）
    metrics.json / history.csv   指标与逐轮记录
    log.txt / curves.png         训练日志与曲线
之后用 6_评估效果.py 做详细评估，或者 9_自动玩.py --dry-run 空跑看实时预测。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ==============================================================================
#  ★★★  训练数据来源  ★★★
# ------------------------------------------------------------------------------
#  "auto"（默认） = dataset/ 下**所有**会话。
#      ★ 2026-10-10 改成默认：转横屏时旧的会话清单已清空，写死路径每次都要手改，
#        改成 auto 之后新采的数据自动进训练集，不会漏。
#      · 自动跳过名字以 _ 或 . 开头的目录（例如 _selftest 自检产物）；
#      · 0 个样本的空会话会被跳过（只建了目录没采到东西的那种）；
#      · 支持子目录（dataset\迭代补采\正确操作 这类会自动展开）。
#
#  字符串          = 只练这一个。可以写绝对路径、相对路径，或者**只写会话名**
#                    （DATA_DIRS = "20261011_204728" 会自动去 dataset\ 下找）。
#  None            = 只挑 dataset/ 里最近修改的那一个会话（老行为，一般不用）。
#
#  路径列表         = 只练列出的那些，可以写绝对路径，也可以写上层目录：
#      DATA_DIRS = [
#          r"D:\deeplearning\make_dataset\paoku\dataset\迭代补采",
#          r"D:\deeplearning\make_dataset\paoku\dataset\20261010_204802",
#      ]
#
#  命令行 --session 仍然可以临时覆盖这里。
# ==============================================================================
# ==============================================================================
#  ★★★  哪些样本**不要**参与"合成 N 负样本"  ★★★
# ------------------------------------------------------------------------------
#  迭代采集的两种样本性质完全不同：
#    · 正确操作会话：模型自己在跑的时候顺手存的，前后帧都是正常跑图，
#      可以照旧合成"换道动画中 / 障碍还远"的 N 负样本。
#    · 碰撞操作会话：你在 12_标注界面.py 里**从撞车那一刻挑帧标的**，
#      它前后那些帧全是撞车 / 被警察追的画面 —— 当成"此刻该不动"的 N
#      是**标签噪声**（用户 2026-09-24 提出）。这类样本只当正样本用。
#
#  默认按「会话名」+「样本 kind」两路排除，一般不用改。
# ==============================================================================
NO_SYNTH_NEG_SESSIONS = ["碰撞操作"]     # 会话名（dataset\迭代补采\碰撞操作）
NO_SYNTH_NEG_KINDS = ["relabel"]         # 12_标注界面.py 人工挑帧写样本时的 kind


#  "auto" = dataset/ 下所有会话（见上面说明）。想只练特定几个会话就换成路径列表。
DATA_DIRS = "20261011_204728"


def newest_session(root: str = "dataset") -> str:
    """dataset/ 下修改时间最新的一个会话目录。"""
    base = root if os.path.isabs(root) else os.path.join(HERE, root)
    if not os.path.isdir(base):
        raise SystemExit(f"[错误] 找不到数据目录 {base}，请先用 2_采集数据.py 采集数据")
    cands = [os.path.join(base, d) for d in os.listdir(base)
             if os.path.isfile(os.path.join(base, d, "meta.jsonl"))]
    if not cands:
        raise SystemExit(f"[错误] {base} 下没有任何会话（没有 meta.jsonl），请先采集数据")
    return max(cands, key=os.path.getmtime)


def list_sessions(root: str = "dataset"):
    """dataset/ 下所有可用的会话目录（含 meta.jsonl 的），新的排前面。"""
    base = root if os.path.isabs(root) else os.path.join(HERE, root)
    if not os.path.isdir(base):
        return []
    ds = [os.path.join(base, d) for d in os.listdir(base)
          if os.path.isfile(os.path.join(base, d, "meta.jsonl"))]
    return sorted(ds, key=os.path.getmtime, reverse=True)


def as_data_list(v):
    """把 DATA_DIRS 归一成 None / "auto" / [路径...]。

    ★ 2026-10-11 踩过的坑：DATA_DIRS 是字符串时**绝不能**直接 list(DATA_DIRS)，
      那会把 "20261011_204728" 拆成 15 个字符 -> paoku\2、paoku\0 … 全报错。
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
        return [s]                     # 单个路径（或裸会话名）
    return list(v)


def resolve_data_path(p: str, root: str) -> str:
    """把 DATA_DIRS 里的一项解析成绝对路径。

    支持：绝对路径 / 相对 paoku\ 的路径 / 只写会话名（自动去 dataset\ 下找）。
    """
    if os.path.isabs(p):
        return p
    under_here = os.path.join(HERE, p)
    if os.path.exists(under_here):
        return under_here
    under_root = os.path.join(HERE, root, p)
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


def expand_sessions(paths, max_depth: int = 2):
    """把用户写的路径展开成「会话目录」列表。

    ★ 会话目录 = 自己带 meta.jsonl 的目录。
      以前只认这一种，所以写 dataset\迭代补采（它的子目录才是会话）会直接报错。
      现在写成上层目录也没关系：自动往下找最多 max_depth 层，把找到的会话都收进来。

    返回 (会话列表, 什么都找不到的路径列表)
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


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--session", nargs="+", default=None,
                    help="数据集会话目录，可以给多个（新旧数据混着练也支持）；"
                         "不给就用 dataset/ 里最新的一个")
    ap.add_argument("--root", default="dataset", help="数据集根目录，默认 dataset")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印这次会用到哪些会话、输出到哪，不真的开始训练")
    args, rest = ap.parse_known_args()

    # 数据来源优先级： 命令行 --session  >  DATA_DIRS  >  只挑最近一个
    # ★ 归一化：字符串=单个路径、"auto"=全都要、列表=多个（见 as_data_list）
    spec = as_data_list(DATA_DIRS)
    auto_mode = (not args.session and spec == "auto")
    if args.session:
        sessions = list(args.session)
    elif auto_mode:
        sessions = [args.root]          # 直接把 dataset/ 交给 expand_sessions 递归展开
    elif spec:
        sessions = list(spec)
    else:
        sessions = [newest_session(args.root)]
    # PyCharm 的 Working directory 不一定对：相对路径按 paoku/ 解析，
    # 只写会话名的（"20261011_204728"）会去 dataset\ 下找
    raw = [os.path.abspath(resolve_data_path(s, args.root)) for s in sessions]
    # ★ 上层目录自动展开成里面的会话（写 dataset\迭代补采 也能用了）
    sessions, bad = expand_sessions(raw)
    if bad:
        print()
        for s in bad:
            print(f"[错误] 这里既不是会话、里面也没找到会话：{s}")
        print("       可用的会话（把路径抄到文件顶部的 DATA_DIRS 即可）：")
        for q in list_sessions(args.root):
            print(f"         {q}   ({count_samples(q)} 个样本)")
        print("\n       也可以在代码里写 DATA_DIRS = \"auto\"（用 dataset/ 下所有会话），")
        print("       或者只写会话名，例如 DATA_DIRS = \"20261011_204728\"；")
        print("       命令行也行：python 5_训练模型.py --session 20261011_204728")
        return 1
    # auto 模式下剔掉 0 样本的空会话（例如只建了目录没采到东西）—— 它们只会让训练日志多一行 0
    if auto_mode:
        empty = [s for s in sessions if count_samples(s) == 0]
        if empty:
            print(f"[数据] 跳过 {len(empty)} 个空会话："
                  + "、".join(os.path.basename(s) for s in empty))
        sessions = [s for s in sessions if count_samples(s) > 0]
    if not sessions:
        print("\n[错误] dataset/ 下没有任何带样本的会话，请先用 2_采集数据.py 采集数据。")
        return 1
    print(f"[数据] 找到 {len(sessions)} 个会话：")
    for s in sessions:
        n = count_samples(s)
        print(f"         {s}   ({n} 个样本)" + ("   ← 空的，先跑 12_标注界面.py 挑帧" if n == 0 else ""))
    if not any(count_samples(s) for s in sessions):
        print("\n[错误] 这些会话里一个样本都没有，没什么可训的。")
        return 1

    # 没给 --out 就自动取名 runs/<会话名>_<时间>
    out = None
    for i, a in enumerate(rest):
        if a == "--out" and i + 1 < len(rest):
            out = rest[i + 1]
        elif a.startswith("--out="):
            out = a.split("=", 1)[1]
    if out is None:
        if len(sessions) == 1:
            name = os.path.basename(os.path.abspath(sessions[0].rstrip("\\/")))
        else:
            name = "%d个会话" % len(sessions)
        out = os.path.join("runs", f"{name}_{time.strftime('%Y%m%d_%H%M')}")
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)

    print("=" * 78)
    print("本地训练")
    print(f"  数据: {[os.path.abspath(s) for s in sessions]}")
    print(f"  输出: {os.path.abspath(out)}")
    print("=" * 78)
    if args.dry_run:
        print("[dry-run] 以上是这次训练会用到的东西，没有真的开始训练。")
        print("          确认无误就去掉 --dry-run 重跑。")
        return 0

    # 把参数拼回去，交给 collector/trainer.py 的 main()
    # argv[0] 用本脚本的名字，这样 --help 里显示的是中文入口
    extra = []
    if NO_SYNTH_NEG_SESSIONS:
        extra += ["--no-synth-neg-sessions", ",".join(NO_SYNTH_NEG_SESSIONS)]
    if NO_SYNTH_NEG_KINDS:
        extra += ["--no-synth-neg-kinds", ",".join(NO_SYNTH_NEG_KINDS)]
    sys.argv = [os.path.abspath(__file__),
                "--sessions", *sessions, "--out", out, *extra, *rest]

    from collector import trainer
    try:
        return trainer.main()
    except MemoryError:
        # DataLoader worker 里的 MemoryError 会在主进程重新抛出。
        # 与其丢一堆堆栈，不如直接告诉他该改哪个参数。
        print()
        print("=" * 78)
        print("[错误] 内存不足，训练被打断。")
        print("=" * 78)
        print("  原因：每个 DataLoader worker 是一个独立进程，光 import torch 就要约 600MB，")
        print("        再乘以 workers 个数；加上预取缓冲（batch_size 越大越多）和模型本身。")
        print("  8 workers + batch 64 实测要 6GB 以上，笔记本上还开着模拟器/PyCharm 就会爆。")
        print()
        print("  按顺序试这几组（越来越省内存）：")
        print("      --workers 4 --batch-size 32 --prefetch 1     约 2.9 GB   （现在的默认值）")
        print("      --workers 2 --batch-size 16 --prefetch 1     约 1.6 GB")
        print("      --workers 0 --batch-size 16                  最省，但会明显变慢")
        print()
        print("  训练前那行 [loader] 会预估内存占用，可以先照着调。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
