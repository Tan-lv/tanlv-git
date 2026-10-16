# -*- coding: utf-8 -*-
"""本地训练入口 —— 在 PyCharm 里直接运行（Run）就能训练，不用配 PYTHONPATH。

    python 4_训练.py --dry-run                # 只看这次会用到哪些会话、输出到哪，不训练
    python 4_训练.py                          # 默认：dataset/ 下所有会话
    python 4_训练.py --session dataset/xxx    # 只练指定的会话（可以给多个）
    python 4_训练.py --out runs/exp03 --epochs 80 --speed-balance 0.5

★ 换训练数据：改下面 DATA_DIRS。默认 "auto" = dataset/ 下所有会话（新采的数据自动进来，
  不用每次手改路径）。想只练某几个会话就写成路径列表，或命令行 --session。

★ 类别（A/D/N/Q/S/W）**不在这里配**：trainer 直接从数据里读，数据里有哪几类就训哪几类。
  所以以后再加动作键（比如这次的 Q）也只要采集时按到就行，这里一个字都不用改。

真正的训练实现在 core/trainer.py，
这里只负责：① 把 gamer_ai/ 加进 sys.path；② 给 --sessions/--out 补默认值。

训练完的产物在 --out 目录下：
    best.pt / best.pt            模型权重
    model_meta.json              部署所需参数（6_自动玩.py 读这个）
    metrics.json / history.csv   指标与逐轮记录
    log.txt / curves.png         训练日志与曲线
之后用 5_评估.py 做详细评估，或者 6_自动玩.py --dry-run 空跑看实时预测。
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
#      · 支持子目录（dataset\迭代数据\正确操作 这类会自动展开）。
#
#  字符串          = 只练这一个。可以写绝对路径、相对路径，或者**只写会话名**
#                    （DATA_DIRS = "20261011_204728" 会自动去 dataset\ 下找）。
#  None            = 只挑 dataset/ 里最近修改的那一个会话（老行为，一般不用）。
#
#  路径列表         = 只练列出的那些，可以写绝对路径，也可以写上层目录：
#      DATA_DIRS = [
#          r"D:\deeplearning\make_dataset\gamer_ai\dataset\迭代数据",
#          r"D:\deeplearning\make_dataset\gamer_ai\dataset\20261010_204802",
#      ]
#
#  命令行 --session 仍然可以临时覆盖这里。
# ==============================================================================

# ==============================================================================
#  【一】换游戏：改这一行   ★★★ 用户 2026-10-15 要求 ★★★
# ------------------------------------------------------------------------------
#  这里填 games\ 下的**文件名**（不带 .py）：
#        "subway_surfers"   -> games\subway_surfers.py   地铁跑酷（当前）
#
#  以后玩别的游戏：把 games\subway_surfers.py 复制成 games\<你的游戏>.py，
#  改掉里面的 ACTION_KEYS / MARKER_KEYS（按键表）和 TRAIN（默认训练参数），
#  再把下面这行换成 <你的游戏> 即可 —— 采集 / 训练 / 部署会自动跟着换。
#
#  看有哪些游戏：python 2_采集.py --list-games
#  数据/模型仍然按"区域档案"分开存，不同游戏各用各的档案，不会串。
# ==============================================================================
GAME = "subway_surfers"


# ==============================================================================
#  【二】训练参数：直接在这儿改   ★★★ 常用参数都列出来了 ★★★
# ------------------------------------------------------------------------------
#  只写你想改的；没写的用 games\<GAME>.py 里的默认值；命令行参数优先级最高。
#  改完直接 Run，不用记命令行参数。
# ==============================================================================
TRAIN = {
    "epochs": 300,              # 训练轮数（一轮约 1.5~4 分钟，看数据量；早停会自动提前结束）
    "batch_size": 32,          # 批大小（显存不够就调小，比如 16）
    "lr": 1e-3,                # 学习率
    "weight_decay": 1e-3,      # 权重衰减（防过拟合）
    "label_smoothing": 0.05,   # 标签平滑
    # ---- 类别不平衡：地铁跑酷踩过的坑就在这儿 ----
    "class_weight_max": 5.0,   # 类别权重上限。0 = 不截断（曾把稀有类权重放大到 218，模型练崩）
    "min_train_samples": 10,   # 训练集里少于这么多个样本的类别直接不参与训练（剔除稀有类）
    # ---- 合成 N 负样本的占比（★ 自动按比例封顶）----
    #   负样本 = 从每个按键样本的胶片条里合成的"换道动画中 / 障碍还远"，标签是 N。
    #   它们会随数据集线性膨胀（上次占到训练条目的 71%），太多会：训练变慢 + 模型变怂。
    #   0.5 = 负样本最多是正样本的一半（N 占训练集 ≤ 33%）
    #   1.0 = 负样本 ≤ 正样本（N ≤ 50%，默认）
    #   0   = 不限（不推荐）
    "neg_ratio": 0.3,
    # ---- 数据来源权重（新旧数据混着练时用）----
    #   写法："20261011_204728=1.0,迭代数据=2.0"   键 = 会话名/路径片段，值 = 权重
    "source_weight": None,
    # ---- 挑最优 / 早停 ----
    "select_metric": "macro_f1",   # 用哪个指标挑 best.pt：macro_f1 或 acc
    "patience": 20,                # 连续多少轮没提升就停
    # 下面这个一般不用改
    "img_size": 224,
}

# ==============================================================================
#  ★★★  哪些样本**不要**参与"合成 N 负样本"  ★★★
# ------------------------------------------------------------------------------
#  迭代采集的两种样本性质完全不同：
#    · 正确操作会话：模型自己在跑的时候顺手存的，前后帧都是正常跑图，
#      可以照旧合成"换道动画中 / 障碍还远"的 N 负样本。
#    · 碰撞操作会话：你在 8_标注界面.py 里**从撞车那一刻挑帧标的**，
#      它前后那些帧全是撞车 / 被警察追的画面 —— 当成"此刻该不动"的 N
#      是**标签噪声**（用户 2026-09-24 提出）。这类样本只当正样本用。
#
#  默认按「会话名」+「样本 kind」两路排除，一般不用改。
# ==============================================================================
NO_SYNTH_NEG_SESSIONS = ["碰撞操作"]     # 会话名（dataset\迭代数据\碰撞操作）
NO_SYNTH_NEG_KINDS = ["relabel"]         # 8_标注界面.py 人工挑帧写样本时的 kind


#  "auto" = dataset/ 下所有会话（见上面说明）。想只练特定几个会话就换成路径列表。
#
#  ★ 2026-10-15 结构整理：现役数据搬进了**区域档案**
#     -> gamer_ai\profiles\cs01\dataset\20261011_204728
#     · 只要当前区域档案还在（自检拉框保存时会自动设），这行用不上：数据直接取档案的 dataset\；
#     · 这行是"档案指针没了"时的兜底，所以写档案里的目录（会自动展开成里面的会话）。
DATA_DIRS = r"profiles\cs01\dataset"


# ★ 2026-10-15 结构整理：下面这套「会话解析/展开」逻辑搬到了 core/sessions.py，
#   因为 5_评估.py 也要用同一套（它以前不会展开档案的 dataset\，直接报错）。
from core.sessions import (as_data_list, count_samples, expand_sessions,  # noqa: E402
                                list_sessions, newest_session, resolve_data_path)


def train_params_to_argv(params: dict) -> list:
    """把 TRAIN 字典变成 trainer 的命令行参数。

        {"epochs": 60, "lr": 1e-3}  ->  ["--epochs", "60", "--lr", "0.001"]
        {"amp": True}               ->  ["--amp"]
        {"amp": False}              ->  ["--no-amp"]
        None 的项直接跳过（= 用默认值）

    ★ 2026-10-15（用户要求"训练参数直接在 py 里改"）：写参数名就行，
      不用记命令行写法；没写的项不会传，交给 games/<游戏>.py 的默认值。
    """
    out = []
    for k, v in (params or {}).items():
        if v is None:
            continue
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            out.append(flag if v else "--" + str(k).replace("_", "-").replace("--", "--no-", 1))
        else:
            out += [flag, str(v)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--session", nargs="+", default=None,
                    help="数据集会话目录，可以给多个（新旧数据混着练也支持）；"
                         "不给就用 dataset/ 里最新的一个")
    ap.add_argument("--root", default="dataset", help="数据集根目录，默认 dataset")
    ap.add_argument("--profile", default=None,
                    help="区域档案名：数据取 gamer_ai/profiles/<名字>/dataset/，"
                         "模型默认也存到该档案的 runs/ 下。不给就用当前档案")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印这次会用到哪些会话、输出到哪，不真的开始训练")
    ap.add_argument("--path", dest="path_hint", default=None,
                    help="也可以用它指定：区域档案目录（含 profile.json）/ 档案名 / 会话目录；"
                         "不给就用当前区域档案")
    args, rest = ap.parse_known_args()

    # ★ 2026-10-15 区域档案：--profile 给定时，数据固定用这个档案的 dataset/，
    #   模型默认也落到这个档案的 runs/ 下 —— 一块区域的数据和模型配套放在一起。
    from core import profiles as P
    #   位置参数 path：档案目录 / 档案名 / 会话目录，三种都能吃
    # ★ 2026-10-16：原来用位置参数 nargs="?" 收路径，结果 "--epochs 1" 里的 "1"
    #   会被 argparse 当成路径（未知选项的值被位置参数吃掉）→ 报"看不懂这个路径：1"。
    #   现在两条路：① --path xxx 显式给；② 自己扫 argv，只认**真的存在**的档案目录/档案名/会话目录。
    _path = getattr(args, "path_hint", None)
    if _path is None:
        for _a in list(rest):
            if _a.startswith("-"):
                continue
            if os.path.isdir(_a) and (os.path.isfile(os.path.join(_a, "profile.json"))
                                      or os.path.isfile(os.path.join(_a, "meta.jsonl"))):
                _path = _a
            else:
                try:
                    if P.exists(P.sanitize(_a)):
                        _path = _a
                except Exception:
                    pass
            if _path:
                rest = [x for x in rest if x != _path]
                break
    if _path:
        if os.path.isdir(_path) and os.path.isfile(os.path.join(_path, "profile.json")):
            args.profile = os.path.basename(os.path.abspath(_path).rstrip("\\/"))
            print("[路径] %s -> 区域档案 %s" % (_path, args.profile))
        elif os.path.isdir(_path) and os.path.isfile(os.path.join(_path, "meta.jsonl")):
            args.session = [os.path.abspath(_path)]
            print("[路径] %s -> 会话目录" % _path)
        elif P.exists(P.sanitize(_path)):
            args.profile = P.sanitize(_path)
            print("[路径] %s -> 区域档案 %s" % (_path, args.profile))
        else:
            _names = "、".join(x.get("name") for x in P.list_profiles()) or "（还没有档案）"
            raise SystemExit("[路径] 看不懂这个路径：%s\n       可以是区域档案目录 / 档案名 / "
                             "会话目录。现有档案：%s" % (_path, _names))
    prof = P.resolve(args.profile)

    # 数据来源优先级： 命令行 --session（你当场指定，最高） >  --profile 的 dataset/
    #                   >  DATA_DIRS  >  只挑最近一个
    #   ★ 输出目录只要给了 --profile 就落在档案的 runs/ 下（数据怎么来的不影响归属）。
    # ★ 归一化：字符串=单个路径、"auto"=全都要、列表=多个（见 as_data_list）
    spec = as_data_list(DATA_DIRS)
    auto_mode = (not args.session and not prof and spec == "auto")
    if prof:
        print("[档案] %s" % P.describe(prof))
        print("       模型目录：%s" % P.runs_root(prof))
    if args.session:
        sessions = list(args.session)
        if prof:
            print("       （数据用你 --session 指定的，不看档案的 dataset/）")
    elif prof:
        sessions = [P.dataset_root(prof)]
        print("       数据目录：%s" % sessions[0])
        # ★ 2026-10-15：AI 自己跑出来的「迭代数据」跟着模型走（runs/<模型>/迭代数据/），
        #   训练时自动一起收进来 —— 不用手动往 DATA_DIRS 里加路径。
        _runs = P.runs_root(prof)
        _iters = []
        if os.path.isdir(_runs):
            for _d in sorted(os.listdir(_runs)):
                _it = os.path.join(_runs, _d, "迭代数据")
                if os.path.isdir(_it):
                    _iters.append(_it)
        if _iters:
            sessions += _iters
            print("       迭代数据：%s"
                  % "、".join(os.path.basename(os.path.dirname(x)) for x in _iters))
    elif auto_mode:
        sessions = [args.root]          # 直接把 dataset/ 交给 expand_sessions 递归展开
    elif spec:
        sessions = list(spec)
    else:
        sessions = [newest_session(args.root)]
    # PyCharm 的 Working directory 不一定对：相对路径按 gamer_ai/ 解析，
    # 只写会话名的（"20261011_204728"）会去 dataset\ 下找
    raw = [os.path.abspath(resolve_data_path(s, args.root)) for s in sessions]
    # ★ 上层目录自动展开成里面的会话（写 dataset\迭代数据 也能用了）
    sessions, bad = expand_sessions(raw)
    if bad:
        print()
        for s in bad:
            print(f"[错误] 这里既不是会话、里面也没找到会话：{s}")
        print("       可用的会话（把路径抄到文件顶部的 DATA_DIRS 即可）：")
        for q in list_sessions(args.root):
            print(f"         {q}   ({count_samples(q)} 个样本)")
        print("\n       也可以在代码里写 DATA_DIRS = \"auto\"（用 dataset/ 下所有会话），")
        print("       或者写档案里的目录，例如 DATA_DIRS = r\"profiles\\cs01\\dataset\"；")
        print("       命令行也行：python 4_训练.py --profile cs01")
        return 1
    # auto 模式下剔掉 0 样本的空会话（例如只建了目录没采到东西）—— 它们只会让训练日志多一行 0
    if auto_mode:
        empty = [s for s in sessions if count_samples(s) == 0]
        if empty:
            print(f"[数据] 跳过 {len(empty)} 个空会话："
                  + "、".join(os.path.basename(s) for s in empty))
        sessions = [s for s in sessions if count_samples(s) > 0]
    if not sessions:
        print("\n[错误] dataset/ 下没有任何带样本的会话，请先用 2_采集.py 采集数据。")
        return 1
    print(f"[数据] 找到 {len(sessions)} 个会话：")
    for s in sessions:
        n = count_samples(s)
        print(f"         {s}   ({n} 个样本)" + ("   ← 空的，先跑 8_标注界面.py 挑帧" if n == 0 else ""))
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
        if prof:
            name = prof                      # 档案里的模型统一叫 <档案名>_<时间>
        if prof:
            out = os.path.join(P.runs_root(prof), f"{name}_{time.strftime('%Y%m%d_%H%M')}")
        else:
            out = os.path.join("runs", f"{name}_{time.strftime('%Y%m%d_%H%M')}")
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)

    # ★ 2026-10-15：把【一】GAME 和【二】TRAIN 起作用。
    #   优先级（后者覆盖前者）：games/<游戏>.py 的 TRAIN  <  本文件的 TRAIN  <  命令行
    from core import gamecfg as _G
    _game = _G.load(GAME)
    print("[游戏] %s" % _G.describe(_game))
    _game_train = dict((_game or {}).get("TRAIN") or {})
    # 合并成一份：游戏文件打底 -> 本文件 TRAIN 覆盖（只覆盖你写了的项）-> 生成参数
    _merged = {**_game_train, **{k: v for k, v in TRAIN.items() if v is not None}}
    _params_argv = train_params_to_argv(_merged)
    # 命令行里写的同名参数优先级最高 —— 打印时必须体现出来，否则你会以为没生效
    #（真正的 argv 里它们排在最后，argparse 取最后一个值，所以确实生效）
    _cli = {}
    for _i, _a in enumerate(rest):
        if _a.startswith("--") and _i + 1 < len(rest) and not rest[_i + 1].startswith("--"):
            _cli[_a[2:].replace("-", "_")] = rest[_i + 1]
    _eff = {**_merged, **_cli}
    _shown = ["%s=%s" % (k, _eff[k]) for k in
              ("epochs", "batch_size", "lr", "class_weight_max", "min_train_samples")
              if k in _eff]
    _from_cli = [k for k in _cli if k in _eff]
    print("[训练参数] %s" % "  ".join(_shown))
    print("           来源：games\\%s.py 打底，4_训练.py 的 TRAIN 覆盖，命令行最高"
          % (_game or {}).get("_name", GAME))
    if _merged.get("source_weight"):
        print("           数据来源权重：%s" % _merged["source_weight"])
    if _from_cli:
        print("           命令行覆盖了：%s"
              % " ".join("--%s %s" % (k.replace("_", "-"), _cli[k]) for k in _from_cli))
    if args.dry_run:
        print("           完整参数（game + py，命令行会再覆盖）：%s" % " ".join(_params_argv))
    print("=" * 78)
    print("本地训练")
    print(f"  数据: {[os.path.abspath(s) for s in sessions]}")
    print(f"  输出: {os.path.abspath(out)}")
    print("=" * 78)
    if args.dry_run:
        print("[dry-run] 以上是这次训练会用到的东西，没有真的开始训练。")
        print("          确认无误就去掉 --dry-run 重跑。")
        return 0

    # 把参数拼回去，交给 core/trainer.py 的 main()
    # argv[0] 用本脚本的名字，这样 --help 里显示的是中文入口
    extra = []
    if NO_SYNTH_NEG_SESSIONS:
        extra += ["--no-synth-neg-sessions", ",".join(NO_SYNTH_NEG_SESSIONS)]
    if NO_SYNTH_NEG_KINDS:
        extra += ["--no-synth-neg-kinds", ",".join(NO_SYNTH_NEG_KINDS)]
    # ★ 2026-10-15：训练代码在 core/trainer.py 里，这里像**调用函数**一样调它
    #   （trainer.train(argv) 就是训练主流程；argv 不传就用命令行参数）。
    argv = ["--sessions", *sessions, "--out", out,
            "--game", (_game or {}).get("_name", GAME) or "",
            *_params_argv, *extra, *rest]

    from core import trainer
    try:
        return trainer.train(argv)
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
