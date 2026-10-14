# -*- coding: utf-8 -*-
r"""模型迭代训练 —— 在**初代模型**上做「旧数据 + 新数据(含人工纠正)」的小学习率微调。

和 5_训练模型.py 的区别（用户 2026-09-24 定）：

    5_训练模型.py       从零训练（ImageNet 预训练起步），只喂 DATA_DIRS 里的数据
    14_模型迭代训练.py   **接着初代模型练**：
                          --resume 初代 best.pt   +   --no-pretrained
                          + 小学习率              +   旧/新数据来源加权
                          + 输入窗定义沿用初代模型（不让底层特征错位）

为什么这条路最稳：
    * 不要从零重训   —— 初代模型已经学会的换道手感（A/D）不该丢，
                        而且它的输入定义、类别顺序都是配套的；
    * 不要只喂修正数据 —— 新的纠正样本只有几千，撑不起重学，会直接过拟合/遗忘；
    * 旧新混合 + 来源加权 + 小 lr —— 底座保住，同时让新纠正的样本有更大发言权。

来源加权怎么起作用：
    给新数据/纠正数据的**损失权重**乘一个系数（默认 2.0），旧数据保持 1.0。
    不是丢掉旧数据，也不是复制新数据，而是让梯度里新样本的声音更大一点。
    （实现在 collector/dataset.py 的 weights + trainer 的加权 loss）

跑起来：
    python 14_模型迭代训练.py --list-runs            # 先看有哪些模型可以接着练（含绝对路径）
    python 14_模型迭代训练.py                       # 用下面写死的配置（= 从初代重新微调 30 轮）
    python 14_模型迭代训练.py --epochs 40 --lr 1e-4  # 临时改（本轮再练多少轮）
    python 14_模型迭代训练.py --continue-from 2 --epochs 20 --lr 1e-4
    python 14_模型迭代训练.py --out runs/迭代_试1

「代次」与目录命名（2026-09-26 按用户要求加）：
    初代模型 = 第 0 代。每次跑出来的结果目录叫  runs/迭代<代次>_<年月日_时分>，
    例如 迭代1_20260926_1205、迭代2_20260926_2010 —— 名字里的数字就是第几代。
    代次 = 底子模型的代次 + 1；练完还会把 generation 写进该目录的 model_meta.json，
    所以下次 --continue-from 2 / --continue-from <绝对路径> 都能准确定位。
    老目录 迭代_20260925_2226 没有代次数字，按第 1 代算。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ==============================================================================
#  ★★★  改这里  ★★★
# ------------------------------------------------------------------------------
#  初代模型：拿它的 best.pt 当底子（不是从零开始）
INIT_RUN = r"D:\deeplearning\make_dataset\paoku\runs\迭代_20260925_2226"

#  旧数据（初代模型当年吃的那批）—— 权重 1.0
OLD_DIRS = [
    r"D:\deeplearning\make_dataset\paoku\dataset\20260918_merged",
   # r"D:\deeplearning\make_dataset\paoku\dataset\20260918_103227",
]

#  新数据 + 人工纠正的数据 —— 权重 NEW_WEIGHT
#    迭代补采 写父目录就行，脚本会自动展开成 正确操作 / 碰撞操作
NEW_DIRS = [
    r"D:\deeplearning\make_dataset\paoku\dataset\20260919_195121",
    r"D:\deeplearning\make_dataset\paoku\dataset\20260920_203729",
    r"D:\deeplearning\make_dataset\paoku\dataset\迭代补采",
]

OLD_WEIGHT = 1.0        # 旧数据损失权重
NEW_WEIGHT = 2.0        # 新数据/纠正数据损失权重（越大越偏向新纠正，建议 1.5~3）

FINE_TUNE_EPOCHS = 30   # 本轮再练多少轮（会换算成 trainer 的 --epochs）
LR = 2e-4               # 微调学习率（从零训练用 1e-3；微调要小一个量级）
PATIENCE = 8            # 多少轮没提升就早停（微调收敛快，别设太大）

#  输入窗定义：沿用**初代模型**的（默认 True）
#    初代模型是 end_offset=40 练出来的；如果你改成 False，就会用当前
#    collector/config.py 的值（现在是 0）—— 输入窗会整体后移 40ms，
#    等于让初代模型重新适应一次输入，不建议在"微调"里做这件事。
KEEP_INIT_INPUT_DEF = True

#  ★ 续训：想「从某个已有模型接着练」就在这里填它 —— 推荐填**绝对路径**。
#    None = 不续训，即每次都是从初代模型重新微调 FINE_TUNE_EPOCHS 轮。
#
#    能填四种写法（都可以，看哪个顺手）：
#      ① 绝对路径   CONTINUE_FROM = r"D:\deeplearning\make_dataset\paoku\runs\迭代2_20260926_1147"
#      ② 相对路径   CONTINUE_FROM = r"runs\迭代2_20260926_1147"
#      ③ 只写代次   CONTINUE_FROM = "2"      （自动取 runs 里最新的 迭代2*）
#      ④ 上次       CONTINUE_FROM = "last"   （取 runs 里代次最高的那个）
#    不知道填哪个？跑  python 14_模型迭代训练.py --list-runs  会列出所有可续训的模型，
#    连绝对路径一起打出来，复制粘贴即可。
#
#    为什么要续训、而不是把 FINE_TUNE_EPOCHS 加大：因为 --resume 默认永远指向初代模型，
#    单纯把轮数从 30 改成 60 会让前 30 轮**原样重算**一遍（每轮 ~710 秒）。
CONTINUE_FROM = None
# CONTINUE_FROM = r"D:\deeplearning\make_dataset\paoku\runs\迭代_20260925_2226"

#  输出目录；None = runs/迭代<代次>_<时间>（代次 = 底子模型的代次 + 1，初代算第 0 代）
#    ★ 2026-09-26 按用户要求：目录名带「第几代」，一眼看出是第几轮迭代出来的
OUT = None
# ==============================================================================


def expand_sessions(paths, max_depth: int = 2):
    """把路径展开成「会话目录」列表（会话 = 自己带 meta.jsonl 的目录）。

    写成上层目录也没关系（比如 dataset\\迭代补采 下面才是 正确操作/碰撞操作）。
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
                continue
            sub = os.path.join(d, name)
            if os.path.isdir(sub) and walk(sub, depth - 1):
                found = True
        return found

    for p in paths:
        if not walk(p, max_depth):
            bad.append(p)
    return out, bad


def count_samples(session: str) -> int:
    try:
        with open(os.path.join(session, "meta.jsonl"), encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except Exception:
        return 0


def best_ckpt_epoch(path: str) -> int:
    """初代模型 best.pt 里记的是第几轮（trainer 的 --resume 会从它的下一轮接着数）。"""
    try:
        import torch
        ck = torch.load(path, map_location="cpu")
        return int(ck.get("epoch", 0))
    except Exception as e:
        print(f"[提示] 读不出初代模型的轮次（{e}），按 0 算")
        return 0


GEN_TAG_RE = re.compile(r"迭代_?(\d{1,3})(?:_|$)")


def run_generation(path: str) -> int:
    """这个模型目录算「第几代」（初代 = 第 0 代）。

    优先读 model_meta.json 里的 "generation"（本脚本练完会写进去），读不到再从目录名猜：
        迭代2_20260926_1147   -> 2   （本脚本的新命名）
        迭代_20260925_2226    -> 1   （老命名，没有代次数字，但确实是迭代出来的）
        20260918_merged_...   -> 0   （初代模型）
    注意 \d{1,3} 后面要求跟 _ 或结尾：否则 迭代_20260925 里的日期 20260925
    会被误当成代次（这条护栏就是为它加的）。
    """
    try:
        with open(os.path.join(path, "model_meta.json"), encoding="utf-8") as fh:
            g = json.load(fh).get("generation")
        if g is not None:
            return int(g)
    except Exception:
        pass
    name = os.path.basename(os.path.normpath(path))
    m = GEN_TAG_RE.search(name)
    if m:
        return int(m.group(1))
    return 1 if name.startswith("迭代") else 0


def ckpt_summary(path: str):
    """(轮次, val_f1, val_acc)。续训真正要看的 val_f1 —— best.pt 就是按它挑的。"""
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("epoch", 0)), ck.get("val_f1"), ck.get("val_acc")
    except Exception:
        return 0, None, None


def list_candidate_runs(root: str):
    """列 runs/ 下所有能当续训底子的目录（best.pt + model_meta.json 都在）。"""
    rows = []
    if not os.path.isdir(root):
        return rows
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        cp, mp = os.path.join(d, "best.pt"), os.path.join(d, "model_meta.json")
        if not (os.path.isfile(cp) and os.path.isfile(mp)):
            continue
        ep, f1, acc = ckpt_summary(cp)
        try:
            with open(mp, encoding="utf-8") as fh:
                m = json.load(fh)
        except Exception:
            m = {}
        # metrics.json 里才有「真实样本」那一栏（model_meta 只记全量口径）
        try:
            with open(os.path.join(d, "metrics.json"), encoding="utf-8") as fh:
                mj = json.load(fh)
        except Exception:
            mj = {}
        rows.append({"dir": d, "name": name, "gen": run_generation(d), "epoch": ep,
                     "val_f1": f1, "val_acc": acc,
                     "test_acc": m.get("test_acc"), "test_f1": m.get("macro_f1"),
                     "real_acc": mj.get("real_only_acc"),
                     "real_f1": mj.get("real_only_macro_f1"),
                     "mtime": os.path.getmtime(d)})
    rows.sort(key=lambda r: (r["gen"], r["mtime"]))
    return rows


def print_candidates(root: str) -> None:
    """--list-runs：把候选模型连同绝对路径打出来，方便复制粘贴。"""
    rows = list_candidate_runs(root)
    if not rows:
        print(f"[提示] {root} 下没有可续训的模型（需要同时有 best.pt 和 model_meta.json）")
        return
    print("可续训的模型（--continue-from 可填「代次」数字，也可以直接复制绝对路径）：")
    print("")
    init_default = os.path.abspath(INIT_RUN if os.path.isabs(INIT_RUN) else os.path.join(HERE, INIT_RUN))
    for r in rows:
        tag = "从零训练出来的（第 0 代）" if r["gen"] == 0 else "第 %d 代迭代" % r["gen"]
        if os.path.abspath(r["dir"]) == init_default:
            tag += "   ← 不填 continue-from 时的默认底子"
        print("  代次 %-3d  %-34s %s" % (r["gen"], r["name"], tag))
        f1v = "%.4f" % r["val_f1"] if r["val_f1"] is not None else "-"
        ta = "%.4f" % r["test_acc"] if r["test_acc"] is not None else "-"
        tf = "%.4f" % r["test_f1"] if r["test_f1"] is not None else "-"
        print("            best.pt 轮次 %-6s val宏F1 %s" % (r["epoch"] + 1, f1v))
        if r["real_acc"] is not None:
            print("            真实样本 Acc %s 宏F1 %.4f   |   全量 Acc %s 宏F1 %s"
                  % ("%.4f" % r["real_acc"], r["real_f1"], ta, tf))
        else:
            print("            只有全量口径（老格式，没记真实样本成绩） Acc %s 宏F1 %s" % (ta, tf))
        print("            绝对路径 %s" % os.path.abspath(r["dir"]))
        print("")
    print("  [两把尺子] 真实样本 = 上线时的真实水平；全量 = 含合成负样本（N 占多数，acc 会虚高）。")
    print("             跨模型比较必须用同一栏，否则会误以为「数据多了反而更差」。")
    print("  例：python 14_模型迭代训练.py --continue-from 2 --epochs 20 --lr 1e-4")
    print("  公正对比几个模型： python tools\\对比模型.py --runs <目录A> <目录B>")


def resolve_continue(spec: str, root: str):
    """把 CONTINUE_FROM / --continue-from 的写法解析成模型目录。返回 (目录, 错误)。

    支持：绝对路径、相对路径（相对本脚本所在目录）、纯数字代次（2 / 2代）、
          last / 上次（取代次最高的那个）、只写目录名（能唯一匹配就行）。
    """
    s = str(spec).strip().strip(chr(34)).strip(chr(39))
    if not s:
        return None, "填的是空字符串"
    if s.lower() in ("last", "newest", "上次", "最新"):
        cands = [r for r in list_candidate_runs(root) if r["gen"] >= 1]
        if not cands:
            return None, "runs 下还没有任何迭代模型（第 1 代都还没练出来）"
        return max(cands, key=lambda r: (r["gen"], r["mtime"]))["dir"], None
    m = re.fullmatch(r"(\d{1,3})\s*代?", s)
    if m:
        gen = int(m.group(1))
        cands = [r for r in list_candidate_runs(root) if r["gen"] == gen]
        if not cands:
            return None, "runs 下没有第 %d 代的模型" % gen
        return max(cands, key=lambda r: r["mtime"])["dir"], None
    p = s if os.path.isabs(s) else os.path.join(HERE, s)
    if os.path.isdir(p):
        return os.path.abspath(p), None
    cands = [r for r in list_candidate_runs(root) if s in r["name"]]
    if len(cands) == 1:
        return cands[0]["dir"], None
    if len(cands) > 1:
        return None, "这个名字匹配到多个模型，请写完整路径"
    return None, "找不到这个目录/模型：%s" % s


def stamp_generation(run_dir: str, gen: int) -> None:
    """把「第几代」写进新模型的 model_meta.json，下次续训就能准确读出来。"""
    p = os.path.join(run_dir, "model_meta.json")
    try:
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        m["generation"] = int(gen)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(m, fh, ensure_ascii=False, indent=2)
        print(f"[代次] 已把 generation={gen} 写进 {p}")
    except Exception as e:
        print(f"[提示] 没能写入 generation 标记（{e}）；下次只能从目录名猜代次")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--init-run", default=INIT_RUN, help="初代模型目录（含 best.pt）")
    ap.add_argument("--old-weight", type=float, default=OLD_WEIGHT)
    ap.add_argument("--new-weight", type=float, default=NEW_WEIGHT)
    ap.add_argument("--epochs", type=int, default=FINE_TUNE_EPOCHS,
                    help="本轮再练多少轮（不是总轮数）")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--continue-from", dest="continue_from", default=CONTINUE_FROM,
                    help="从哪个模型接着练：绝对路径 / 相对路径 / 代次(如 2) / last。"
                         "不填 = 每次从初代模型重新微调")
    ap.add_argument("--list-runs", action="store_true",
                    help="只列出可续训的模型（代次/轮次/指标/绝对路径），不训练")
    ap.add_argument("--no-keep-input", dest="keep_input", action="store_false",
                    default=KEEP_INIT_INPUT_DEF,
                    help="不沿用初代模型的输入窗定义，改用当前 config（会整体后移 40ms）")
    args, rest = ap.parse_known_args()

    RUNS_DIR = os.path.join(HERE, "runs")
    if args.list_runs:
        print_candidates(RUNS_DIR)
        return 0

    from collector.config import Config
    cfg = Config()

    # ---------- 1) 初代模型 ----------
    init_run = args.init_run if os.path.isabs(args.init_run) else os.path.join(HERE, args.init_run)
    continuing = False
    if args.continue_from:
        cf, err = resolve_continue(args.continue_from, RUNS_DIR)
        if cf is None:
            print(f"[错误] --continue-from 解析失败：{err}")
            print(f"       你填的是：{args.continue_from}")
            print("       用 --list-runs 看看有哪些能选。")
            return 1
        init_run, continuing = cf, True
    ckpt = os.path.join(init_run, "best.pt")
    meta_p = os.path.join(init_run, "model_meta.json")
    if not os.path.isfile(ckpt) or not os.path.isfile(meta_p):
        print(f"[错误] 初代模型不完整，缺少 best.pt 或 model_meta.json：{init_run}")
        return 1
    with open(meta_p, encoding="utf-8") as fh:
        im = json.load(fh)
    # 本次要练的是第几代 = 底子的代次 + 1（初代模型算第 0 代）
    gen = run_generation(init_run) + 1

    print("=" * 84)
    print("模型迭代训练（在初代模型上微调，不从零重训）")
    print("=" * 84)
    print(f"  {'续训底子' if continuing else '初代模型'} : {init_run}")
    if continuing:
        print("             （--continue-from：从这个模型的 best.pt 接着练，不是从初代重来）")
    print(f"  本次代次 : 第 {gen} 代（底子 = 第 {gen - 1} 代；第 0 代就是初代模型本身）")
    print(f"             最佳轮次 {best_ckpt_epoch(ckpt)} · "
          f"test_acc {im.get('test_acc', float('nan')):.4f} · "
          f"macro-F1 {im.get('macro_f1', float('nan')):.4f}")

    # ---------- 2) 输入定义必须能对上（对不上权重根本装不进去）----------
    bad = []
    if int(im.get("in_frames", -1)) != cfg.input_frames:
        bad.append(f"输入帧数：初代 {im.get('in_frames')} vs 现在 {cfg.input_frames}")
    if int(im.get("img_size", -1)) != 224:
        bad.append(f"img_size：初代 {im.get('img_size')} vs 现在 224")
    if list(im.get("labels", [])) != list(cfg.labels):
        bad.append(f"类别：初代 {im.get('labels')} vs 现在 {list(cfg.labels)}")
    if bad:
        print("\n[错误] 初代模型和当前配置对不上，微调没意义（权重装不进去）：")
        for b in bad:
            print("        " + b)
        print("       解决：改 collector/config.py 让它们一致，或者换一个匹配的初代模型。")
        return 1

    # 输入窗定义（默认沿用初代模型）
    keep = args.keep_input
    end_off = int(im.get("input_end_offset_ms", cfg.input_end_offset_ms)) if keep \
        else int(cfg.input_end_offset_ms)
    stride = int(im.get("input_stride", cfg.input_stride)) if keep else int(cfg.input_stride)
    print(f"\n  输入定义 : 6 帧 x stride {stride}，窗口结束于按键前 {end_off}ms"
          + ("（沿用初代模型）" if keep else "（用当前 config，输入窗会变）"))
    if keep and end_off != int(cfg.input_end_offset_ms):
        print(f"             [注意] 当前 config 是 {int(cfg.input_end_offset_ms)}ms，"
              f"微调先用初代模型的 {end_off}ms，免得底层特征错位；")
        print(f"                    想换成 config 的值就加 --no-keep-input")

    # ---------- 3) 数据 ----------
    old_raw, new_raw = list(OLD_DIRS), list(NEW_DIRS)
    # 同一个路径写两遍没意义（去重）
    seen = set()
    old_raw = [p for p in old_raw if not (p in seen or seen.add(p))]
    seen = set()
    new_raw = [p for p in new_raw if not (p in seen or seen.add(p))]

    old_sess, bad_o = expand_sessions(old_raw)
    new_sess, bad_n = expand_sessions(new_raw)
    for p in bad_o + bad_n:
        print(f"[错误] 这里既不是会话、里面也没有会话：{p}")
        return 1

    print(f"\n  旧数据（权重 {args.old_weight}）:")
    n_old = 0
    for s in old_sess:
        n = count_samples(s)
        n_old += n
        print(f"         {os.path.relpath(s, HERE):<46} {n:>6}")
    print(f"  新数据 + 纠正（权重 {args.new_weight}）:")
    n_new = 0
    for s in new_sess:
        n = count_samples(s)
        n_new += n
        print(f"         {os.path.relpath(s, HERE):<46} {n:>6}")
    print(f"  ------------------------------------------------------------")
    print(f"  合计 {n_old + n_new} 个样本（旧 {n_old} + 新 {n_new}）")

    if "20260918_merged" in " ".join(old_raw):
        for p in old_raw:
            if p.endswith("20260918_103227"):
                print("\n  [提醒] dataset\\20260918_merged 本身就是 0917+103227 合并出来的，")
                print("         再单独加 20260918_103227 会让那段时期**算两遍**（重复计权）。")
                print("         想更干净就把 OLD_DIRS 里 103227 那一行删掉/注释掉。")

    # ---------- 4) 组装 trainer 参数 ----------
    sessions = old_sess + new_sess
    if not sessions:
        print("[错误] 没找到任何会话")
        return 1

    # 来源加权：规则按路径子串匹配，先命中先算；旧数据保持 1.0 就不用写规则
    rules = []
    if abs(args.new_weight - 1.0) > 1e-9:
        for s in new_sess:
            rules.append(f"{os.path.basename(s)}={args.new_weight}")
    if abs(args.old_weight - 1.0) > 1e-9:
        for s in old_sess:
            rules.append(f"{os.path.basename(s)}={args.old_weight}")

    init_epoch = best_ckpt_epoch(ckpt)
    # ★ trainer 里 start_epoch = ck["epoch"] + 1，所以总轮数要再加 1，
    #   否则 range(start, total) 是空的 —— 一轮都不会练（2026-09-24 踩过）。
    total_epochs = init_epoch + 1 + max(1, args.epochs)

    out = args.out
    if out is None:
        # ★ 目录名带「第几代」：迭代2_20260926_1147
        out = os.path.join("runs", f"迭代{gen}_{time.strftime('%Y%m%d_%H%M')}")
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)

    extra = [
        "--resume", ckpt,                 # ★ 关键：从初代模型接着练
        "--no-pretrained",                # 已经有初代权重了，不需要 ImageNet
        "--lr", str(args.lr),             # ★ 小学习率
        "--epochs", str(total_epochs),    # trainer 的轮次是累计的，这里换算
        "--patience", str(args.patience),
        "--end-offset", str(end_off),     # 沿用初代模型的输入窗定义
        "--input-stride", str(stride),
        "--select-metric", "macro_f1",
    ]
    if rules:
        extra += ["--source-weight", ",".join(rules)]

    print(f"\n  第 {gen} 代微调计划 : "
          f"lr {args.lr} · 再练 {args.epochs} 轮（trainer 从第 {init_epoch + 1} 轮接着跑，"
          f"跑到第 {total_epochs} 轮）· 早停 {args.patience} 轮")
    if rules:
        print(f"  来源加权 : {','.join(rules)}")
    print(f"  输出     : {os.path.abspath(out)}")
    print("=" * 84)

    sys.argv = [os.path.abspath(__file__),
                "--sessions", *sessions, "--out", out, *extra, *rest]

    from collector import trainer
    try:
        rc = trainer.main()
    except MemoryError:
        print()
        print("[错误] 内存不足。微调同样吃内存，按顺序试：")
        print("      --workers 2 --batch-size 16 --prefetch 1")
        print("      --workers 0 --batch-size 16")
        return 2
    # ★ 练完把「第几代」写进新模型的 model_meta.json —— 下次续训就能准确认出代次，
    #   而不是靠猜目录名（目录名是可以被改的，元数据更可靠）。
    if rc == 0:
        stamp_generation(out, gen)
    return rc


if __name__ == "__main__":
    sys.exit(main())
