# -*- coding: utf-8 -*-
"""采集后复核：删除坏样本 / 改正标错的标签。

★ 默认打开**浏览器界面**（直观、大图、点一下就改）：

    python 3_复核.py                  # 复核当前区域档案里最新的会话（自动开浏览器）
    python 3_复核.py <会话目录>        # 指定会话
    python 3_复核.py --only A         # 只看某一类
    python 3_复核.py --port 47910     # 换端口

浏览器里：左侧按类别过滤 + 样本列表，中间大图（点图或按 f 切换 适应窗口/1:1），
底部：改成 [1 A][2 D][3 N][4 Q][5 S][6 W]、上一个/下一个、删除。
快捷键：← → 翻页 · 1~6 或 A/D/N/Q/S/W 改标签 · Del 删除 · f 缩放。
改动**立即落盘**（改标签会搬 jpg + 改 meta.jsonl 那一行；删除会同时删文件和行）。

不想用浏览器：加 --win 用原来的 cv2 小窗口

    python 3_复核.py --win
    python 3_复核.py --win --no-window     # 只统计，不开窗口

为什么要复核：菜单画面、死亡结算画面、明显标错的样本会教坏模型。
采集时按 X 能现场撤销最近几个，事后就用这个工具一批批看过去。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core.review import main as win_main        # noqa: E402  (cv2 窗口版)


def _count(d: str) -> int:
    mp = os.path.join(d, "meta.jsonl")
    if not os.path.isfile(mp):
        return 0
    with open(mp, encoding="utf-8") as fh:
        return sum(1 for x in fh if x.strip())


def _candidates():
    """(档案, 当前档案里的会话列表) —— 拿不到就返回 (None, [])。"""
    try:
        from core import profiles as P
        from core.sessions import expand_sessions
    except Exception:
        return None, []
    prof = P.resolve()
    if not prof:
        return None, []
    cands, _bad = expand_sessions([P.dataset_root(prof)])
    return prof, cands


def newest_session():
    """当前区域档案里最新的、**有样本**的会话；没有样本就退回最新的那个。"""
    _prof, cands = _candidates()
    if not cands:
        return None
    with_data = [c for c in cands if _count(c) > 0]
    return max(with_data or cands, key=os.path.getmtime)


def pick_session() -> str:
    """交互式挑会话（cv2 版用；浏览器版默认不问）。"""
    prof, cands = _candidates()
    if prof:
        try:
            from core import profiles as P
            print("[档案] %s" % P.describe(prof))
        except Exception:
            pass
    if len(cands) == 1:
        print("       复核会话：%s" % os.path.basename(cands[0]))
        return cands[0]
    if len(cands) > 1:
        cands.sort(key=os.path.getmtime, reverse=True)
        print("       这个档案里有 %d 个会话：" % len(cands))
        for i, d in enumerate(cands, 1):
            print("         %d) %s   (%d 个样本)" % (i, os.path.basename(d), _count(d)))
        try:
            a = input("       复核哪一个？（直接回车 = 最新的有样本那个）: ").strip()
        except EOFError:
            a = ""
        if not a:
            return newest_session() or cands[0]
        if a.isdigit() and 1 <= int(a) <= len(cands):
            return cands[int(a) - 1]
        return a
    try:
        return input("要复核哪个会话？（例如 profiles/cs01/dataset/xxx）: ").strip()
    except EOFError:
        return ""


if __name__ == "__main__":
    argv = sys.argv[1:]

    # ---------- ① 默认：浏览器界面（直观、大图、点一下改） ----------
    if "--win" not in argv and "--cv2" not in argv:
        from core.review_web import main as web_main
        has_sess = bool([a for a in argv if not a.startswith("-")])
        if not has_sess and newest_session() is None:
            s = pick_session()          # 一个会话都没有：问一下（也会告诉你先去采集）
            if not s:
                raise SystemExit("没找到会话：先跑 2_采集.py 采一批，或把会话目录当参数传进来")
            argv = [s] + argv
        sys.exit(web_main(argv))

    # ---------- ② 老界面：--win 走 cv2 小窗口 ----------
    argv = [a for a in argv if a not in ("--win", "--cv2")]
    has_sess = bool(argv) and not argv[0].startswith("-")
    if not has_sess and not any(a in ("-h", "--help") for a in argv):
        s = pick_session()
        if not s:
            raise SystemExit("没选会话，退出")
        argv = [s] + argv
    sys.argv = [sys.argv[0]] + argv
    win_main()
