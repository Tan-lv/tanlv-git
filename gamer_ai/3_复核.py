# -*- coding: utf-8 -*-
"""采集后复核：删除坏样本 / 改正标错的标签。

在 PyCharm 里直接右键 Run 就行：
  · 不给参数 -> 默认复核**当前区域档案**里的会话（只有一个就直接用，多个列出来让你选）
  · 也可以 Run -> Edit Configurations -> Parameters 里填  dataset/run01

为什么要复核：菜单画面、死亡结算画面、明显标错的样本会教坏模型。
采集时按 X 能现场撤销最近几个，事后就用这个工具一批批看过去。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core.review import main          # noqa: E402


def pick_session() -> str:
    """不给参数时挑一个会话：优先当前区域档案，其次让你手填。"""
    try:
        from core import profiles as P
        from core.sessions import expand_sessions
    except Exception:
        P = None
    if P is not None:
        prof = P.resolve()
        if prof:
            cands, _bad = expand_sessions([P.dataset_root(prof)])
            print("[档案] %s" % P.describe(prof))
            if len(cands) == 1:
                print("       复核会话：%s" % os.path.basename(cands[0]))
                return cands[0]
            if cands:
                cands.sort(key=os.path.getmtime, reverse=True)
                print("       这个档案里有 %d 个会话：" % len(cands))
                for i, d in enumerate(cands, 1):
                    print("         %d) %s" % (i, os.path.basename(d)))
                s = input("       复核哪一个？（直接回车 = 最新那个）: ").strip()
                if not s:
                    return cands[0]
                if s.isdigit() and 1 <= int(s) <= len(cands):
                    return cands[int(s) - 1]
                return s
    return input("要复核哪个会话？（例如 dataset/run01）: ").strip()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        s = pick_session()
        if s:
            sys.argv.append(s)
        else:
            raise SystemExit("没选会话，退出")
    main()
