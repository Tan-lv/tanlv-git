# -*- coding: utf-8 -*-
r"""区域档案管理（列出 / 切换 / 新建 / 删除）。

一块画面区域 = 一个档案目录 gamer_ai/profiles/<名字>/，里面有：

    geometry.json   这块区域（自检时鼠标拉框选出来的）
    profile.json    档案信息（创建时间 / 标定分辨率 / 备注）
    dataset/        用这块区域采到的数据
    runs/           用这些数据训出来的模型

用法（在 PyCharm 里直接 Run = 列出来看看）：

    python tools\管理区域档案.py                     # 列出所有档案，标出当前用的那个
    python tools\管理区域档案.py --use cs01       # 切换当前档案
    python tools\管理区域档案.py --new 竖屏OLD        # 新建一个空档案（之后自检拉框标定）
    python tools\管理区域档案.py --rename 旧名 新名    # 改名（目录/元信息/当前指针一起改）
    python tools\管理区域档案.py --show cs01      # 看它的路径和内容
    python tools\管理区域档案.py --delete 竖屏OLD     # 删除（会连数据/模型一起删）
    python tools\管理区域档案.py --delete 竖屏OLD --keep-files   # 只删档案信息，留数据

★ 平时不用跑它：1_自检.py 里拉框保存时会自动建档并设为当前档案。
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from core import profiles as P        # noqa: E402


def list_all() -> None:
    cur = P.current()
    items = P.list_profiles()
    print("=" * 78)
    print("区域档案（gamer_ai/profiles/）")
    print("=" * 78)
    if not items:
        print("  （还没有档案）")
        print("")
        print("  怎么建：跑 1_自检.py -> 窗口里鼠标拖两个对角 -> 按 s")
        print("          -> 弹出的命名窗里起个名字 -> 点【确定保存】")
        return
    print("  %-22s %-10s %-8s %-11s %s" % ("名字", "数据会话", "模型", "标定", "备注"))
    for it in items:
        nm = it.get("name") or os.path.basename(it["dir"])
        mark = "★当前" if nm == cur else ""
        res = it.get("game_res")
        res_s = ("%sx%s" % tuple(res)) if res else "-"
        print("  %-22s %-10d %-8d %-11s %s  %s"
              % (nm, it["n_dataset"], it["n_runs"], res_s,
                 it.get("note", "")[:26], mark))
    print("")
    print("  当前档案：%s" % (cur or "（无，用老位置 gamer_ai/dataset + gamer_ai/runs）"))
    print("  切换 --use 名字    详情 --show 名字    删除 --delete 名字")


def show(name: str) -> None:
    d = P.profile_dir(name)
    if not P.exists(name):
        raise SystemExit("[档案] 没有这个档案：%s" % d)
    print("档案 %s" % name)
    print("  目录   %s" % d)
    print("  取景   %s" % P.geometry_path(name))
    print("  数据   %s" % P.dataset_root(name))
    print("  模型   %s" % P.runs_root(name))
    print("  %s" % P.describe(name))
    ds = P.dataset_root(name)
    if os.path.isdir(ds):
        sess = [x for x in os.listdir(ds) if not x.startswith("_")]
        tail = " ..." if len(sess) > 8 else ""
        print("  会话 %d 个：%s%s" % (len(sess), ", ".join(sess[:8]), tail))
    rs = P.runs_root(name)
    if os.path.isdir(rs):
        runs = [x for x in os.listdir(rs) if not x.startswith("_")]
        tail = " ..." if len(runs) > 8 else ""
        print("  模型 %d 个：%s%s" % (len(runs), ", ".join(runs[:8]), tail))


def main() -> int:
    ap = argparse.ArgumentParser(description="区域档案管理")
    ap.add_argument("--list", action="store_true", help="列出所有档案（默认动作）")
    ap.add_argument("--use", default=None, help="把某个档案设为当前档案")
    ap.add_argument("--new", default=None, help="新建一个空档案")
    ap.add_argument("--rename", nargs=2, metavar=("旧名", "新名"), default=None,
                    help="给档案改名（目录、profile.json、当前档案指针一起改）")
    ap.add_argument("--note", default="", help="配合 --new：备注")
    ap.add_argument("--show", default=None, help="看某个档案的详情")
    ap.add_argument("--delete", default=None, help="删除某个档案")
    ap.add_argument("--keep-files", action="store_true",
                    help="配合 --delete：只删档案信息，保留 dataset/、runs/")
    args = ap.parse_args()

    if args.rename:
        old, new = args.rename
        d = P.rename(old, new)
        print("[档案] %s -> %s" % (old, new))
        print("       新目录：%s" % d)
        print("       里面的数据/模型原样不动；当前档案指针已同步。")
        return 0
    if args.new:
        d = P.create(args.new, note=args.note)
        print("[档案] 已新建：%s" % d)
        print("       下一步：跑 1_自检.py --profile %s 拉框标定这块区域"
              % P.sanitize(args.new))
        return 0
    if args.use:
        nm = P.set_current(args.use)
        print("[档案] 当前档案 -> %s" % nm)
        print("       2_采集数据 / 5_训练模型 / 6_评估效果 / 9_自动玩 / 10_迭代补数据"
              " 不加 --profile 时都会用它")
        return 0
    if args.show:
        show(args.show)
        return 0
    if args.delete:
        d = P.delete(args.delete, keep_files=args.keep_files)
        print("[档案] 已删除：%s%s" % (d, "（数据/模型保留）" if args.keep_files else ""))
        return 0
    list_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
