# -*- coding: utf-8 -*-
"""干跑 14_模型迭代训练.py：拦掉真正的 trainer，只看它算出来的轮次/学习率/来源权重。"""
import importlib.util
import os
import sys
import types

HERE = r"D:\deeplearning\make_dataset\paoku"
sys.path.insert(0, HERE)
os.chdir(HERE)

import collector  # noqa

fake = types.ModuleType("collector.trainer")


def fake_main():
    a = list(sys.argv)
    print("  --- trainer 实际收到 ---")
    for k in ("--resume", "--lr", "--epochs", "--patience", "--end-offset",
              "--input-stride", "--select-metric", "--source-weight", "--out"):
        if k in a:
            print("     %-16s %s" % (k, a[a.index(k) + 1]))
    print("     sessions        %d 个" % (a.count("\\") and 0 or (len(a) - 1 - a.index("--sessions"))))
    return 1


fake.main = fake_main
sys.modules["collector.trainer"] = fake

spec = importlib.util.spec_from_file_location("iter_train", os.path.join(HERE, "14_模型迭代训练.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CASES = [
    ("① 默认（从初代重新微调 30 轮）", []),
    ("② 只用代次数字：--continue-from 1 --epochs 20",
     ["--continue-from", "1", "--epochs", "20"]),
    ("③ 绝对路径：--continue-from <迭代_20260925_2226 的绝对路径>",
     ["--continue-from", r"D:\deeplearning\make_dataset\paoku\runs\迭代_20260925_2226",
      "--epochs", "20"]),
    ("④ --continue-from last", ["--continue-from", "last", "--epochs", "20"]),
    ("⑤ 填错路径（看报错提示）", ["--continue-from", r"runs\这个目录不存在"]),
]
for title, extra in CASES:
    print("=" * 78)
    print(title)
    sys.argv = ["14_模型迭代训练.py"] + extra
    try:
        mod.main()
    except SystemExit as e:
        print("  SystemExit:", e)
    print()
