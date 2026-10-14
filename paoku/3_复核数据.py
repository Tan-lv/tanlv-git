# -*- coding: utf-8 -*-
"""采集后复核：删除坏样本 / 改正标错的标签。

在 PyCharm 里运行这个文件，然后在下方的 Run 窗口里会提示输入会话目录。
更简单的用法：Run -> Edit Configurations -> Parameters 填  dataset/run01
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from collector.review import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        s = input("要复核哪个会话？（例如 dataset/run01）: ").strip()
        if s:
            sys.argv.append(s)
    main()
