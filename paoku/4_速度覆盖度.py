# -*- coding: utf-8 -*-
"""速度覆盖度报告

在 PyCharm 里直接右键 Run；需要参数时用 Run -> Edit Configurations -> Parameters。

用法（Parameters）: dataset/run01 dataset/run02
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from collector.coverage import main

if __name__ == "__main__":
    main()
