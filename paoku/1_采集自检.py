# -*- coding: utf-8 -*-
"""采集前自检

在 PyCharm 里直接右键 -> Run 这个文件即可。
（不要直接运行 collector/selftest.py，那个文件用的是包内相对导入，直接跑会报 ImportError）

确认能抓到活的游戏画面、模拟器有没有卡死、参数是否正常。
采集之前必须先跑通这个。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from collector.selftest import main

if __name__ == "__main__":
    main()
