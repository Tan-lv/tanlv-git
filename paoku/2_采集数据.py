# -*- coding: utf-8 -*-
"""开始采集数据

在 PyCharm 里直接右键 -> Run 这个文件即可。
（不要直接运行 collector/recorder.py，那个文件用的是包内相对导入，直接跑会报 ImportError）

采集的按键（每个键 = 一个动作类）：
    W 跳   S 滚   A 左移   D 右移   Q ★动作键（要在模拟器按键映射里绑好）
    N 纯标记键：标记"此刻该什么都不做"（游戏里没映射，按下去游戏无反应）

快捷键：F8 暂停/继续采集  F9 自动补 N 开关  F7 导出调试条  F10/Esc 退出
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from collector.recorder import main

if __name__ == "__main__":
    main()
