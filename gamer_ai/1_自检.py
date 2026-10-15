# -*- coding: utf-8 -*-
"""采集前自检

在 PyCharm 里直接右键 -> Run 这个文件即可。
（不要直接运行 core/selftest.py，那个文件用的是包内相对导入，直接跑会报 ImportError）

确认能抓到活的游戏画面、模拟器有没有卡死、参数是否正常。
采集之前必须先跑通这个。

★ 2026-10-14：跑它的时候会**弹出一个窗口，让你用鼠标拉框选区域**（拖两个对角）。
   选完会把这块区域写进 core/config.py 的 crop_rel —— 之后
   2_采集.py（采集）和 6_自动玩.py（部署）就都用这块区域，
   换分辨率/横竖屏也不用改参数（会自动按比例换算到当时的分辨率）。
   窗口里：黄框=你拖的，蓝框=吸附到 240:293 后真正用的，绿框=检测到的游戏画面；
           右侧实时显示落盘帧和模型实际吃到的 224x224。按键 r重选 l锁定比例 s保存 q退出。
   不想弹窗：加 --no-pick；想临时指定：--pick-corners x1,y1,x2,y2。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core.selftest import main

if __name__ == "__main__":
    main()
