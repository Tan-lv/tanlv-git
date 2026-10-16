# -*- coding: utf-8 -*-
"""开始采集数据

在 PyCharm 里直接右键 -> Run 这个文件即可。
（不要直接运行 core/recorder.py，那个文件用的是包内相对导入，直接跑会报 ImportError）

采集的按键（每个键 = 一个动作类）：
    W 跳   S 滚   A 左移   D 右移   Q ★动作键（要在模拟器按键映射里绑好）
    N 纯标记键：标记"此刻该什么都不做"（游戏里没映射，按下去游戏无反应）
（要加/改按键：改 games/<游戏>.py 的 ACTION_KEYS，或运行加 --keys w,s,a,d,q,e,n）

快捷键：F8 暂停/继续采集  F9 自动补 N 开关  F7 导出调试条  F10/Esc 退出
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ==============================================================================
#  ★★★  数据存到哪：改下面这几行（改完直接 Run，不用记命令行参数）  ★★★
# ------------------------------------------------------------------------------
#  数据目录就是"区域档案"里的 dataset/：
#      gamer_ai/profiles/<档案名>/dataset/<会话名>/
#
#  PROFILE  —— 存到哪个区域档案（最常用）
#      None        = 用【当前区域档案】（1_自检 拉框保存时自动设的那个）
#      "sys_gamer" = 固定存到 gamer_ai/profiles/sys_gamer/dataset/
#      看有哪些档案：python tools\管理区域档案.py
#
#  DATA_ROOT —— 直接指定数据根目录（绕开档案，一般不用）
#      None                          = 跟着 PROFILE 走
#      r"profiles\sys_gamer\dataset" = 就存这里（相对 gamer_ai/，也可写绝对路径）
#
#  SESSION —— 这次采集的会话名
#      None        = 用时间戳（推荐，一次采集一个目录，互不覆盖）
#      "run01"     = 固定名字（**会续着往同一个目录里存**，别多用）
#
#  命令行给的参数优先级最高（--profile / --root / --session 会覆盖上面）。
# ==============================================================================
PROFILE = None
DATA_ROOT = None
SESSION = None


if __name__ == "__main__":
    # 把上面的常量变成参数；命令行里已经写了的以命令行为准
    argv = sys.argv[1:]
    extra = []
    for flag, val in (("--profile", PROFILE), ("--root", DATA_ROOT), ("--session", SESSION)):
        if val and flag not in argv:
            extra += [flag, str(val)]
    if extra:
        print("[采集] 用 2_采集.py 开头设置的参数：%s" % " ".join(extra))

    from core.recorder import main
    sys.argv = [sys.argv[0]] + extra + argv
    main()
