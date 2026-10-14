# make_dataset

这是你的 PyCharm 工程根目录。

**里面只有一个项目：[paoku](paoku/README.md)** —— 让 AI 看着屏幕玩《地铁跑酷》。

> 以前的 `xz/`（YOLO 目标检测抽帧）已经不在了，本文档也跟着更新过了。
> 以前的 `paoku/remote/`（上传服务器训练）也已经删掉，现在**全部在本机训练**。

---

## 我该打开哪份文档？

| 我想…… | 看这个 |
|---|---|
| **第一次上手，从头做一遍** | **[paoku/README.md](paoku/README.md)** ← 先看这个 |
| 跑不起来 / 报错了 / 忘了某个键干什么 | [paoku/常见问题.md](paoku/常见问题.md) |
| 想搞懂「动作偏早」「连按发不出来」这些深层问题 | [paoku/进阶说明.md](paoku/进阶说明.md) |
| 想调参数（存几帧、往前存多久、撤销次数……） | [paoku/collector/README.md](paoku/collector/README.md) |
| 想知道为什么能直接读游戏内存、不用 hook | [paoku/tools/README.md](paoku/tools/README.md) |
| 想了解这套方案相比旧版改了什么 | [paoku/项目分析报告.md](paoku/项目分析报告.md)（历史资料） |

---

## 目录长什么样

```
make_dataset/
├── README.md                     ← 本文件
├── .idea/                        PyCharm 工程配置（解释器已指向 wasd311）
│
└── paoku/                        唯一的项目
    ├── README.md                 ★ 上手教程，先看这个
    ├── 常见问题.md                  报错排查 + 按键速查
    ├── 进阶说明.md                  深入原理与实测数据
    ├── 项目分析报告.md               对旧方案的完整分析（历史资料）
    │
    ├── 1_采集自检.py  …  9_自动玩.py   9 个入口脚本，按流程顺序编号
    ├── collector/                采集 / 训练 / 评估的实现代码（不用直接运行）
    ├── tools/                    逆向工具 + 游戏内存读取服务
    ├── dataset/                  采集到的数据（每个会话一个子目录）
    └── runs/                     训练出来的模型
```

**你在 PyCharm 里只会去运行 `paoku/` 下那 9 个编号脚本，别的都不用碰。**

---

## Python 解释器

全项目统一用 conda 环境 **`wasd311`**（Python 3.11.16）：

* 位置：`D:\Anaconda\envs\wasd311`
* 装了：torch 2.5.1+cu124（CUDA，能用显卡）、opencv、numpy、PIL、matplotlib、**frida 17.17.0**
* PyCharm 里切换：**File → Settings → Project → Python Interpreter → 选 `wasd311`**

> 还有另一个环境 `wasd_local`（Python 3.10）是老版本，留着当退路。
> 它**装不了 frida**（frida 17.x 要求 Python 3.11 以上），所以采集时想读游戏内存
> 就必须用 `wasd311`。确认新环境没问题后可以 `conda env remove -n wasd_local` 省 5 GB。

---

## 一句话流程

```
1_采集自检  →  2_采集数据  →  3_复核数据  →  5_训练模型  →  6_评估效果  →  9_自动玩
              （边玩边录）     （挑掉坏的）    （本机训练）    （看准不准）    （让它自己玩）
```

慢慢来，第 2 步要多做几轮（多玩几局），后面才练得出来。详细步骤见
[paoku/README.md](paoku/README.md)。

---

## 几个提醒

* **这个目录没有用 git 管理**。删掉的东西找不回来，删之前先确认。
* `dataset/` 已经占了好几 GB 磁盘，其中 `dataset/_selftest` 只是自检用的假数据，可以删。
* 采集数据是**最花时间**的一步，别急着训练 —— 见主文档里的「采集建议」。
