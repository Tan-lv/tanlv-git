# tools —— 逆向 / hook 工具

> **这份文档回答的是「凭什么能直接读游戏内存」。**
> * 第一次上手 → 先看 [`../README.md`](../README.md)
> * 状态服务起不来 → [`../常见问题.md`](../常见问题.md)
>
> 你平时**不需要打开这个目录**：采集时加 `--hook-state`，状态服务会自动拉起。

## 现在只有这 6 个（2026-10-15 清理后）

| 文件 | 干什么 |
|---|---|
| `state_server.py` | **游戏状态服务**：frida 读 IL2CPP 内存（inPlay / alive / paused / chasing）。`--hook-state` 会自动拉起它 |
| `il2cpp_state.js` | 上面那个服务注入的 frida 脚本 |
| `re_probe.py` | frida 探针公共库（state_server 用） |
| `管理区域档案.py` | **区域档案管理**：列出 / 切换 / 新建 / 改名 / 删除 |
| `鼠标选裁剪框.py` | 单独拉框选区域（`--png` 可以不开模拟器） |
| `对比模型.py` | 两三个模型在同一批干净样本上对比 |

> **已经删掉的**（都用不上了，2026-10-15 清理）：
> · 过时工具：`取景标定.py` `模型取景重指向.py` `click_log.py` `抓窗口整图.py` `上机体检.py`
>   （取景现在全自动；抓整窗被 `1_自检` 的 `auto_geometry.png` 覆盖）
> · 内存逆向探针与一次性脚本：`frida_probe.js` `hook_probe.py` `il2cpp_re.js` `chase_monitor.*`
>   `collision_trace.py` `state_watch.py` `dedup_*/rebuild_*/repair_meta.py` `run_*.bat`
>   （当年挖内存字段用的，字段已经写进 `il2cpp_state.js` 了）
> 需要旧版本可以从 git 取：`git log --all -- paoku/tools/<文件名>`。

**这个目录下的脚本用 `wasd311` 环境跑**（Python 3.11 + frida 17.17.0）。
状态服务不用手动启动：`2_采集.py --hook-state` 会自己拉起来（跑不起来看 `runs/state_server.log`）。

### 目录

| 节 | 内容 |
|---|---|
| 一、hook 可行性（实测结论） | 为什么 hook ARM 代码走不通 |
| 二、x86 侧 hook 探针（备用方案） | `frida_probe.js` |
| 三、纯内存读取游戏状态（现在的主用方案） | ★ 五步定位原理 |
| 四、前置条件（现在大部分是自动的） | adb / frida-server / 端口转发 |
| 五、解释器说明 | 为什么必须是 wasd311 |
| 六、局限 | 什么情况下会失效 |

### 目录里都是什么文件（2026-10-15 清理后）

**① 采集 / 自动玩会在后台用到的（别动、别改名）**

| 文件 | 作用 |
|---|---|
| `state_server.py` | ★ 游戏状态服务：frida 读内存拿 `inPlay/alive/chasing/chaseDist`。采集加 `--hook-state` 时会**自动拉起它** |
| `il2cpp_state.js` | 状态服务加载的 frida 脚本（IL2CPP 内存定位） |
| `re_probe.py` | frida 探针公共库（`state_server.py` 用） |

> ⚠️ `state_server.py` 和 `il2cpp_state.js` 的路径是写死在采集代码里的，**挪位置会直接起不来**。

**② 手动用的小工具**

| 文件 | 作用 |
|---|---|
| `管理区域档案.py` | 区域档案：列出 / 切换 / 新建 / 改名 / 删除 |
| `鼠标选裁剪框.py` | 单独拉框选区域（`--png` 可不开模拟器） |
| `对比模型.py` | 两三个模型在同一批干净样本上对比 |

**③ 已经删掉的（2026-10-15）**

下面这些都用不上了，直接从 `tools/` 删除（它们的"为什么不用了"见上面那段）：

| 文件 | 当年的作用 |
|---|---|
| `hook_probe.py` + `frida_probe.js` + `run_hook_probe.bat` | 早期 x86 侧 hook 探针（实测走不通） |
| `il2cpp_re.js` | 定位内存地址用的 frida 脚本 |
| `state_watch.py` / `chase_monitor.py` + `.js` | 盯状态变化、盯「被警察追击」的监视器 |
| `collision_trace.py` | 区分「被抓 vs 被追击」的高频内存记录 —— 结论已写进 `7_迭代采集.py` |
| `popup_capture.py` / `click_log.py` / `fix_frida.py` | 抓弹窗 / 点按日志 / frida 修复 |
| `dedup_by_id.py` `dedup_samples.py` `rebuild_samples.py` `repair_meta.py` `fix_missing_samples.py` `merge_sessions.py` | 早期修 `meta.jsonl` 的数据维护脚本 |
| `make_synth_dataset.py` | 造合成数据集做对照实验 |
| `cam_compare.py` | CAM 聚焦度对比（`5_评估.py --cam` 已覆盖） |
| `run_state_server.bat` / `run_hook_probe.bat` | 双击启动脚本（状态服务现在自动拉起） |
| `取景标定.py` `模型取景重指向.py` `抓窗口整图.py` `上机体检.py` | 手工标定 / 改元数据取景 / 抓整窗 / 上机体检（取景已全自动） |

> 要找旧版本：`git log --all -- paoku/tools/<文件名>`（改名/删除前都提交过）。

**③ 数据维护小工具（按需手动跑）**

| 文件 | 作用 |
|---|---|
| `repair_meta.py` | 修 `meta.jsonl` 里的字段（动手前把原文件备份到**它旁边**：`meta.jsonl.bak`） |
| `merge_sessions.py` | 把多个会话合并成一个 |
| `rebuild_samples.py` | 按 meta 重建样本 |
| `dedup_samples.py` / `dedup_by_id.py` | 去重 |
| `fix_missing_samples.py` | 清掉「meta 里有记录但图片已丢」的行 |
| `make_synth_dataset.py` | 生成合成数据集（旁路实验用） |

**④ 分析 / 评估工具（效果不理想时用）**

| 文件 | 作用 |
|---|---|
| `对比模型.py` | 把多个模型放到**同一批干净样本**上公正对比（含「上线口径」触发率） |
| `上机体检.py` | 逐帧滚动回放：动作率 / 提前量 / 漏动 + 阈值扫描 |
| `鼠标选裁剪框.py` | ★ **鼠标拖两个对角**就能选裁剪框；默认写进取景文件 `gamer_ai/geometry.json`（存相对比例，换分辨率/横竖屏都有效），加 `--save-to-config` 才额外写 `core/config.py` |
| `取景标定.py` | 实时标定裁剪框（整窗+框 / 落盘画面 / CNN 实际输入 三方并排，键盘微调） |
| `模型取景重指向.py` | 把模型元数据里的取景参数改到新分辨率（可回滚） |
| `抓窗口整图.py` | 绕过尺寸检查抓整个模拟器窗口 —— 画面/分辨率不对时先用它看一眼 |
| `cam_compare.py` | 两个模型的 CAM 聚焦度对比（顶10%质量、熵、热点面积） |
| `管理区域档案.py` | 列出/切换/新建/删除**区域档案**（`gamer_ai/profiles/<名字>/` = 一块区域 + 它的数据 + 它的模型）。平时不用跑：自检拉框保存时会自动建档 |

**⑤ 备份与归档（平时不用看）**

| 位置 | 内容 |
|---|---|
| [`../../旧仓库/旧数据/_meta_backups/`](../../旧仓库/README.md) | `meta.jsonl` 的历史备份（去重/重建之前留的）——2026-10-15 已搬进旧仓库 |
| [`../../旧仓库/旧探针/`](../../旧仓库/旧探针/README.md) | 早期一次性探针与调试产物（内存逆向的 `_rev_*.py`/`_revy.txt`/`_chase_log4.txt`、`_strip.py`、旧版标注界面存档、几个临时小工具、一份旧脚本备份）。2026-10-15 已搬进 `旧仓库/旧探针/`，**每份文件是什么、能不能删都写在那个目录的 README 里** |

---

## 一、hook 可行性（实测结论）

雷电用 **libhoudini** 把 ARM 翻译成 x86。实测（2026-09-13）：

| 手段 | 结果 |
|---|---|
| Hook x86 原生函数（libEGL / libGLESv2 / libaudioclient / libaaudio / libc） | ✅ **可用**，实测挂上了 6 个函数并拿到实时计数 |
| 读 / 扫描 ARM 库内存 | ✅ 可用（能读到 `libil2cpp.so` 的 ELF 头） |
| **Hook ARM 函数**（`libil2cpp.so` 里的游戏逻辑） | ❌ `libil2cpp.so` / `libunity.so` **不在 Frida 的模块列表里**，且内存映射只有 `r--p` / `rw-p`，**没有 r-xp** |
| Java 层 hook | ❌ 这个 32 位进程里 Frida 的 Java bridge 没起来（`Java` 未定义） |
| ARM 侧文件操作走 x86 libc | ❌ 挂 `open/openat` 后 20 秒 25432 次调用里，游戏自己的路径 **0 次** —— ARM 代码直接发 syscall，绕过了 x86 libc |

### 唯一的门：houdini 的翻译缓存

```
0d420000-11380000  rwxp  [anon:Mem_0x20000000]      <- 63MB，可读可写可执行
```

ARM 代码本身不可执行，但 **houdini 翻译出来的 x86 代码全在这个 RWX 区里**，而且可以改。
往这个区里搜 ARM 绝对地址是能命中的：

```
get_CanJump   ABS=0xcb5c4df8  命中 1 处 -> 0xd509070
get_Grounded  ABS=0xcb5c3dd4  命中 4 处 -> 0xdb9b7b4 ...
```

**但 dump 出来看，这些命中在数据表里**（8 字节一条记录的 `{ARM地址, 0, 缓存内指针}`），
不是翻译后的代码本体。要真正把 ARM 地址映射到 x86 翻译块，得先逆向 houdini 的缓存布局 ——
这是个独立的大工程，所以最后没走这条路。

---

## 二、x86 侧 hook 探针（备用方案）

`frida_probe.js` + `hook_probe.py`：把 x86 侧的 GL / 音频边界全部挂上，每 2 秒汇报一次计数。

```powershell
# 双击也行：
tools\run_hook_probe.bat

# 或命令行：
D:\Anaconda\envs\wasd311\python.exe -u tools\hook_probe.py
```

实测输出（游戏停在菜单时）：

```
[OK]   libEGL.so!eglSwapBuffers            -> eglSwapBuffers
[OK]   libGLESv2.so!glDrawArrays           -> glDrawArrays
[OK]   libGLESv2.so!glDrawElements         -> glDrawElements
[OK]   libGLESv2.so!glBindTexture          -> glBindTexture
[OK]   libaudioclient.so!AudioTrack::write
[OK]   libaaudio.so!AAudioStreamWrite
```

**用途**：进关卡正常玩一局（撞死、重开），看哪个计数在「撞车 / Continue? / 结算」时刻
发生**突变**。菜单和跑酷的场景复杂度差很多，绘制调用数大概率不同。

实测基线：约 **160 FPS**，每帧 1 次 `glDrawArrays`、约 56 次 `glDrawElements`、
约 44 次 `glBindTexture`。音频两个入口都是 0 次 —— 游戏当前没走这两条路。

---

## 三、纯内存读取游戏状态（现在的主用方案）

### 原理：五步定位，不执行任何 ARM 代码

全程只有 `readU32` / `readCString` / `Memory.scan`：

```
1. 类名字符串在 global-metadata.dat 的映射区 (0xc0404000 ~ 0xc1348000)
   扫 "GameController\0" 拿到字符串地址

2. 扫所有可写区，找「值等于该字符串地址」的 4 字节指针
   这些位置就是 Il2CppClass.name 字段（结构偏移 +0x08）

3. 强校验确认（三条同时成立才算）：
     struct+0x08 -> 解引用 == "GameController"
     struct+0x0C -> 解引用是可读字符串（"SYBO.Subway"）
     struct+0x3C -> 值 == 结构自身地址（klass 自指）
   => GameController 的 Il2CppClass

4. 扫内存找「值等于该 Il2CppClass」的对象，就是 GameController 实例
   再用强校验筛掉误命中：+0x04 必须是 0(monitor)，
   且 +0x10/+0x14/+0x18 三个指针指向的对象的 klass 必须都等于 GlobalBool 类

5. 读那三个 GlobalBool 的 _value（在对象 +0x10）：
     +0x10  _isInPlaystate   1 = 正在跑酷
     +0x14  _isAlive         0 = 撞死了
     +0x18  _runPaused       1 = 暂停
```

依据来自 `dump.cs` 里 `GameController`（TypeDefIndex 835）：

```csharp
public class GameController : MonoBehaviour {
    private GlobalBool _isInPlaystate; // 0x10
    private GlobalBool _isAlive;       // 0x14
    private GlobalBool _runPaused;     // 0x18
}
public abstract class GlobalVariable<T> {
    private GlobalValueChanged<T> OnChanged;    // 0x08
    private GlobalValueListeners<T> _listeners; // 0x0C
    private T _value;                           // 0x10
}
```

### 用法

**一个环境就够了** —— `wasd311` 里同时有 torch 和 frida，不再需要「另开一个进程用别的 Python」。

```powershell
# 推荐：采集器加 --hook-state，状态服务会自动拉起
python 2_采集.py --session run01 --hook-state

# 想单独看状态：双击 tools\run_state_server.bat
```

启动时打印：

```
[前置] 检查模拟器 / adb / frida-server
      adb 设备: emulator-5554	device
      frida-server: 运行中
      端口转发: tcp:27042 -> 设备
[目标] pid=2945
[定位] {'ok': True, 'gc': '0xb8a684c8', 'inst': '0xade28820'}
[就绪] 采集器可连 127.0.0.1:47800 取状态。
[心跳] 已运行     10s   inPlay=0 alive=1 paused=0   已响应查询 5 次
```

> 这个窗口要**一直开着** —— 它是服务，不是跑完就结束的脚本。心跳停了才说明它真的挂了。

### 比画面判断强在哪

| | 画面判断（core/gamestate.py） | IL2CPP 内存读取 |
|---|---|---|
| 判据 | 帧间差异（「画面在动」） | 游戏自己的变量（「在跑酷」） |
| 开局识别 | 等画面动起来，约 0.5s | 精确 |
| **撞车时刻** | 等画面静止，**慢 3~4 秒** | `alive` 由 1 变 0 的**那一帧** |
| 依赖 | 无 | 模拟器 + adb + frida-server + 状态服务 |
| 失效场景 | 有动画的弹窗会误判 | 游戏更新版本后类/字段偏移可能变 |

撞车时刻精确之后，回滚逻辑就干净了：**从 `alive` 翻转那一刻往前删 0.5 秒**，
而不是「猜最后 4 秒」。

---

## 四、前置条件（现在大部分是自动的）

`state_server.py` 启动时会自动检查并补齐这四样：

1. adb server（没起就 start-server）
2. 设备连接（没设备就 connect 127.0.0.1:5555）
3. **frida-server**（没跑就用 `su` 拉起 —— 模拟器每次重启都会丢，必须重来）
4. 端口转发 tcp:27042

手动重拉 frida-server（万一自动失败）：

```powershell
D:\leidian\LDPlayer9\adb.exe shell "su -c 'nohup /data/local/tmp/frida-server -D &'"
```

---

## 五、解释器说明

`wasd311`（Python 3.11.16）里已经装好 **frida 17.17.0**，和模拟器上的
frida-server 版本一致，**直接跑就行**。

* **frida 主程序和设备上 frida-server 的版本必须一致**，升级任意一边都要同步另一边，
  否则连不上。
* 旧的 `wasd_local` 是 Python 3.10，**装不了 frida 17.x**（需要 3.11+）。
  硬装能装上但一导入就报 `ImportError: cannot import name 'NotRequired' from 'typing'`。
  所以当时的做法是新建 `wasd311` 而不是原地升级（原地升级会让 230 个 cp310
  专属扩展全部作废，要重下 4.8 GB）。
* 脚本里保留了「自动换解释器」的兜底逻辑（子进程 + 转发输出，不用 `os.execv`，
  后者会让 PyCharm 以为进程结束而回收控制台），正常情况下不会触发。

---

## 六、局限

* 只对**当前这个版本的 APK** 有效。游戏一更新，`Il2CppClass` 布局和字段偏移都可能变，
  需要重新用 Il2CppDumper 跑一遍 dump 确认。
* 类不唯一时（同名类）需要额外区分，目前只处理了 GameController 这种唯一的。
* 状态服务挂了会自动回退到画面判断（采集器启动时会打印警告）。
