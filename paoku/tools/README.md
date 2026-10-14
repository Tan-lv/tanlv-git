# tools —— 逆向 / hook 工具

> **这份文档回答的是「凭什么能直接读游戏内存」。**
> * 第一次上手 → 先看 [`../README.md`](../README.md)
> * 状态服务起不来 → [`../常见问题.md`](../常见问题.md)
>
> 你平时**不需要打开这个目录**：采集时加 `--hook-state`，状态服务会自动拉起。

**这个目录下的脚本用 `wasd311` 环境跑**（Python 3.11 + frida 17.17.0）。
最省事的方式是直接**双击 `run_*.bat`**，里面的解释器路径是写死的。

### 目录

| 节 | 内容 |
|---|---|
| 一、hook 可行性（实测结论） | 为什么 hook ARM 代码走不通 |
| 二、x86 侧 hook 探针（备用方案） | `frida_probe.js` |
| 三、纯内存读取游戏状态（现在的主用方案） | ★ 五步定位原理 |
| 四、前置条件（现在大部分是自动的） | adb / frida-server / 端口转发 |
| 五、解释器说明 | 为什么必须是 wasd311 |
| 六、局限 | 什么情况下会失效 |

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
python 2_采集数据.py --session run01 --hook-state

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

| | 画面判断（collector/gamestate.py） | IL2CPP 内存读取 |
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
