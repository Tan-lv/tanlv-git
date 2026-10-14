# -*- coding: utf-8 -*-
"""采集配置 —— 所有可调参数集中在这里。

训练与部署必须使用完全相同的 game_crop / frame_w / input_* 参数,
否则会重新引入"训练与推理分布不一致"的问题。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple


# ======================================================================
#  ★ 键位定义（唯一来源）—— 以后加新按键只改这两张表，别的脚本一个字都不用动
# ----------------------------------------------------------------------
#  ACTION_KEYS  动作键：游戏里真的会响应。按下即采集该动作的样本；
#               上线时模型预测到这一类会真的按这个键（9_自动玩.py 靠 key_for_label 查）
#  MARKER_KEYS  纯标记键：游戏里没有映射，只用来标"此刻该什么都不做"
#
#  类别顺序 labels 由这两张表自动推出（按字母序），和 trainer 的
#  sorted(cnt.keys()) 保持一致 —— 所以这里加一个键，数据集/训练/标注界面全都自动跟上。
# ======================================================================
ACTION_KEYS: Dict[str, str] = {
    "w": "W",      # 跳
    "s": "S",      # 滚
    "a": "A",      # 左移
    "d": "D",      # 右移
    "q": "Q",      # ★ 2026-10-10 新增
    # "e": "E",    # 2026-10-11 曾短暂加过 E 键，样本太少（只有 6 个）已去掉；
    #              # 以后想恢复，取消注释这一行即可（labels/key_map/上线发键会自动跟上）
}
MARKER_KEYS: Dict[str, str] = {
    "n": "N",      # 标记"此刻该什么都不做"（Subway Surfers 没映射，按下去游戏无反应）
}


def key_for_label(label: str) -> Optional[str]:
    """类别 -> 要按下的键字符（小写）；不是动作类（例如 N）返回 None。

    发键的地方（9_自动玩.py / 10_迭代补数据.py）统一走这个函数，
    所以加动作键只需要往 ACTION_KEYS 里加一行。
    """
    lab = str(label).upper()
    for ch, lb in ACTION_KEYS.items():
        if lb == lab:
            return ch
    return None


@dataclass
class Config:
    # ---------------- 窗口 / 抓帧 ----------------
    window_keyword: str = "雷电模拟器"
    # 抓帧后端: "printwindow" (推荐, 被遮挡也能抓) / "bitblt" (更快, 但窗口被遮挡时可能抓到别的画面)
    backend: str = "printwindow"
    # 安卓游戏画面分辨率（LDPlayer）
    #   ★ 2026-10-10：用户把模拟器转成**横屏**并清空了旧的竖屏数据，这里改成 1280x720。
    #     竖屏时代的旧值是 (720, 1280)；以后要切回去，这两行一起改。
    game_res: Tuple[int, int] = (1280, 720)
    # 在 game_res 坐标系里的裁剪框 (x, y, w, h)。
    #   ★ 比例必须保持 720:880（= 240:293）！一旦比例变了，落盘帧尺寸跟着变，
    #     而前处理是「短边缩到 224 + 中心裁 224」，模型看到的有效视野会完全不同。
    #   横屏取 590x720 这种【同比例竖窗】；具体放在哪用 tools\取景标定.py 边看边调。
    game_crop: Tuple[int, int, int, int] = (345, 0, 590, 720)
    # 抓帧上限（帧率）。0 = 不限速。PrintWindow 实测可达 ~80fps
    capture_max_fps: float = 40.0
    # 环形缓冲保留时长（秒）。必须 > (save_pre_ms + save_post_ms)/1000, 建议 >= 3
    ring_seconds: float = 5.0

    # ---------------- 落盘格式 ----------------
    output_root: str = "dataset"
    # 落盘帧宽度（高度按裁剪框比例自动计算）。一个样本 = 一张横向胶片条 JPEG
    frame_w: int = 240
    # 胶片条中相邻两帧之间插入的分隔线宽度（像素），便于人工肉眼核对
    sep: int = 2
    # JPEG 编码参数。实测（22 帧 5322x293 的胶片条，一张 1031KB @ q95/4:4:4）：
    #   q95,4:4:4 = 1031KB(100%)   q90,4:4:4 = 797KB(77%)
    #   q95,4:2:0 =  688KB( 67%)   q90,4:2:0 = 509KB(49%)  <- 默认
    #   q85,4:2:0 =  389KB( 38%)
    # 模型输入最终会缩到 224x224，q90 的编码误差(mean|diff|=2.7/255)完全可忽略，
    # 但能把数据集体积砍掉一半。
    jpeg_quality: int = 90
    # 色度采样：0=4:4:4(最高)  1=4:2:2  2=4:2:0(默认，通用做法)
    jpeg_subsampling: int = 2
    # 一个样本保存的时间窗（相对按键时刻）
    # save_post_ms 要覆盖「换道动画」的时长（约 300ms），否则无法在事后构造
    # 「角色已经在动了 -> 不该再按键」的训练样本，也无法复盘动作效果。
    #
    # save_pre_ms 要覆盖「障碍已经很近了但还不到该动」的那一段（见 approach_neg_*），
    # 否则治不了模型「动作偏早」（障碍还远就换道）。原来的 300ms 太短：
    # 6 帧输入窗口跨度 167ms，最早只能取到按键前 146ms 的画面，证据不足。
    # 提到 800ms 后最早能取到约 -633ms。代价是胶片条从 22 帧涨到 37 帧，
    # 单样本从约 450KB 涨到约 750KB。
    save_pre_ms: int = 800
    save_post_ms: int = 400
    # 保存窗口内的目标采样帧率
    save_fps: float = 30.0

    # ---------------- 模型输入（离线可改，不需要重新采集）----------------
    # 输入给模型的连续帧数
    input_frames: int = 6
    # 输入窗口的结束位置相对按键时刻往前偏移多少毫秒（保证取到的都是"按键之前"的画面）
    #
    #  【已改成 0】训练和推理两端现在完全一致：窗口结束于"当前帧"。
    #  取 0 的收益：① 修掉训练(-40ms)/推理(0ms)的错位 ② 输入窗后移 67ms 缓解动作偏早
    #             ③ 标注界面里黄框最后一帧正好落在当前帧上，所见即所得
    #  风险很小：按键到画面可见变化有 100ms+ 延迟，2 帧内画面基本没变。
    #
    #  （下面是当初决定改动时的分析，留作记录）
    #  这个值在训练/推理两端必须一致，之前**不一致**：
    #      训练（dataset.py）：窗口结束于 按键前 input_end_offset_ms 毫秒 = -40ms
    #      推理（9_自动玩.py）：窗口结束于"此刻" = 0ms
    #    后果：训练教的是"看到这个画面 → 40ms 后该按了"，推理问的是"看到这个画面 → 现在按吗"，
    #    模型因此系统性早按。按 30fps 离散化，实际差约 2 帧 = 67ms。
    #    这正是「动作偏早」的一个隐藏来源（另一个来源是行为克隆本身没有人的反应延迟）。
    #
    #  改成 0 可以同时做三件事：
    #    ① 修掉训练/推理不一致  ② 输入窗后移 67ms，缓解动作偏早  ③ 标注界面里黄框贴住当前帧
    #  风险很小：按键到画面产生可见变化有 100ms 以上延迟，2 帧内画面基本没变，
    #  不会让模型学到"看到自己已经在动"。
    #
    #  改完必须重新训练（输入定义变了）。但**不需要重新采集** ——
    #  样本 meta 里只存了 37 帧的时间戳，取哪 6 帧是训练时从这里现算的，
    #  所以新旧样本会统一按新窗口取帧，已标注的数据不会白做。
    input_end_offset_ms: int = 0
    # 输入帧之间在"保存帧序列"中的步长
    input_stride: int = 1

    # ---------------- 从"动作动画"合成的 N 类负样本 ----------------
    # 按键之后 save_post_ms 内的帧显示"角色已经在换道"，把这些帧当作模型输入、
    # 标签设为 N，就教会了模型"已经在动了就别再按" —— 直接治你说的重复触发。
    # 前提是 save_post_ms 足够长（默认 400ms，覆盖换道动画）。
    animation_neg_enabled: bool = True
    # 负样本的输入窗口结束于"按键后多少 ms"（可多个，取多个采样点）
    animation_neg_end_ms: Tuple[int, ...] = (200, 350)
    # 只有当这个按键之后至少这么久没有下一次按键，才认为「当时确实不该动」
    animation_neg_min_gap_ms: int = 700

    # ---------------- 从「障碍已出现但还不该动」合成的 N 类负样本 ----------------
    # 治模型的「动作偏早」。模型只学过「按键前那一小段画面 -> 动作」，从没见过
    # 「障碍已经出现、但还没到该动的时候」，于是把动作一路往更早的时刻泛化，
    # 结果是障碍还远就换道。用 7_时序诊断.py 可以量化这个偏差。
    #
    # 做法：每个按键样本，把「更早的那一段窗口」也拿出来、标签设成 N。
    # 前提是 save_pre_ms 够长（默认 800ms -> 最早可用的窗口结束位置约 -633ms）。
    approach_neg_enabled: bool = True
    # 负样本的输入窗口结束于「按键前多少 ms」（取值都是负数，可多个）。
    # 最靠近正样本的那一档最关键 —— 它把决策边界压到该动之前一点点。
    # 正样本窗口 = 结束于按键前 input_end_offset_ms、往前跨
    # (input_frames-1)*input_stride*1000/save_fps 毫秒；负样本不要超过
    # -(input_end_offset_ms + 跨度 + 一帧)，否则会和正样本画面重叠、标签打架。
    approach_neg_end_ms: Tuple[int, ...] = (-250, -350, -500)
    # 只有当「这个按键之前足够久都没按过键」时，那段画面才确实该什么都不做。
    # 实际要求 = |end| + 输入窗口跨度 + 这个余量。
    approach_neg_quiet_margin_ms: int = 250
    # ---------------- 迭代补采：哪些样本不参与"合成 N 负样本" ----------------
    #  ★ 2026-09-24（用户）：碰撞操作会话里的样本是人工从"撞车那一刻"挑帧标的，
    #    它前后那些帧全是撞车 / 被警察追的画面 —— 把它们当成"此刻该不动"的 N
    #    负样本是**标签噪声**。所以这类样本只当正样本用，不在它周围合成负样本。
    #    按会话名排除（默认"碰撞操作"）：
    no_synth_neg_sessions: Tuple[str, ...] = ("碰撞操作",)
    #    按样本 kind 排除（默认"relabel" = 12_标注界面.py 人工挑帧标出来的）：
    no_synth_neg_kinds: Tuple[str, ...] = ("relabel",)

    # ---------------- 采集行为 ----------------
    # 采集哪些类别。N = "此刻正确的动作是什么都不做"。
    # N 有两个来源：
    #   1) 采集时你**手动按键盘上的 n** 标记的"此刻该什么都不做"的画面
    #      （和 WASD 一样：按下才记）
    #   2) 训练时从每个按键样本的"后置窗口"自动合成的"换道动画中"画面
    #      （见 animation_neg_* —— 这是抑制"移动中重复触发"的关键，不用你管）
    #  类别列表：★ 由上面的 ACTION_KEYS / MARKER_KEYS 自动推出（按字母序），
    #  不再手写 —— 以后加键只要改 ACTION_KEYS，这里、采集横幅、标注界面、
    #  训练类别全都自动跟上。（顺序按字母序 = trainer 的 sorted(cnt.keys())）
    labels: Tuple[str, ...] = tuple(sorted(set(ACTION_KEYS.values())
                                          | set(MARKER_KEYS.values())))
    # 按键 -> 类别（pynput 的 char 为小写）。同样由那两张表拼出来。
    #   动作键必须**在模拟器按键映射里绑好**，否则采到的样本画面上什么都没发生。
    key_map: Dict[str, str] = field(default_factory=lambda: {**ACTION_KEYS, **MARKER_KEYS})
    # 同一个类别的最小采样间隔（毫秒），防止抖动重复
    key_debounce_ms: int = 120
    # N 类自动补采样：**默认关闭**，改成和 WASD 一样手动按 n 键。
    # 想恢复"空闲时自动补 N"就设成 True（采集时按 F9 也能临时切换）。
    n_auto_enabled: bool = False
    # ---- 下面 4 个只在开了自动补 N（n_auto_enabled=True 或按 F9）时生效 ----
    # 距离上一次方向键（WASD）至少这么久，才算"真的空闲了"
    n_idle_after_ms: int = 500
    # 两次自动补 N 之间的最小间隔。原来 400ms 太密（2.5 个/秒），N 会被灌爆，
    # 模型就偏向"什么都不做" —— 该换道的时候不动，直接撞死。
    n_auto_interval_ms: int = 1500
    # 按键后多久之内不补 N。必须盖住"换道动画 + 模型输入窗口"：
    # N 样本喂给模型的是该点往前 207ms~40ms 的那 6 帧
    # （由 input_frames / input_stride / input_end_offset_ms / save_fps 决定），
    # 500ms 保证这 6 帧整体落在换道动画结束之后，不和 WASD 的按键样本画面重叠。
    n_suppress_after_key_ms: int = 500
    # 反向避让：候选点选中后先不落盘，再等这么久。这期间如果又按了 WASD，
    # 说明那一刻其实是"马上要按键"的前奏，正确标签是按键而不是"什么都不做"，
    # 直接把这个候选丢掉不采。
    n_guard_after_key_ms: int = 350
    # X 键撤销：每按一次，删掉**最近落盘**的 undo_count 个样本（按采集时间倒序），
    # 连按就一批一批往前退。用于处理按错键、误采菜单画面、手滑等失误。
    #
    # 旧做法是"删掉撤回点往前 10 秒内的所有样本"，有个坑：撤回点往回退的同时，
    # "现在"还在往后长。如果两次按 X 之间又采了很久，一次就会吞掉整整那一段
    # （实测吞掉过 229 个样本）。改成按个数删之后，每次删多少完全可预期。
    undo_count: int = 5
    # **连续**撤销次数上限：防止手滑一直按，一口气把数据一路删光。
    # 注意这不是"整场总额度" —— 它限制的是"当下这一串连按"最多几下。
    # 停手 undo_reset_after_ms 毫秒没有再按 X，次数就重新计满，整场可以无限次撤销。
    # （只有手动按 X 计数，撞车自动回滚不占额度）
    undo_max_times: int = 5
    # 连续撤销的"冷却重置"时间：距上次成功撤销超过这么久，次数重新计满。
    undo_reset_after_ms: int = 5000

    # ---------------- 速度估计 ----------------
    # 游戏速度随存活时间上升。记录每个样本的滚动速度，用于：
    #   1) 采集时状态栏实时显示速度   2) 采完统计速度覆盖度   3) 训练时按速度重加权
    record_speed: bool = True

    # ---------------- IL2CPP 内存读取状态（比画面判断精确）----------------
    # 需要先另开一个进程跑 tools/state_server.py（它持有 Frida 会话）。
    # 打开后优先用它：inPlay / alive / paused 直接来自游戏自己的变量，
    # alive 由 1 变 0 的那一帧就是撞车时刻。
    hook_state: bool = False
    hook_state_host: str = "127.0.0.1"
    hook_state_port: int = 47800

    # ---------------- 游戏状态自动检测（开关录制）----------------
    # 跑酷中跑道在滚动，帧间差异大；菜单/弹窗/死亡界面是静止的。
    # 打开后：检测到开局自动开始记，检测到一局结束自动暂停并回滚最后几秒。
    # 手动 F8 仍然有效（F8 暂停时不受自动状态影响）。
    auto_state: bool = True
    # 实测：静止的菜单/弹窗 = 0.45；真实跑酷 = 10~40（同一套度量下）。
    # 阈值取在两者中间，且偏向"宁可多录"（误判成静止会导致完全不记录，代价更大）。
    state_motion_on: float = 5.0     # 超过这个帧间差异 -> 认为是跑酷中
    state_motion_off: float = 1.5    # 低于这个值 -> 认为已经不在跑酷
    state_on_ms: int = 500           # 需连续超过 on 阈值这么久才确认开局
    state_off_ms: int = 1200         # 需连续低于 off 阈值这么久才确认结束
    # 一局结束时回滚多少毫秒的样本。撞到障碍 -> 被警察抓住 -> 弹出 Continue?
    # 大约 3~4 秒，这段时间的样本标签没有意义，直接删掉。
    crash_rollback_ms: int = 4000

    # ---------------- 数据质量守护 ----------------
    # 连续多少帧完全相同就判定"画面冻结"并告警（模拟器卡死 / 游戏暂停）
    freeze_warn_frames: int = 60
    # 画面方差低于该值判定为黑屏/无效
    black_std_threshold: float = 1.0

    # ---------------- 会话 ----------------
    session_name: str = ""            # 留空则用时间戳
    note: str = ""

    # ------------------------------------------------------------------
    @property
    def crop_w(self) -> int:
        return int(self.game_crop[2])

    @property
    def crop_h(self) -> int:
        return int(self.game_crop[3])

    @property
    def frame_h(self) -> int:
        """按裁剪框比例算出的落盘帧高度。"""
        return int(round(self.frame_w * self.crop_h / self.crop_w))

    @property
    def strip_width(self) -> int:
        """一张胶片条里单帧占用的横向像素（含分隔线）。"""
        return self.frame_w + self.sep

    @property
    def saved_frames(self) -> int:
        """一个样本保存多少帧。"""
        span = (self.save_pre_ms + self.save_post_ms) / 1000.0
        return max(1, int(round(span * self.save_fps)) + 1)

    def validate(self) -> None:
        assert self.backend in ("printwindow", "bitblt"), f"未知 backend: {self.backend}"
        assert self.crop_w > 0 and self.crop_h > 0, "game_crop 尺寸非法"
        assert self.crop_w <= self.game_res[0] and self.crop_h <= self.game_res[1], \
            f"game_crop {self.game_crop} 超出 game_res {self.game_res}"
        # ★ 2026-10-10：原来只查了「框的宽高」，没查「框的起点」—— 标定时把框往右下
        #   推一点就会越界，然后抓到的是一块越界的黑边，到训练才发现。这里补上。
        assert self.game_crop[0] >= 0 and self.game_crop[1] >= 0, \
            f"game_crop 的起点不能为负: {self.game_crop}"
        assert self.game_crop[0] + self.crop_w <= self.game_res[0] \
            and self.game_crop[1] + self.crop_h <= self.game_res[1], \
            f"game_crop {self.game_crop} 越界：x+w 应为 {self.game_crop[0] + self.crop_w} " \
            f"<= {self.game_res[0]}，y+h 应为 {self.game_crop[1] + self.crop_h} <= {self.game_res[1]}"
        need = (self.save_pre_ms + self.save_post_ms) / 1000.0 + 0.5
        assert self.ring_seconds >= need, \
            f"ring_seconds={self.ring_seconds} 太小, 至少要 {need:.1f} 秒"
        assert self.input_frames >= 1
        max_idx = (self.input_frames - 1) * self.input_stride
        assert max_idx < self.saved_frames, \
            f"input_frames={self.input_frames} * stride={self.input_stride} 超出保存帧数 {self.saved_frames}"

    # ---------------- 读写 ----------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("game_res", "game_crop", "labels"):
            d[k] = list(d[k])
        return d

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        """从 JSON 覆盖默认配置。"""
        cfg = cls()
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            valid = {f.name for f in fields(cls)}
            for k, v in raw.items():
                if k in valid:
                    if k in ("game_res", "game_crop", "labels", "animation_neg_end_ms",
                              "approach_neg_end_ms", "no_synth_neg_sessions",
                              "no_synth_neg_kinds"):
                        v = tuple(v)
                    setattr(cfg, k, v)
        cfg.validate()
        return cfg
