# -*- coding: utf-8 -*-
"""读取"胶片条"格式的多帧数据集。

一个样本 = 一张横向拼接的 JPEG（frame_w * n 帧，中间用 sep 像素分隔）
           + meta.jsonl 里的一行元数据（每帧相对按键时刻的毫秒偏移）

关键点：输入给模型的帧是离线从保存窗口里"切"出来的，所以
input_frames / input_end_offset_ms / input_stride 随时可以改，
不需要重新采集数据。
"""
from __future__ import annotations

import bisect
import json
import os
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .config import Config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ======================================================================
#  元数据
# ======================================================================
def load_session(root: str, skip_missing: bool = True) -> List[dict]:
    """读取一个会话目录下的 meta.jsonl。

    ★ 2026-09-23：meta.jsonl 里记着胶片条文件名，但文件可能在磁盘上丢了
      （删除/移动过数据集、写盘失败……）。以前这会一路传到 DataLoader 里变成
      FileNotFoundError，**训到第一个 epoch 才炸**（训练白等）。
      现在这里就检查一遍，缺文件的样本直接跳过并报数，训练照常跑。
    """
    path = os.path.join(root, "meta.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 {path}")
    out = []
    n_missing = 0
    missing_eg: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                n_missing += 1
                continue
            m["_root"] = root
            if skip_missing:
                rel = str(m.get("file", "")).replace("/", os.sep).replace("\\", os.sep)
                if rel and not os.path.isfile(os.path.join(root, rel)):
                    n_missing += 1
                    if len(missing_eg) < 5:
                        missing_eg.append(str(m.get("id", "?")))
                    continue
            out.append(m)
    if n_missing:
        print(f"[data] ⚠ {os.path.basename(str(root))}: 有 {n_missing} 条记录的文件在磁盘上找不到"
              f"（例如 {', '.join(missing_eg)}），已跳过；"
              f"想彻底清掉就跑 tools\\fix_missing_samples.py")
    return out


def load_sessions(roots: Sequence[str]) -> List[dict]:
    samples: List[dict] = []
    for r in roots:
        samples.extend(load_session(r))
    return samples


# ======================================================================
#  帧选择（训练与部署共用，必须完全一致）
# ======================================================================
def select_window_indices(meta: dict, frames: int, stride: int,
                         end_offset_ms: float) -> List[int]:
    """从胶片条里挑 frames 帧，要求每帧的时间戳都 <= end_offset_ms（相对按键时刻）。

    end_offset_ms 为负 = 按键之前（正常样本）；
    为正 = 按键之后（从换道动画里合成的 N 类负样本）。
    """
    times = meta["frame_times_ms"]
    n = len(times)
    stride = max(1, int(stride))
    cand = [i for i in range(n) if times[i] <= end_offset_ms + 1e-6]
    if not cand:                      # 窗口太短时退化为"最早可用的帧"
        cand = [0]

    idx: List[int] = []
    pos = len(cand) - 1
    while len(idx) < frames and pos >= 0:
        idx.append(cand[pos])
        pos -= stride
    while len(idx) < frames:          # 仍然不够就复制最早的一帧
        idx.append(cand[0])
    return sorted(idx)                # 恢复成时间顺序（旧 -> 新）


def select_input_indices(meta: dict, cfg: Config) -> List[int]:
    """正常样本：窗口结束于按键前 input_end_offset_ms，即全部帧都在按键之前。"""
    return select_window_indices(meta, cfg.input_frames, cfg.input_stride,
                                 -cfg.input_end_offset_ms)


def window_available(meta: dict, frames: int, stride: int,
                     end_offset_ms: float) -> bool:
    """该胶片条里能不能凑出一个**完整**的窗口（不靠复制最早一帧凑数）。

    select_window_indices 在帧不够时会退化成复制最早一帧，那对负样本是致命的
    （会把「更早」的窗口做成一堆重复帧）。所以合成负样本前必须先卡一下：
    旧数据 save_pre_ms=300 时，按键前 250ms 的窗口就是不完整的，直接跳过。
    """
    times = meta["frame_times_ms"]
    stride = max(1, int(stride))
    cand = sum(1 for t in times if t <= end_offset_ms + 1e-6)
    return cand >= (frames - 1) * stride + 1


def synth_neg_allowed(m: dict, cfg: Config) -> bool:
    """这个样本能不能在它周围合成 N 类负样本？

    ★ 2026-09-24（用户）：**碰撞操作**样本是人工从"撞车那一刻"挑帧标的，
      它前后那些帧全是撞车/被警察追的画面，标成"此刻该不动"的 N 是标签噪声。
      所以这类样本只当正样本用，不在它身上合成负样本。

    注意：它的 t_anchor **仍要参与"空档"判断**（anchors 是全体的），
    否则别的样本会误以为"附近没人按过键"，反而合成出脏负样本。
    """
    if m.get("no_synth_neg"):
        return False
    kinds = set(getattr(cfg, "no_synth_neg_kinds", None) or ())
    if str(m.get("kind", "")) in kinds:
        return False
    sess = set(getattr(cfg, "no_synth_neg_sessions", None) or ())
    root = str(m.get("_root", "")).rstrip("\\/")
    if root and sess and os.path.basename(root) in sess:
        return False
    return True


def build_animation_negatives(all_samples: Sequence[dict], cfg: Config,
                              labels: Optional[Sequence[str]] = None
                              ) -> List[Tuple[int, float]]:
    """找出可以从"换道动画"合成 N 类负样本的按键样本。

    返回 [(样本下标, end_offset_ms), ...]。

    判据（很重要，否则会引入标签噪声）：只有当这个按键之后至少
    animation_neg_min_gap_ms 毫秒内没有再按任何键，才认为"当时确实不该动"。
    否则那段时间里人其实又做了决策，硬标成 N 是错的。
    """
    if not getattr(cfg, "animation_neg_enabled", True):
        return []
    if labels is not None and "N" not in labels:
        return []
    ends = list(getattr(cfg, "animation_neg_end_ms", ()) or [])
    if not ends:
        return []
    need_ms = max(ends)
    key_idx = [i for i, m in enumerate(all_samples) if m["label"] != "N"]
    anchors = sorted(all_samples[i]["t_anchor"] for i in key_idx)
    out: List[Tuple[int, float]] = []
    for i in key_idx:
        m = all_samples[i]
        if not synth_neg_allowed(m, cfg):
            continue                      # ★ 碰撞操作/人工挑帧样本：不在它身上合成
        times = m["frame_times_ms"]
        if not times or times[-1] < need_ms:
            continue                      # 后置窗口不够长（旧数据），跳过
        t = m["t_anchor"]
        k = bisect.bisect_right(anchors, t + 1e-6)
        gap_ms = (anchors[k] - t) * 1000.0 if k < len(anchors) else 1e9
        if gap_ms < getattr(cfg, "animation_neg_min_gap_ms", 700):
            continue
        for e in ends:
            out.append((i, float(e)))
    return out


def build_approach_negatives(all_samples: Sequence[dict], cfg: Config,
                             labels: Optional[Sequence[str]] = None
                             ) -> List[Tuple[int, float]]:
    """找出可以从「障碍已出现但还不该动」合成 N 类负样本的按键样本。

    返回 [(样本下标, end_offset_ms), ...]（end_offset_ms 为负 = 按键之前）。

    治的是模型的「动作偏早」：模型只学过「按键前那一小段画面 -> 动作」，会把
    动作往更早的时刻泛化，表现为障碍还远就换道。这里把更早的那截窗口标成 N，
    显式教它「看到障碍但还早，先别动」。

    两个必要条件（否则是标签噪声）：
      1) 窗口必须完整（save_pre_ms 够长）—— 旧数据会自动跳过；
      2) 这一段之前必须安静：上一个按键要在窗口起点再往前 quiet_margin 以上。
         否则那段画面其实是上一次动作的动画，标成 N 会和正样本打架。
    """
    if not getattr(cfg, "approach_neg_enabled", False):
        return []
    if labels is not None and "N" not in labels:
        return []
    ends = [float(e) for e in (getattr(cfg, "approach_neg_end_ms", ()) or ())]
    if not ends:
        return []
    stride = max(1, int(cfg.input_stride))
    span = ((cfg.input_frames - 1) * stride * 1000.0
            / max(float(cfg.save_fps), 1e-6))
    margin = float(getattr(cfg, "approach_neg_quiet_margin_ms", 250))
    key_idx = [i for i, m in enumerate(all_samples) if m["label"] != "N"]
    anchors = sorted(all_samples[i]["t_anchor"] for i in key_idx)
    out: List[Tuple[int, float]] = []
    for i in key_idx:
        m = all_samples[i]
        if not synth_neg_allowed(m, cfg):
            continue                      # ★ 碰撞操作/人工挑帧样本：不在它身上合成
        t = m["t_anchor"]
        k = bisect.bisect_left(anchors, t - 1e-6)
        prev_gap_ms = (t - anchors[k - 1]) * 1000.0 if k > 0 else 1e9
        for e in ends:
            if prev_gap_ms < abs(e) + span + margin:
                continue                      # 前面刚按过键，这段不干净
            if not window_available(m, cfg.input_frames, stride, e):
                continue                      # 采集窗口不够长（旧数据）
            out.append((i, e))
    return out


def negatives_for_subset(all_samples: Sequence[dict], split_idx: Sequence[int],
                         cfg: Config, labels: Optional[Sequence[str]] = None
                         ) -> List[Tuple[int, float]]:
    """先在全体样本上判定空档，再映射成子集内的局部下标（避免跨子集漏判）。

    两类负样本合并返回：换道动画中（按键之后）+ 障碍还远（按键之前）。
    """
    glob = (build_animation_negatives(all_samples, cfg, labels)
            + build_approach_negatives(all_samples, cfg, labels))
    pos = {g: loc for loc, g in enumerate(split_idx)}
    return [(pos[g], e) for g, e in glob if g in pos]


def negative_breakdown(all_samples: Sequence[dict], split_idx: Sequence[int],
                       cfg: Config, labels: Optional[Sequence[str]] = None
                       ) -> Dict[str, int]:
    """统计一个子集里两类负样本各有多少（给训练日志用）。"""
    pos = set(split_idx)
    anim = build_animation_negatives(all_samples, cfg, labels)
    appr = build_approach_negatives(all_samples, cfg, labels)
    return {"anim": sum(1 for g, _ in anim if g in pos),
            "approach": sum(1 for g, _ in appr if g in pos)}


# 胶片条解码缓存（每个进程一份）。
#
# 为什么要它：一个按键样本会派生 1 个正样本 + 1~2 个负样本，它们**共用同一张胶片条**。
# 没有缓存时每个条目都要重新解码一次 780KB 的 JPEG（实测 23ms），占单条目耗时一半以上。
# 配合 GroupShuffleSampler（让同一条目的条目连续出现），解码次数能降到 1/2.5 左右。
#
# 上限只留 3 条，不是拍脑袋 —— 分组打乱保证同一张胶片条的条目**连续**出现，
# 所以缓存留着"当前这一张"就够了。实测上限 2 和上限 48 的命中率完全一样。
#
# ⚠️ 别调大：DataLoader 每个 worker 是一个独立进程，各有一份缓存。
#    每条解压后 7.9MB（8954x293x3），上限 N 条 -> 每个 worker 占 N x 7.9MB。
#    8 workers x 48 条 = 3.0GB —— 实测直接把内存打爆，报 MemoryError。
#    现在 8 workers x 3 条 = 190MB，安全。
_STRIP_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_STRIP_CACHE_MAX = 3


def clear_strip_cache() -> None:
    """清空解码缓存（换数据集 / 长时间训练时手动调一下）。"""
    _STRIP_CACHE.clear()


def read_strip(meta: dict, indices: Sequence[int]) -> np.ndarray:
    """从胶片条里取出指定帧，返回 (K, H, W, 3) uint8。带解码缓存。"""
    fw, fh, sep = meta["frame_w"], meta["frame_h"], meta["sep"]
    path = os.path.join(meta["_root"], meta["file"])
    img = _STRIP_CACHE.get(path)
    if img is None:
        img = np.asarray(Image.open(path).convert("RGB"))
        _STRIP_CACHE[path] = img
        while len(_STRIP_CACHE) > _STRIP_CACHE_MAX:
            _STRIP_CACHE.popitem(last=False)   # 丢最旧的
    else:
        _STRIP_CACHE.move_to_end(path)
    out = np.empty((len(indices), fh, fw, 3), dtype=np.uint8)
    for j, i in enumerate(indices):
        x = i * (fw + sep)
        out[j] = img[0:fh, x:x + fw]
    return out


# ======================================================================
#  变换
# ======================================================================
def _v2():
    from torchvision.transforms import v2
    return v2


def build_train_transform(img_size: int = 224):
    """训练增强。v2 会对 (T,C,H,W) 的所有时间帧施加**同一套**随机参数，
    保证时序一致（不会出现每帧各裁一块的错位）。

    **故意不加 hue 抖动**，两个原因：
      1) 快：实测单次 ColorJitter 里 hue 就占 210ms/307ms（要转 HSV），
         去掉后整个 __getitem__ 从 407ms 降到约 172ms，一个 epoch 快一倍多。
      2) 对：这游戏里颜色是有语义的（列车/障碍/金币靠颜色区分），
         随机色相抖动会破坏这些线索。亮度/对比度/饱和度抖动足够覆盖
         「模拟器亮度不同」「白天/黑夜关卡」这类真实差异。
    """
    v2 = _v2()
    return v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(img_size: int = 224):
    v2 = _v2()
    return v2.Compose([
        v2.Resize(img_size, antialias=True),
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def to_video_tensor(frames: np.ndarray) -> torch.Tensor:
    """(K,H,W,3) uint8 -> (K,3,H,W) uint8 tensor"""
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)


def flatten_frames(x: torch.Tensor) -> torch.Tensor:
    """(K,3,H,W) -> (K*3,H,W) 通道维堆叠（早融合）"""
    k, c, h, w = x.shape
    return x.reshape(k * c, h, w)


# ======================================================================
#  Dataset
# ======================================================================
class GroupShuffleSampler(Sampler):
    """按「胶片条」分组打乱：同一张胶片条派生的条目连续出现。

    一个按键样本会派生 1 个正样本 + 1~2 个负样本（换道动画 / 障碍还远），
    它们用的是**同一张**胶片条。默认逐条目乱序时，每个条目都要重新解码一次
    （780KB JPEG，实测 23ms，占单条目耗时一半以上）。
    改成先打乱组、组内按顺序发，配合 read_strip 的解码缓存，
    一张胶片条只解码一次 —— 解码次数降到约 1/2.5。

    随机性没有丢：每个 epoch 重新打乱组顺序（种子 = seed x 大质数 + 第几次）。
    """

    def __init__(self, groups: Sequence[Sequence[int]], seed: int = 42):
        self.groups = [list(g) for g in groups]
        self.seed = int(seed)
        self._iters = 0

    def __iter__(self):
        import random
        self._iters += 1
        order = list(range(len(self.groups)))
        random.Random(self.seed * 1000003 + self._iters).shuffle(order)
        for i in order:
            for j in self.groups[i]:
                yield j

    def __len__(self) -> int:
        return sum(len(g) for g in self.groups)


class MultiFrameDataset(Dataset):
    def __init__(self, samples: Sequence[dict], cfg: Config,
                 transform=None, img_size: int = 224,
                 labels: Optional[Sequence[str]] = None,
                 negatives: Optional[Sequence[Tuple[int, float]]] = None,
                 clip_index_range: Optional[Tuple[float, float]] = None,
                 weights: Optional[Sequence[float]] = None):
        self.cfg = cfg
        self.samples = list(samples)
        # ★ 每样本损失权重（来源加权：新数据 / 人工纠正的数据可以给更高权重）。
        #   给了就 __getitem__ 多返回一个 w，训练循环按 w 加权算 loss；
        #   长度 = 真实样本数（合成负样本继承它来源样本的权重）。
        self.weights: Optional[List[float]] = None
        if weights is not None:
            _w = list(weights)
            self.weights = _w + [_w[src] if src < len(_w) else 1.0
                                 for src, _e in (list(negatives or []))]
        self.transform = transform if transform is not None else build_eval_transform(img_size)
        # labels 必须由外部统一传入：否则某个 split 恰好缺一类时映射会错位
        self.labels = list(labels) if labels is not None else sorted({s["label"] for s in self.samples})
        self.label_to_idx: Dict[str, int] = {lab: i for i, lab in enumerate(self.labels)}
        # 合成负样本：(子集内下标, 输入窗口结束于按键后多少 ms)，标签统一为 N
        self.negatives: List[Tuple[int, float]] = list(negatives or [])
        if "N" not in self.label_to_idx:
            self.negatives = []
        self.clip_index_range = clip_index_range

    def __len__(self) -> int:
        return len(self.samples) + len(self.negatives)

    def __getitem__(self, i: int):
        if i < len(self.samples):
            meta = self.samples[i]
            end_ms = -self.cfg.input_end_offset_ms     # 按键之前
            label = meta["label"]
            kind = "real"
        else:
            src, end_ms = self.negatives[i - len(self.samples)]
            meta = self.samples[src]
            label = "N"                                # 角色已经在换道动画中 -> 不该再按
            kind = "anim_neg"
        idx = select_window_indices(meta, self.cfg.input_frames, self.cfg.input_stride, end_ms)
        frames = read_strip(meta, idx)                       # (K,H,W,3)
        x = to_video_tensor(frames)                          # (K,3,H,W)
        x = self.transform(x)                                # 同一套参数作用于所有帧
        x = flatten_frames(x)                                # (K*3,H,W)
        y = self.label_to_idx[label]
        if self.weights is not None:
            return x, y, float(self.weights[i] if i < len(self.weights) else 1.0)
        return x, y

    def negative_count(self) -> int:
        return len(self.negatives)

    def strip_groups(self) -> List[List[int]]:
        """把条目按来源胶片条分组（配合 GroupShuffleSampler + 解码缓存）。"""
        groups: Dict[str, List[int]] = {}
        for i, m in enumerate(self.samples):
            groups.setdefault(m["file"], []).append(i)
        for local, (src, _e) in enumerate(self.negatives):
            groups.setdefault(self.samples[src]["file"], []).append(len(self.samples) + local)
        return list(groups.values())

    # ---------- 统计 ----------
    def class_counts(self) -> Dict[str, int]:
        from collections import Counter
        c = Counter(s["label"] for s in self.samples)
        return {lab: c.get(lab, 0) for lab in self.labels}


# ======================================================================
#  划分（重点：按时间分块，避免相邻帧泄漏到不同集合）
# ======================================================================
def split_indices(samples: Sequence[dict], train_ratio=0.7, val_ratio=0.15,
                  mode: str = "time_balanced", seed: int = 42,
                  boundary_gap: int = 0, segment_seconds: float = 15.0,
                  n_segments: Optional[int] = None) -> Dict[str, List[int]]:
    """返回 {'train': [...], 'val': [...], 'test': [...]}（样本下标）。

    mode="time_balanced"（默认，推荐）:
        先把时间轴切成若干"段"（默认每段 ~15 秒），整段是划分的最小单位，
        **同一段内的样本永远待在同一个集合里**；再把各段贪心地分配给三个集合，
        使每个集合的类别分布尽量贴近目标比例。
        这样既避免了相邻帧泄漏（跨段的帧在时间上已相距很远），
        又不会像纯时间分块那样"把整个类别切进某一个集合"。

    mode="time_block":
        按时间排序后直接切三段连续区间。相邻帧不泄漏，但如果数据是
        "先集中录 A、再集中录 D"这种类别分块的采集方式，某一类可能整个
        落进单一集合 —— 旧版没有这个问题只是因为用了随机划分（会泄漏）。

    mode="random":
        原来的做法，仅用于对照（会造成相邻帧泄漏）。
    """
    n = len(samples)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    idx = list(range(n))
    ratios = {"train": train_ratio, "val": val_ratio,
              "test": max(0.0, 1.0 - train_ratio - val_ratio)}

    if mode == "random":
        import random
        rnd = random.Random(seed)
        rnd.shuffle(idx)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        return {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    order = sorted(idx, key=lambda i: samples[i]["t_anchor"])

    if mode == "time_block":
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        out = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
        if boundary_gap > 0:
            for k in ("train", "val", "test"):
                out[k] = out[k][boundary_gap:-boundary_gap] if len(out[k]) > 2 * boundary_gap else out[k]
        return out

    if mode != "time_balanced":
        raise ValueError(f"未知 mode: {mode}")

    # ---------- 切段 + 整段贪心分配 ----------
    labels_all = sorted({s["label"] for s in samples})
    total_by_c = Counter(s["label"] for s in samples)
    t_first = samples[order[0]]["t_anchor"]
    t_last = samples[order[-1]]["t_anchor"]
    span = max(t_last - t_first, 1e-6)

    def assign(n_seg: int) -> Dict[str, List[int]]:
        seg = span / n_seg
        segments: Dict[int, List[int]] = {}
        for i in order:
            s = min(n_seg - 1, int((samples[i]["t_anchor"] - t_first) / seg))
            segments.setdefault(s, []).append(i)
        budget = {k: ratios[k] * n for k in ratios}
        target = {k: {c: ratios[k] * total_by_c[c] for c in labels_all} for k in ratios}
        cur = {k: Counter() for k in ratios}
        out = {k: [] for k in ratios}
        for s in sorted(segments):
            cnt = Counter(samples[i]["label"] for i in segments[s])
            size = len(segments[s])
            best_k, best_score = None, -1e18
            for k in ratios:
                if budget[k] <= 0 or len(out[k]) >= budget[k] + size:
                    continue
                # 这一段能补上多少"类别缺口"，得分就多高
                score = 0.0
                for c in labels_all:
                    if cnt.get(c, 0) == 0:
                        continue
                    want = target[k][c] - cur[k][c]
                    score += cnt[c] * want / max(target[k][c], 1.0)
                over = len(out[k]) + size - budget[k]
                if over > 0:                       # 超出配额要惩罚
                    score -= 5.0 * over
                if score > best_score:
                    best_k, best_score = k, score
            if best_k is None:                     # 兜底：给剩余空间最大的集合
                best_k = max(ratios, key=lambda k: budget[k] - len(out[k]))
            out[best_k].extend(segments[s])
            cur[best_k].update(cnt)
        for k in out:
            out[k].sort()
        return out

    base = int(n_segments) if n_segments else max(12, min(4000, int(span / float(segment_seconds))))
    result = assign(min(4000, base))
    # 如果还有集合缺类别，就自动把时间段切细再试（段越细，类别越容易被摊到各集合）
    for factor in (2, 4, 8, 16):
        if all(any(samples[i]["label"] == c for i in result[k])
               for k in result for c in labels_all):
            break
        result = assign(min(4000, base * factor))
    return result
