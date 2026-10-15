# -*- coding: utf-8 -*-
r"""区域档案（profile）：把「一块画面区域 + 它的数据 + 它的模型」绑在一个目录里。

用户 2026-10-15 的想法：自检截图 -> 鼠标框选 -> 起个名字 -> 确认保存，
然后 gamer_ai/profiles/<名字>/ 下自动出现：

    geometry.json    这块区域（相对比例，换分辨率/横竖屏也能自己对准）
    profile.json     档案信息（名字 / 创建时间 / 标定分辨率 / 备注）
    dataset/         以后用这块区域采到的数据，全放这里
    runs/            用这些数据训出来的模型，全放这里

于是"换一块区域" = 换一个档案目录：采集、训练、部署三边的数据/模型/取景一起走，
不会互相串（以前全挤在 gamer_ai/dataset 和 gamer_ai/runs 里，靠文件名区分，很容易搞混）。

没有选任何档案时，一切仍然用老位置：gamer_ai/dataset、gamer_ai/runs、gamer_ai/geometry.json。

当前档案记在 gamer_ai/profiles/_current.json，`--profile 名字` 可以临时覆盖。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from typing import List, Optional

PAOKU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(PAOKU_DIR, "profiles")
CURRENT_FILE = os.path.join(PROFILES_DIR, "_current.json")
GEOMETRY_NAME = "geometry.json"
INDEX_NAME = "profile.json"
KEYS_NAME = "keys.json"          # ★ 这块区域的按键表（和分辨率/截图区域放一起）


def sanitize_game(name: str) -> str:
    """游戏名 -> 安全的文件名（和 gamecfg.sanitize 同一套规则）。"""
    s = str(name or "").strip()
    for ch in '<>:"/\\|?* ':
        s = s.replace(ch, "_")
    return s.strip(".")

# Windows 文件名里不允许的字符；顺便挡掉路径穿越
_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize(name: str) -> str:
    """把用户起的名字变成安全的目录名（中文可以用，非法字符换成下划线）。"""
    s = _BAD.sub("_", str(name or "").strip())
    s = s.strip(" .")
    if len(s) > 48:
        s = s[:48]
    return s


def profile_dir(name: str) -> str:
    return os.path.join(PROFILES_DIR, sanitize(name))


def exists(name: str) -> bool:
    """目录在 + 有 profile.json，才算一个档案。"""
    return bool(name) and os.path.isfile(os.path.join(profile_dir(name), INDEX_NAME))


def geometry_path(name: str) -> str:
    return os.path.join(profile_dir(name), GEOMETRY_NAME)


def dataset_root(name: str) -> str:
    return os.path.join(profile_dir(name), "dataset")


def runs_root(name: str) -> str:
    return os.path.join(profile_dir(name), "runs")


def events_root(name: str) -> str:
    """迭代补数据（10_）录下来的待标注事件也放档案里。"""
    return os.path.join(profile_dir(name), "events")


def create(name: str, geometry: dict = None, note: str = "",
           game: str = None, keys: dict = None) -> str:
    """建一个档案（已存在就补齐缺的目录），返回档案目录。

    ★ 2026-10-15 用户要求："设计按键时，把这份按键也生成在分辨率和截图区域的文件里"。
      所以档案除了 geometry.json（分辨率/截图区域），还会记下：
        · profile.json 里的 game（哪个游戏）+ keys（按键表）
        · 单独的 keys.json —— 肉眼可读、可以直接手改，采集/训练/部署都读它
      这样"选档案"就等于同时选定了 分辨率 + 截图区域 + 按键表，不用再管第二件事。
    """
    nm = sanitize(name)
    if not nm:
        raise ValueError("区域名字不能为空")
    d = profile_dir(nm)
    for sub in ("dataset", "runs", "events"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    p = os.path.join(d, INDEX_NAME)
    info = {}
    if os.path.isfile(p):
        try:
            info = json.load(open(p, encoding="utf-8"))
        except Exception:
            info = {}
    info.setdefault("name", nm)
    info.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    info["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if note:
        info["note"] = note
    if geometry:
        info["game_res"] = geometry.get("game_res")
        info["crop_rel"] = geometry.get("crop_rel")
        info["calibrated_at"] = geometry.get("saved_at")
    if game:
        info["game"] = sanitize_game(game)
    if keys:
        info["keys"] = {str(k): str(v) for k, v in keys.items()}
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    if geometry:
        with open(os.path.join(d, GEOMETRY_NAME), "w", encoding="utf-8") as fh:
            json.dump(geometry, fh, ensure_ascii=False, indent=2)
    if keys:
        with open(os.path.join(d, KEYS_NAME), "w", encoding="utf-8") as fh:
            json.dump({"game": info.get("game"),
                       "keys": {str(k): str(v) for k, v in keys.items()},
                       "note": "按键表：键 -> 类别。改这里就能给这块区域换/加按键"
                               "（采集监听、训练类别、上线发键都读它）",
                       "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                      fh, ensure_ascii=False, indent=2)
    return d


def keys_of(name: str) -> dict:
    """读档案里的按键表（keys.json > profile.json 里的 keys）；没有返回 {}。"""
    if not name:
        return {}
    p = os.path.join(profile_dir(name), KEYS_NAME)
    if os.path.isfile(p):
        try:
            return {str(k): str(v) for k, v in
                    (json.load(open(p, encoding="utf-8")).get("keys") or {}).items()}
        except Exception:
            pass
    info = load(name) or {}
    return {str(k): str(v) for k, v in (info.get("keys") or {}).items()}


def game_of(name: str) -> Optional[str]:
    """档案属于哪个游戏；没记返回 None。"""
    info = load(name) or {}
    return info.get("game")


def load(name: str) -> Optional[dict]:
    p = os.path.join(profile_dir(name), INDEX_NAME)
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def list_profiles() -> List[dict]:
    """列出所有档案（按修改时间新 -> 旧）。"""
    if not os.path.isdir(PROFILES_DIR):
        return []
    out = []
    for d in os.listdir(PROFILES_DIR):
        if d.startswith("_") or d.startswith("."):
            continue
        full = os.path.join(PROFILES_DIR, d)
        if not os.path.isfile(os.path.join(full, INDEX_NAME)):
            continue
        info = load(d) or {"name": d}
        ds = os.path.join(full, "dataset")
        rs = os.path.join(full, "runs")
        n_sess = len([x for x in os.listdir(ds)
                      if os.path.isdir(os.path.join(ds, x)) and not x.startswith("_")]) \
            if os.path.isdir(ds) else 0
        n_run = len([x for x in os.listdir(rs)
                     if os.path.isdir(os.path.join(rs, x)) and not x.startswith("_")]) \
            if os.path.isdir(rs) else 0
        info.update({"dir": full, "n_dataset": n_sess, "n_runs": n_run,
                     "mtime": os.path.getmtime(full)})
        out.append(info)
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def current() -> Optional[str]:
    """当前档案名；没有返回 None。"""
    if not os.path.isfile(CURRENT_FILE):
        return None
    try:
        nm = json.load(open(CURRENT_FILE, encoding="utf-8")).get("name")
    except Exception:
        return None
    return nm if exists(nm) else None


def set_current(name: str) -> str:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    nm = sanitize(name)
    with open(CURRENT_FILE, "w", encoding="utf-8") as fh:
        json.dump({"name": nm, "set_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  fh, ensure_ascii=False, indent=2)
    return nm


def resolve(name: str = None, create_if_missing: bool = False) -> Optional[str]:
    """决定这次用哪个档案：命令行 --profile 优先，其次当前档案，都没有返回 None。"""
    nm = sanitize(name) if name else current()
    if not nm:
        return None
    if not exists(nm):
        if create_if_missing:
            create(nm)
        else:
            raise SystemExit("[档案] 找不到区域档案 %r —— 先跑 1_自检.py 建一个，"
                             "或者用 tools\\管理区域档案.py 看看有哪些" % nm)
    return nm


def describe(name: str) -> str:
    info = load(name) or {}
    s = "档案 %s" % name
    if info.get("game"):
        s += "  游戏 %s" % info["game"]
    if info.get("keys"):
        s += "  按键 %s" % " ".join("%s=%s" % (k, v) for k, v in info["keys"].items())
    if info.get("game_res"):
        s += "  标定于 %sx%s" % tuple(info["game_res"])
    if info.get("crop_rel"):
        s += "  crop_rel=%s" % (tuple(round(float(v), 4) for v in info["crop_rel"]),)
    if info.get("note"):
        s += "  (%s)" % info["note"]
    return s


def rename(old: str, new: str) -> str:
    """给档案改名（目录 + profile.json + "当前档案"指针一起改）。返回新目录。

    ★ 为什么要有这个函数：档案名同时出现在三个地方 ——
      ① 目录名 gamer_ai/profiles/<名字>/
      ② <目录>/profile.json 里的 name 字段
      ③ gamer_ai/profiles/_current.json（当前档案指针）
      只改目录名，另外两处就会对不上（列表里显示旧名、当前档案直接"消失"）。
    """
    o, n = sanitize(old), sanitize(new)
    if not o or not n:
        raise SystemExit("[档案] 改名要给旧名字和新名字")
    if not exists(o):
        raise SystemExit("[档案] 没有这个档案：%s\n       先跑 python tools\\管理区域档案.py 看看有哪些" % profile_dir(o))
    if o == n:
        return profile_dir(n)
    if os.path.exists(profile_dir(n)):
        raise SystemExit("[档案] 新名字已被占用：%s" % profile_dir(n))
    was_current = (current() == o)
    os.rename(profile_dir(o), profile_dir(n))          # 同盘改名，几 GB 也是瞬间完成
    p = os.path.join(profile_dir(n), INDEX_NAME)
    info = {}
    if os.path.isfile(p):
        try:
            info = json.load(open(p, encoding="utf-8"))
        except Exception:
            info = {}
    info["name"] = n
    info["renamed_from"] = o
    info["renamed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    if was_current:
        set_current(n)
    return profile_dir(n)


def delete(name: str, keep_files: bool = False) -> str:
    """删掉一个档案。keep_files=True 时只删档案元信息，留下 dataset/runs 目录。"""
    d = profile_dir(name)
    if not os.path.isdir(d):
        raise SystemExit("[档案] 没有这个档案：%s" % d)
    if keep_files:
        for f in (INDEX_NAME, GEOMETRY_NAME):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                os.remove(fp)
    else:
        shutil.rmtree(d)
    if os.path.isfile(CURRENT_FILE) and current() is None:
        os.remove(CURRENT_FILE)
    return d
