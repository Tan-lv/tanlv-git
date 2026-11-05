# -*- coding: utf-8 -*-
r"""游戏配置的加载与切换（games/<游戏名>.py 的读写入口）。

为什么单独一层：这个项目不止用来玩《地铁跑酷》。每个游戏的**按键表**和
**默认训练参数**都不一样，所以一个游戏一个文件，放在 gamer_ai/games/ 下；
这里负责把它们加载进来，供采集(2_)/训练(5_)/部署(9_) 使用。

    games/subway_surfers.py     地铁跑酷
    games/<别的游戏>.py         复制一份改改就是新游戏

当前游戏记在 gamer_ai/games/_current.json，`--game 名字` 可以临时覆盖。
没有指定、也没有当前游戏时，退回 core/config.py 里的默认按键表。
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from typing import Dict, List, Optional, Tuple

PAOKU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_DIR = os.path.join(PAOKU_DIR, "games")
CURRENT_FILE = os.path.join(GAMES_DIR, "_current.json")


def list_games() -> List[str]:
    """games/ 下可用的游戏名（= 文件名，不含 .py）。"""
    if not os.path.isdir(GAMES_DIR):
        return []
    out = []
    for f in sorted(os.listdir(GAMES_DIR)):
        if f.endswith(".py") and not f.startswith("_"):
            out.append(f[:-3])
    return out


def sanitize(name: str) -> str:
    """游戏名 = 安全的文件名。"""
    s = str(name or "").strip()
    for ch in '<>:"/\\|?* ':
        s = s.replace(ch, "_")
    return s.strip(".")


def load(name: str = None) -> Optional[dict]:
    """加载一个游戏配置，返回 dict（NAME/ACTION_KEYS/MARKER_KEYS/TRAIN/NOTES）。

    name 为空时用"当前游戏"；当前游戏也没有就返回 None（调用方退回默认按键表）。
    找不到指定名字时**报错退出**（而不是悄悄用默认值）—— 按键表错了，采到的数据
    标签就是错的，这种错误必须当场发现。
    """
    nm = sanitize(name) if name else current()
    if not nm:
        return None
    path = os.path.join(GAMES_DIR, nm + ".py")
    if not os.path.isfile(path):
        raise SystemExit("[游戏] 没有这个游戏配置：%s ；可用的：%s ；新建一个就复制 games/subway_surfers.py 改改名字" % (path, ", ".join(list_games()) or "（一个都没有）"))
    spec = importlib.util.spec_from_file_location("games_" + nm, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = {k: getattr(mod, k) for k in dir(mod) if not k.startswith("_")}
    g["_name"] = nm
    g["_path"] = path
    return g


def current() -> Optional[str]:
    """当前游戏名。

    ① 有 _current.json 且指向一个存在的游戏 -> 用它；
    ② 没设过（或指向的已经没了）：games/ 下只有一个游戏配置时就用它 ——
       ★ 2026-10-15：不然老模型（model_meta.json 里没有 game 字段）会退回 config.py
       的默认表，而那份表跟你正在玩的游戏可能已经不一样了（"按键和数据对不上"的来源之一）；
    ③ 有多个游戏又没设过 -> None（这时必须用 --game/--keys 明确指定）。
    """
    nm = None
    if os.path.isfile(CURRENT_FILE):
        try:
            nm = json.load(open(CURRENT_FILE, encoding="utf-8")).get("name")
        except Exception:
            nm = None
    if nm and nm in list_games():
        return nm
    # ★ 2026-10-15：还没设过"当前游戏"时 —— 如果 games/ 下只有一个游戏配置，就用它。
    #   否则老模型（model_meta.json 里没有 game 字段）会退回 config.py 的默认表，
    #   而那份表跟你正在玩的游戏可能已经不一样了（正是"按键和数据集不对称"的来源之一）。
    games = list_games()
    if len(games) == 1:
        return games[0]
    return None


def set_current(name: str) -> str:
    os.makedirs(GAMES_DIR, exist_ok=True)
    nm = sanitize(name)
    with open(CURRENT_FILE, "w", encoding="utf-8") as fh:
        json.dump({"name": nm, "set_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  fh, ensure_ascii=False, indent=2)
    return nm


def parse_keys(spec: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """解析命令行的 --keys，返回 (ACTION_KEYS, MARKER_KEYS)。

    写法：`w,s,a,d,q,n`   —— 纯键名，类别名 = 大写（n 自动算标记键）
          `w=跳,a=左,n=N`  —— 显式指定类别名；类别名是 N 的算标记键
    """
    act: Dict[str, str] = {}
    mrk: Dict[str, str] = {}
    for item in str(spec or "").replace(";", ",").split(","):
        it = item.strip()
        if not it:
            continue
        if "=" in it:
            k, lab = it.split("=", 1)
            k, lab = k.strip(), lab.strip()
        else:
            k, lab = it, it.upper()
        if not k or not lab:
            raise SystemExit("[按键] --keys 写法不对：%r（应该像 w,s,a,d,q,n 或 w=W,n=N）" % it)
        if lab.upper() == "N":
            mrk[k] = lab.upper()
        else:
            act[k] = lab.upper()
    return act, mrk


def apply_keys(cfg, game: dict = None, keys: str = None) -> dict:
    """把按键表装进 cfg（key_map / labels）。返回生效的 {键: 类别}。

    优先级：命令行 --keys  >  games/<游戏>.py  >  core/config.py 的默认表
    """
    if keys:
        act, mrk = parse_keys(keys)
    elif game and (game.get("ACTION_KEYS") or game.get("MARKER_KEYS")):
        act = dict(game.get("ACTION_KEYS") or {})
        mrk = dict(game.get("MARKER_KEYS") or {})
    else:
        return {}
    cfg.key_map = {**act, **mrk}
    cfg.labels = tuple(sorted(set(cfg.key_map.values())))
    cfg.validate()
    return dict(cfg.key_map)


def validate_keymap(key_map) -> List[str]:
    """按键表自检，返回警告列表（空 = 没问题）。

    ★ 2026-11-04（用户："我在这里面添加了 G 按键的数据采集和监听，为什么采集时
      没有收集到 G 数据"）：采集端监听拿到的是 **key.lower()**，
      所以表里写成 "Q": "G" 这种**大写源键**永远匹配不上 ——
      按 G 时 keyid 是 "g"，查不到就静默跳过，一条样本都不产生，日志里毫无提示。
      这个坑太隐蔽，所以在加载按键表时主动检查一次。
    """
    out: List[str] = []
    seen = {}
    for k in (key_map or {}):
        ks = str(k)
        if len(ks) != 1:
            out.append("源键 %r 不是单个字符 —— 永远不会被匹配（应该只写一个字符，如 'g'）" % ks)
            continue
        low = ks.lower()
        if ks != low:
            out.append("源键 %r 是**大写**，但监听拿到的是小写 %r —— 它永远不会触发；"
                       "如果本意是那个键，请改成 %r"
                       % (ks, low, low))
        if low in seen:
            out.append("源键 %r 和 %r 归一化后都是 %r，只有一条会生效"
                       % (seen[low], ks, low))
        else:
            seen[low] = ks
    by_class: Dict[str, List[str]] = {}
    for k, v in (key_map or {}).items():
        by_class.setdefault(str(v), []).append(str(k))
    for cls, keys in by_class.items():
        if len(keys) > 1:
            out.append("类别 %r 被多个按键映射：%s（采集都能收；上线发键只会用其中一个）"
                       % (cls, ", ".join(repr(x) for x in keys)))
    return out


def train_defaults(name: str = None) -> dict:
    """这个游戏的默认训练参数字典（没有就返回 {}）。"""
    g = load(name)
    return dict(g.get("TRAIN") or {}) if g else {}


def align_with_model(meta: dict, game: str = None, keys: str = None,
                     allow_mismatch: bool = False) -> dict:
    """让**模型 / 按键表 / 目标数据集**三边对齐。

    ★ 2026-10-15（用户反馈："不同游戏的模型、按键、数据集会对不上"）：
      以前 7_迭代采集.py 只看 Config() 里的默认按键表 —— 模型如果是另一个游戏训练出来的，
      它会**静默按错键**（按下去的键在这个游戏里是别的动作），采出来的数据标签全是错的；
      更糟的是 write_sample 遇到没见过的类别会直接 KeyError。

    规则（一句话：**模型说了算**）：
      ① 按键表：命令行 --keys > 模型自带 key_map > 模型训练时那个游戏(meta["game"])的表
                 > 命令行 --game 的表 > 当前游戏 > config.py 默认表
      ② 类别：模型类别 ∪ 按键表类别（并集，保证计数/目录/写样本都不会缺）
      ③ 校验：模型每个要发键的类别都必须能查到按键，否则报错（除非 allow_mismatch）
      ④ 校验：模型是 A 游戏、你指定 B 游戏 = 明显搞错了 -> 报错（除非 allow_mismatch）

    返回 {"game": 游戏名或 None, "key_map": {类别: 键}, "labels": [...], "notes": [...]}
    notes 里每一条都是给人看的说明/警告。
    """
    from .config import Config as _C
    model_labels = [str(x).upper() for x in (meta.get("labels") or [])]
    model_km = {str(k).upper(): v for k, v in (meta.get("key_map") or {}).items()}
    mgame = meta.get("game")
    notes: List[str] = []

    # ---- 定游戏名：命令行 > 模型自带 > 当前游戏 ----
    gname = game or mgame or current()
    if game and mgame and sanitize(game) != sanitize(mgame):
        msg = ("模型是 %r 游戏训练出来的，你却指定了 --game %r —— 按键表会对不上，采出来的数据标签会是错的" % (mgame, game))
        if not allow_mismatch:
            raise SystemExit("[对齐] %s\n       确认要这么做就加 --allow-mismatch" % msg)
        notes.append("[警告] " + msg)
    g = load(gname) if gname else None

    # ---- 定按键表 ----
    act: Dict[str, str] = {}
    if keys:
        _a, _m = parse_keys(keys)
        act = {**_m, **_a}
        src = "命令行 --keys"
    elif model_km:
        # ★ 2026-11-04（用户："为什么这个自动玩的脚本不能自己读模型有几个按键，然后自己适配"）：
        #   模型的 model_meta.json 里 key_map 是**「类别 -> 键」**（{"W": "w"}），
        #   而下面统一按**「键 -> 类别」**（{"w": "W"}）处理 —— 原来直接 dict(model_km)，
        #   结果第 238 行转出来是 {"W": "W"}：**键字符退化成了类别名**，
        #   上线时发的是大写 A/D/W（游戏里没绑 Shift 组合），所以角色一动不动。
        #   ADB 通道里 ch.lower() 恰好掩盖了这个问题。
        act = {str(ch): str(lb).upper() for lb, ch in model_km.items()}
        src = "模型自带 key_map（自动转成 键->类别）"
    elif g and (g.get("ACTION_KEYS") or g.get("MARKER_KEYS")):
        act = {**(g.get("MARKER_KEYS") or {}), **(g.get("ACTION_KEYS") or {})}
        src = "games/%s.py 的按键表" % (g.get("_name") or gname)
    else:
        # Config.key_map 本来就是「键 -> 类别」，别再翻转一次（同上一个问题）
        act = dict(_C().key_map)
        src = "core/config.py 默认表"
    key_map = {str(lb).upper(): ch for ch, lb in act.items()}

    # ---- 校验①：模型要发键的类别必须查得到键 ----
    unpressable = [lb for lb in model_labels if lb != "N" and lb not in key_map]
    if unpressable:
        msg = ("模型会输出 %s 这几类，但当前按键表里没有对应的键 —— 发不出去；请换 --game / --keys 指定正确的表" % unpressable)
        if not allow_mismatch:
            raise SystemExit("[对齐] %s" % msg)
        notes.append("[警告] " + msg)

    # ---- 类别并集（计数/目录/写样本都不会缺） ----
    labels = sorted(set(model_labels) | set(key_map.keys()))
    g_labels = sorted(set(((g or {}).get("ACTION_KEYS") or {}).values())
                      | set(((g or {}).get("MARKER_KEYS") or {}).values()))
    if g_labels:
        only_game = [x for x in g_labels if x not in model_labels]
        only_model = [x for x in model_labels if x not in g_labels]
        if only_game:
            notes.append("[提示] 按键表里有 %s，模型没训过这几类 —— 它永远不会预测出来" % only_game)
        if only_model:
            notes.append("[提示] 模型有 %s，按键表里没有这几类 —— 采集时按不到它们" % only_model)

    # ---- 校验②：目标数据集是不是另一个游戏的 ----
    for root in (meta.get("_ds_roots") or []):
        sp = os.path.join(root, "session.json")
        if not os.path.isfile(sp):
            continue
        try:
            dg = (json.load(open(sp, encoding="utf-8")).get("config") or {}).get("game")
        except Exception:
            dg = None
        if dg and mgame and sanitize(dg) != sanitize(mgame):
            msg = ("目标数据集 %s 是 %r 游戏采的，而模型是 %r —— 混在一个数据集里训练，模型会学乱" % (os.path.basename(root), dg, mgame))
            if not allow_mismatch:
                raise SystemExit("[对齐] %s\n       确认要混就加 --allow-mismatch" % msg)
            notes.append("[警告] " + msg)

    notes.insert(0, "[对齐] 按键表来源：%s；游戏：%s" % (src, gname or "（未指定）"))
    return {"game": sanitize(gname) if gname else None, "key_map": key_map,
            "labels": labels, "notes": notes, "source": src}


def describe(game: dict) -> str:
    if not game:
        return "(未指定游戏，用 core/config.py 里的默认按键表)"
    act = game.get("ACTION_KEYS") or {}
    mrk = game.get("MARKER_KEYS") or {}
    s = "游戏 %s" % (game.get("NAME") or game.get("_name"))
    if act:
        s += "  动作键 " + " ".join("%s=%s" % (k, v) for k, v in act.items())
    if mrk:
        s += "  标记键 " + " ".join("%s=%s" % (k, v) for k, v in mrk.items())
    return s
