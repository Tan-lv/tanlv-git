# -*- coding: utf-8 -*-
r"""清掉 meta.jsonl 里「胶片条文件已经不在磁盘上」的坏记录。

背景：meta.jsonl 记着文件名，但文件可能被删/移动/写盘失败。这种坏记录会让
训练在第一个 epoch 才炸 FileNotFoundError（白等）。加载器现在会跳过，
这个工具是把记录**彻底清掉**，让样本数也对得上。

用法:
    python tools\fix_missing_samples.py                 # 只扫描，不改（默认）
    python tools\fix_missing_samples.py --apply          # 真改（先备份 meta.jsonl.bak_<时间>）
    python tools\fix_missing_samples.py --apply --dir "dataset\迭代补采\正确操作"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_sessions(root: str):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "_"))]
        if "meta.jsonl" in filenames:
            out.append(dirpath)
    return sorted(out)


def check(session: str):
    """返回 (好记录, 坏记录, 坏记录里的 id, 没被任何记录引用的文件数)。"""
    mp = os.path.join(session, "meta.jsonl")
    good, bad = [], []
    used = set()
    with open(mp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                bad.append(line)
                continue
            rel = str(m.get("file", "")).replace("/", os.sep)
            if rel:
                used.add(os.path.normcase(os.path.normpath(os.path.join(session, rel))))
            if rel and os.path.isfile(os.path.join(session, rel)):
                good.append(m)
            else:
                bad.append(m)
    orphan = 0
    for dirpath, dirnames, filenames in os.walk(session):
        for fn in filenames:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.normcase(os.path.normpath(os.path.join(dirpath, fn)))
            if p not in used:
                orphan += 1
    return good, bad, orphan


def main() -> int:
    ap = argparse.ArgumentParser(description="清理 meta.jsonl 里的坏记录")
    ap.add_argument("--apply", action="store_true", help="真的修改（默认只扫描）")
    ap.add_argument("--dir", default=os.path.join(HERE, "dataset"), help="要扫描的根目录")
    ap.add_argument("--depth", type=int, default=3, help="往下找几层会话（默认 3）")
    args = ap.parse_args()

    root = args.dir if os.path.isabs(args.dir) else os.path.join(HERE, args.dir)
    if not os.path.isdir(root):
        print(f"[错误] 目录不存在：{root}")
        return 1
    sessions = find_sessions(root)
    sessions = [s for s in sessions if s.count(os.sep) - root.count(os.sep) < args.depth]
    print(f"扫描 {root}  找到 {len(sessions)} 个会话\n")
    total_bad = 0
    for s in sessions:
        good, bad, orphan = check(s)
        rel = os.path.relpath(s, HERE)
        if not bad and not orphan:
            print(f"  OK   {rel}   样本 {len(good)}")
            continue
        total_bad += len(bad)
        ids = [str(m.get("id", "?")) if isinstance(m, dict) else "坏行" for m in bad[:8]]
        print(f"  ⚠   {rel}   好样本 {len(good)}  坏记录 {len(bad)}  没被引用的图 {orphan}")
        if ids:
            print(f"        坏记录示例: {', '.join(ids)}")
        if args.apply and bad:
            mp = os.path.join(s, "meta.jsonl")
            bak = mp + ".bak_" + time.strftime("%Y%m%d_%H%M%S")
            shutil.copy2(mp, bak)
            with open(mp, "w", encoding="utf-8") as fh:
                for m in good:
                    fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            print(f"        已清理（备份 {os.path.basename(bak)}），现在 {len(good)} 个样本")
    print(f"\n合计坏记录 {total_bad} 条" + ("" if args.apply else "   （只扫描，没改；要改加 --apply）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
