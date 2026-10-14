# -*- coding: utf-8 -*-
"""修复 meta.jsonl 与实际图片不一致的问题。

采集器异常退出（崩溃 / 强杀 / 断电）后可能出现两类不一致：

  1. **索引里有、磁盘上没有** —— 撤销已经把 JPEG 删了，但 meta.jsonl 还没重写完就崩了。
     这种条目必须删掉，否则训练时读图会报错。
  2. **同一个 id / 文件出现多次** —— 撤销没跑完，类别编号没重算，下一个样本复用了同一个编号，
     把前一个的 JPEG 覆盖了。只保留最后一次（图片内容对应的就是最后一次）。

用法：
    python tools/repair_meta.py                    # 只检查，不改
    python tools/repair_meta.py --fix              # 全部会话都修（会先备份）
    python tools/repair_meta.py --session dataset/xxx --fix

修复前会自动把原文件存成 meta.jsonl.bak。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PAOKU = os.path.dirname(HERE)


def scan(session_dir: str):
    mp = os.path.join(session_dir, "meta.jsonl")
    rows = [json.loads(l) for l in io.open(mp, encoding="utf-8") if l.strip()]
    missing, keep = [], []
    seen = {}
    for r in rows:
        fp = os.path.join(session_dir, r["file"])
        if not os.path.isfile(fp):
            missing.append(r)
            continue
        # 同一个 file 出现多次：后面的覆盖前面的，保留最后一条
        if r["file"] in seen:
            keep[seen[r["file"]]] = None
        seen[r["file"]] = len(keep)
        keep.append(r)
    dup = len(rows) - len(missing) - sum(1 for x in keep if x is not None)
    return mp, rows, [x for x in keep if x is not None], missing, dup


def main() -> int:
    ap = argparse.ArgumentParser(description="修复 meta.jsonl 与图片不一致")
    ap.add_argument("--fix", action="store_true", help="真的修（默认只检查）")
    ap.add_argument("--session", default=None, help="指定会话目录；不给就检查 dataset/ 下所有")
    args = ap.parse_args()

    if args.session:
        d = args.session if os.path.isabs(args.session) else os.path.join(PAOKU, args.session)
        sessions = [d]
    else:
        root = os.path.join(PAOKU, "dataset")
        sessions = [os.path.join(root, d) for d in sorted(os.listdir(root))
                    if os.path.isfile(os.path.join(root, d, "meta.jsonl"))]

    bad = 0
    for s in sessions:
        mp, rows, keep, missing, dup = scan(s)
        name = os.path.basename(s.rstrip("\\/"))
        if not missing and not dup:
            print(f"[OK]   {name:<22} {len(rows)} 条，索引与图片一致")
            continue
        bad += 1
        print(f"[!!]   {name:<22} {len(rows)} 条 -> 删 {len(missing)} 条缺图、"
              f"去重 {dup} 条 -> 剩 {len(keep)} 条")
        if missing[:3]:
            print(f"         缺图示例: {[m['file'] for m in missing[:3]]}")
        if args.fix:
            shutil.copy2(mp, mp + ".bak")
            with io.open(mp, "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cnt = Counter(r["label"] for r in keep)
            print(f"         已修（原文件备份为 meta.jsonl.bak）  类别: "
                  f"{dict(sorted(cnt.items()))}")
    if not bad:
        print("\n所有会话都没问题")
    elif not args.fix:
        print(f"\n{bad} 个会话有问题，加 --fix 修复")
    return 0


if __name__ == "__main__":
    sys.exit(main())
