# -*- coding: utf-8 -*-
"""清理 补采_标注 里的重复样本：每个事件只保留"当前 cursor_t + label"那一条。"""
import json, os, glob, shutil, sys
root = r"D:\deeplearning\make_dataset\paoku"
sess = os.path.join(root, "dataset", "补采_标注")
evroot = os.path.join(root, "events")
mp = os.path.join(sess, "meta.jsonl")

# 1) 每个事件当前应该保留哪一个
want = {}
for ej in sorted(glob.glob(os.path.join(evroot, "*", "event.json"))):
    m = json.load(open(ej, encoding="utf-8"))
    if not m.get("label") or m.get("cursor_t") is None:
        continue
    want[os.path.basename(os.path.dirname(ej))] = (round(float(m["cursor_t"]), 3), m["label"])
print("事件 %d 个，各自应保留 (锚点, 标签)：" % len(want))
for k2, v in want.items():
    print("   %-28s %s  %s" % (k2, v[0], v[1]))

# 2) 读 meta，按 t_anchor 分组，每组保留 id 最大的那条
rows = []
for line in open(mp, encoding="utf-8"):
    if line.strip(): rows.append(json.loads(line))
keep, drop = [], []
by_anchor = {}
for m in rows:
    by_anchor.setdefault(round(float(m["t_anchor"]), 3), []).append(m)
want_anchors = set(v[0] for v in want.values())
for anchor, group in by_anchor.items():
    group.sort(key=lambda x: x["id"])
    if anchor in want_anchors:
        keep.append(group[-1])
        drop += group[:-1]
    else:
        drop += group            # 锚点已经不是当前的了（重标到别处留下的）
print()
print("meta 共 %d 条 -> 保留 %d 条，删除 %d 条" % (len(rows), len(keep), len(drop)))
print("保留的：")
for m in keep: print("   %-12s %s  %s" % (m["id"], m["label"], m["file"]))

# 3) 执行（先备份）
if os.path.exists(mp):
    shutil.copy(mp, mp + ".bak_dedup")
with open(mp, "w", encoding="utf-8") as fh:
    for m in keep:
        fh.write(json.dumps(m, ensure_ascii=False) + "\n")
for m in drop:
    p = os.path.join(sess, m["file"].replace("/", os.sep))
    try:
        if os.path.isfile(p): os.remove(p)
    except Exception as e:
        print("   删除失败", p, e)
print()
print("已重写 meta.jsonl（备份 meta.jsonl.bak_dedup），当前样本数：", len(keep))
from collections import Counter
print("按标签：", dict(Counter(m["label"] for m in keep)))