# -*- coding: utf-8 -*-
"""只保留被 event.json 的 sample_id 引用的样本（精确 ID 匹配，不做浮点比较）。"""
import json, os, glob, shutil
from collections import Counter
root = r"D:\deeplearning\make_dataset\paoku"
sess = os.path.join(root, "dataset", "补采_标注")
mp = os.path.join(sess, "meta.jsonl")
keep_ids = {}
for ej in sorted(glob.glob(os.path.join(root, "events", "*", "event.json"))):
    e = json.load(open(ej, encoding="utf-8"))
    if e.get("sample_id"):
        keep_ids[e["sample_id"]] = os.path.basename(os.path.dirname(ej))
print("event.json 引用的样本 id:", sorted(keep_ids))
rows = [json.loads(l) for l in open(mp, encoding="utf-8") if l.strip()]
keep = [m for m in rows if m["id"] in keep_ids]
drop = [m for m in rows if m["id"] not in keep_ids]
print("meta 共 %d 条 -> 保留 %d，删除 %d" % (len(rows), len(keep), len(drop)))
for m in keep: print("   保留 %-11s %s  <- %s" % (m["id"], m["label"], keep_ids[m["id"]]))
if os.path.isfile(mp): shutil.copy(mp, mp + ".bak_dedup2")
with open(mp, "w", encoding="utf-8") as fh:
    for m in keep: fh.write(json.dumps(m, ensure_ascii=False) + "\n")
n = 0
for m in drop:
    p = os.path.join(sess, m["file"].replace("/", os.sep))
    if os.path.isfile(p): os.remove(p); n += 1
# 扫一遍有没有 meta 没引用到的孤儿 jpg
ref = set(m["file"].replace("/", os.sep) for m in keep)
orphan = 0
for p in glob.glob(os.path.join(sess, "*", "*.jpg")):
    rel = os.path.relpath(p, sess)
    if rel not in ref: os.remove(p); orphan += 1
print("删除 jpg: meta 里的 %d 个 + 孤儿 %d 个" % (n, orphan))
print("最终样本数:", len(keep), " 按标签:", dict(Counter(m["label"] for m in keep)))