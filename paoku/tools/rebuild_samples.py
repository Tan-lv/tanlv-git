# -*- coding: utf-8 -*-
"""按 event.json 重建 补采_标注 的全部样本（event.json 是唯一事实来源）。"""
import json, os, glob, shutil, sys
import numpy as np
from PIL import Image
root = r"D:\deeplearning\make_dataset\paoku"
sys.path.insert(0, root)
from collector.config import Config
from collector.recorder import write_sample

CFG = Config(); CFG.validate()
sess = os.path.join(root, "dataset", "补采_标注")
evroot = os.path.join(root, "events")
mp = os.path.join(sess, "meta.jsonl")

def frame_time(m, idx):
    n = int(m["n_frames"]); t0, t1 = float(m["t0"]), float(m["t1"])
    return t0 + (t1 - t0) * idx / max(n - 1, 1)

def resample(m, anchor):
    n = int(m["n_frames"]); t0, t1 = float(m["t0"]), float(m["t1"])
    src = [t0 + (t1 - t0) * i / max(n - 1, 1) for i in range(n)]
    want = CFG.saved_frames; pre, post = CFG.save_pre_ms/1000.0, CFG.save_post_ms/1000.0
    a0, a1 = anchor - pre, anchor + post
    trunc = (a0 < t0 - 1e-6) or (a1 > t1 + 1e-6)
    times, idxs = [], []
    for i in range(want):
        tg = a0 + (a1 - a0) * i / max(want - 1, 1)
        j = min(range(n), key=lambda k: abs(src[k] - tg))
        times.append(tg); idxs.append(j)
    return times, idxs, trunc

# 清空旧样本（先备份 meta）
if os.path.isfile(mp): shutil.copy(mp, mp + ".bak_rebuild")
for f in glob.glob(os.path.join(sess, "*", "*.jpg")): os.remove(f)
open(mp, "w", encoding="utf-8").close()
counters = {lb: 0 for lb in CFG.labels}

done = 0
for ej in sorted(glob.glob(os.path.join(evroot, "*", "event.json"))):
    m = json.load(open(ej, encoding="utf-8"))
    lab = m.get("label")
    if not lab or m.get("cursor_t") is None:
        print("  跳过（没标）", os.path.basename(os.path.dirname(ej))); continue
    name = os.path.basename(os.path.dirname(ej))
    n = int(m["n_frames"])
    idx = int(m.get("cursor_idx")) if m.get("cursor_idx") is not None else \
          min(range(n), key=lambda k: abs(frame_time(m, k) - float(m["cursor_t"])))
    anchor = frame_time(m, idx)
    times, idxs, trunc = resample(m, anchor)
    frames = [np.asarray(Image.open(os.path.join(evroot, name, "frames", "%04d.jpg" % j)).convert("RGB"))
              for j in idxs]
    meta = write_sample(sess, counters, CFG, lab, "relabel", "", anchor, times, frames, CFG.save_fps, trunc)
    with open(mp, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
    m["sample_id"] = meta["id"]; m["sample_file"] = meta["file"]
    m["anchor_mismatch_ms"] = round((anchor - float(m["cursor_t"])) * 1000, 1)
    json.dump(m, open(ej, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  重建 %-28s %s  idx=%-4d -> %s" % (name, lab, idx, meta["file"]))
    done += 1
print()
print("重建 %d 条，jpg 数 = %d" % (done, len(glob.glob(os.path.join(sess, "*", "*.jpg")))))
from collections import Counter
print("按标签：", dict(Counter(json.loads(l)["label"] for l in open(mp, encoding="utf-8") if l.strip())))