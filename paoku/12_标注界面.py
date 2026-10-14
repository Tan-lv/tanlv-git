# -*- coding: utf-8 -*-
r"""补采标注 · 本地标注界面（浏览器 UI）

给 10_迭代补数据.py 录下来的事件做个可视化标注界面：
    python 12_标注界面.py
浏览器自动打开 http://127.0.0.1:47900

为什么需要一个本地服务：浏览器不能直接读写磁盘。这个文件只做两件事 ——
把 events/ 里的帧发给浏览器、把你选的帧**用采集器同一段代码**（write_sample）
写成标准胶片条塞进 dataset\补采_标注。界面本身是纯 HTML/CSS/JS。

用法：
    python 12_标注界面.py                        # 默认
    python 12_标注界面.py --events events --session dataset\补采_标注
    python 12_标注界面.py --port 47900 --no-open
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

from collector.config import Config
from collector.recorder import write_sample

HTML_NAME = "标注界面.html"
ARGS = None
CFG: Config = None
COUNTERS = {}
LOCK = threading.Lock()


# ==============================================================================
def init_session(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    sp = os.path.join(root, "session.json")
    if not os.path.isfile(sp):
        conf = CFG.to_dict() if hasattr(CFG, "to_dict") else {}
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump({"session": os.path.basename(root.rstrip("\\/")), "config": conf},
                      fh, ensure_ascii=False, indent=2)
    mp = os.path.join(root, "meta.jsonl")
    if not os.path.isfile(mp):
        open(mp, "w", encoding="utf-8").close()
    # write_sample 不会建 <标签>/ 子目录，这里先建好
    for lb in CFG.labels:
        os.makedirs(os.path.join(root, lb), exist_ok=True)


def load_counters(root: str) -> dict:
    from collections import Counter
    c = Counter()
    mp = os.path.join(root, "meta.jsonl")
    if os.path.isfile(mp):
        with open(mp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    c[json.loads(line)["label"]] += 1
                except Exception:
                    pass
    # 关键：所有标签都要预置 0。write_sample 会直接取 counters[label]，
    # 空会话里只有 {} 会 KeyError。
    return {lb: int(c.get(lb, 0)) for lb in CFG.labels}


def event_dir(name: str) -> str:
    return os.path.join(ARGS.events, name)


def read_event(name: str) -> dict:
    with open(os.path.join(event_dir(name), "event.json"), encoding="utf-8") as fh:
        return json.load(fh)


def write_event(name: str, meta: dict) -> None:
    with open(os.path.join(event_dir(name), "event.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def has_events(d: str) -> int:
    """目录 d 下有几个 <子目录>/event.json。"""
    try:
        n = 0
        for s in os.listdir(d):
            if os.path.isfile(os.path.join(d, s, "event.json")):
                n += 1
        return n
    except Exception:
        return 0


def candidate_dirs() -> list:
    """paoku/ 下所有像"事件目录"的目录（含 <子目录>/event.json），加上当前目录。"""
    out = [os.path.abspath(ARGS.events)]
    try:
        for n in sorted(os.listdir(HERE)):
            d = os.path.join(HERE, n)
            if d.startswith(ARGS.events) or ARGS.events.startswith(d):
                pass
            if os.path.isdir(d) and n not in ("dataset", "collector", "tools", "__pycache__", ".idea"):
                if has_events(d):
                    a = os.path.abspath(d)
                    if a not in out:
                        out.append(a)
    except Exception:
        pass
    return out


def browse(path: str) -> dict:
    path = os.path.abspath(path or HERE)
    if not os.path.isdir(path):
        raise ValueError("不是目录: " + path)
    subs = []
    for n in sorted(os.listdir(path)):
        p = os.path.join(path, n)
        if os.path.isdir(p):
            subs.append({"name": n, "path": p, "events": has_events(p)})
    parent = os.path.dirname(path.rstrip("\\/")) or path
    return {"path": path, "parent": parent, "subs": subs, "events": has_events(path)}


def list_events() -> list:
    out = []
    if not os.path.isdir(ARGS.events):
        return out
    for name in sorted(os.listdir(ARGS.events)):
        ej = os.path.join(ARGS.events, name, "event.json")
        if not os.path.isfile(ej):
            continue
        try:
            m = read_event(name)
        except Exception:
            continue
        out.append({"name": name, "kind": m.get("kind"), "n_frames": m.get("n_frames"),
                    "contaminated": bool(m.get("contaminated")),
                    "labeled": bool(m.get("labeled")), "discarded": bool(m.get("discarded")),
                    "label": m.get("label"), "cursor_idx": m.get("cursor_idx"),
                    "t_anchor": m.get("t_anchor"), "pre_ms": m.get("pre_ms"),
                    "post_ms": m.get("post_ms")})
    return out


def real_counts(session: str) -> dict:
    """盘上**真实**的样本数（按文件统计，不用内存计数器 —— 那个只增不减）。"""
    from collections import Counter
    c = Counter()
    mp = os.path.join(session, "meta.jsonl")
    if os.path.isfile(mp):
        with open(mp, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    c[json.loads(line)["label"]] += 1
                except Exception:
                    pass
    return dict(c)


def remove_sample(session: str, sample_id: str) -> dict:
    """把某个样本从 meta.jsonl 里删掉，并删掉它的 jpg。返回被删掉的 meta。"""
    if not sample_id:
        return {}
    mp = os.path.join(session, "meta.jsonl")
    if not os.path.isfile(mp):
        return {}
    keep, gone = [], {}
    with open(mp, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                m = json.loads(line)
            except Exception:
                keep.append(line)
                continue
            if m.get("id") == sample_id:
                gone = m
            else:
                keep.append(line)
    if not gone:
        return {}
    with open(mp, "w", encoding="utf-8") as fh:
        fh.writelines(keep)
    p = os.path.join(session, gone.get("file", "").replace("/", os.sep))
    try:
        if os.path.isfile(p):
            os.remove(p)
    except Exception:
        pass
    return gone


def frame_time(meta: dict, idx: int) -> float:
    n = int(meta["n_frames"])
    t0, t1 = float(meta["t0"]), float(meta["t1"])
    return t0 + (t1 - t0) * idx / max(n - 1, 1)


def resample(meta: dict, anchor: float):
    """按采集器规则从事件帧里重采样出标准窗口 [anchor-pre, anchor+post]。"""
    n = int(meta["n_frames"])
    t0, t1 = float(meta["t0"]), float(meta["t1"])
    src = [t0 + (t1 - t0) * i / max(n - 1, 1) for i in range(n)]
    want = CFG.saved_frames
    pre, post = CFG.save_pre_ms / 1000.0, CFG.save_post_ms / 1000.0
    a0, a1 = anchor - pre, anchor + post
    truncated = (a0 < t0 - 1e-6) or (a1 > t1 + 1e-6)
    times, idxs = [], []
    for i in range(want):
        tg = a0 + (a1 - a0) * i / max(want - 1, 1)
        j = min(range(n), key=lambda k: abs(src[k] - tg))
        times.append(tg)
        idxs.append(j)
    return times, idxs, truncated


# ==============================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "paoku-label/1.0"

    def log_message(self, fmt, *a):
        pass

    # ---------------------------------------------------------------- 工具
    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                p = os.path.join(HERE, HTML_NAME)
                with open(p, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")

            if u.path == "/api/list":
                evs = list_events()
                done = sum(1 for e in evs if e["labeled"])
                return self._json({"events": evs, "total": len(evs), "done": done,
                                   "session": os.path.abspath(ARGS.session),
                                   "events_dir": os.path.abspath(ARGS.events),
                                   "saved": sum(real_counts(ARGS.session).values()),
                                   "by_label": real_counts(ARGS.session),
                                   "saved_frames": CFG.saved_frames,
                                   "save_pre_ms": CFG.save_pre_ms,
                                   "save_post_ms": CFG.save_post_ms,
                                   # 输入窗参数以**当前 config** 为准（训练用的就是这个），
                                   # 不是 event.json 里那个（那是采集时那个旧模型的 meta）
                                   "input_frames": CFG.input_frames,
                                   "input_stride": CFG.input_stride,
                                   "input_end_offset_ms": CFG.input_end_offset_ms})

            if u.path == "/api/dirs":
                return self._json({"current": os.path.abspath(ARGS.events),
                                   "candidates": candidate_dirs(),
                                   "home": HERE})

            if u.path == "/api/browse":
                return self._json(browse(q.get("dir", [HERE])[0]))

            if u.path == "/api/event":
                name = q.get("name", [""])[0]
                m = read_event(name)
                m["_resampled"] = []
                return self._json(m)

            if u.path == "/api/window":
                # 返回某个锚点会生成的标准 37 帧窗口（帧下标 + 相对毫秒）
                name = q.get("name", [""])[0]
                idx = int(q.get("idx", ["0"])[0])
                m = read_event(name)
                a = frame_time(m, idx)
                times, idxs, trunc = resample(m, a)
                return self._json({"anchor": a, "truncated": trunc,
                                   "frame_idx": idxs,
                                   "ms": [round((t - a) * 1000) for t in times]})

            if u.path == "/frames":
                name = q.get("name", [""])[0]
                i = int(q.get("i", ["0"])[0])
                thumb = int(q.get("thumb", ["0"])[0])
                p = os.path.join(event_dir(name), "frames", "%04d.jpg" % i)
                im = Image.open(p).convert("RGB")
                if thumb:
                    w = 64
                    im = im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=88 if not thumb else 70)
                return self._send(200, buf.getvalue(), "image/jpeg")

            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except Exception:
            return self._json({"error": "bad json"}, 400)
        try:
            if u.path == "/api/label":
                return self._label(body)
            if u.path == "/api/setdir":
                d = os.path.abspath(body.get("dir", ""))
                if not os.path.isdir(d):
                    return self._json({"error": "目录不存在: " + d}, 400)
                ARGS.events = d
                n = len(list_events())
                print(f"[换目录] {d}  事件 {n} 个", flush=True)
                return self._json({"ok": True, "current": d, "events": n})

            if u.path == "/api/save_all":
                return self._save_all()

            if u.path == "/api/delete":
                name = body["name"]
                src = os.path.join(ARGS.events, name)
                if not os.path.isdir(src):
                    return self._json({"error": "不存在: " + name}, 400)
                trash = os.path.join(ARGS.events, "_deleted")
                os.makedirs(trash, exist_ok=True)
                dst = os.path.join(trash, name)
                if os.path.exists(dst):
                    import time as _t
                    dst += "_" + _t.strftime("%H%M%S")
                import shutil
                shutil.move(src, dst)
                print(f"[删除] {name} -> _deleted/", flush=True)
                return self._json({"ok": True, "moved_to": dst})

            if u.path == "/api/discard":
                m = read_event(body["name"])
                m["labeled"] = True
                m["discarded"] = True
                write_event(body["name"], m)
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def _save_all(self):
        """把**全部已标注**的事件按 event.json 记录的锚点重新生成样本。
        用当前 config 的窗口参数重切；每个事件仍然只有 1 条（先删旧的）。"""
        # ★ 批量操作前先备份 meta.jsonl（jpg 可从 events\ 的原始帧 + event.json 重建）
        mp = os.path.join(ARGS.session, "meta.jsonl")
        if os.path.isfile(mp):
            import shutil as _sh, time as _t
            bak = mp + ".bak_" + _t.strftime("%m%d_%H%M%S")
            try:
                _sh.copy(mp, bak)
                print(f"[全量] 已备份原 meta -> {os.path.basename(bak)}", flush=True)
            except Exception:
                pass
        names = []
        for nm in sorted(os.listdir(ARGS.events)):
            d = os.path.join(ARGS.events, nm)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "event.json")):
                names.append(nm)
        done, skipped, errs = 0, 0, []
        for name in names:
            m = read_event(name)
            lab = m.get("label")
            if not lab or m.get("cursor_idx") is None:
                skipped += 1
                continue
            try:
                idx = int(m["cursor_idx"])
                anchor = frame_time(m, idx)
                times, idxs, trunc = resample(m, anchor)
                frames = [np.asarray(Image.open(os.path.join(event_dir(name), "frames",
                                                           "%04d.jpg" % j)).convert("RGB"))
                          for j in idxs]
                with LOCK:
                    old = m.get("sample_id")
                    if old:
                        remove_sample(ARGS.session, old)
                    meta = write_sample(ARGS.session, COUNTERS, CFG, lab, "relabel", "",
                                        anchor, times, frames, CFG.save_fps, trunc)
                    # ★ 人工从"撞车那一刻"挑帧标的样本：它前后帧全是撞车/追击画面，
                    #   训练时**不能**在它周围合成"此刻该不动"的 N 负样本（标签噪声）。
                    meta["no_synth_neg"] = True
                    # ★ 2026-09-25：人工挑帧的样本以前**不记速度**，于是迭代补采那
                    #   9000+ 个样本在训练端"没有速度信息"，速度覆盖度和
                    #   --speed-balance 全都看不见它们。这里用同一套速度估计补上。
                    try:
                        from collector.speed import speed_px_per_sec
                        _sp, _r = speed_px_per_sec(frames, times)
                        meta["speed_px_s"] = round(float(_sp), 1)
                    except Exception:
                        pass
                    with open(os.path.join(ARGS.session, "meta.jsonl"), "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    m["sample_id"] = meta["id"]
                    m["sample_file"] = meta["file"]
                    write_event(name, m)
                done += 1
                print(f"[全量] {name} {lab} -> {meta['file']}", flush=True)
            except Exception as e:
                errs.append(f"{name}: {str(e)[:60]}")
        c = real_counts(ARGS.session)
        print(f"[全量] 完成：{done} 个事件，盘上共 {sum(c.values())} 条", flush=True)
        return self._json({"ok": True, "done": done, "skipped": skipped, "errors": errs,
                           "total": sum(c.values()), "by_label": c})

    def _label(self, body):
        name = body["name"]
        label = str(body["label"]).upper()
        idx = int(body["idx"])
        if label not in list(CFG.labels):
            return self._json({"error": "未知标签 " + label}, 400)
        m = read_event(name)
        anchor = frame_time(m, idx)
        # 同一事件 + 同标签 + 同锚点 = 没有任何变化，直接跳过，别再删了重写
        if (m.get("sample_id") and m.get("label") == label
                and m.get("cursor_idx") == idx):
            return self._json({"ok": True, "id": m["sample_id"], "label": label,
                               "file": m.get("sample_file"), "skipped": True})
        times, idxs, trunc = resample(m, anchor)
        frames = []
        for j in idxs:
            p = os.path.join(event_dir(name), "frames", "%04d.jpg" % j)
            # write_sample 要的是 numpy 数组（和采集器一致），不是 PIL Image
            frames.append(np.asarray(Image.open(p).convert("RGB")))
        with LOCK:
            # ★ 一个事件只对应一个样本：重标时先删掉上一次的，避免重复堆积
            old = m.get("sample_id")
            if old:
                removed = remove_sample(ARGS.session, old)
                if removed:
                    print(f"[替换] {name} 删掉旧样本 {old} ({removed.get('file')})", flush=True)
            meta = write_sample(ARGS.session, COUNTERS, CFG, label, "relabel", "",
                                anchor, times, frames, CFG.save_fps, trunc)
            # ★ 同上：人工挑帧样本不参与负样本合成
            meta["no_synth_neg"] = True
            # ★ 并补记速度（否则这个样本在速度统计/--speed-balance 里是隐形的）
            try:
                from collector.speed import speed_px_per_sec
                _sp, _r = speed_px_per_sec(frames, times)
                meta["speed_px_s"] = round(float(_sp), 1)
            except Exception:
                pass
            with open(os.path.join(ARGS.session, "meta.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            m["labeled"] = True
            m["label"] = label
            m["cursor_t"] = round(anchor, 4)
            m["cursor_idx"] = idx
            m["sample_id"] = meta["id"]            # ← 记住这次用的是哪个样本
            m["sample_file"] = meta["file"]
            write_event(name, m)
        print(f"[样本] {name}  标签 {label}  游标 idx={idx} ({anchor - m['t_anchor']:+.3f}s)  "
              f"-> {meta['file']}", flush=True)
        return self._json({"ok": True, "id": meta["id"], "label": label,
                           "file": meta["file"], "truncated": bool(trunc)})


# ==============================================================================
def main():
    global ARGS, CFG, COUNTERS
    ap = argparse.ArgumentParser(description="补采标注界面")
    ap.add_argument("--events", default=os.path.join(HERE, "events"), help="事件目录")
    ap.add_argument("--session", default=os.path.join(HERE, r"dataset\迭代补采\碰撞操作"),
                    help="标注样本写进哪个会话（默认和 10_迭代补数据.py 对齐）")
    ap.add_argument("--port", type=int, default=47900)
    ap.add_argument("--no-open", action="store_true", help="不要自动开浏览器")
    ARGS = ap.parse_args()

    CFG = Config()
    CFG.validate()
    init_session(ARGS.session)
    COUNTERS = load_counters(ARGS.session)

    evs = list_events()
    print(f"[事件] {os.path.abspath(ARGS.events)}  共 {len(evs)} 个，待标注 {sum(1 for e in evs if not e['labeled'])}")
    print(f"[会话] {os.path.abspath(ARGS.session)}  已有样本 {sum(COUNTERS.values())}")
    print(f"[胶片条] {CFG.saved_frames} 帧，[按键前 {CFG.save_pre_ms}ms, 后 {CFG.save_post_ms}ms]")
    print("[防重复] 一个事件只对应一个样本：重标会先删掉上一次的（已启用）")
    print("         —— 看不到这行说明你跑的是旧进程，请重启本脚本")

    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    url = f"http://127.0.0.1:{ARGS.port}/"
    print(f"[界面] {url}   （Ctrl+C 停止）")
    if not ARGS.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止] 已保存的历史样本在", os.path.abspath(ARGS.session))


if __name__ == "__main__":
    main()
