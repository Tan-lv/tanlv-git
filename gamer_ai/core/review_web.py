# -*- coding: utf-8 -*-
r"""会话复核 · 浏览器界面（3_复核.py --web 调它）。

和 8_标注界面.py 的分工：
    · 8_标注界面.py  = 标注【事件】（迭代数据/原始事件/ 里的一次次失误，一帧帧挑）
    · 本文件        = 复核【已有会话】（dataset/<会话>/ 里成型的样本，看胶片条、改标签、删坏样本）

为什么要有浏览器版：胶片条很长很窄，cv2 小窗口里看不清；浏览器里能放大看、
按类别跳着看、点一下就改标签/删除，比键盘窗口直观得多。

用法：
    python 3_复核.py --web                     # 复核当前区域档案里的会话
    python 3_复核.py --web <会话目录>           # 指定会话
    python 3_复核.py --web --only A            # 只看某一类
    python 3_复核.py --web --port 47910        # 换端口

改动是**立即落盘**的（改标签会搬动 jpg + 改 meta.jsonl 那一行；删除会删文件 + 行）。
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAOKU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_NAME = "复核界面.html"          # 页面（在 gamer_ai/ 下，可自由改样式）
LOCK = threading.Lock()
ARGS = None


def read_samples(session: str):
    mp = os.path.join(session, "meta.jsonl")
    out = []
    if not os.path.isfile(mp):
        return out
    with open(mp, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            out.append(m)
    return out


def session_cfg(session: str):
    """按会话的 session.json 建一份 Config（取 input_frames / stride / end_offset 等）。"""
    from dataclasses import fields
    from core.config import Config
    cfg = Config()
    p = os.path.join(session, "session.json")
    if os.path.isfile(p):
        try:
            raw = json.load(open(p, encoding="utf-8")).get("config", {})
        except Exception:
            raw = {}
        valid = {f.name for f in fields(Config)}
        for k, v in raw.items():
            if k in valid:
                if v is not None and k in ("game_res", "game_crop", "labels", "crop_rel",
                                           "animation_neg_end_ms", "approach_neg_end_ms"):
                    v = tuple(v)
                setattr(cfg, k, v)
    cfg.validate()
    return cfg


def find_sample(session: str, sid: str):
    for s in read_samples(session):
        if s.get("id") == sid:
            return s
    return None


def window_of(session: str, sid: str) -> dict:
    """这个样本**训练时真正吃的 6 帧**是哪几帧（用和训练完全相同的选窗规则）。"""
    from core.dataset import select_window_indices
    m = find_sample(session, sid)
    if m is None:
        return {"error": "找不到样本 %s" % sid}
    cfg = session_cfg(session)
    K, stride = int(cfg.input_frames), int(cfg.input_stride)
    end = -float(cfg.input_end_offset_ms)
    idx = select_window_indices(m, K, stride, end)
    times = m.get("frame_times_ms") or []
    return {"K": K, "stride": stride, "end_offset_ms": float(cfg.input_end_offset_ms),
            "frame_idx": list(idx),
            "ms": [round(float(times[i]), 1) if i < len(times) else None for i in idx],
            "n_frames": m.get("n_frames"), "frame_w": m.get("frame_w")}


def counts(session: str):
    from collections import Counter
    return dict(Counter(s.get("label", "?") for s in read_samples(session)))


def relabel(session: str, sid: str, new_label: str) -> dict:
    """改一个样本的标签：搬 jpg 到新类别目录 + 改 meta.jsonl 里那一行的 label/id/file。"""
    mp = os.path.join(session, "meta.jsonl")
    with LOCK:
        lines = []
        with open(mp, encoding="utf-8") as fh:
            lines = fh.readlines()
        hit = -1
        rec = None
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            if m.get("id") == sid:
                hit, rec = i, m
                break
        if rec is None:
            return {"error": "找不到样本 %s" % sid}
        old_rel = str(rec.get("file", "")).replace("/", os.sep)
        old_path = os.path.join(session, old_rel)
        num = sid.rsplit("_", 1)[-1]
        d = os.path.join(session, new_label)
        os.makedirs(d, exist_ok=True)
        nid, npath = None, None
        for _ in range(100):
            cand = "%s_%s" % (new_label, num)
            cp = os.path.join(d, cand + ".jpg")
            if not os.path.exists(cp) or cand == sid:
                nid, npath = cand, cp
                break
            num = "%06d" % (int(num) + 1)
        if nid is None:
            return {"error": "目标编号都被占了" }
        if os.path.isfile(old_path) and os.path.abspath(old_path) != os.path.abspath(npath):
            os.replace(old_path, npath)
        rec["label"] = new_label
        rec["id"] = nid
        rec["file"] = "%s/%s.jpg" % (new_label, nid)
        lines[hit] = json.dumps(rec, ensure_ascii=False) + "\n"
        with open(mp, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    return {"ok": True, "id": nid, "label": new_label, "file": rec["file"]}


def delete(session: str, sid: str) -> dict:
    import json as _j
    mp = os.path.join(session, "meta.jsonl")
    with LOCK:
        lines = open(mp, encoding="utf-8").readlines()
        keep, gone = [], None
        for line in lines:
            if not line.strip():
                continue
            try:
                m = _j.loads(line)
            except Exception:
                keep.append(line); continue
            if m.get("id") == sid:
                gone = m
            else:
                keep.append(line)
        if gone is None:
            return {"error": "找不到样本 %s" % sid}
        open(mp, "w", encoding="utf-8").writelines(keep)
        p = os.path.join(session, str(gone.get("file", "")).replace("/", os.sep))
        if os.path.isfile(p):
            os.remove(p)
    return {"ok": True, "id": sid}


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>会话复核</title><style>
body{margin:0;font:14px/1.5 "Microsoft YaHei",sans-serif;background:#14181d;color:#e6edf3}
header{padding:8px 12px;background:#1b2129;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.k{color:#7d8590} .v{font-weight:600}
main{display:flex;height:calc(100vh - 46px)}
#left{flex:1;display:flex;align-items:center;justify-content:center;background:#0d1117;overflow:auto}
#strip{max-width:100%;image-rendering:crisp-edges}
#right{width:270px;border-left:1px solid #262c36;padding:10px;overflow:auto}
button{margin:3px;padding:6px 10px;background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;cursor:pointer}
button:hover{background:#30363d} button.lab{width:44px;font-weight:700}
button.on{background:#2d7dd2;border-color:#2d7dd2}
button.del{background:#5a1d1d;border-color:#7d2b2b}
.row{display:flex;flex-wrap:wrap} #list div{padding:3px 6px;cursor:pointer;border-radius:4px}
#list div:hover{background:#21262d} #list div.cur{background:#2d7dd2}
</style></head><body>
<header><span class="k">会话</span><span class="v" id="sess"></span>
<span class="k">进度</span><span class="v" id="prog"></span>
<span class="k">当前</span><span class="v" id="cur"></span>
<span class="k">真实标签</span><span class="v" id="lab"></span></header>
<main><div id="left"><img id="strip" alt=""></div>
<div id="right"><div class="row" id="labels"></div>
<div class="row"><button onclick="go(-1)">← 上一个</button><button onclick="go(1)">下一个 →</button></div>
<div class="row"><button class="del" onclick="del()">删除这个样本 (Del)</button></div>
<div class="row"><button onclick="filter(null)">全部</button><span id="fbtns"></span></div>
<h3 style="margin:10px 0 4px">样本列表</h3><div id="list"></div></div></main>
<script>
var S={list:[],i:0,only:null,labels:[]};
function api(p,o){return fetch(p,o).then(function(r){return r.json()})}
function load(){api("/api/list").then(function(d){
  S.sess=d.session;S.labels=d.labels;
  S.list=d.samples.filter(function(s){return !S.only||s.label===S.only});
  document.getElementById("sess").textContent=d.session.split(/[\\/]/).pop();
  var lb=document.getElementById("labels");lb.innerHTML="";
  d.labels.forEach(function(L){var b=document.createElement("button");b.className="lab";
    b.textContent=L;b.onclick=function(){setLab(L)};lb.appendChild(b)});
  var fb=document.getElementById("fbtns");fb.innerHTML="";
  d.labels.forEach(function(L){var b=document.createElement("button");b.textContent=L;
    b.onclick=function(){filter(L)};fb.appendChild(b)});
  render()})}
function render(){
  var n=S.list.length;
  document.getElementById("prog").textContent=(n? (S.i+1):0)+" / "+n;
  if(!n){document.getElementById("strip").src="";return}
  var s=S.list[S.i];
  document.getElementById("strip").src="/strip?id="+encodeURIComponent(s.id)+"&t="+Date.now();
  document.getElementById("cur").textContent=s.id+"  ("+s.kind+")";
  document.getElementById("lab").textContent=s.label;
  var list=document.getElementById("list");list.innerHTML="";
  S.list.forEach(function(x,k){var d=document.createElement("div");
    d.textContent=(k+1)+". "+x.id;if(k===S.i)d.className="cur";
    d.onclick=function(){S.i=k;render()};list.appendChild(d)})}
function go(d){if(!S.list.length)return;S.i=(S.i+d+S.list.length)%S.list.length;render()}
function filter(L){S.only=L;load()}
function setLab(L){if(!S.list.length)return;var s=S.list[S.i];
  api("/api/relabel",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({id:s.id,label:L})}).then(function(r){
      if(r.error){alert(r.error);return} s.label=L;s.id=r.id; go(1)})}
function del(){if(!S.list.length)return;var s=S.list[S.i];
  if(!confirm("删除 "+s.id+" ？（文件和 meta 行都会删掉）"))return;
  api("/api/delete",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({id:s.id})}).then(function(r){
      if(r.error){alert(r.error);return} S.list.splice(S.i,1); if(S.i>=S.list.length)S.i=0; render()})}
document.addEventListener("keydown",function(e){
  if(e.key==="ArrowLeft"){go(-1);e.preventDefault()}
  else if(e.key==="ArrowRight"||e.key===" "){go(1);e.preventDefault()}
  else if(e.key==="Delete"){del()}
  else{var i=S.labels.indexOf(e.key); if(i<0)i="123456".indexOf(e.key);
    if(i>=0&&S.labels[i])setLab(S.labels[i]) }
})
load();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _bytes(self, b, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                # ★ 2026-10-15：页面改成独立的 复核界面.html（和 8_标注界面.py + 标注界面.html
                #   一个套路）—— 排版/交互都在那个文件里改，不用动 Python。
                hp = os.path.join(PAOKU_DIR, HTML_NAME)
                if os.path.isfile(hp):
                    return self._bytes(open(hp, "rb").read(), "text/html; charset=utf-8")
                return self._bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/api/list":
                ss = read_samples(ARGS.session)
                if ARGS.only:
                    ss = [s for s in ss if s.get("label") == ARGS.only]
                return self._json({"session": ARGS.session,
                                   "total": len(ss), "total_all": len(read_samples(ARGS.session)),
                                   "labels": list(ARGS.labels), "by_label": counts(ARGS.session),
                                   "samples": [{"id": s.get("id"), "label": s.get("label"),
                                                "kind": s.get("kind", ""),
                                                "n_frames": s.get("n_frames")} for s in ss]})
            if u.path == "/api/window":
                return self._json(window_of(ARGS.session, q.get("id", [""])[0]))

            if u.path == "/frame":
                # 从胶片条里裁出**某一帧**（复核页里显示"模型吃的那 6 帧"用）
                sid = q.get("id", [""])[0]
                i = int(q.get("i", ["0"])[0])
                m = find_sample(ARGS.session, sid)
                if m is None:
                    return self._json({"error": "找不到样本 %s" % sid}, 404)
                p = os.path.join(ARGS.session, str(m.get("file", "")).replace("/", os.sep))
                if not os.path.isfile(p):
                    return self._json({"error": "文件不在：%s" % p}, 404)
                from PIL import Image
                im = Image.open(p).convert("RGB")
                fw = int(m.get("frame_w", 240))
                sep = int(m.get("sep", 2))
                x = max(0, i) * (fw + sep)
                im = im.crop((x, 0, x + fw, im.height))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=90)
                return self._bytes(buf.getvalue(), "image/jpeg")

            if u.path == "/strip":
                sid = q.get("id", [""])[0]
                for s in read_samples(ARGS.session):
                    if s.get("id") == sid:
                        p = os.path.join(ARGS.session, str(s.get("file", "")).replace("/", os.sep))
                        if os.path.isfile(p):
                            return self._bytes(open(p, "rb").read(), "image/jpeg")
                        break
                return self._json({"error": "没找到 %s" % sid}, 404)
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        try:
            if u.path == "/api/relabel":
                lab = str(body.get("label", "")).upper()
                if lab not in ARGS.labels:
                    return self._json({"error": "未知标签 " + lab}, 400)
                r = relabel(ARGS.session, str(body.get("id", "")), lab)
                print("[复核] %s -> %s %s" % (body.get("id"), lab, r))
                return self._json(r)
            if u.path == "/api/delete":
                r = delete(ARGS.session, str(body.get("id", "")))
                print("[复核] 删除 %s %s" % (body.get("id"), r))
                return self._json(r)
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)


def main(argv=None) -> int:
    import argparse
    from core import profiles as P
    from core.sessions import expand_sessions
    ap = argparse.ArgumentParser(description="会话复核（浏览器界面）")
    ap.add_argument("session", nargs="?", default=None, help="会话目录；不给就用当前区域档案里的")
    ap.add_argument("--only", default=None, help="只看某一类，例如 --only A")
    ap.add_argument("--port", type=int, default=47910)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args(argv)
    ses = a.session
    if not ses:
        prof = P.resolve()
        cands, _bad = ([], [])
        if prof:
            cands, _bad = expand_sessions([P.dataset_root(prof)])
        if not cands:
            raise SystemExit("[复核] 没找到会话：给个会话目录，或先跑一次采集")
        # 优先挑**有样本**的会话（迭代数据/碰撞操作 这类可能还是空的，别看它新就挑它）
        with_data = [c for c in cands if len(read_samples(c)) > 0]
        pool = with_data or cands
        ses = max(pool, key=os.path.getmtime)
        print("[复核] 用当前区域档案里最新的会话：%s" % ses)
        if len(cands) > 1:
            print("       这个档案里还有别的会话（想看哪个就把路径当参数传）：")
            for c in sorted(cands, key=os.path.getmtime, reverse=True)[:8]:
                print("         %s" % c)
    ses = ses if os.path.isabs(ses) else os.path.join(PAOKU_DIR, ses)
    if not os.path.isfile(os.path.join(ses, "meta.jsonl")):
        raise SystemExit("[复核] 这里没有 meta.jsonl：%s" % ses)
    global ARGS
    ARGS = a
    ARGS.session = ses
    try:
        from core.config import Config
        ARGS.labels = list(Config().labels)
    except Exception:
        ARGS.labels = ["A", "D", "N", "Q", "S", "W"]
    url = "http://127.0.0.1:%d/" % a.port
    print("[复核] 会话 %s" % ses)
    print("[复核] 类别 %s   共 %d 个样本   %s" % (ARGS.labels, len(read_samples(ses)), url))
    print("       浏览器里：← → 翻页，A/D/N/Q/S/W 改标签，Del 删除（改完立即落盘）")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
