/*
 * 追击/碰撞事件监听 agent。
 *
 * 思路：游戏的状态开关几乎都是 GlobalVariable<bool>（类名 GlobalBool）。
 * 启动时枚举出**全部** GlobalBool 实例，之后在 agent 内部按固定频率轮询，
 * 任何一个位发生翻转就带时间戳 send() 回 Python。
 * 这样"警察出现"和"撞车"必然表现为某几个位在某一刻翻转。
 *
 * 同时轮询 GameController 实例的前 0x100 字节（防止标志不是 GlobalBool 而是普通字段）。
 */
var META_MAGIC = "AF 1B B1 FA";
function u32(p){ try{ return p.readU32(); } catch(e){ return 0; } }
function ascii(s){ var o=[]; for(var i=0;i<s.length;i++) o.push(("0"+s.charCodeAt(i).toString(16)).slice(-2)); o.push("00"); return o.join(" "); }
function le(v){ return [v&0xff,(v>>8)&0xff,(v>>16)&0xff,(v>>>24)&0xff].map(function(b){return ("0"+b.toString(16)).slice(-2);}).join(" "); }
function readUtf8(a,n){ try { return ptr(a).readUtf8String(n); } catch(e){ return null; } }
function readCStrAt(a,max){
    var bytes=[];
    for (var i=0;i<(max||72);i++){ var b; try{ b=a.add(i).readU8(); }catch(e){ break; } if(b===0) break; bytes.push(b); }
    if (!bytes.length) return null;
    var s=""; for (var j=0;j<bytes.length;j++) s += String.fromCharCode(bytes[j]);
    return s;
}
function findMeta(){
    var byFile=null, byMagic=null;
    Process.enumerateRanges("r--").concat(Process.enumerateRanges("rw-")).forEach(function(r){
        if (!byFile && r.file && r.file.path && r.file.path.indexOf("global-metadata.dat")>=0) byFile=r;
        if (!byMagic && r.size>1024*1024){ try{ if (Memory.scanSync(r.base,r.size,META_MAGIC).length) byMagic=r; }catch(e){} }
    });
    var r = byFile || byMagic; return r ? {base:r.base, size:r.size} : null;
}
var META=null;
function findClass(name){
    if(!META) META=findMeta();
    if(!META) return null;
    var lit; try{ lit=Memory.scanSync(META.base,META.size,ascii(name)); }catch(e){ return null; }
    var ranges=Process.enumerateRanges("rw-");
    for(var i=0;i<lit.length;i++){
        var refs=[], va=lit[i].address.toUInt32();
        ranges.forEach(function(r){ try{ Memory.scanSync(r.base,r.size,le(va)).forEach(function(x){refs.push(x.address);}); }catch(e){} });
        for(var j=0;j<refs.length;j++){
            var k=refs[j].sub(8);
            if(readUtf8(u32(k.add(8)), name.length)!==name) continue;
            if(u32(k.add(0x3C))!==k.toUInt32()) continue;
            return k;
        }
    }
    return null;
}
function asciiRaw(s){ var o=[]; for(var i=0;i<s.length;i++) o.push(("0"+s.charCodeAt(i).toString(16)).slice(-2)); return o.join(" "); }
function nameLike(s){ return !!s && s.length>0 && s.length<72 && /^[A-Za-z_][A-Za-z0-9_<>,.` ]*$/.test(s); }
/* 在 metadata 字符串表里按关键词搜类名（子串匹配，不能加结尾 00） */
function scanNames(keywords){
    if(!META) META=findMeta();
    if(!META) return [];
    var found = {};
    keywords.forEach(function(kw){
        var hits; try{ hits = Memory.scanSync(META.base, META.size, asciiRaw(kw)); }catch(e){ return; }
        hits.forEach(function(h){
            var q = h.address;
            for (var i=0;i<96;i++){ var prev; try{ prev=q.sub(1).readU8(); }catch(e){ break; } if(prev===0) break; q=q.sub(1); }
            var s = readCStrAt(q, 72);
            if (nameLike(s) && s.indexOf(kw) >= 0) found[s] = true;
        });
    });
    return Object.keys(found).sort().map(function(n){ return { name:n, klass: findClass(n)!==null }; });
}
function instancesOf(klass){
    var out=[];
    Process.enumerateRanges("rw-").forEach(function(r){
        try{ Memory.scanSync(r.base,r.size,le(klass.toUInt32())).forEach(function(x){out.push(x.address);}); }catch(e){}
    });
    return out;
}
function classNameOf(k){
    if (u32(k.add(0x3C)) !== k.toUInt32()) return null;
    var n = readCStrAt(u32(k.add(8)), 64);
    return (n && n.length>0 && n.length<64)? n : null;
}

var GC = { klass:null, inst:null };
var P = { gbs:[], last:null, sigLast:null, t0:0, timer:null, polls:0, flips:0, hz:40,
          watch:[], wlast:null, wticks:0,
          caddr:null, track:{}, trackList:[] };

/* 追击相关的类：数它们的"实例引用数"。游戏若在撞车时新建追击对象，这个数会跳。*/
var WATCH_CLASSES = ['ChaserBehaviour','ChaserGraphics','Guard','ChaseTargetController','ChaseNode','Player'];
var TRACK_CLASS = 'ChaserBehaviour';
var TRACK_BYTES = 0x100;
var TRACK_MAX = 400;

function readState(){
    function val(off){ var gb=u32(GC.inst.add(off)); return gb ? (u32(ptr(gb).add(0x10))&0xff) : -1; }
    return [val(0x10), val(0x14), val(0x18)];
}
function snapshot(){
    var a = new Array(P.gbs.length);
    for (var i=0;i<P.gbs.length;i++) a[i] = u32(P.gbs[i].add(0x10)) & 0xff;
    return a;
}
function sigSnapshot(){
    var a = new Array(0x100/4);
    for (var i=0;i<a.length;i++) a[i] = u32(GC.inst.add(i*4));
    return a;
}

rpc.exports = {
    start: function(hz, keywords){
        META = null;
        GC.klass = findClass("GameController");
        if(!GC.klass) return { ok:false, err:"找不到 GameController" };
        GC.inst = null;
        instancesOf(GC.klass).forEach(function(p){
            var a=u32(p.add(0x10)), b=u32(p.add(0x14)), c=u32(p.add(0x18));
            if(!a||!b||!c) return;
            var ka=u32(ptr(a)), kb=u32(ptr(b)), kc=u32(ptr(c));
            if(!ka || ka!==kb || kb!==kc) return;
            GC.inst = p;
        });
        if(!GC.inst) return { ok:false, err:"找不到 GameController 实例" };
        var gbK = findClass("GlobalBool");
        if(!gbK) return { ok:false, err:"找不到 GlobalBool" };
        P.gbs = instancesOf(gbK);
        P.last = snapshot();
        P.sigLast = sigSnapshot();
        P.t0 = Date.now(); P.polls = 0; P.flips = 0; P.hz = hz||40;
        var st = readState();
        send({ type:"init", gbCount:P.gbs.length, inst:GC.inst.toString(),
               gcClass:GC.klass.toString(), gbClass:gbK.toString(), state:{inPlay:st[0],alive:st[1],paused:st[2]} });
        if (keywords && keywords.length){
            try { send({ type:"names", items: scanNames(keywords) }); }
            catch(e){ send({ type:"names", err:String(e) }); }
        }
        // 解析要监视的类
        P.watch = [];
        WATCH_CLASSES.forEach(function(cn){
            var k = findClass(cn);
            if (k) P.watch.push({ name: cn, klass: k, n: instancesOf(k).length });
        });
        send({ type:"watch", items: P.watch.map(function(w){ return { name:w.name, n:w.n, klass:w.klass.toString() }; }) });
        P.wlast = P.watch.map(function(w){ return w.n; });

        var period = Math.max(8, Math.round(1000/P.hz));
        P.timer = setInterval(function(){
            var cur = snapshot();
            for (var i=0;i<cur.length;i++){
                if (cur[i] !== P.last[i]){
                    P.flips++;
                    send({ type:"flip", t:Date.now()-P.t0, gb:P.gbs[i].toString(), from:P.last[i], to:cur[i] });
                }
            }
            P.last = cur;
            var sig = sigSnapshot();
            for (var k=0;k<sig.length;k++){
                if (sig[k] !== P.sigLast[k]){
                    // 过滤掉"缓慢递增的计数器"（分数/距离），否则一帧一条把日志刷爆
                    var d = sig[k] - P.sigLast[k];
                    if (d > 0 && d < 0x100) { P.sigLast[k] = sig[k]; continue; }
                    send({ type:"gc", t:Date.now()-P.t0, off:"0x"+(k*4).toString(16),
                           from:("0000000"+P.sigLast[k].toString(16)).slice(-8),
                           to:("0000000"+sig[k].toString(16)).slice(-8) });
                }
            }
            P.sigLast = sig;
            P.polls++;
            // 每 ~1 秒数一次追击类实例引用数（全内存扫，比较慢，不能跟 40Hz 主循环同频）
            if ((P.polls % Math.max(1, Math.round(P.hz))) === 0 && P.watch.length){
                for (var wi=0; wi<P.watch.length; wi++){
                    var list = instancesOf(P.watch[wi].klass);
                    var nn = list.length;
                    if (nn !== P.wlast[wi]){
                        send({ type:"count", t:Date.now()-P.t0, name:P.watch[wi].name,
                               from:P.wlast[wi], to:nn });
                        P.wlast[wi] = nn;
                    }
                    // 对 TRACK_CLASS 做地址集合 diff：新增的那个地址就是刚生成的追击对象
                    if (P.watch[wi].name === TRACK_CLASS){
                        var cur = {}, i2;
                        for (i2=0;i2<list.length;i2++) cur[list[i2].toString()] = list[i2];
                        if (P.caddr){
                            for (var a in cur){ if (!P.caddr[a]) send({type:"newobj", t:Date.now()-P.t0, addr:a}); }
                            for (var b in P.caddr){ if (!cur[b]) send({type:"delobj", t:Date.now()-P.t0, addr:b}); }
                        }
                        P.caddr = cur;
                        // 对象是复用的（不会销毁重建），所以跟踪**全部**目标的内存头部，
                        // 谁变了就说明谁承载了"正在追"的状态。
                        var all = Object.keys(cur);
                        if (P.trackList.length !== all.length) P.trackList = all.slice(0, TRACK_MAX);
                        // 跟踪这些新对象的内存头部，任何变化都报出来
                        var nw = TRACK_BYTES/4;
                        for (i2=0;i2<P.trackList.length;i2++){
                            var ad = P.trackList[i2], obj = cur[ad];
                            if (!obj) continue;
                            var curw = [];
                            for (var q=0;q<nw;q++) curw.push(u32(obj.add(q*4)));
                            var oldw = P.track[ad];
                            if (oldw){
                                for (var q2=0;q2<nw;q2++){
                                    if (curw[q2] !== oldw[q2]) send({type:"obj", t:Date.now()-P.t0, addr:ad,
                                        off:"0x"+(q2*4).toString(16),
                                        from:("0000000"+oldw[q2].toString(16)).slice(-8),
                                        to:("0000000"+curw[q2].toString(16)).slice(-8)});
                                }
                            } else if (P.trackCount === undefined || P.trackCount < 3) {
                                P.trackCount = (P.trackCount||0) + 1;
                                send({type:"objdump", t:Date.now()-P.t0, addr:ad,
                                      words:curw.map(function(v){return ("0000000"+v.toString(16)).slice(-8);})});
                            }
                            P.track[ad] = curw;
                        }

                    }
                }
            }
        }, period);
        return { ok:true, gbCount:P.gbs.length, inst:GC.inst.toString() };
    },

    stop: function(){ if(P.timer){ clearInterval(P.timer); P.timer=null; } return { polls:P.polls, flips:P.flips }; },

    /* 事后把翻转过的 GlobalBool 反查"属于哪个类"：扫内存找谁存了指向它的指针，
       再从那个位置往前找对象起始（首个字是合法类指针）。 */
    owners: function(addrs){
        var out = [];
        addrs.forEach(function(a){
            var gb = ptr(parseInt(a,16)), va = gb.toUInt32(), found = [];
            Process.enumerateRanges("rw-").forEach(function(r){
                try{ Memory.scanSync(r.base,r.size,le(va)).forEach(function(x){
                    var holder = x.address;
                    for (var back=0; back<=0x200; back+=4){
                        var P2 = holder.sub(back), cand = u32(P2);
                        if (!cand || cand < 0x1000) continue;
                        var nm = classNameOf(ptr(cand));
                        if (nm){ found.push({ obj:P2.toString(), cls:nm, off:"0x"+back.toString(16) }); break; }
                    }
                }); }catch(e){}
            });
            out.push({ gb:a, owners:found.slice(0,6) });
        });
        return out;
    }
};
