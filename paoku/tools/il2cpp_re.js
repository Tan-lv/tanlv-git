/*
 * IL2CPP 现场侦察 agent —— 一次性逆向工具，不参与采集。
 *
 * 定位方式与 il2cpp_state.js 保持一致（按 global-metadata.dat 文件名 + 0x3C 自引用校验）。
 * 旧版本写死 metadata 区间、并用 readCString(96) 比对类名，在本 build 上两个都会失败。
 */
var META_MAGIC = "AF 1B B1 FA";

function u32(p){ try{ return p.readU32(); } catch(e){ return 0; } }
function ascii(s){ var o=[]; for(var i=0;i<s.length;i++) o.push(("0"+s.charCodeAt(i).toString(16)).slice(-2)); o.push("00"); return o.join(" "); }
function asciiRaw(s){ var o=[]; for(var i=0;i<s.length;i++) o.push(("0"+s.charCodeAt(i).toString(16)).slice(-2)); return o.join(" "); }
function le(v){ return [v&0xff,(v>>8)&0xff,(v>>16)&0xff,(v>>>24)&0xff].map(function(b){return ("0"+b.toString(16)).slice(-2);}).join(" "); }
function readUtf8(a,n){ try { return ptr(a).readUtf8String(n); } catch(e){ return null; } }
/* 读到 NUL 为止 —— 不能用 readUtf8String(固定长度)：那是定长读，短字符串会串到下一个串 */
function readCStrAt(a, max){
    var p = ptr(a);                       // 兼容数字地址：数字没有 .add()
    var bytes = [];
    for (var i=0;i<(max||72);i++){
        var b; try { b = p.add(i).readU8(); } catch(e){ break; }
        if (b === 0) break;
        bytes.push(b);
    }
    if (!bytes.length) return null;
    var s = "";                       // 手写 ASCII 解码：Frida 的 QuickJS 里没有 TextDecoder
    for (var j=0;j<bytes.length;j++) s += String.fromCharCode(bytes[j]);
    return s;
}
function nameLike(s){ return !!s && s.length>0 && s.length<72 && /^[A-Za-z_][A-Za-z0-9_<>,.` ]*$/.test(s); }

var META = { base:null, size:0, how:"" };
var GC = { klass:null, inst:null };

function findMeta(){
    var byFile=null, byMagic=null;
    Process.enumerateRanges("r--").concat(Process.enumerateRanges("rw-")).forEach(function(r){
        if (!byFile && r.file && r.file.path && r.file.path.indexOf("global-metadata.dat")>=0) byFile=r;
        if (!byMagic && r.size>1024*1024){
            try { if (Memory.scanSync(r.base, r.size, META_MAGIC).length) byMagic=r; } catch(e){}
        }
    });
    var r = byFile || byMagic;
    if (!r) return false;
    META.base=r.base; META.size=r.size;
    META.how = byFile ? ("按文件名 "+r.file.path) : ("按魔数 "+r.base);
    return true;
}

function findClass(name){
    if (!META.base && !findMeta()) return null;
    var lit;
    try { lit = Memory.scanSync(META.base, META.size, ascii(name)); } catch(e){ return null; }
    var ranges = Process.enumerateRanges("rw-");
    for (var i=0;i<lit.length;i++){
        var refs=[], va=lit[i].address.toUInt32();
        ranges.forEach(function(r){ try{ Memory.scanSync(r.base,r.size,le(va)).forEach(function(x){refs.push(x.address);}); }catch(e){} });
        for (var j=0;j<refs.length;j++){
            var k = refs[j].sub(8);
            if (readUtf8(u32(k.add(8)), name.length) !== name) continue;
            if (u32(k.add(0x3C)) !== k.toUInt32()) continue;
            return k;
        }
    }
    return null;
}

/* 扫某个类的全部实例：内存里凡是存着「该类地址」的地方，就当作一个对象的 klass 字段 */
function instancesOf(klass){
    var out = [];
    Process.enumerateRanges("rw-").forEach(function(r){
        try { Memory.scanSync(r.base, r.size, le(klass.toUInt32())).forEach(function(x){ out.push(x.address); }); } catch(e){}
    });
    return out;
}

function scanNames(keywords){
    if (!META.base && !findMeta()) return [];
    var found = {};
    keywords.forEach(function(kw){
        var pat; try { pat = asciiRaw(kw); } catch(e){ return; }   // 子串搜索：**不能**加结尾的 00
        var hits; try { hits = Memory.scanSync(META.base, META.size, pat); } catch(e){ return; }
        hits.forEach(function(h){
            var p = h.address;
            for (var i=0;i<96;i++){
                var prev; try { prev = p.sub(1).readU8(); } catch(e){ break; }
                if (prev === 0) break;
                p = p.sub(1);
            }
            var s = readCStrAt(p, 72);
            if (nameLike(s) && s.indexOf(kw) >= 0) found[s] = true;
        });
    });
    return Object.keys(found).sort().map(function(n){ return { name:n, klass: findClass(n) !== null }; });
}

rpc.exports = {
    init: function(){
        META.base = null;
        if (!findMeta()) return { ok:false, err:"没找到 global-metadata.dat" };
        GC.klass = findClass("GameController");
        if (!GC.klass) return { ok:false, err:"找不到 GameController 类", how:META.how };
        GC.inst = null;
        instancesOf(GC.klass).forEach(function(p){
            var a=u32(p.add(0x10)), b=u32(p.add(0x14)), c=u32(p.add(0x18));
            if (!a||!b||!c) return;
            var ka=u32(ptr(a)), kb=u32(ptr(b)), kc=u32(ptr(c));
            if (!ka || ka!==kb || kb!==kc) return;
            GC.inst = p;
        });
        return { ok: !!GC.inst, how: META.how, klass: GC.klass.toString(),
                 inst: GC.inst ? GC.inst.toString() : null, err: GC.inst ? "" : "找到类但没找到实例" };
    },

    read: function(){
        if (!GC.inst) return { ok:false, err:"没有实例" };
        function val(off){ var gb=u32(GC.inst.add(off)); return gb ? (u32(ptr(gb).add(0x10))&0xff) : -1; }
        return { ok:true, inPlay:val(0x10), alive:val(0x14), paused:val(0x18) };
    },

    /* 字段表枚举：本 build 的 Il2CppClass 布局与所有已知版本都不一样，
       0x30~0x70 之间没有任何指针能解释成 FieldInfo 数组，暂时不做。
       需要字段信息时用 dumpInst / instancesOf 现场对比。 */
    fields: function(className){
        if (!className || className === "GameController") return { err:"本 build 字段表布局未确定，请用 --dump 直接看实例内存" };
        var k = findClass(className);
        return k ? { err:"本 build 字段表布局未确定", klass:k.toString() } : { err:"找不到类 "+className };
    },

    scanNames: function(keywords){ return scanNames(keywords); },

    /* 用已知字段名反推 FieldInfo 布局：
       搜 metadata 里的字段名字符串 -> 找谁存了指向它的指针（= FieldInfo.name 所在）
       -> 把该结构前后各 3 个字的窗口打出来，据此确定 name/type/offset 的位置和 stride */
    fieldProbe: function(fieldName){
        if (!META.base && !findMeta()) return { err: "没有 metadata" };
        var lits; try{ lits = Memory.scanSync(META.base, META.size, ascii(fieldName)); }
        catch(e){ return { err: String(e) }; }
        var rw = Process.enumerateRanges("rw-"), out = [];
        for (var i=0;i<lits.length;i++){
            var va = lits[i].address.toUInt32();
            rw.forEach(function(r){ try{ Memory.scanSync(r.base, r.size, le(va)).forEach(function(x){
                var H = x.address, w = [];
                for (var k=-3;k<7;k++) w.push(("0000000"+u32(H.add(k*4)).toString(16)).slice(-8));
                out.push({ holder:H.toString()+" (就是 name 字段)", words:w });
            }); }catch(e){} });
        }
        return { field: fieldName, literalCount: lits.length, hits: out };
    },

    /* 反查字段名的归属：搜 metadata 里的字段名字符串 -> 找指向它的 FieldInfo.name
       -> 读出 parent（声明类）和 offset。对没有 seed 字段名的类也用这招。 */
    fieldOwners: function(substr){
        if (!META.base && !findMeta()) return { err: "没有 metadata" };
        var rw = Process.enumerateRanges("rw-");
        var names = {}, out = [];
        var hits; try{ hits = Memory.scanSync(META.base, META.size, asciiRaw(substr)); }catch(e){ return {err:String(e)}; }
        hits.forEach(function(h){
            var q = h.address;
            for (var i=0;i<96;i++){ var prev; try{ prev=q.sub(1).readU8(); }catch(e){ break; } if(prev===0) break; q=q.sub(1); }
            var s = readCStrAt(q, 72);           // readCStrAt 现在兼容 NativePointer
            if (nameLike(s) && s.indexOf(substr) >= 0) names[s] = true;
        });
        var list = Object.keys(names);
        var seen = {};
        list.forEach(function(nm){
            var lits; try{ lits = Memory.scanSync(META.base, META.size, ascii(nm)); }catch(e){ return; }
            if (!lits.length) return;
            var va = lits[0].address.toUInt32();
            rw.forEach(function(r){ try{ Memory.scanSync(r.base, r.size, le(va)).forEach(function(x){
                var F = x.address, parent = u32(F.add(8));
                var pn = classNameOf(ptr(parent));
                if (!pn) return;
                var off = u32(F.add(0x0C));
                if (off > 0x8000) return;
                var key = pn + "|" + off + "|" + nm;
                if (seen[key]) return; seen[key] = 1;
                out.push({ cls:pn, off:"0x"+off.toString(16), field:nm });
            }); }catch(e){} });
        });
        return { substr: substr, nameCount: list.length, owners: out };
    },

    /* 枚举任意类的实例字段。实测 Il2CppClass 里 fields 指针固定在 +0x40，
       FieldInfo = {name@0, type@4, parent@8, offset@0xC, token@0x10}，stride 0x14。
       不需要字段个数：一直往后读，直到某个条目不像 FieldInfo 为止。 */
    fieldsOfClass: function(className){
        var klass = findClass(className);
        if (!klass) return { err: "找不到类 " + className };
        var fp = u32(klass.add(0x40));
        if (!fp) return { err: className + " 没有字段表" };
        var fields = [], bad = 0;
        for (var i=0;i<400;i++){
            var fi = ptr(fp).add(i*0x14);
            var nm = readCStrAt(u32(fi), 64);
            var off = u32(fi.add(0x0C));
            if (!nameLike(nm) || off > 0x8000){
                if (++bad > 2) break;             // 容忍个别坏条目，不要一遇到就停
                fields.push({ name:"<坏条目 "+i+">", off:off, parent:u32(fi.add(8)) });
                continue;
            }
            bad = 0;
            fields.push({ name:nm, off:off, parent:u32(fi.add(8)) });
        }
        return { klass: klass.toString(), fieldsPtr: ptr(fp).toString(), count: fields.length,
                 fields: fields };
    },

    /* 用已知字段名定位该类的 FieldInfo 数组，顺便反查类里"fields 指针"的偏移，然后列出全部字段。
       这比猜类布局可靠：FieldInfo = {name@0, type@4, parent@8, offset@0xC, token@0x10}, stride 0x14 */
    fieldArrayOf: function(className, seedField){
        var klass = findClass(className);
        if (!klass) return { err: "找不到类 " + className };
        var lits; try{ lits = Memory.scanSync(META.base, META.size, ascii(seedField)); }catch(e){ return {err:String(e)}; }
        var k32 = klass.toUInt32(), F = null;
        Process.enumerateRanges("rw-").forEach(function(r){
            if (F) return;
            try{ Memory.scanSync(r.base, r.size, le(lits[0].address.toUInt32())).forEach(function(x){
                if (!F && u32(x.address.add(8)) === k32) F = x.address;
            }); }catch(e){}
        });
        if (!F) return { err: "没找到 " + seedField + " 的 FieldInfo（parent 不等于该类）" };
        // 往前退到数组起点，往后数出个数
        var start = F;
        while (true){
            var prev = start.sub(0x14);
            if (u32(prev.add(8)) !== k32) break;
            start = prev;
        }
        var count = 0, cur = start;
        while (u32(cur.add(8)) === k32 && count < 500){ count++; cur = cur.add(0x14); }
        var fields = [];
        for (var i=0;i<count;i++){
            var fi = start.add(i*0x14);
            fields.push({ name: readCStrAt(u32(fi), 64), off: u32(fi.add(0x0C)), typePtr: u32(fi.add(4)) });
        }
        // 反查类里哪个偏移指向这个数组
        var ptrOff = null;
        for (var o=0; o<=0x140; o+=4){ if (u32(klass.add(o)) === start.toUInt32()) { ptrOff = "0x"+o.toString(16); break; } }
        return { klass: klass.toString(), arrayStart: start.toString(), ptrOff: ptrOff, count: count, fields: fields };
    },

    /* 快速模式：只把名字捞出来，不逐个 findClass（那个要对每个名字全内存扫，很慢） */
    scanNamesRaw: function(keywords){
        if (!META.base && !findMeta()) return [];
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
        return Object.keys(found).sort();
    },

    classesFor: function(names){
        return names.map(function(n){ var k = findClass(n); return { name:n, klass: k ? k.toString() : null, instCount: k ? instancesOf(k).length : 0 }; });
    },

    dumpInst: function(words){
        if (!GC.inst) return { err:"没有实例" };
        var out = [];
        for (var i=0;i<Math.max(4, (words||0x200)/4);i++) out.push(u32(GC.inst.add(i*4)));
        return { inst: GC.inst.toString(), u32: out };
    },

    /* dump 任意对象的前 n 字节（配合 classesFor 找警察对象用） */
    dumpAt: function(addr, words){
        var a = ptr(parseInt(addr, 16)), out = [];
        for (var i=0;i<Math.max(4,(words||0x100)/4);i++) out.push(u32(a.add(i*4)));
        return { at: a.toString(), u32: out };
    },

    instances: function(className, maxN){
        var k = findClass(className);
        if (!k) return { err:"找不到类 "+className };
        var all = instancesOf(k).map(function(p){
            var w=[]; for(var i=0;i<8;i++) w.push(("0000000"+u32(p.add(i*4)).toString(16)).slice(-8));
            return { at:p.toString(), head:w.join(" ") };
        });
        return { name:className, klass:k.toString(), count:all.length, items:all.slice(0, maxN||10) };
    }
};
