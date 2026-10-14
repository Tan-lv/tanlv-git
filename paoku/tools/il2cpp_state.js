/*
 * 纯内存读取游戏状态 —— 不 hook、不执行任何 ARM 代码。
 *
 * ============================ 2026-09 重要修正 ============================
 * 旧版把 metadata 区间写死成 META_LO=0xc0404000 .. META_HI=0xc1348000。
 * 这个区间**长度是对的**（正好等于 global-metadata.dat 的大小 16007168 字节），
 * 但基址是上一次运行时的 ASLR 结果。游戏一重新加载，映射整体平移
 * （实测从 0xc0404000 挪到 0xc06bc000），这段写死的区间就跨进了未映射空洞，
 * Memory.scanSync 抛异常 -> 被 try/catch 吞掉 -> findClass 恒返回 null
 * -> locate() 永远报"找不到类" -> 采集端静默退化成画面判断。
 *
 * 现在改成：先按**文件路径**找到真正的 global-metadata.dat 映射；
 * 找不到再退回按 IL2CPP 魔数 0xFAB11BAF 全地址空间扫描。
 * 这样游戏升级、换模拟器实例、ASLR 变化都不会再失效。
 *
 * 另外修了一个隐蔽的坑：Frida 17 的 readCString(n) 会**读满 n 字节**而不是
 * 在 NUL 处停下，用它比对类名会永远失败。这里改用 readUtf8String(长度)。
 * =========================================================================
 *
 * 原理（IL2CPP 内存布局，本 build 实测）：
 *   1. global-metadata.dat 里有全部类名字符串
 *   2. 扫 rw- 内存找"指向该字符串的指针"，它的位置就是 Il2CppClass.name（+0x08）
 *      => 类基址 = 该指针位置 - 8
 *   3. 强校验：name 精确匹配 + 类基址 +0x3C 处指向自己
 *   4. 扫 rw- 找 klass 指针 == 该类地址 的对象，就是实例
 *   5. GameController 实例 +0x10/+0x14/+0x18 = 三个 GlobalVariable<bool>*
 *      （类名 GlobalBool），值在 GlobalBool 对象 +0x10（0=假 1=真）
 */
var META_MAGIC = "AF 1B B1 FA";     // global-metadata.dat 头部魔数 0xFAB11BAF

function u32(p){ try{ return p.readU32(); } catch(e){ return 0; } }
function ascii(s){ var o=[]; for(var i=0;i<s.length;i++) o.push(("0"+s.charCodeAt(i).toString(16)).slice(-2)); o.push("00"); return o.join(" "); }
function le(v){ return [v&0xff,(v>>8)&0xff,(v>>16)&0xff,(v>>>24)&0xff].map(function(b){return ("0"+b.toString(16)).slice(-2);}).join(" "); }
function readUtf8(a, n){ try { return ptr(a).readUtf8String(n); } catch(e){ return null; } }
function rwRanges(){ return Process.enumerateRanges("rw-"); }

var META = { base:null, size:0, how:"" };

/* 定位 global-metadata.dat：优先按文件名，其次按魔数。 */
function findMeta(){
    var byFile = null, byMagic = null;
    Process.enumerateRanges("r--").concat(Process.enumerateRanges("rw-")).forEach(function(r){
        if (!byFile && r.file && r.file.path && r.file.path.indexOf("global-metadata.dat") >= 0) byFile = r;
        if (!byMagic && r.size > 1024*1024){
            try { if (Memory.scanSync(r.base, r.size, META_MAGIC).length) byMagic = r; } catch(e){}
        }
    });
    var r = byFile || byMagic;
    if (!r) return false;
    META.base = r.base; META.size = r.size;
    META.how  = byFile ? ("按文件名: " + r.file.path) : ("按魔数: " + r.base);
    return true;
}

function findClass(name){
    if (!META.base && !findMeta()) return null;
    var lit;
    try { lit = Memory.scanSync(META.base, META.size, ascii(name)); } catch(e){ return null; }
    var ranges = rwRanges();
    for (var i=0;i<lit.length;i++){
        var refs = [], va = lit[i].address.toUInt32();
        ranges.forEach(function(r){
            try { Memory.scanSync(r.base, r.size, le(va)).forEach(function(x){ refs.push(x.address); }); } catch(e){}
        });
        for (var j=0;j<refs.length;j++){
            var k = refs[j].sub(8);
            if (readUtf8(u32(k.add(8)), name.length) !== name) continue;
            if (u32(k.add(0x3C)) !== k.toUInt32()) continue;
            return k;
        }
    }
    return null;
}

/* 追击者（警察）对象。内存里有 200 多个位置存着 ChaserBehaviour 类指针，
   其中只有一个是真实例 —— 判据：它的 +0x50 (_chaserGraphics) 指向 ChaserGraphics 类的对象。
   真实例 +0x64 = _currentDistanceToCharacter (float)
           +0x68 = _killed                   (bool, 1 = 警察正在追) */
var CHASER = { klass:null, cgrClass:null, inst:null };

/* ---------------- 复活弹窗（Continue?）----------------
   逆向结果（FieldInfo: name@0 type@4 parent@8 offset@0xC）：
     ReviveChoiceInfo    AdReviveOffered    @ +0x14
                         BonusReviveOffered @ +0x15
     GameModeReviveData  AllowKeyRevive     @ +0x11
   难点：instancesOf 找出的地址大多是假阳性（内存别处也存着类指针）。
   这里的过滤条件：+0x14 和 +0x15 两个字节必须 <=1（bool 该有的值），
   否则基本可以断定不是真对象。 */
var RV = { rci:null, gmrd:null, rciList:[], gmrList:[] };

function locateRevive(){
    RV.rci = findClass("ReviveChoiceInfo");
    RV.gmrd = findClass("GameModeReviveData");
    function cands(k){
        var out = [];
        if (!k) return out;
        try {
            Process.enumerateRanges("rw-").forEach(function(r){
                try { Memory.scanSync(r.base, r.size, le(k.toUInt32())).forEach(function(x){
                    var p = x.address;
                    if ((u32(p.add(0x14)) & 0xff) <= 1 && (u32(p.add(0x15)) & 0xff) <= 1) out.push(p);
                }); } catch(e){}
            });
        } catch(e){}
        return out;
    }
    RV.rciList = cands(RV.rci);
    RV.gmrList = cands(RV.gmrd);
}

function locateChaser(){
    CHASER.inst = null;
    var cb = findClass("ChaserBehaviour");
    var cg = findClass("ChaserGraphics");
    if (!cb || !cg) return;
    CHASER.klass = cb; CHASER.cgrClass = cg;
    var hits = [];
    Process.enumerateRanges("rw-").forEach(function(r){
        try { Memory.scanSync(r.base, r.size, le(cb.toUInt32())).forEach(function(x){ hits.push(x.address); }); } catch(e){}
    });
    hits.forEach(function(O){
        var g = u32(O.add(0x50));
        if (g && u32(ptr(g)) === cg.toUInt32()) CHASER.inst = O;
    });
}

var STATE = { gcClass:null, gbClass:null, inst:null, ok:false, err:"" };

function locate(){
    try {
        META.base = null;
        if (!findMeta()){ STATE.ok = false; STATE.err = "没找到 global-metadata.dat"; return; }
        var ranges = rwRanges();
        STATE.gcClass = findClass("GameController");
        STATE.gbClass = findClass("GlobalBool");
        STATE.inst = null;
        if (!STATE.gcClass || !STATE.gbClass) { STATE.err = "找不到类"; return; }
        var insts = [];
        ranges.forEach(function(r){ try{
            Memory.scanSync(r.base, r.size, le(STATE.gcClass.toUInt32())).forEach(function(x){ insts.push(x.address); });
        }catch(e){} });
        insts.forEach(function(p){
            var a=u32(p.add(0x10)), b=u32(p.add(0x14)), c=u32(p.add(0x18));
            if (!a||!b||!c) return;
            if (u32(ptr(a))!==STATE.gbClass.toUInt32()) return;
            if (u32(ptr(b))!==STATE.gbClass.toUInt32()) return;
            if (u32(ptr(c))!==STATE.gbClass.toUInt32()) return;
            // 三个状态变量都指向 GlobalBool 就够了；旧的 "u32(p+4)===0" 条件
            // 在本 build 上会把唯一的合法实例过滤掉，所以不再作为硬条件。
            STATE.inst = p;
        });
        STATE.ok = !!STATE.inst;
        if (!STATE.ok) STATE.err = "找到类但没找到实例";
        locateChaser();
        locateRevive();
    } catch (e) { STATE.ok = false; STATE.err = String(e); }
}

rpc.exports = {
    locate: function(){ locate(); return { ok: STATE.ok, err: STATE.err, how: META.how,
        gc: STATE.gcClass ? STATE.gcClass.toString() : null,
        gb: STATE.gbClass ? STATE.gbClass.toString() : null,
        inst: STATE.inst ? STATE.inst.toString() : null }; },
    read: function(){
        if (!STATE.ok) locate();
        if (!STATE.ok) return { ok:false, err: STATE.err };
        try {
            function val(off){ var gb = u32(STATE.inst.add(off)); return gb ? (u32(ptr(gb).add(0x10)) & 0xff) : -1; }
            var chasing = -1, dist = -1.0;
            if (CHASER.inst){
                chasing = u32(CHASER.inst.add(0x68)) & 0xff;
                try { dist = ptr(CHASER.inst.add(0x64)).readFloat(); } catch(e){ dist = -1.0; }
            }
            // 复活弹窗：把候选对象的两个 bool 汇总起来
            var adOffer = -1, keyOffer = -1, allowKey = -1;
            if (RV.rciList.length){
                adOffer = 0; keyOffer = 0;
                RV.rciList.forEach(function(p){
                    if (u32(p.add(0x14)) & 0xff) adOffer = 1;
                    if (u32(p.add(0x15)) & 0xff) keyOffer = 1;
                });
            }
            if (RV.gmrList.length){
                allowKey = 0;
                RV.gmrList.forEach(function(p){ if (u32(p.add(0x11)) & 0xff) allowKey = 1; });
            }
            return { ok:true, inPlay: val(0x10), alive: val(0x14), paused: val(0x18),
                     chasing: chasing, chaseDist: Math.round(dist * 100) / 100,
                     adOffer: adOffer, keyOffer: keyOffer, allowKey: allowKey,
                     rciN: RV.rciList.length, gmrN: RV.gmrList.length,
                     chaser: CHASER.inst ? CHASER.inst.toString() : null,
                     inst: STATE.inst.toString() };
        } catch (e) { STATE.ok = false; return { ok:false, err:String(e) }; }
    }
};
