/*
 * 地铁跑酷 · x86 侧 hook 探针
 *
 * 为什么只能 hook x86 侧：
 *   雷电用 libhoudini 把 ARM 翻译成 x86。libil2cpp.so / libunity.so 是 r--p（不可执行），
 *   Frida 看不到它们，Interceptor 无从下手。但 x86 的系统库（libEGL / libGLESv2 /
 *   libaudioclient / libaaudio / libc）都是正常可执行的，可以随便挂。
 *
 * 这个探针把所有"可能和游戏状态相关"的 x86 边界都挂上，周期性汇报计数。
 * 用法：进关卡前启动，正常玩一局（撞死、重开），看哪些计数在"撞车/结束"时刻发生突变。
 */
var counters = {};
function bump(k, n) { counters[k] = (counters[k] || 0) + (n === undefined ? 1 : n); }

function hookExport(libName, symName, tag) {
    var m = Process.findModuleByName(libName);
    if (!m) { console.log("[跳过] " + libName + " 不存在"); return null; }
    var p = m.findExportByName(symName);
    if (!p) { console.log("[跳过] " + libName + "!" + symName + " 找不到"); return null; }
    try {
        Interceptor.attach(p, { onEnter: function () { bump(tag); } });
        console.log("[OK]   " + libName + "!" + symName + "  -> " + tag);
        return p;
    } catch (e) { console.log("[失败] " + libName + "!" + symName + " : " + e); return null; }
}

console.log("=========== 挂载 x86 侧探针 ===========");

/* ---- 1. 图形：帧节奏 ---- */
hookExport("libEGL.so", "eglSwapBuffers", "eglSwapBuffers");
hookExport("libGLESv2.so", "glDrawArrays", "glDrawArrays");
hookExport("libGLESv2.so", "glDrawElements", "glDrawElements");
hookExport("libGLESv2.so", "glBindTexture", "glBindTexture");

/* ---- 2. 音频：不同路径都试 ---- */
var ac = Process.findModuleByName("libaudioclient.so");
if (ac) {
    ["_ZN7android10AudioTrack5writeEPKvjb",
     "_ZN7android10AudioTrack5writeEPKvm",
     "_ZN7android10AudioTrack5writeEPKvmb"].forEach(function (n) {
        var p = ac.findExportByName(n);
        if (p) {
            try {
                Interceptor.attach(p, {
                    onEnter: function (a) {
                        bump("AudioTrack.write");
                        counters._lastAudioBytes = a[1].toInt32();
                    }
                });
                console.log("[OK]   libaudioclient.so!" + n);
            } catch (e) { console.log("[失败] " + n + " : " + e); }
        }
    });
}
hookExport("libaaudio.so", "AAudioStream_write", "AAudioStream_write");

/* ---- 3. 文件：只看游戏自己的路径 ---- */
var libc = Process.findModuleByName("libc.so");
if (libc) {
    var po = libc.findExportByName("open");
    if (po) Interceptor.attach(po, {
        onEnter: function (a) {
            try {
                var s = a[0].readCString();
                if (s && s.indexOf("subwaysurf") >= 0) { bump("game.open"); send({ ev: "open", path: s }); }
            } catch (e) {}
        }
    });
}

/* ---- 4. 周期性汇报 ---- */
var t0 = Date.now();
var prev = {};
setInterval(function () {
    var sec = (Date.now() - t0) / 1000;
    var parts = [], delta = {};
    Object.keys(counters).forEach(function (k) {
        if (k[0] === "_") return;
        delta[k] = counters[k] - (prev[k] || 0);
        prev[k] = counters[k];
        parts.push(k + "=" + counters[k] + "(+" + delta[k] + ")");
    });
    send({ ev: "tick", sec: Math.round(sec), delta: delta, total: counters });
}, 2000);

console.log("=========== 探针就绪，每 2 秒汇报一次 ===========");
