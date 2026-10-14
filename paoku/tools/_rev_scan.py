# -*- coding: utf-8 -*-
import sys, time, subprocess
sys.path.insert(0, r"D:\deeplearning\make_dataset\paoku\tools")
import re_probe as R, frida
KW = ["Continue", "Revive", "Resurrect", "SecondChance", "Reborn", "Rescue", "Retry", "Refill"]
def restart():
    def sh(c): subprocess.run([R.ADB, "shell", c], capture_output=True, timeout=30)
    sh("su -c 'killall frida-server'"); time.sleep(2)
    sh("su -c 'nohup /data/local/tmp/frida-server -D > /data/local/tmp/fs.log 2>&1 &'"); time.sleep(4)
    subprocess.run([R.ADB,"forward","--remove-all"], capture_output=True)
    subprocess.run([R.ADB,"forward","tcp:27042","tcp:27042"], capture_output=True)
R.ensure_ready(27042)
js = open(r"D:\deeplearning\make_dataset\paoku\tools\il2cpp_re.js", encoding="utf-8").read()
sc = None
for i in range(4):
    try:
        dev = frida.get_device_manager().add_remote_device(R.FRIDA_HOST)
        pid = [p.pid for p in dev.enumerate_processes() if R.GAME_KEY in p.name.lower()][0]
        s = dev.attach(pid); sc = s.create_script(js); sc.load(); break
    except Exception as e:
        print("  注入失败%d: %s" % (i+1, str(e)[:50]), flush=True); restart()
if sc is None: print("FRIDA_WEDGED"); sys.exit(2)
print("init:", sc.exports_sync.init().get("ok"), flush=True)
names = sc.exports_sync.scanNamesRaw(KW)
print("候选 %d 个" % len(names), flush=True)
open(r"D:\deeplearning\make_dataset\paoku\tools\_revy.txt","w",encoding="utf-8").write("\n".join(names))
low = [n for n in names if n[:1].isupper() and "." not in n]
print("像类名的 %d 个，逐个查是否已加载（慢）…" % len(low), flush=True)
res = sc.exports_sync.classesFor(low[:40])
live = [x for x in res if x["klass"]]
print("已加载 %d 个:" % len(live), flush=True)
for x in live: print("   LIVE  %-42s %s" % (x["name"], x["klass"]), flush=True)
print("--- 未加载的前 30 个 ---", flush=True)
for x in res: 
    if not x["klass"]: print("   --    " + x["name"], flush=True)
s.detach()