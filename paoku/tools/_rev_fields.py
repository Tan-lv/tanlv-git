# -*- coding: utf-8 -*-
import sys, time, subprocess
sys.path.insert(0, r"D:\deeplearning\make_dataset\paoku\tools")
import re_probe as R, frida
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
        print("  注入失败%d" % (i+1), flush=True); restart()
if sc is None: print("WEDGED"); sys.exit(2)
sc.exports_sync.init()
for nm in ["ContinueKey", "AllowKeyRevive", "AllowKeyRevives", "CanRevive", "BonusReviveOffered", "AdReviveOffered"]:
    try:
        d = sc.exports_sync.fieldProbe(nm)
        hits = d.get("hits", [])
        print("%-24s 字面量 %s 处，指针命中 %d" % (nm, d.get("literalCount"), len(hits)), flush=True)
        for h in hits[:4]:
            print("      holder=%s  附近: %s" % (h["holder"][:24], " ".join(h["words"][3:9])), flush=True)
    except Exception as e:
        print("%-24s 查询失败 %s" % (nm, str(e)[:50]), flush=True)
s.detach()