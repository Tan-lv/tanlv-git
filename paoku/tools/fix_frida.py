# -*- coding: utf-8 -*-
import subprocess, time, socket
ADB = r"D:\leidian\LDPlayer9\adb.exe"
def sh(c, t=30): subprocess.run([ADB,"shell",c], capture_output=True, timeout=t)
def ps():
    return subprocess.run([ADB,"shell","ps -A | grep -i frida"], capture_output=True, timeout=20).stdout.decode(errors="replace")
print("杀掉全部 frida-server ...", flush=True)
sh("su -c 'killall -9 frida-server'"); time.sleep(3)
o = ps().strip()
print("  剩余:", o if o else "（清干净了）", flush=True)
print("只起一个 ...", flush=True)
sh("su -c 'nohup /data/local/tmp/frida-server -D > /data/local/tmp/fs.log 2>&1 &'"); time.sleep(5)
o = ps()
print("  现在 frida-server 行数:", len([l for l in o.splitlines() if "frida-server" in l]), flush=True)
subprocess.run([ADB,"forward","--remove-all"], capture_output=True)
subprocess.run([ADB,"forward","tcp:27042","tcp:27042"], capture_output=True)
time.sleep(1)
for port in (27042, 47800):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
        print("  端口 %d 通" % port, flush=True)
    except Exception as e:
        print("  端口 %d 不通（%s）" % (port, "状态服务没起，需要重跑 12_标注界面.py 或任何会拉起它的脚本" if port==47800 else str(e)[:40]), flush=True)