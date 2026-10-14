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
JS = open(r"D:\deeplearning\make_dataset\paoku\tools\il2cpp_re.js", encoding="utf-8").read() + (
  "\n" +
  "rpc.exports.resolve = function(hexaddr){\n" +
  "  var k = ptr(parseInt(hexaddr, 16));\n" +
  "  var nm = readCStrAt(u32(k.add(8)), 64);\n" +
  "  var list = instancesOf(k);\n" +
  "  var insts = [];\n" +
  "  for (var i=0;i<list.length;i++){ var p=list[i];\n" +
  "    insts.push({ at:p.toString(), b11:u32(p.add(0x11))&0xff,\n" +
  "                 b14:u32(p.add(0x14))&0xff, b15:u32(p.add(0x15))&0xff }); }\n" +
  "  return { name:nm, klass:k.toString(), count:list.length, insts:insts.slice(0,5) };\n" +
  "};\n")
sc = None
for i in range(4):
    try:
        dev = frida.get_device_manager().add_remote_device(R.FRIDA_HOST)
        pid = [p.pid for p in dev.enumerate_processes() if R.GAME_KEY in p.name.lower()][0]
        s = dev.attach(pid); sc = s.create_script(JS); sc.load(); break
    except Exception as e:
        print("  注入失败%d" % (i+1), flush=True); restart()
if sc is None: print("WEDGED"); sys.exit(2)
sc.exports_sync.init()
for a in ["0xb3a62e08", "0xb988b6e8", "0xb8f57cd8"]:
    try:
        d = sc.exports_sync.resolve(a)
        print("类 %s = %-34s 实例引用 %d" % (a, d["name"], d["count"]), flush=True)
        for ins in d["insts"][:4]:
            print("     obj %s   +0x11=%d  +0x14=%d  +0x15=%d" % (
                  ins["at"], ins["b11"], ins["b14"], ins["b15"]), flush=True)
    except Exception as e:
        print("  %s 解析失败 %s" % (a, str(e)[:60]), flush=True)
s.detach()