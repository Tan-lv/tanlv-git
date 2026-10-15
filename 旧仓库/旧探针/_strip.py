# -*- coding: utf-8 -*-
"""删掉已废弃的复活/点击 PLAY 代码（方法 + 参数 + 状态变量）。"""
import re, shutil
p = r"D:\deeplearning\make_dataset\paoku\10_迭代补数据.py"
src = open(p, encoding="utf-8").read()
shutil.copy(p, p + ".bak_norevive")
lines = src.split("\n")

# 1) 按名字删掉整个方法（从含 def 的那行，到下一个同级 def 或文件结束）
drop = {"_click_frac", "_try_revive", "_on_game_over", "_revive_worker", "_popup_now", "_parse_xy"}
out, i, removed = [], 0, []
while i < len(lines):
    m = re.match(r"^    def (\w+)\(", lines[i])
    if m and m.group(1) in drop:
        removed.append(m.group(1))
        i += 1
        while i < len(lines) and not re.match(r"^    def \w+\(", lines[i]) and not re.match(r"^# =====", lines[i]):
            i += 1
        continue
    out.append(lines[i]); i += 1
lines = out

# 2) 删掉重复的 _focus_window（保留第一个），以及参数/状态变量
txt = "\n".join(lines)
parts = txt.split("    def _focus_window(self) -> None:")
if len(parts) == 3:
    # 保留第一份：把第二份的正文去掉
    second = parts[2]
    nxt = second.find("\n    def ")
    parts[2] = second[nxt:] if nxt > 0 else ""
    txt = "    def _focus_window(self) -> None:".join(parts)
    removed.append("_focus_window(重复的一份)")

for flag in ["--auto-revive", "--no-auto-revive", "--revive-xy", "--play-xy",
             "--revive-delay", "--revive-wait", "--max-revive",
             "--popup-white-level", "--popup-white"]:
    r2 = re.compile(r"^ *ap\.add_argument\(\s*\"" + re.escape(flag) + r"\".*?(?=^ *ap\.add_argument\()",
                    re.S | re.M)
    n = len(r2.findall(txt))
    if n:
        txt = r2.sub("", txt); removed.append(flag + " x%d" % n)
for var in ["self._reviving = False", "self.n_revived = 0", "self.n_restarted = 0",
            "self._last_pop = 0", "self._last_wlog = 0", "self._last_white = 0.0"]:
    txt = txt.replace("        " + var + "\n", "")

# 3) 清理可能残留的空 if 块
txt = txt.replace("                # （复活/点 PLAY 的逻辑已按要求移除，这里只做撞车与追击的判卷）\n", "")
open(p, "w", encoding="utf-8").write(txt)
print("删掉:", removed)