# -*- coding: utf-8 -*-
r"""把项目打成可分发的包（不含数据图片、不含 Python 环境）。

用法：
    python tools\打包发布.py --dry-run              只报告会打/会跳过什么，不复制
    python tools\打包发布.py                        复制到 D:\deeplearning\_package\gamer_ai_项目包_<日期>
    python tools\打包发布.py --out E:\包  --name mypack
    python tools\打包发布.py --emulator D:\leidian\LDPlayer9              连雷电一起复制（含 Android 磁盘）
    python tools\打包发布.py --emulator D:\leidian\LDPlayer9 --emulator-no-vms   只要雷电程序

规则（按用户要求：数据图片和环境不打包，模型要带上）：
    排除  profiles/<档案>/dataset、profiles/<档案>/sessions   数据图片（原机 2.58 GB，占大头）
    排除  任意层的 迭代数据 / 原始事件 目录                    迭代补数据抓的帧
    排除  旧仓库/旧找图复活、旧仓库/旧数据                      旧图片
    排除  *.mp4 *.avi（录像）、__pycache__、*.pyc、.idea
    排除  .git（超过 300 MB 时；小的话保留历史）
    保留  runs/ 下的 best.pt last.pt model_meta.json 评估结果 曲线
    保留  所有代码、文档、profiles 的 geometry.json/keys.json/profile.json
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))        # gamer_ai/tools
GAMER = os.path.dirname(HERE)                            # gamer_ai
ROOT = os.path.dirname(GAMER)                            # 仓库根 make_dataset

SKIP_DIR_NAMES = {"__pycache__", ".idea", ".ipynb_checkpoints", ".pytest_cache",
                  ".mypy_cache", ".vscode"}
SKIP_EXT = {".pyc", ".pyo", ".mp4", ".avi", ".mov", ".tmp", ".log"}
FRAME_DIRS = {"迭代数据", "原始事件"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f GB" % n


def dir_size(p: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def force_rmtree(p: str) -> None:
    """删目录：git 对象是只读的，不先清属性删不掉。"""
    def onerr(func, path, _exc):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            pass
    if os.path.isdir(p):
        shutil.rmtree(p, onerror=onerr)


def copy_one(sp: str, dp: str) -> None:
    """复制单个文件；目标已存在且只读时先改属性（否则 copy2 会 PermissionError）。"""
    if os.path.exists(dp):
        try:
            os.chmod(dp, stat.S_IWRITE)
        except OSError:
            pass
    try:
        shutil.copy2(sp, dp)
    except OSError:
        shutil.copyfile(sp, dp)      # 兜底：至少把内容写进去


def skip_reason(rel: str, is_dir: bool, keep_git: bool):
    """rel 用 / 分隔、相对仓库根。返回跳过原因字符串，None = 要打包。"""
    parts = rel.split("/")
    name = parts[-1]
    if name in SKIP_DIR_NAMES:
        return "缓存/IDE 配置"
    if not is_dir and os.path.splitext(name)[1].lower() in SKIP_EXT:
        return "录像/日志"
    if parts[0] == ".git":
        return None if keep_git else "git 历史（太大）"
    # 数据图片：gamer_ai/profiles/<档案>/dataset|sessions
    if len(parts) >= 4 and parts[0] == "gamer_ai" and parts[1] == "profiles" \
            and parts[3] in ("dataset", "sessions"):
        return "数据图片"
    if name in FRAME_DIRS:
        return "迭代数据帧"
    if rel.startswith("旧仓库/旧找图复活") or rel.startswith("旧仓库/旧数据"):
        return "旧数据图片"
    return None


def copy_emulator(src: str, dst: str, no_vms: bool, dry: bool):
    """拷雷电目录。总是跳过日志/录像/快照这类可再生垃圾。"""
    emu_skip_dirs = {"log", "Logs", "video", "Snapshots", "operationRecords", "imagecache"}
    total = kept = 0
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src).replace("\\", "/")
        rel = "" if rel == "." else rel
        if no_vms and (rel == "vms" or rel.startswith("vms/")):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in emu_skip_dirs]
        out_dir = os.path.join(dst, *rel.split("/")) if rel else dst
        if not dry:
            os.makedirs(out_dir, exist_ok=True)
        for f in files:
            sp = os.path.join(root, f)
            dp = os.path.join(out_dir, f)
            try:
                sz = os.path.getsize(sp)
            except OSError:
                continue
            if dry:
                kept += 1
                total += sz
                continue
            try:
                copy_one(sp, dp)
                kept += 1
                total += sz
            except OSError as e:
                print("  [跳过] %s (%s)" % (sp, e))
    print("  雷电：复制 %d 个文件，%s" % (kept, human(total)))


def main() -> int:
    ap = argparse.ArgumentParser(description="打包项目（不含数据图片和环境）")
    ap.add_argument("--src", default=ROOT, help="仓库根目录（默认自动取本脚本的上上级）")
    ap.add_argument("--out", default=r"D:\deeplearning\_package", help="输出到哪个目录")
    ap.add_argument("--name", default=None, help="包目录名，默认 gamer_ai_项目包_<日期>")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不复制")
    ap.add_argument("--keep-old", action="store_true",
                    help="保留目标目录里已有的文件（默认会先清空，避免只读文件导致半旧半新）")
    ap.add_argument("--emulator", default=None,
                    help=r"把雷电模拟器也复制进去，给安装目录，如 D:\leidian\LDPlayer9")
    ap.add_argument("--emulator-no-vms", action="store_true",
                    help="雷电只复制程序，不含 Android 磁盘 vms/（体积从 34GB 降到约 2GB）")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    name = a.name or ("gamer_ai_项目包_" + time.strftime("%Y%m%d"))
    dst = os.path.join(os.path.abspath(a.out), name)
    if not os.path.isdir(src):
        print("源目录不存在：%s" % src)
        return 1

    # .git 超过 300MB 就不带（历史里有大文件）
    keep_git = True
    git_dir = os.path.join(src, ".git")
    if os.path.isdir(git_dir):
        g = dir_size(git_dir)
        keep_git = g <= 300 * 1024 * 1024
        print("[git] .git = %s -> %s" % (human(g), "保留（历史跟包走）" if keep_git else "跳过（太大）"))

    print("[源] %s" % src)
    print("[目标] %s%s" % (dst, "   （dry-run，不写盘）" if a.dry_run else ""))
    if not a.dry_run:
        if os.path.isdir(dst) and not a.keep_old:
            # ★ 2026-11-01：以前直接往已有目录里覆盖，遇到只读文件（git 对象）会整片失败，
            #   产出半新半旧的目录。现在默认先清空，保证每次都是干净完整的一份。
            print("[清空] 目标目录已存在，先删掉再重建")
            force_rmtree(dst)
        os.makedirs(dst, exist_ok=True)

    kept_n = kept_b = 0
    skipped = {}      # 原因 -> [文件数, 字节, [样例路径...]]
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src).replace("\\", "/")
        rel_root = "" if rel_root == "." else rel_root
        # 目标目录在源里时别把自己拷进去
        if os.path.abspath(root).startswith(os.path.abspath(dst)):
            dirs[:] = []
            continue
        keep_dirs = []
        for d in dirs:
            rel = (rel_root + "/" + d) if rel_root else d
            r = skip_reason(rel, True, keep_git)
            if r is None:
                keep_dirs.append(d)
            else:
                p = os.path.join(root, d)
                sz = dir_size(p)
                rec = skipped.setdefault(r, [0, 0, []])
                rec[0] += 1
                rec[1] += sz
                if len(rec[2]) < 6:
                    rec[2].append(rel)
        dirs[:] = keep_dirs
        out_dir = os.path.join(dst, *rel_root.split("/")) if rel_root else dst
        if not a.dry_run and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        for f in files:
            rel = (rel_root + "/" + f) if rel_root else f
            r = skip_reason(rel, False, keep_git)
            sp = os.path.join(root, f)
            try:
                sz = os.path.getsize(sp)
            except OSError:
                continue
            if r is not None:
                rec = skipped.setdefault(r, [0, 0, []])
                rec[0] += 1
                rec[1] += sz
                if len(rec[2]) < 6:
                    rec[2].append(rel)
                continue
            if not a.dry_run:
                try:
                    copy_one(sp, os.path.join(out_dir, f))
                except OSError as e:
                    print("  [失败] %s (%s)" % (rel, e))
                    continue
            kept_n += 1
            kept_b += sz

    print("")
    print("======== 复制进包 ========")
    print("  文件 %d 个，%s" % (kept_n, human(kept_b)))
    print("")
    print("======== 排除 ========")
    for reason, (n, b, samples) in sorted(skipped.items(), key=lambda kv: -kv[1][1]):
        print("  %-16s %5d 项  %10s   例：%s" % (reason, n, human(b), samples[0] if samples else ""))

    if a.emulator:
        print("")
        print("======== 雷电模拟器 ========")
        print("  源：%s   模式：%s" % (a.emulator, "只程序（不含 vms/）" if a.emulator_no_vms else "完整（含 Android 磁盘）"))
        copy_emulator(a.emulator, os.path.join(dst, os.path.basename(a.emulator.rstrip("\\/"))),
                      a.emulator_no_vms, a.dry_run)

    if not a.dry_run:
        man = os.path.join(dst, "打包清单.txt")
        with open(man, "w", encoding="utf-8") as fp:
            fp.write("项目包：%s\n打包时间：%s\n原机器：%s\n\n" %
                     (name, time.strftime("%Y-%m-%d %H:%M:%S"), src))
            fp.write("包含：文件 %d 个，%s\n\n排除：\n" % (kept_n, human(kept_b)))
            for reason, (n, b, samples) in sorted(skipped.items(), key=lambda kv: -kv[1][1]):
                fp.write("  %-16s %5d 项  %10s\n" % (reason, n, human(b)))
                for s in samples:
                    fp.write("        %s\n" % s)
        print("")
        print("清单已写：%s" % man)
        print("下一步打包成 zip：")
        print('  7z a -tzip "%s.zip" ".\\*" -mx=5 -mmt' % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
