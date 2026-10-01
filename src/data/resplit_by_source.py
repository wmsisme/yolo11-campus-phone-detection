"""按「源组」重新切分 phone_usage_dataset，消除 train/val 泄漏。

问题（实测）：原 split 把**同一视频的相邻帧**和**同图增广版**散落到 train 与 val，
dhash 近重复率 val 7.0% / test 13.9% → 指标虚高。

分组规则（每个组只进一个 split）：
  aug_X_<src>            -> 组 = <src>                （增广与其源帧必须同折）
  cvat_<...>_videoNN_frame_N -> 组 = videoNN           （同一视频所有帧同折）
  videoNNN_frame_N       -> 组 = videoNNN
  existing_positive_*_<id>_<h1>_<h2> -> 组 = 该文件名本身（实测每个都唯一，无变体）
  phone_source_background_negative_*_<h> -> 组 = 文件名本身
  videoXXX_frame_N       -> 组 = videoXXX

⚠️ 2026-10-01 修复的漏洞（第一版分组键的 bug，导致重切后 test 仍有 9.03% 逐像素同源）：
  第一版只去掉了 roboflow 的 `.rf.<hash>` 后缀，**把"源 id 之后的帧号"留在了键里**。
  但 `<publisher>-<id>-...-mp4<帧号>` 这类是**视频抽帧**：同一个源视频的 4510 / 4585 / 442 …
  是**同一段视频的不同帧**，本该同组。帧号留在键里 → 每帧各自成组 → 相邻帧被分到
  train 与 test 两侧；而 `aug_blur_*` 又把相邻帧差异抹平，于是精确 dhash 相撞
  （实测 test 324/3589 = 9.03%、val 24/3589 = 0.67%）。
  **修法：先剥增广前缀，再去掉"源 id 之后"的一切（帧号/分辨率/page 号）**——
  见 `_drop_after_source_id()`。
  启示：做"按源分组"时，键必须落在**源实体**（一段视频/一张图）上，而不是"文件名"上；
  文件名里常常混着帧号、页号、分辨率等**同一实体内部的编号**。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "手机数据集", "phone_usage_dataset")
DST = os.path.join(REPO, "手机数据集", "phone_usage_split")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

RE_AUG = re.compile(r"^aug_[a-z]+_(.+)$", re.I)
RE_CVAT = re.compile(r"^cvat_(.+?)_frame_\d+$")
RE_VIDEO = re.compile(r"^(video[^-]*?)_frame_\d+$")
RE_FRAME = re.compile(r"_frame_\d+$")
RE_RF = re.compile(r"_jpg\.rf\.[0-9a-f]+$|\.rf\.[0-9a-f]+$", re.I)
# 形如 <publisher>-<id>[-(1080p|2160p)]-<...> ：源实体标识 = 到 <id> 为止
RE_SOURCE_ID = re.compile(r"^([A-Za-z][A-Za-z0-9]*-\d+)")
# 单独出现的 tail 编号（如 video109 -> 109、fig85 -> 85）
RE_TAIL_NUM = re.compile(r"[_-](\d{1,6})$")


def _drop_after_source_id(name: str) -> str:
    """把「源 id 之后」的帧号/分辨率/page 号等丢掉，让同一源实体的所有帧归一组。

    例：
      pexels-anna-shvets-12692098-Original-mp4510 -> pexels-anna-shvets-12692098
      production_id-4326534-2160p-mp4114          -> production_id-4326534
    """
    m = RE_SOURCE_ID.match(name)
    if not m:
        return name
    prefix = m.group(1)
    rest = name[len(prefix):]
    if rest == "":
        return name
    # 仅当"源 id 之后还跟着东西"时才截断（此时后半段必是内容编号/介质描述）
    if re.match(r"^[-\d]", rest):
        return prefix
    return name


def group_key(stem: str) -> str:
    s = stem
    m = RE_AUG.match(s)                    # 增广 -> 跟源图同组
    if m:
        s = m.group(1)
    m = RE_CVAT.match(s)                   # cvat_*_videoNN_frame_N -> videoNN
    if m:
        vm = re.search(r"video(\d+)", m.group(1))
        return f"vid{vm.group(1)}" if vm else m.group(1)
    m = RE_VIDEO.match(s)                  # videoNNN_frame_N -> videoNNN
    if m:
        vm = re.search(r"video(\d+)", m.group(1))
        return f"vid{vm.group(1)}" if vm else m.group(1)
    # 视频抽帧型（pexels-xxx-<id>-mp4<帧号> / production_id-<id>-2160p-mp4<帧号>）：
    # 必须截到「源 id」为止，否则同一视频的不同帧会各自成组（第一版的 bug）
    s = _drop_after_source_id(RE_RF.sub("", s))
    s = RE_FRAME.sub("", s)                # 其余去掉帧号
    return s


def collect():
    """返回 [(split, img_path, lbl_path, stem, group)]"""
    rows = []
    for sp in ("train", "val", "test"):
        idir = os.path.join(SRC, "images", sp)
        ldir = os.path.join(SRC, "labels", sp)
        for p in sorted(glob.glob(os.path.join(idir, "*"))):
            if os.path.splitext(p)[1].lower() not in EXTS:
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            lbl = os.path.join(ldir, stem + ".txt")
            rows.append((sp, p, lbl, stem, group_key(stem)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正落盘（默认只 dry-run）")
    args = ap.parse_args()

    rows = collect()
    print(f"读取到 {len(rows)} 张（含标注配对检查）")
    missing_lbl = sum(1 for r in rows if not os.path.exists(r[2]))
    print(f"  缺标注的图: {missing_lbl}")

    groups = defaultdict(list)
    for r in rows:
        groups[r[4]].append(r)
    print(f"分组数: {len(groups)}（原始 split 共 {len(rows)} 张）")

    sizes = Counter(len(v) for v in groups.values())
    print(f"  组大小分布(前8): {sorted(sizes.items())[:8]}")
    biggest = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]
    print("  最大组:")
    for k, v in biggest:
        print(f"    {k[:56]:58s} {len(v):5d} 张  (原split: {Counter(x[0] for x in v)})")

    # 泄漏检查：原来的 split 里，同一组是否被劈开过
    leaked = {k: v for k, v in groups.items() if len({x[0] for x in v}) > 1}
    print(f"\n**原 split 中被劈开的组: {len(leaked)} 个 "
          f"（涉及 {sum(len(v) for v in leaked.values())} 张）—— 这些就是泄漏源**")
    for k, v in sorted(leaked.items(), key=lambda kv: -len(kv[1]))[:5]:
        print(f"    {k[:56]:58s} {Counter(x[0] for x in v)}")

    # 目标划分：70/15/15，按「组大小降序 + 轮转」分配，保证各折分布相近
    target = {"train": 0.70, "val": 0.15, "test": 0.15}
    total = len(rows)
    quota = {k: int(total * v) for k, v in target.items()}
    assign = {}
    acc = Counter()
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        # 选当前最"欠额"的 split
        best = min(quota, key=lambda sp: acc[sp] / max(quota[sp], 1))
        assign[k] = best
        acc[best] += len(v)
    print(f"\n计划划分（按组整体分配）: {dict(acc)}  (目标 {quota})")

    # 负样本（空标注）也按组走，检查各折负样本比例
    neg = Counter()
    for r in rows:
        if os.path.exists(r[2]) and os.path.getsize(r[2]) == 0:
            neg[assign[r[4]]] += 1
    print(f"  各折负样本(空标注)张数: {dict(neg)}")

    if not args.apply:
        print("\n[dry-run] 未落盘。确认无误后加 --apply 执行。")
        return 0

    # ---------- 落盘 ----------
    if os.path.exists(DST):
        print(f"目标已存在，先移除: {DST}")
        shutil.rmtree(DST)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", sp), exist_ok=True)

    n = Counter()
    for k, v in groups.items():
        sp = assign[k]
        for _osp, img, lbl, stem, _g in v:
            di = os.path.join(DST, "images", sp, os.path.basename(img))
            dl = os.path.join(DST, "labels", sp, stem + ".txt")
            try:
                os.link(img, di)            # 同盘硬链接，零拷贝
            except OSError:
                shutil.copy2(img, di)
            if os.path.exists(lbl):
                try:
                    os.link(lbl, dl)
                except OSError:
                    shutil.copy2(lbl, dl)
            n[sp] += 1
    print(f"\n落盘完成: {dict(n)}")

    # data.yaml（相对路径，不写死绝对路径——避免重蹈上游 path 失效）
    yml = os.path.join(DST, "data.yaml")
    with open(yml, "w", encoding="utf-8") as f:
        f.write("train: images/train\nval: images/val\ntest: images/test\n\n")
        f.write("nc: 2\nnames: ['in_hand', 'on_ear']\n")
    print(f"data.yaml 已写入（相对路径）: {yml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
