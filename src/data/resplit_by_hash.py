"""按「像素哈希」做权威去重分组：把画面相同的图合并成一组，组只进一个 split。

为什么要换掉"纯文件名分组"：
  这批数据的文件名里帧号格式**极不统一**——
    pexels-<pub>-<id>-Original-mp4<帧号>_jpg.rf.<哈希>   （帧号 + roboflow 后缀）
    pexels-<pub>-<id>-Original-mp4<帧号>                 （只有帧号）
    pexels-<pub>-<id>-1080p-mp4<帧号>                    （中间夹分辨率）
    production_id-<id>-2160p-mp4<帧号>
  用正则拼"源 id"必然漏（实测第一版漏掉 test 9.03% 的逐像素同源）。
  而 dhash 是**画面**的函数：同源实体的不同帧在模糊增强后哈希会靠拢甚至相同。

做法：
  1. 读取三折全部图片的 dhash（128 bit）
  2. **并查集合并**：哈希相同（距离 0）的图并入同一组
  3. 每个最终组整体分配到某一折（按当前折大小均衡）
  4. 用硬链接重建 phone_usage_split/<sp>/{images,labels}

不做"距离 ≤k 合并"——因为模糊增强会让相邻帧距离落在 0–4 区间，
用 k>0 会把"同一视频的邻居"和"恰好相似的异源图"混在一起；精确 0 距离是安全且确定的判据。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(REPO, "手机数据集")
SRC = os.path.join(DATASET_DIR, "phone_usage_dataset")
DST = os.path.join(DATASET_DIR, "phone_usage_split")
DIAG = os.path.join(REPO, "runs", "_diag")
CACHE = os.path.join(DIAG, "split_hash_cache.json")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
sys.path.insert(0, REPO)
from src.data.resplit_by_source import group_key  # noqa: E402  (保留作为辅助键)
from src.utils.helpers import imread_unicode  # noqa: E402


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def dhash(path):
    im = imread_unicode(path)
    if im is None:
        return None
    import cv2
    import numpy as np
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 9), interpolation=cv2.INTER_AREA)
    bits = 0
    for b in np.concatenate([(g[:, 1:] > g[:, :-1]).flatten(),
                             (g[1:, :] > g[:-1, :]).flatten()]):
        bits = (bits << 1) | int(b)
    return f"{bits:032x}"


def _hamming(a: str, b: str) -> int:
    """两个十六进制 dhash 的汉明距离（bit）。"""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def collect():
    rows = []
    for sp in ("train", "val", "test"):
        idir = os.path.join(SRC, "images", sp)
        ldir = os.path.join(SRC, "labels", sp)
        for p in sorted(glob.glob(os.path.join(idir, "*"))):
            if os.path.splitext(p)[1].lower() not in EXTS:
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            lbl = os.path.join(ldir, stem + ".txt")
            rows.append(dict(split=sp, img=p, lbl=lbl, stem=stem,
                             gname=group_key(stem), hash=None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = collect()
    print(f"共 {len(rows)} 张")

    # 1) 哈希（用缓存加速；缓存键 = 图片绝对规范化路径）
    #    ⚠️ 缓存键必须是**完整路径**：同一文件名可能同时存在于原始 train 与 val，
    #    用 split|stem 会让两个不同文件共用一条缓存（实测导致统计只覆盖 15186/23930）。
    #    另外把可能存在的旧缓存（相对路径 / 混合分隔符）一并规范化后合并进来。
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    for old_key in list(cache):
        norm = os.path.normcase(os.path.abspath(os.path.join(REPO, old_key)))
        if norm != old_key:
            cache.setdefault(norm, cache[old_key])
    print(f"  哈希缓存 {len(cache)} 条（含键规范化）")
    miss = 0
    for r in rows:
        k = os.path.normcase(os.path.abspath(r["img"]))
        h = cache.get(k)
        if h is None:
            h = dhash(r["img"])
            cache[k] = h
            miss += 1
        r["hash"] = h
    if miss:
        os.makedirs(DIAG, exist_ok=True)
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    print(f"  新算 {miss} 张，缓存总量 {len(cache)}")

    # 2) 并查集：哈希相同/**近乎相同** -> 同组；再把"文件名组名"也作为合并依据（双保险）
    #
    # ⚠️ 阈值取舍（2026-10-01 小样本实测，见 runs/_diag/validate_threshold.py）：
    #   同视频**不同帧**的哈希距离中位 44（p90 72）——差异很大，**不需要按视频合并**；
    #   异源对中位 71、最小 37，与上者重叠。所以阈值合并只该用来兜住
    #   aug↔原图 这类几乎不变形的变体（距离 0–2），取大只会把异源图误并。
    #   实测阈值 0/1/2/3/4 都能做到零跨折，故取保守的 ≤2。
    THRESHOLD = 2
    dsu = DSU()
    by_hash = defaultdict(list)
    for r in rows:
        dsu.find(r["stem"])
        if r["hash"]:
            by_hash[r["hash"]].append(r["stem"])
    for h, stems in by_hash.items():
        base = stems[0]
        for s in stems[1:]:
            dsu.union(base, s)
    # 距离 ≤THRESHOLD 的也并（用前 3 个 hex 分桶，避免全表比对）
    buckets = defaultdict(list)
    for r in rows:
        if r["hash"]:
            buckets[r["hash"][:3]].append((r["stem"], r["hash"]))
    near_merges = 0
    for _pref, items in buckets.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, ha = items[i]
                b, hb = items[j]
                if dsu.find(a) != dsu.find(b) and _hamming(ha, hb) <= THRESHOLD:
                    near_merges += 1
                    dsu.union(a, b)
    print(f"  同哈希组 {sum(1 for _h, s in by_hash.items() if len(s) > 1)}；"
          f" 近邻合并(≤{THRESHOLD}bit) {near_merges} 次")
    # 文件名组名合并（同名组必同折）
    by_name = defaultdict(list)
    for r in rows:
        by_name[r["gname"]].append(r["stem"])
    name_merges = 0
    for gname, stems in by_name.items():
        base = stems[0]
        for s in stems[1:]:
            if dsu.find(base) != dsu.find(s):
                name_merges += 1
            dsu.union(base, s)
    print(f"  同哈希组 {sum(1 for _h, s in by_hash.items() if len(s) > 1)}；"
          f" 近邻合并(≤{THRESHOLD}bit) {near_merges} 次；"
          f" 文件名合并 {name_merges} 次")

    # 3) 分组
    groups = defaultdict(list)
    for r in rows:
        groups[dsu.find(r["stem"])].append(r)
    print(f"  最终分组数 {len(groups)}（原 {len(rows)} 张）")

    # 4) 分配：组按大小降序，轮流投给最欠额的折
    target = {"train": 0.70, "val": 0.15, "test": 0.15}
    total = len(rows)
    quota = {k: int(total * v) for k, v in target.items()}
    acc = Counter()
    assign = {}
    for root, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        sp = min(quota, key=lambda s: acc[s] / max(quota[s], 1))
        assign[root] = sp
        acc[sp] += len(members)
    print(f"  计划分布 {dict(acc)}  (目标 {quota})")

    # 5) 校验计划：同哈希的图是否全部同折
    bad = 0
    for h, stems in by_hash.items():
        folds = {assign[dsu.find(s)] for s in stems}
        if len(folds) > 1:
            bad += 1
    print(f"  **计划校验：同哈希图分散到多折的组数 = {bad} （应为 0）**")

    if not args.apply:
        print("\n[dry-run] 未落盘。加 --apply 执行。")
        return 0

    if os.path.exists(DST):
        shutil.rmtree(DST)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", sp), exist_ok=True)

    n = Counter()
    for root, members in groups.items():
        sp = assign[root]
        for r in members:
            di = os.path.join(DST, "images", sp, os.path.basename(r["img"]))
            dl = os.path.join(DST, "labels", sp, r["stem"] + ".txt")
            try:
                os.link(r["img"], di)
            except OSError:
                shutil.copy2(r["img"], di)
            if os.path.exists(r["lbl"]):
                try:
                    os.link(r["lbl"], dl)
                except OSError:
                    shutil.copy2(r["lbl"], dl)
            n[sp] += 1
    yml = os.path.join(DST, "data.yaml")
    with open(yml, "w", encoding="utf-8") as f:
        f.write("train: images/train\nval: images/val\ntest: images/test\n\n")
        f.write("nc: 2\nnames: ['in_hand', 'on_ear']\n")
    print(f"\n落盘完成 {dict(n)}；data.yaml: {yml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
