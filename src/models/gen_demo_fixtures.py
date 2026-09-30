"""生成静态 Demo 的示例图与自检基准（docs/demo/selftest.json）。

为什么需要它：
* `docs/demo/samples/*.jpg` —— 页面上的「点击体验」示例图。**不放人脸可识别的照片**，
  统一选自 Roboflow 公开数据集（CC BY 4.0）：Smart School v5 的课堂监控视角远景，
  人脸不可辨认，且与项目定位（校园场景）一致。
* `docs/demo/selftest.json` —— 浏览器自检的**基准值**：由 Python 参考实现
  （`src/models/onnx_infer.py`）在导出好的 ONNX 上跑出来的真实检测框。
  页面带 `?selftest=1` 打开时，会用同一批图跑一遍浏览器内推理，与基准逐框比对，
  把 PASS/FAIL 写进 DOM —— `tests/test_webdemo.py` 就是靠它做端到端断言的。

用法::

    python -m src.models.gen_demo_fixtures              # 用默认模型与默认示例图
    python -m src.models.gen_demo_fixtures --max-size 900
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]

# 示例图 → 输出文件名（顺序即页面上的展示顺序）
# 四张都取自 Roboflow **Smart School v5**（项目自己的公开小数据集，CC BY 4.0）的
# 教室/走廊监控视角远景：人物在画面中小到不可辨认，不涉及肖像隐私。
#
# ⚠️ 必须同时包含**横图**与**竖图**样例：真实用户照片几乎都不是 1:1，而 letterbox 的
# 灰边补偿分 x/y 两个轴——横图只有 pad[1] ≠ 0，竖图才有 pad[0] ≠ 0。2026-09-30 实测教训：
# 最初四张全是 640×640（pad 恒为 0），我把 `- pad[0]` 删掉后自检**照样 PASS**；
# 换成一张 640×400 的横图后仍然 PASS（那张只让 pad[1] ≠ 0，x 轴补偿还是没跑到）。
# 补上竖图后，同样的破坏立刻被自检抓住。生成脚本末尾会强制校验这一点。
SAMPLE_SPECS = [
    {"src": "smart school.v5i.yolov11/valid/images/image173_jpg.rf.027cfcf16eea2c29c3c242bbfd870cda.jpg",
     "out": "sample-classroom.jpg"},       # 教室，检测到 4 个手机
    {"src": "smart school.v5i.yolov11/test/images/image35_jpg.rf.3a3eaabbe52f24b868e71a3887ad5ac9.jpg",
     "out": "sample-classroom-2.jpg"},     # 教室，检测到 3 个手机
    {"src": "smart school.v5i.yolov11/train/images/image31_jpg.rf.e4d221a59f3745cb837e5017ce226e4a.jpg",
     "out": "sample-corridor.jpg"},        # 走廊，同时命中「使用手机的人」与「手机」两类
    {"src": "smart school.v5i.yolov11/valid/images/image173_jpg.rf.027cfcf16eea2c29c3c242bbfd870cda.jpg",
     "out": "sample-classroom-wide.jpg",   # 16:10 横裁 → 覆盖 pad[1]（y 轴灰边）
     "crop": [0, 0, 640, 400]},
    {"src": "smart school.v5i.yolov11/train/images/image31_jpg.rf.e4d221a59f3745cb837e5017ce226e4a.jpg",
     "out": "sample-corridor-portrait.jpg",  # 5:8 竖裁 → 覆盖 pad[0]（x 轴灰边）
     "crop": [100, 0, 500, 640]},
]

NAMES = ["People using cellphone", "cellphone"]
CONF = 0.25
IOU = 0.7


def imread(path: Path):
    from src.utils.helpers import imread_unicode

    img = cv2.imread(str(path))
    return img if img is not None else imread_unicode(path)


def write_jpeg(path: Path, img, max_size: int, quality: int = 85) -> None:
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"编码失败：{path}")
    buf.tofile(str(path))


def pick_fallback(data_root: Path, detector, exclude: set[str]) -> Path | None:
    """兜底：挑一张「检测到 ≥2 个目标」的课堂远景（宽 ≥ 500）。"""
    best = None
    for split in ("valid", "test", "train"):
        d = data_root / split / "images"
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg")):
            if p.name in exclude:
                continue
            img = imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            if w < 500:
                continue
            dets = detector.infer(img, CONF, IOU)
            if len(dets) >= 2:
                best = p
                break
        if best:
            break
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成静态 Demo 的示例图与自检基准")
    ap.add_argument("--model", type=Path, default=REPO_ROOT / "docs/demo/model/phone-yolo11n.onnx")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs/demo")
    ap.add_argument("--max-size", type=int, default=900)
    ap.add_argument("--school-root", type=Path,
                    default=REPO_ROOT / "data_source/smart school.v5i.yolov11")
    args = ap.parse_args(argv)

    from src.models.onnx_infer import OnnxDetector
    from src.utils.helpers import save_json

    if not args.model.is_file():
        print(f"模型不存在：{args.model}（先跑 src.models.export_onnx）", file=sys.stderr)
        return 2

    detector = OnnxDetector(args.model, NAMES)
    out_dir: Path = args.out_dir
    used: list[dict] = []

    for spec in SAMPLE_SPECS:
        rel, out_name = spec["src"], spec["out"]
        src = args.school_root / Path(rel).relative_to("smart school.v5i.yolov11")
        if not src.is_file():
            src = pick_fallback(args.school_root, detector, {s["src"] for s in SAMPLE_SPECS})
            if src is None:
                print(f"找不到可用示例图：{rel}", file=sys.stderr)
                return 3
            print(f"[补位] {rel} 不存在，改用 {src.name}")
        img = imread(src)
        if img is None:
            print(f"图片读不出来：{src}", file=sys.stderr)
            return 4
        if spec.get("crop"):
            x1, y1, x2, y2 = spec["crop"]
            img = img[y1:y2, x1:x2]
        dst = out_dir / "samples" / out_name
        write_jpeg(dst, img, args.max_size)
        # 基准值必须跑**最终落盘的那份图**——转码/缩放会轻微改变结果
        final = imread(dst)
        dets = detector.infer(final, CONF, IOU)
        used.append({
            "image": out_name,
            "source": f"{src.parent.parent.name}/{src.name}（Roboflow Smart School v5, CC BY 4.0）",
            "size": [final.shape[1], final.shape[0]],
            "conf": CONF,
            "iou": IOU,
            "counts": {NAMES[0]: sum(1 for d in dets if d.cls_id == 0),
                       NAMES[1]: sum(1 for d in dets if d.cls_id == 1)},
            "boxes": [d.to_dict() for d in dets],
        })
        print(f"{out_name}: {dst.stat().st_size / 1024:.0f} KB, {final.shape[1]}x{final.shape[0]}, "
              f"检测 {len(dets)} 个")

    spec = {
        "model": args.model.name,
        "classes": NAMES,
        "conf": CONF,
        "iou": IOU,
        "iouTolerance": 0.85,
        "note": "基准值由 Python 参考实现（src/models/onnx_infer.py）在导出的 ONNX 上生成；"
                "浏览器自检必须复现出同样的框（IoU ≥ iouTolerance）。",
        "models": [{"id": "n-fp32", "cases": used}],
    }

    # 自把关：letterbox 的灰边分 x/y 两轴，横图只覆盖 pad[1]、竖图才覆盖 pad[0]，
    # 缺任何一种，"灰边补偿"那段代码就没有回归覆盖（踩过两次，见 SAMPLE_SPECS 注释）
    has_wide = any(c["size"][0] > c["size"][1] for c in used)
    has_tall = any(c["size"][0] < c["size"][1] for c in used)
    if not (has_wide and has_tall):
        print(f"✗ 示例图缺少{'横图' if not has_wide else '竖图'} —— "
              f"letterbox 在 {'y' if not has_wide else 'x'} 轴上的灰边补偿不会被测试覆盖，"
              f"请补一张对应的裁剪样例", file=sys.stderr)
        return 5

    save_json(spec, out_dir / "selftest.json")
    print(f"自检基准已写入 {out_dir / 'selftest.json'}（{len(used)} 个用例："
          f"方形 {sum(1 for c in used if c['size'][0] == c['size'][1])} / "
          f"横图 {sum(1 for c in used if c['size'][0] > c['size'][1])} / "
          f"竖图 {sum(1 for c in used if c['size'][0] < c['size'][1])}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
