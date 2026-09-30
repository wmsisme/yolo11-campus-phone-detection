"""ONNX 导出的数值一致性测试：**导出前后必须是同一个模型**。

静态 Demo 走的是 ONNX + onnxruntime-web，与训练时用的 PyTorch 是两条完全不同的执行路径。
一旦导出口径跑偏（opset、输入尺寸、letterbox 方式），页面上会"看起来正常但结果是错的"，
所以这里用真实验证集图片做逐框比对：

1. 同一张 letterbox 张量喂给 PyTorch 与 ONNX → 原始输出最大偏差要小；
2. 全链路（各自预处理 + 解码 + NMS）→ 框数一致、逐框 IoU ≥ 0.9。

⚠️ PyTorch 侧必须用 `rect=False`：`model.predict()` 默认走 **rect 矩形 letterbox**（auto=True），
而导出的 ONNX 是固定 640×640 方形输入——不统一口径，比对会得出"模型不一致"的错误结论
（实测 366×157 这种极端长宽比图片会明显对不上）。

缺少数据集 / 权重 / onnx 文件时跳过。
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "experiments" / "exp3_reorganized_yolo11n" / "best.pt"
ONNX = REPO_ROOT / "docs" / "demo" / "model" / "phone-yolo11n.onnx"
IMAGES_DIR = REPO_ROOT / "data_source" / "reorganized_phone_dataset_yolo" / "reorganized_dataset" / "val" / "images"
NAMES = ["People using cellphone", "cellphone"]
N_IMAGES = 8

pytest.importorskip("onnxruntime", reason="需要 onnxruntime")
pytest.importorskip("torch", reason="需要 torch 才能做 PyTorch 侧比对")

pytestmark = pytest.mark.skipif(
    not (WEIGHTS.is_file() and ONNX.is_file() and IMAGES_DIR.is_dir()),
    reason="需要训练权重、导出的 ONNX 与验证集图片",
)


def _images() -> list[pathlib.Path]:
    from src.utils.helpers import imread_unicode

    files = sorted(p for p in IMAGES_DIR.glob("*.jpg"))
    picked = []
    for p in files:
        if imread_unicode(p) is not None:
            picked.append(p)
        if len(picked) >= N_IMAGES:
            break
    return picked


@pytest.fixture(scope="module")
def detectors():
    from ultralytics import YOLO

    from src.models.onnx_infer import OnnxDetector

    return YOLO(str(WEIGHTS)), OnnxDetector(ONNX, NAMES)


def test_onnx_contract():
    """输入/输出形状是前端依赖的硬契约（前端按 [1, 4+nc, 8400] 解码）。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: np.zeros((1, 3, 640, 640), dtype=np.float32)})[0]
    assert list(inp.shape) == [1, 3, 640, 640], f"输入形状变了：{inp.shape}"
    assert list(out.shape) == [1, 4 + len(NAMES), 8400], f"输出形状变了：{out.shape}"


def test_raw_output_matches_pytorch(detectors):
    """同一张 letterbox 张量，PyTorch 与 ONNX 的原始输出要几乎一致。"""
    import torch

    from src.models.onnx_infer import preprocess
    from src.utils.helpers import imread_unicode

    yolo, onnx_det = detectors
    net = yolo.model.float().eval()
    worst = 0.0
    for p in _images():
        img = imread_unicode(p)
        tensor, _, _ = preprocess(img, 640)
        with torch.no_grad():
            y_torch = net(torch.from_numpy(tensor))[0].numpy()
        y_onnx = onnx_det.session.run(None, {onnx_det.input_name: tensor})[0]
        worst = max(worst, float(np.abs(y_torch - y_onnx).max()))
    assert worst < 0.02, f"ONNX 与 PyTorch 原始输出偏差过大：{worst:.4f}"


def test_full_pipeline_matches_pytorch(detectors):
    """全链路逐框比对：框数一致 + 最小 IoU ≥ 0.9。"""
    from src.models.onnx_infer import iou
    from src.utils.helpers import imread_unicode

    yolo, onnx_det = detectors
    checked = 0
    for p in _images():
        img = imread_unicode(p)
        # rect=False：与导出的方形输入口径对齐（见模块 docstring）
        r = yolo.predict(img, imgsz=640, rect=False, conf=0.25, iou=0.7, verbose=False)[0]
        want = [(tuple(float(v) for v in b.xyxy[0]), int(b.cls[0])) for b in r.boxes]
        got = onnx_det.infer(img, 0.25, 0.7)

        assert len(want) == len(got), f"{p.name}: 框数不一致 torch={len(want)} onnx={len(got)}"
        for box, cid in want:
            best = max([iou(box, d.xyxy) for d in got if d.cls_id == cid] or [0.0])
            assert best >= 0.9, f"{p.name}: 类别 {cid} 的框对不上（IoU={best:.3f}）"
        checked += 1
    assert checked == N_IMAGES, f"实际只比对了 {checked} 张图"
