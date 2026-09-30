"""ONNX 推理的 **参考实现**（numpy），与前端 `docs/demo/app.js` 逐步骤对齐。

它的存在有两个目的：
1. 作为前后端口径的**规范**——前端 JS 里的 letterbox / 解码 / NMS 必须与这里一致，
   否则同一张图在 Python 与浏览器里会给出不同的框；
2. 作为**测试基准**——`tests/test_onnx_parity.py` 用它把 ONNX 与 PyTorch 的预测
   逐框对比（IoU ≥ 0.9），确保导出的模型没有跑偏。

契约（导出时固定，见 `export_onnx.py`）：
  * 输入 `images`：float32 `[1,3,S,S]`，RGB、/255、letterbox（填充值 114）
  * 输出 `output0`：float32 `[1,4+nc,8400]`，行 0-3 为 cx,cy,w,h（S 尺度），
    行 4.. 为各类别置信度（已 sigmoid）；**模型内不含 NMS**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PAD_VALUE = 114  # ultralytics letterbox 默认填充灰度


@dataclass
class Detection:
    """一个检测框（坐标为原图像素）。"""

    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str

    def to_dict(self) -> dict:
        return {
            "xyxy": [round(float(v), 2) for v in self.xyxy],
            "conf": round(float(self.conf), 4),
            "cls_id": int(self.cls_id),
            "cls_name": self.cls_name,
        }


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """等比缩放 + 居中灰边，返回 (填充后图像, 缩放比 r, (dw, dh))。

    与 ultralytics `LetterBox(size, auto=False)` 在 size 为 32 的整数倍时等价：
    r = min(size/w, size/h)，dw/dh 取半宽半高（不取整，去 pad 时不引入偏差）。
    """
    h, w = image.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = round(w * r), round(h * r)
    if (w, h) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return padded, r, (dw, dh)


def preprocess(image_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """BGR uint8 图像 → 模型输入张量 `[1,3,S,S]`（RGB、/255、CHW）。"""
    padded, r, pad = letterbox(image_bgr, size)
    rgb = padded[:, :, ::-1]  # BGR → RGB
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor[None, ...], r, pad


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """按类别无关的贪心 NMS，返回保留下来的下标（输入为 xyxy）。"""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


def decode(
    output: np.ndarray,
    r: float,
    pad: tuple[float, float],
    shape: tuple[int, int],
    names: list[str],
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    max_det: int = 300,
) -> list[Detection]:
    """把 `[1,4+nc,8400]` 的原始输出解码成原图坐标系下的检测框。"""
    preds = np.squeeze(output, axis=0).T  # (8400, 4+nc)
    boxes_xywh = preds[:, :4]
    scores_all = preds[:, 4:]
    cls_ids = scores_all.argmax(axis=1)
    confs = scores_all[np.arange(len(cls_ids)), cls_ids]

    mask = confs >= conf_thres
    if not mask.any():
        return []
    boxes_xywh, confs, cls_ids = boxes_xywh[mask], confs[mask], cls_ids[mask]

    # cxcywh → xyxy（imgsz 尺度）
    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # 去掉 letterbox 的灰边并缩放回原图
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy /= r
    h, w = shape
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

    # ultralytics 是**按类别**做 NMS（同一位置两类都留），这里逐类执行
    keep: list[int] = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep.extend(idx[nms(xyxy[idx], confs[idx], iou_thres)])
    keep = sorted(keep, key=lambda i: -confs[i])[:max_det]

    return [
        Detection(
            xyxy=tuple(float(v) for v in xyxy[i]),
            conf=float(confs[i]),
            cls_id=int(cls_ids[i]),
            cls_name=names[int(cls_ids[i])] if int(cls_ids[i]) < len(names) else str(int(cls_ids[i])),
        )
        for i in keep
    ]


class OnnxDetector:
    """薄封装：一次加载模型，多张图复用。"""

    def __init__(self, model_path: str | Path, names: list[str], imgsz: int = 640):
        import onnxruntime as ort

        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.names = names
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(
        self,
        image_bgr: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
    ) -> list[Detection]:
        tensor, r, pad = preprocess(image_bgr, self.imgsz)
        out = self.session.run(None, {self.input_name: tensor})[0]
        return decode(out, r, pad, image_bgr.shape[:2], self.names, conf_thres, iou_thres)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """两个 xyxy 框的交并比（测试里做逐框匹配用）。"""
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)
