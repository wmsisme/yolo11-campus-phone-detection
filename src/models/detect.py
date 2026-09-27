"""YOLO11 推理/检测模块：加载模型、检测、标注、生成报告"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.utils.helpers import logger, Timer


def load_model(model_path: str = "best.pt") -> YOLO:
    """加载 YOLO11 模型

    Args:
        model_path: 模型权重文件路径，支持相对路径（相对于项目根）

    Returns:
        加载好的 YOLO 模型
    """
    path = Path(model_path)
    if not path.is_absolute():
        from src.utils.helpers import get_project_root
        # 尝试从实验目录或 models 目录查找
        candidates = [
            get_project_root() / "src" / "models" / model_path,
            get_project_root() / model_path,
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    if path.exists():
        logger.info(f"加载模型: {path}")
    else:
        logger.warning(f"本地模型 {path} 不存在，使用预训练权重: {model_path}")
        path = Path(model_path)  # 让 YOLO 自己处理

    return YOLO(str(path))


def detect_image(
    model: YOLO,
    image: np.ndarray | str | Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict]:
    """对单张图片进行目标检测

    Args:
        model: 已加载的 YOLO 模型
        image: 图片路径 (str/Path) 或 numpy 数组 (BGR)
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值

    Returns:
        检测结果列表，每项为 {"class": str, "class_id": int, "confidence": float,
                           "bbox": [x1, y1, x2, y2], "area": float}
    """
    with Timer("inference"):
        results = model.predict(image, conf=conf_threshold, iou=iou_threshold, verbose=False)

    detections = []
    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            class_names = model.names if model.names else {}
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = xyxy
                detections.append({
                    "class": class_names.get(cls_id, f"class_{cls_id}"),
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "area": round((x2 - x1) * (y2 - y1), 1),
                })

    return detections


def draw_boxes(
    image: np.ndarray,
    detections: list[dict],
    color_map: Optional[dict] = None,
) -> np.ndarray:
    """在图片上绘制检测框

    Args:
        image: OpenCV BGR 格式图片
        detections: detect_image() 返回的检测列表
        color_map: 类别名 → BGR 颜色，默认手机红色、使用者蓝色

    Returns:
        标注后的图片 (BGR)
    """
    if color_map is None:
        color_map = {
            "People using cellphone": (0, 0, 255),  # 红色
            "cellphone": (255, 0, 0),  # 蓝色
        }

    img = image.copy()

    for det in detections:
        cls_name = det["class"]
        color = color_map.get(cls_name, (0, 255, 0))
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"{cls_name}: {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def generate_report(detections: list[dict], image_name: str = "") -> dict:
    """根据检测结果生成统计报告

    Returns:
        包含计数、置信度分布、明细的字典
    """
    # 按类别统计
    class_counts: dict[str, int] = {}
    class_confidences: dict[str, list[float]] = {}

    for det in detections:
        cls_name = det["class"]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        if cls_name not in class_confidences:
            class_confidences[cls_name] = []
        class_confidences[cls_name].append(det["confidence"])

    # 类别平均置信度
    class_avg_conf = {
        k: round(sum(v) / len(v), 4) for k, v in class_confidences.items()
    }

    total_detections = len(detections)
    avg_confidence = (
        round(sum(d["confidence"] for d in detections) / total_detections, 4)
        if total_detections > 0
        else 0
    )

    report = {
        "image": image_name,
        "timestamp": datetime.now().isoformat(),
        "total_detections": total_detections,
        "average_confidence": avg_confidence,
        "class_counts": class_counts,
        "class_avg_confidence": class_avg_conf,
        "detections": detections,
    }

    return report


def report_to_markdown(report: dict) -> str:
    """将报告转为 Markdown 格式"""
    lines = [
        f"# 检测报告",
        f"",
        f"- **图片**: {report['image']}",
        f"- **检测时间**: {report['timestamp']}",
        f"- **总检测数**: {report['total_detections']}",
        f"- **平均置信度**: {report['average_confidence']:.2%}",
        f"",
        f"## 类别统计",
        f"",
        f"| 类别 | 数量 | 平均置信度 |",
        f"|------|------|-----------|",
    ]

    for cls_name in report.get("class_counts", {}):
        count = report["class_counts"][cls_name]
        avg_c = report.get("class_avg_confidence", {}).get(cls_name, 0)
        lines.append(f"| {cls_name} | {count} | {avg_c:.2%} |")

    if report.get("detections"):
        lines.extend([
            "",
            "## 检测明细",
            "",
            "| # | 类别 | 置信度 | 位置 (x1,y1,x2,y2) |",
            "|---|------|--------|-------------------|",
        ])
        for i, det in enumerate(report["detections"], 1):
            bbox = f"({det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]})"
            lines.append(f"| {i} | {det['class']} | {det['confidence']:.2%} | {bbox} |")

    return "\n".join(lines)


def report_to_json(report: dict, output_path: Path) -> None:
    """将报告保存为 JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"检测报告已保存: {output_path}")
