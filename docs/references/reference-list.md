# 参考来源

## 数据集

| 名称 | 来源 | 说明 |
|------|------|------|
| **`phone_usage`（当前生产集）** | COCO 派生的手机使用数据集 | 约 1.7 万张，类别：**in_hand（手持手机）/ on_ear（贴耳手机）**（CC BY 4.0）。本地保存、**不随仓库分发** |
| Smart School v5 | [Roboflow Universe](https://universe.roboflow.com/) | **早期数据集**（141 张，类别 People using cellphone / cellphone）。已不用于生产训练 |
| University-Outdoor v1 | [Roboflow Universe](https://universe.roboflow.com/) | ~~重组数据集来源之一~~（该重组集已弃用，见下） |
| University ver 2 / ver 3 | [Roboflow Universe](https://universe.roboflow.com/) | ~~重组数据集来源之一~~（同上） |
| School v1 | [Roboflow Universe](https://universe.roboflow.com/) | ~~重组数据集来源之一~~（同上） |

> ⚠️ **上述 4 个数据集合并成的 `reorganized_phone_dataset_yolo`（22,879 张）已整体弃用**：
> 审计发现其标注被污染——大量 `cellphone` 框套在人脸/头部上（`cellphone` 框中 77.2% 边长 >25%，
> 而「People using cellphone」中位边长仅 13.5%，语义倒挂），导致模型**把人脸判成手机**
> （真实负样本误报 67.7%）。详见 `docs/exec-plans/tech-debt-tracker.md` #015/#016
> 与 `reports/final-report.md` 第 9 节「阶段七」。
>
> **当前生产训练使用 `phone_usage` 数据集**（本地保存，需自行获取）。**所有数据集均不随本仓库分发**。

## 模型与框架

| 名称 | 版本 | 来源 | 说明 |
|------|------|------|------|
| YOLO11 | 8.3.x | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 最新 YOLO 目标检测框架 |
| PyTorch | 2.x | [pytorch.org](https://pytorch.org/) | 深度学习框架 |
| Streamlit | 1.28+ | [streamlit.io](https://streamlit.io/) | Python Web 应用框架 |
| OpenCV | 4.8+ | [opencv.org](https://opencv.org/) | 计算机视觉库 |

## 论文

| 论文 | 说明 |
|------|------|
| [YOLOv8: Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/) | YOLO 系列官方文档 |
| You Only Look Once: Unified, Real-Time Object Detection (Redmon et al., 2016) | YOLO 开创性论文 |

## 代码参考

| 来源 | 说明 |
|------|------|
| [Ultralytics YOLO 训练示例](https://docs.ultralytics.com/modes/train/) | 训练参数与数据格式参考 |
| [Ultralytics YOLO 预测示例](https://docs.ultralytics.com/modes/predict/) | 推理与后处理参考 |
| [Streamlit 官方文档](https://docs.streamlit.io/) | Web 界面开发参考 |

## 教程

| 来源 | 说明 |
|------|------|
| [Roboflow YOLO11 Tutorial](https://blog.roboflow.com/yolov11/) | YOLO11 训练教程 |
| [Ultralytics Quickstart](https://docs.ultralytics.com/quickstart/) | 快速上手指南 |
