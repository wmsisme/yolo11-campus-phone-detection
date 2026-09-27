# 参考来源

## 数据集

| 名称 | 来源 | 说明 |
|------|------|------|
| Smart School v5 | [Roboflow Universe](https://universe.roboflow.com/) | 校园场景手机使用检测数据集，141 张图片，标注类别：People using cellphone, cellphone（CC BY 4.0） |
| University-Outdoor v1 | [Roboflow Universe](https://universe.roboflow.com/) | 重组数据集来源之一 |
| University ver 2 / ver 3 | [Roboflow Universe](https://universe.roboflow.com/) | 重组数据集来源之一 |
| School v1 | [Roboflow Universe](https://universe.roboflow.com/) | 重组数据集来源之一 |

> 上述 4 个数据集经类别映射合并为 `reorganized_phone_dataset_yolo`（train 18,800 / val 1,730 / test 2,349，共 22,879 张），统一为 2 类。
> **所有数据集均不随本仓库分发**，需自行从 Roboflow 下载。

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
