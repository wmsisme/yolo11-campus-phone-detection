# 可靠性说明

## 环境依赖

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | 推荐 3.10 或 3.11 |
| ultralytics | 8.3.0+ | YOLO11 训练与推理 |
| torch | 2.0.0+ | PyTorch 后端 |
| streamlit | 1.28.0+ | Web 界面 |
| opencv-python | 4.8.0+ | 图像处理与框绘制 |
| pillow | 10.0.0+ | 图片读写 |
| matplotlib | 3.7.0+ | 训练曲线绘制 |
| pytest | 7.0.0+ | 冒烟测试 |

## 最低可运行要求

### 硬件

- **CPU 训练**：可行但较慢（100 epochs 约 2-4 小时），建议 `--device cpu`
- **GPU 训练**：推荐 NVIDIA GPU (≥4GB VRAM) 或 Apple M1/M2/M3
- **推理**：CPU 即可流畅运行，单图 < 3 秒

### 网络

- 首次训练需从 ultralytics 自动下载 YOLO11 预训练权重 (~5-10MB)
- 如遇网络问题，可手动下载 .pt 文件放到项目根目录

## 样例输入输出

### 输入

```bash
python -m src.models.train --model yolo11n --epochs 100 --batch 16 --exp_name exp0_yolo11n_baseline
```

### 输出

训练完成后，`experiments/exp0_yolo11n_baseline/` 包含：
```
exp0_yolo11n_baseline/
├── config.yaml         # 实验配置
├── metrics.json        # {"mAP50": 0.xxx, "mAP50-95": 0.xxx, "precision": 0.xxx, "recall": 0.xxx}
├── best.pt             # 最佳模型权重
└── training_curves.png # 训练曲线图 (loss, mAP, Precision, Recall)
```

## 失败处理

| 失败场景 | 处理方式 |
|---------|---------|
| 模型下载失败 | 手动下载 .pt 文件，放到项目根目录；或设置 HF_ENDPOINT 镜像 |
| 训练 OOM | 减小 `--batch`（如 16 → 8 → 4），减小 `--imgsz`（如 640 → 320） |
| 数据集未准备 | 先运行 `python -m src.data.dataset` |
| 无 GPU | 使用 `--device cpu` |
| Web Demo 模型加载失败 | 自动回退到 ultralytics 预训练权重 |

## 已知限制

1. ~~重组数据集（22,879 张）~~ **该数据集已因标注污染（人脸被标成 cellphone）整体弃用**；当前生产集为 `phone_usage`（约 1.7 万张，COCO 派生）。历史说明：该集原用于缓解 141 张小样本的泛化瓶颈，但相当比例来自通用"户外/自拍"场景，与真实课堂环境的分布仍有差异
2. 数据集来自 Roboflow，场景可能偏向特定校园环境
3. 仅检测"手持手机/贴耳手机"两类使用状态，不检测其他电子设备，也**不输出"人"**
4. 模型在光线暗、遮挡严重场景下精度可能下降
5. exp5 (yolo11m) 训练中断，仅完成 1 个 epoch，其权重与指标不具备参考价值
6. 仓库不携带数据集与训练产物（`data_source/`、`data/`、`runs/`），复现训练需自行下载数据集
7. 本机实测推理 0.78~0.85s/图（RTX 4070 Laptop，640×640，含前后处理），未达 0.5s/图 目标——该耗时含图片解码与报告生成，未单独剥离模型前向时间
