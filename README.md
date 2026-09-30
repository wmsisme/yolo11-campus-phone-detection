# 校园手机使用检测系统

基于 **YOLO11** 的校园场景手机使用行为检测系统。上传校园场景图片，自动检测画面中**使用手机的人**和**手机**的位置，并生成检测统计报告（可导出 Markdown）。

> 🚀 **在线 Demo（点开即用，无需安装）**：<https://wmsisme.github.io/yolo11-campus-phone-detection/demo/>
> 纯前端推理 —— 模型在你的浏览器里跑（onnxruntime-web / WebAssembly），**图片不上传服务器、不需要 API Key**。
> 源码与说明见 [`docs/demo/`](docs/demo/README.md)。

## 应用场景

- **课堂管理**：检测学生上课是否使用手机
- **校园监控**：识别校园公共区域的手机使用行为
- **考试监考辅助**：发现考场中出现的手机

## 技术栈

| 组件 | 技术 |
|:-----|:-----|
| 目标检测 | YOLO11 (nano / small / medium 三尺度可切换) |
| 推理框架 | ultralytics + PyTorch（全部本地运行，无需 API key） |
| 前端界面 | Streamlit 三栏布局 |
| 图像处理 | OpenCV + Pillow |
| 数据集 | Roboflow：Smart School v5 + 多数据集重组（University-Outdoor / University ver 2 / ver 3 / School v1） |
| 实验追踪 | Markdown + metrics.json + 训练曲线图 |
| 语言 | Python 3.10+ |

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/wmsisme/yolo11-campus-phone-detection.git
cd yolo11-campus-phone-detection

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 Web Demo（仓库已内置训练好的权重，无需先训练）
streamlit run src/web/app.py
```

浏览器会打开 <http://localhost:8501>。上传一张校园场景图片 → 点击「开始检测」→ 右侧查看统计报告并可导出 Markdown。

> **只想看效果？** 用上面的[在线 Demo](https://wmsisme.github.io/yolo11-campus-phone-detection/demo/)（纯前端，零安装），
> 或本地起静态服务：`python -m http.server 8080 --directory docs`，然后打开 <http://127.0.0.1:8080/demo/>。

> **本仓库不包含数据集**（体积原因，见 `.gitignore`）。Web Demo 所需的训练权重、
> 实验配置与指标、训练曲线均已入库，因此**开箱即可运行推理**；
> 若要复现训练，请按下面「运行方式 / 准备数据集」下载数据集。

## 项目结构

```text
.
├── AGENTS.md                     # AI Agent 开发手册（Harness 规则）
├── ARCHITECTURE.md               # 系统架构图 + 数据流说明
├── README.md                     # 本文件
├── LICENSE                       # MIT（含 ultralytics AGPL-3.0 等第三方声明）
├── requirements.txt              # 依赖声明
├── yolo11n.pt / yolo11s.pt       # 预训练权重（复现训练用，推理无需）
├── src/
│   ├── data/dataset.py           # 数据集准备（→ YOLO 标准格式）
│   ├── models/
│   │   ├── train.py              # YOLO11 训练入口（argparse 驱动）
│   │   ├── detect.py             # 推理 / 绘图 / 报告生成
│   │   ├── export_onnx.py        # 导出 ONNX（前端推理用，int8 带自检）
│   │   ├── onnx_infer.py         # ONNX 推理的 Python 参考实现（与前端逐步骤对齐）
│   │   └── gen_demo_fixtures.py  # 生成静态 Demo 的示例图与自检基准
│   ├── web/app.py                # Streamlit Web Demo（三栏布局）
│   └── utils/
│       ├── helpers.py            # 路径 / 日志 / JSON / YAML / 计时器（含 imread_unicode）
│       └── metrics.py            # mAP 提取 + 训练曲线 + 实验对比图
├── tests/                        # 冒烟测试 + ONNX 一致性 + 静态 Demo 端到端
├── experiments/                  # 每个实验一个目录（config + metrics + 曲线 + best.pt）
│   ├── exp0_yolo11n_baseline/
│   ├── exp1_yolo11s_augment/
│   ├── exp2_yolo11m_hyperparams/
│   ├── exp3_reorganized_yolo11n/     ← 静态 Demo 用的模型（mAP@50 0.729）
│   ├── exp4_reorganized_yolo11s/     ← 推荐模型（mAP@50 0.726）
│   └── exp5_reorganized_yolo11m/
├── docs/
│   ├── demo/                     # 纯前端静态 Demo（GitHub Pages 发布源）
│   │   ├── index.html / app.js / style.css
│   │   ├── model/phone-yolo11n.onnx   # 10.1 MB，浏览器直接下载运行
│   │   ├── samples/              # 示例图（Roboflow Smart School v5，CC BY 4.0）
│   │   ├── selftest.json         # 自检基准（由 Python 参考实现生成）
│   │   └── vendor/ort/           # onnxruntime-web 本地副本（不依赖 CDN）
│   ├── product-specs/index.md    # 功能列表与验收标准
│   ├── exec-plans/               # 当前计划 + 技术债追踪
│   ├── QUALITY_SCORE.md          # 质量评分卡
│   ├── RELIABILITY.md            # 依赖与可复现说明
│   ├── SECURITY.md               # 安全与数据边界
│   └── references/               # 数据集 / 论文 / 代码来源
└── reports/final-report.md       # 最终报告（16 部分）
```

## 运行方式

### 1. 准备数据集（仅复现训练时需要）

本仓库不含数据集。需自行从 Roboflow Universe 下载以下数据集，解压到 `data_source/`：

| 数据集 | 用途 |
|:-------|:-----|
| `smart school.v5i.yolov11` | 原始小数据集（141 张） |
| `University-Outdoor.v1i.yolov11`、`University ver 2.v3i.yolov11`、`University ver 3.v1i.yolov11`、`school.v1i.yolov11` | 重组数据集来源（合并后 22,879 张） |

```bash
# 原始 141 张数据集 → data/phone_detection/（自动生成 data.yaml）
python -m src.data.dataset
```

重组数据集 `data_source/reorganized_phone_dataset_yolo/reorganized_dataset/` 需先自行合并四份数据（类别统一为 `People using cellphone` / `cellphone`），
`data.yaml` 由训练脚本自动生成（`path` 字段按当前机器自动锚定）。

### 2. 训练模型

```bash
# 原始小数据集（mAP@50 约 0.25~0.30）
python -m src.models.train --model yolo11n --epochs 100 --batch 16 --exp_name exp0_yolo11n_baseline
python -m src.models.train --model yolo11s --epochs 100 --batch 16 --exp_name exp1_yolo11s_augment
python -m src.models.train --model yolo11m --epochs 150 --batch 8  --exp_name exp2_yolo11m_hyperparams

# 重组数据集（mAP@50 约 0.73，含 mAP 早停）
python -m src.models.train --model yolo11n --dataset reorganized --epochs 100 --batch 16 \
    --exp_name exp3_reorganized_yolo11n --early_stop_map 0.90
python -m src.models.train --model yolo11s --dataset reorganized --epochs 100 --batch 16 \
    --exp_name exp4_reorganized_yolo11s
```

训练结果自动落盘到 `experiments/<exp_name>/`（`config.yaml` + `metrics.json` + `best.pt` + `training_curves.png`）。

常用参数：`--device 0|cpu`、`--imgsz 640`、`--lr 0.01`、`--resume`（断点续训）、
`--dataset smart_school|reorganized`、`--early_stop_map <阈值>`。

### 3. 启动 Web Demo

```bash
streamlit run src/web/app.py
```

界面按「模型变体名」自动到 `experiments/` 里查找对应训练权重（`yolo11n` → exp3、
`yolo11s` → exp4）；若找不到训练权重会**明确提示**已退回通用 COCO 预训练权重，
并声明该权重不具备手机检测能力，避免把 person/bicycle 误读成手机。

### 4. 静态 Demo（浏览器内推理，无需 Python）

`docs/demo/` 是一个**纯前端**版本：模型导出为 ONNX，由 onnxruntime-web 在访客浏览器里执行。

```bash
# 起一个静态服务（浏览器限制 file:// 下的 fetch，不能直接双击 html）
python -m http.server 8080 --directory docs
# 打开 http://127.0.0.1:8080/demo/
```

导出模型与生成自检基准：

```bash
python -m src.models.export_onnx --weights experiments/exp3_reorganized_yolo11n/best.pt \
    --out docs/demo/model/phone-yolo11n.onnx
python -m src.models.gen_demo_fixtures
```

细节（口径对齐、自检机制、为什么不发 int8）见 [`docs/demo/README.md`](docs/demo/README.md)。

### 5. 运行测试

```bash
python -m pytest tests/ -v
```

其中两组是本次新增的**真实验证**（不是文件存在性检查）：

| 测试 | 验证什么 |
|:--|:--|
| `tests/test_onnx_parity.py` | 导出的 ONNX 与 PyTorch **逐框一致**（同一批验证集图片，IoU ≥ 0.9） |
| `tests/test_webdemo.py` | 起本地服务 + 无头 Edge 真跑静态 Demo，断言浏览器内推理结果与 Python 基准一致、零 JS 报错 |

## 实验结果

### 原始 Smart School v5 数据集（141 张，仅 phone 类）

| 实验 | 模型 | Epochs | mAP@50 | mAP@50-95 | Precision | Recall |
|:-----|:-----|:-------|:-------|:----------|:----------|:-------|
| exp0 | YOLO11n | 100 | 0.258 | 0.133 | 0.482 | 0.269 |
| exp1 | YOLO11s | 100 | 0.293 | 0.162 | 0.570 | 0.261 |
| exp2 | YOLO11m | 150 | 0.252 | 0.128 | 0.502 | 0.255 |

> 三轮训练 mAP@50 均不足 0.30，远低于 0.70 的质量目标 → 定位核心瓶颈为**样本绝对数量不足**。

### 重组数据集（合并 4 个 Roboflow 数据集，22,879 张）

| 实验 | 模型 | Epochs | mAP@50 | mAP@50-95 | Precision | Recall |
|:-----|:-----|:-------|:-------|:----------|:----------|:-------|
| exp3 | YOLO11n | 61（早停） | **0.729** | 0.503 | 0.717 | 0.862 |
| exp4 | YOLO11s | 100 | **0.726** | **0.507** | 0.722 | **0.883** |
| exp5 | YOLO11m | 1（中断） | 0.549 | 0.301 | 0.621 | 0.653 |

**结论**：扩充数据后 mAP@50 从 0.30 → 0.73（+143%），**数据量而非模型尺度才是核心瓶颈**；
yolo11s 在精度与速度间最均衡，为推荐方案。exp5 因训练中断仅完成 1 轮，其权重无参考价值。

### 质量目标达成情况

| 指标 | 目标 | 实测（exp4） | 达成 |
|:-----|:-----|:-------------|:-----|
| mAP@50 | ≥ 0.70 | 0.726 | ✅ |
| mAP@50-95 | ≥ 0.45 | 0.507 | ✅ |
| Precision | ≥ 0.75 | 0.722 | ❌ 差 0.028 |
| Recall | ≥ 0.65 | 0.883 | ✅ |
| 冒烟测试 | 5+/5 通过 | 10 passed, 1 skipped | ✅ |

## 主要功能

| 功能 | 描述 |
|:-----|:-----|
| 图片上传检测 | 上传 JPG/PNG 校园场景图片，自动检测手机和使用者 |
| 多模型切换 | 在 YOLO11n/s/m 之间切换，对比检测效果 |
| 置信度调节 | 滑块调整检测阈值 (0.05 – 0.95) |
| 统计报告 | 自动生成类别计数、置信度分布柱状图 |
| 检测明细 | 每个检测框的类别、置信度、位置、面积 |
| 报告导出 | 一键导出 Markdown 格式检测报告 |

## 许可与数据来源

本项目原创代码以 **MIT License** 授权，详见 [LICENSE](LICENSE)（含第三方组件声明）。

- 数据集来自 [Roboflow Universe](https://universe.roboflow.com/)（CC BY 4.0），**不随本仓库分发**，详见 `docs/references/reference-list.md`。
- 代码基于 [ultralytics](https://github.com/ultralytics/ultralytics)（**AGPL-3.0**）构建——若将本项目作为网络服务对外提供，需按 AGPL-3.0 提供完整源码，或向 Ultralytics 申请商业授权。
- 本项目的使用须遵守 `docs/SECURITY.md` 中的数据边界：**禁止上传包含学生面部可识别信息、身份证件等隐私图片**。
