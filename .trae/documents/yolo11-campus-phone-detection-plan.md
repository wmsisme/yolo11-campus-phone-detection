# 校园手机使用检测系统 — 基于 YOLO11 的重建计划

## 一、项目概述

**项目名称**：基于 YOLO11 的校园场景手机使用检测系统\
**选题方向**：方向 B（目标检测类项目）+ 方向 E（智能视觉应用系统）\
**难度等级**：Level 3（竞赛水准）\
**应用场景**：校园监控 / 课堂管理 — 自动检测画面中是否有人在用手机，以及手机的位置\
**输入**：校园场景图片（单张 / 批量 / 摄像头帧）\
**输出**：检测框（类别：`cellphone`、`People using cellphone`）+ 检测报告

**与旧项目的区别**：

| <br />    | 旧项目 (CLIP/BLIP-2)      | 新项目 (YOLO11)         |
| --------- | ---------------------- | -------------------- |
| 任务        | 图文检索 + VQA             | 目标检测                 |
| 模型        | CLIP ViT-B/32 + BLIP-2 | YOLO11 (n/s/m 多尺度实验) |
| 数据集匹配度    | 数据是 YOLO 标注，任务与数据不匹配   | 数据天生是 YOLO 格式，完全匹配   |
| Recall\@5 | 0.0065                 | 目标 mAP\@50 ≥ 0.70    |
| 可训练性      | 微调未真正运行                | 完整训练+验证+测试流程         |
| 实验对比      | 仅 baseline             | 3+ 组实验（不同模型大小 / 超参）  |

***

## 二、新项目目录结构

```
d:\本科\作业\视觉ai\期末作业\
├── AGENTS.md                          # 重写：AI Agent 工作规则（YOLO11 版）
├── ARCHITECTURE.md                    # 重写：新系统架构
├── README.md                          # 重写：安装与运行说明
├── requirements.txt                   # 重写：ultralytics, streamlit, torch 等
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                 # 数据集加载、按 YOLO 格式组织、划分
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                   # YOLO11 训练入口（命令行参数驱动）
│   │   └── detect.py                  # 推理/检测函数封装
│   ├── web/
│   │   ├── __init__.py
│   │   └── app.py                     # Streamlit Web Demo（上传+实时检测+统计）
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py                 # 项目路径、日志、配置保存
│       └── metrics.py                 # mAP/Precision/Recall 计算与可视化
│
├── tests/
│   ├── __init__.py
│   └── test_smoke.py                  # 冒烟测试（模型加载、单图推理、Web 导入）
│
├── data/
│   └── phone_detection/               # 处理后的 YOLO 格式数据集
│       ├── data.yaml                  # YOLO 数据集配置文件
│       ├── train/images/              # 训练集图片（软链接或复制）
│       ├── train/labels/              # 训练集标注 .txt
│       ├── val/images/                # 验证集图片
│       ├── val/labels/                # 验证集标注
│       ├── test/images/               # 测试集图片
│       └── test/labels/               # 测试集标注
│
├── data_source/                       # 保留原始数据集不动
│
├── experiments/
│   ├── exp0_yolo11n_baseline/         # 实验 0：YOLO11n 基线（无额外增强）
│   ├── exp1_yolo11s_augment/          # 实验 1：YOLO11s + 数据增强
│   └── exp2_yolo11m_hyperparams/      # 实验 2：YOLO11m + 调参
│
├── docs/                              # Harness 文档（内容重写）
│   ├── product-specs/
│   │   └── index.md                   # 功能列表与验收标准（目标检测版）
│   ├── exec-plans/
│   │   ├── active/
│   │   │   └── current-plan.md        # 重写：当前进度追踪
│   │   └── tech-debt-tracker.md       # 重写：技术债记录
│   ├── QUALITY_SCORE.md               # 重写：质量评分卡
│   ├── RELIABILITY.md                 # 重写：可靠性说明
│   ├── SECURITY.md                    # 重写：安全与边界
│   └── references/
│       └── reference-list.md          # 重写：参考来源
│
├── reports/
│   └── final-report.md                # 重写：最终报告（16 部分完整）
│
└── runs/                              # YOLO 自动生成的训练输出目录
```

***

## 三、实施步骤（共 10 步）

### Step 0：备份旧项目 + 初始化新结构

**操作**：

1. 将旧的 `src/`、`tests/`、`scripts/` 移动到 `_archive/`（备份，不删除）
2. 清空 `data/images/` 和 `data/annotations.json`
3. 创建新的 `src/` 子目录结构（data, models, web, utils）
4. 创建 `data/phone_detection/` 目录树
5. **重写** **`requirements.txt`**：

   * `ultralytics>=8.3.0`（YOLO11）

   * `torch>=2.0.0`

   * `streamlit>=1.28.0`

   * `opencv-python>=4.8.0`

   * `pillow>=10.0.0`

   * `matplotlib>=3.7.0`

   * `pytest`

### Step 1：数据准备脚本

**文件**：`src/data/dataset.py`

**功能**：

1. 扫描 `data_source/smart school.v5i.yolov11/` 下的 train/valid/test 三部分
2. 按 YOLO 标准格式复制/软链接到 `data/phone_detection/train|val|test/{images,labels}/`
3. 生成 `data/phone_detection/data.yaml`：

   ```yaml
   path: <绝对路径>/data/phone_detection
   train: train/images
   val: val/images
   test: test/images
   nc: 2
   names: ["People using cellphone", "cellphone"]
   ```
4. 输出数据集统计：各类别数量、图像尺寸分布

**数据增强策略（训练时由 YOLO11 内置）**：

* Mosaic 拼接（4 图拼 1 图）

* 随机水平翻转

* HSV 色彩抖动

* 尺度缩放

* 这些在 YOLO11 训练参数中配置，不需要单独实现

### Step 2：训练脚本

**文件**：`src/models/train.py`

**功能**：

* argparse 参数驱动：`--model`(yolo11n/s/m)、`--epochs`、`--batch`、`--imgsz`、`--exp_name`

* 调用 `ultralytics.YOLO` 训练

* 训练完成后：

  * 自动评估验证集，保存指标到 `experiments/{exp_name}/metrics.json`

  * 保存训练曲线图（loss、mAP）到 `experiments/{exp_name}/`

  * 记录配置到 `experiments/{exp_name}/config.yaml`

**使用示例**：

```bash
python -m src.models.train --model yolo11n --epochs 100 --batch 16 --imgsz 640 --exp_name exp0_yolo11n_baseline
python -m src.models.train --model yolo11s --epochs 100 --batch 16 --imgsz 640 --exp_name exp1_yolo11s_augment
python -m src.models.train --model yolo11m --epochs 150 --batch 8 --imgsz 640 --exp_name exp2_yolo11m_hyperparams
```

**三组实验设计**：

| 实验   | 模型               | Epochs | 增强        | 预期 mAP\@50  | 目的   |
| ---- | ---------------- | ------ | --------- | ----------- | ---- |
| exp0 | YOLO11n (nano)   | 100    | 默认增强      | \~0.55-0.65 | 轻量基线 |
| exp1 | YOLO11s (small)  | 100    | 默认增强      | \~0.65-0.75 | 精度提升 |
| exp2 | YOLO11m (medium) | 150    | 默认增强 + 调参 | \~0.70-0.80 | 最优模型 |

### Step 3：推理/检测模块

**文件**：`src/models/detect.py`

**功能**：

* `load_model(model_path)` — 加载训练好的 YOLO11 模型

* `detect_image(model, image_path)` — 单张图片检测，返回检测结果列表

* `detect_batch(model, image_dir)` — 批量检测

* `draw_boxes(image, results)` — 在图片上绘制检测框

* `generate_report(results)` — 生成检测统计报告（JSON/Markdown）

### Step 4：工具函数

**文件**：

* `src/utils/helpers.py` — 项目根路径、日志、YAML/JSON 存取、计时器

* `src/utils/metrics.py` — 从 YOLO 验证结果中提取 mAP\@50、mAP\@50-95、Precision、Recall，绘制对比图

### Step 5：Streamlit Web Demo

**文件**：`src/web/app.py`

**页面设计**（单页三栏）：

```
┌──────────────┬──────────────────────┬──────────────────┐
│  左侧面板     │     中间主区域         │   右侧面板         │
│              │                      │                  │
│ 📁 上传图片   │  检测结果画布           │ 📊 检测统计        │
│ (拖拽或浏览)  │  (带检测框的图片)       │ - 检测到手机: N个  │
│              │                      │ - 使用手机的人: M个│
│ 🎛️ 模型选择  │                      │ - 置信度分布       │
│ - Nano      │                      │ - 每类计数柱状图   │
│ - Small     │                      │                  │
│ - Medium    │                      │ 📋 检测明细        │
│              │                      │ (每个框的类别+置信度)│
│ 🔍 置信度阈值 │                      │                  │
│              │                      │ 💾 导出报告        │
│ 🚀 开始检测  │                      │                  │
│              │                      │                  │
│ 📸 示例图片  │                      │                  │
└──────────────┴──────────────────────┴──────────────────┘
```

**核心交互**：

1. 用户上传图片 / 从示例选择 → 点击"开始检测"
2. 中间显示带检测框的图片（不同颜色：红色=使用手机的人，蓝色=手机）
3. 右侧显示统计面板（计数、置信度、明细表格）
4. 支持导出检测报告为 Markdown

**Streamlit session\_state 管理**：模型懒加载，首次使用时加载，切换模型时重新加载。

### Step 6：冒烟测试

**文件**：`tests/test_smoke.py`

**测试用例**：

1. `test_import_modules` — 所有模块能正常导入
2. `test_dataset_structure` — 数据集目录结构正确、data.yaml 存在
3. `test_model_load` — YOLO11n 能成功加载（不需要训练好的权重，直接用预训练）
4. `test_single_inference` — 单张图片推理不报错，返回有效结果
5. `test_web_import` — Streamlit app 模块能导入

### Step 7：重写 Harness 文档

**需要重写的文件**（按优先级）：

| 文件                                       | 关键变化                               |
| ---------------------------------------- | ---------------------------------- |
| `AGENTS.md`                              | 技术路线从 CLIP→YOLO11，任务从检索→检测，开发步骤重写  |
| `ARCHITECTURE.md`                        | 架构图改为 YOLO11 + Streamlit，数据流改为检测流程 |
| `README.md`                              | 安装、运行、示例更改为 YOLO11 风格              |
| `docs/product-specs/index.md`            | 功能改为上传检测、统计报告、模型切换                 |
| `docs/exec-plans/active/current-plan.md` | 重写为 Step 0-10 进度追踪                 |
| `docs/exec-plans/tech-debt-tracker.md`   | 清空旧记录，新记录真实错误                      |
| `docs/QUALITY_SCORE.md`                  | 指标改为 mAP\@50、Precision、Recall、推理速度 |
| `docs/RELIABILITY.md`                    | 依赖改为 ultralytics，硬件要求更新            |
| `docs/SECURITY.md`                       | 更新数据来源声明                           |
| `docs/references/reference-list.md`      | 更新为 YOLO11、smart school 数据集等       |

### Step 8：运行实验并记录

**操作顺序**：

1. 运行 `python -m src.models.train --exp_name exp0_yolo11n_baseline ...`
2. 记录指标 → 更新 `current-plan.md`
3. 运行 `python -m src.models.train --exp_name exp1_yolo11s_augment ...`
4. 记录指标 → 对比表格
5. 运行 `python -m src.models.train --exp_name exp2_yolo11m_hyperparams ...`
6. 记录指标 → 三组完整对比
7. 将最优模型权重复制到 `src/models/best.pt`

### Step 9：错误收集与分析

**运行过程中必须记录**（写入 `tech-debt-tracker.md`）：

* 训练中的问题（如 OOM、收敛慢、某个类别 AP 低）

* 误检/漏检典型案例（截图+分析）

* 依赖版本问题

* Web Demo 中的边界情况

目标：至少 5 条真实错误记录，每条包含现象、原因、修复、状态。

### Step 10：撰写最终报告

**文件**：`reports/final-report.md`

按作业要求的 16 个部分完整填写：

1. 项目基本信息
2. 项目背景与应用场景
3. 任务类型
4. 输入输出说明
5. 模型与技术路线
6. 系统架构
7. Agent Harness 设计
8. AI Agent 使用过程
9. 项目实现过程
10. 测试与验证
11. 结果展示与分析
12. 错误分析与经验积累
13. Harness 作用反思
14. 项目局限与改进
15. 总结
16. 可选：演示视频

***

## 四、Harness 设计要点（满足 25 分评分标准）

| Harness 要素 | 实现方式                                                           |
| ---------- | -------------------------------------------------------------- |
| 文件结构       | AGENTS.md + ARCHITECTURE.md + README.md + docs/\* + reports/\* |
| Agent 规则   | AGENTS.md 定义：开发前读 current-plan.md，每次修改后更新进度，禁止伪造指标             |
| 上下文管理      | current-plan.md 持续更新已完成/进行中/下一步，给下个 Agent 的提醒                  |
| 工具边界       | SECURITY.md 声明：不上传隐私图片、不提交 API key、不伪造结果、所有代码本地运行              |
| 实验记录       | 每个实验一个子目录：config.yaml + metrics.json + 训练曲线图                   |
| 测试验证       | tests/test\_smoke.py 至少 5 个测试用例                                |
| 错误积累       | tech-debt-tracker.md 记录 ≥ 5 条真实错误                              |
| 交接机制       | 每个 Step 完成后更新 current-plan.md，注明给下个 Agent 的提醒                  |

***

## 五、质量目标

| 指标           | 目标值            | 说明                    |
| ------------ | -------------- | --------------------- |
| mAP\@50      | ≥ 0.70         | 在验证集上                 |
| mAP\@50-95   | ≥ 0.45         | 多 IoU 阈值平均            |
| Precision    | ≥ 0.75         | 减少误检                  |
| Recall       | ≥ 0.65         | 减少漏检                  |
| 推理时间         | < 0.5s/图 (GPU) | 或 < 3s/图 (CPU)        |
| 冒烟测试通过       | 5/5 pass       | pytest                |
| Harness 文档齐全 | 11/11 文件       | 内容与项目一致               |
| 错误记录 ≥ 5 条   | ≥ 5            | 含图/原因/修复              |
| 实验组数 ≥ 3     | ≥ 3            | 每组有 metrics.json + 曲线 |

***

## 六、技术选型依据

| 选项                   | 理由                            |
| -------------------- | ----------------------------- |
| YOLO11（非 YOLOv8）     | 最新版，精度更高，训练更稳定，官方文档完善         |
| nano/small/medium 三级 | 形成对比实验，展示模型选择理由               |
| Streamlit（非 Gradio）  | 更灵活布局，可做三栏设计，统计图表丰富           |
| ultralytics 框架       | 一站式训练/验证/导出，减少代码量，聚焦核心任务      |
| 不用 BLIP-2/VQA        | 旧项目的 VQA 依赖大模型且效果未知，新项目纯检测更可控 |

***

## 七、确认点

* [x] 方案：YOLO11 重建

* [x] 数据：仅用 smart school + YOLO11 内置增强

* [ ] 确认上述 10 步计划和文件结构是否 OK

* [ ] 确认质量目标是否合理

