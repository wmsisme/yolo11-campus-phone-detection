# 项目开发指导：基于 YOLO11 的校园手机使用检测系统

> 本文件是专为 AI Agent 编写的开发手册。\
> **你的任务**：在遵守以下规则的前提下，协助开发者完成一个 Level 3 智能视觉项目。\
> **核心原则**：先读规则，再行动；每次修改后更新进度；绝不伪造结果。

---

## 1. 项目目标

开发一个 **Web 应用**，允许用户：

- **上传校园场景图片**，自动检测画面中是否有人在**使用手机**，以及**手机的位置**。
- **切换不同模型**（YOLO11n/s/m）对比检测效果。
- **查看检测报告**（检测框数量、类别分布、置信度统计）。
- **导出检测报告**为 Markdown 格式。

项目需包含**完整模型训练**、**多组实验对比**、**错误分析**和**完整的 Harness 文档**。

---

## 2. 技术路线（强制）

| 模块 | 技术选型 | 说明 |
|:-----|:---------|:-----|
| 目标检测 | YOLO11 (n/s/m 三尺度) | 使用 smart school 数据集训练，含数据增强 |
| 数据来源 | Roboflow Smart School v5 | 校园场景手机使用检测标注数据集 (141 张) |
| 前端界面 | Streamlit | 三栏布局：上传/控制 → 检测画布 → 统计报告 |
| 后端推理 | Python + PyTorch + ultralytics | 所有模型本地运行 |
| 实验追踪 | 手动记录 Markdown + metrics.json + 训练曲线图 | 每个实验的配置、指标、可视化 |

---

## 3. 强制 Harness 文件结构

你**必须**维护以下文件。任何修改必须同步更新对应文件。

```text
.
├── AGENTS.md                          # 你当前正在读的文件
├── ARCHITECTURE.md                    # 系统架构图 + 数据流说明
├── README.md                          # 人类可读的安装与运行说明
├── src/
│   ├── data/
│   │   └── dataset.py                 # 数据集准备 (smart school → YOLO 格式)
│   ├── models/
│   │   ├── train.py                   # YOLO11 训练入口
│   │   └── detect.py                  # 推理/检测/报告模块
│   ├── web/
│   │   └── app.py                     # Streamlit Web Demo
│   └── utils/
│       ├── helpers.py                 # 路径/日志/配置工具
│       └── metrics.py                 # mAP 提取与可视化
├── tests/
│   └── test_smoke.py                  # 冒烟测试 (6 个测试类)
├── docs/
│   ├── product-specs/
│   │   └── index.md                   # 功能列表与验收标准
│   ├── exec-plans/
│   │   ├── active/
│   │   │   └── current-plan.md        # 当前进度
│   │   └── tech-debt-tracker.md       # 技术债与已知错误
│   ├── QUALITY_SCORE.md               # 质量评分卡
│   ├── RELIABILITY.md                 # 依赖与可复现说明
│   ├── SECURITY.md                    # 安全与数据边界
│   └── references/
│       └── reference-list.md          # 数据集、论文、代码来源
├── experiments/                       # 每个实验一个子目录
│   ├── exp0_yolo11n_baseline/
│   ├── exp1_yolo11s_augment/
│   └── exp2_yolo11m_hyperparams/
└── reports/
    └── final-report.md                # 最终报告（16 部分）
```

---

## 4. AI Agent 工作规则（必须遵守）

### 4.1 上下文管理

- **开始任务前**：读取 `docs/exec-plans/active/current-plan.md`，了解已完成和下一步。
- **每次对话/提交**：只负责一个明确子任务。完成后更新 `current-plan.md` 的状态。
- **避免长上下文丢失**：如果任务跨越多个文件，先生成概要计划再逐个实现。

### 4.2 工具边界（来自 `SECURITY.md`）

- ✅ 允许：创建/修改 `src/` 下的代码、`tests/` 测试、`experiments/` 记录。
- ❌ 禁止：直接修改 `docs/product-specs/` 或 `docs/QUALITY_SCORE.md` 中的验收标准（除非开发者明确要求）。
- ❌ 禁止：删除任何原始数据文件或未备份的代码。
- ❌ 禁止：提交 API key、隐私图片或伪造实验指标。

### 4.3 实验与验证机制

- 任何涉及模型训练的操作，**自动记录配置**（保存 config.yaml 到实验目录）。
- 每次修改后，运行 `python -m pytest tests/ -v` 确保基本功能不崩溃。
- 对于目标检测模型，必须输出**定量指标**（mAP@50, mAP@50-95, Precision, Recall）并记录到 `experiments/<exp_name>/metrics.json`。
- 对于误检/漏检案例，保存截图并写入 `tech-debt-tracker.md`。

### 4.4 错误经验积累

当遇到以下情况时，**必须**记录到 `docs/exec-plans/tech-debt-tracker.md`：

- 模型检测结果明显不合理（如把书本识别为手机）。
- 代码报错（由你或开发者引入）。
- 依赖库版本冲突。
- 训练过程中 OOM 或收敛异常。
- AI 生成代码后测试失败。

记录格式：

```markdown
### 错误 #N - YYYY-MM-DD
- **现象**：描述具体错误
- **原因**：根因分析
- **建议修复**：具体方案
- **状态**：待处理 / 已修复 / 已知限制
```

---

## 5. 开发步骤（按顺序执行）

### Step 0：环境搭建

- 安装依赖：`pip install -r requirements.txt`
- 核心依赖：`ultralytics`, `torch`, `streamlit`, `opencv-python`, `pillow`, `matplotlib`, `pytest`

### Step 1：数据准备

- 运行 `python -m src.data.dataset` 将 smart school 数据集组织为 YOLO 标准格式
- 验证 `data/phone_detection/data.yaml` 正确

### Step 2：基线训练 (exp0)

- 运行：`python -m src.models.train --model yolo11n --epochs 100 --batch 16 --exp_name exp0_yolo11n_baseline`
- 记录 mAP@50 到实验目录

### Step 3：对比实验 (exp1 + exp2)

- exp1：`python -m src.models.train --model yolo11s --epochs 100 --batch 16 --exp_name exp1_yolo11s_augment`
- exp2：`python -m src.models.train --model yolo11m --epochs 150 --batch 8 --exp_name exp2_yolo11m_hyperparams`

### Step 4：Web Demo

- 复制最优模型：`copy experiments\exp1_yolo11s_augment\best.pt src\models\best.pt`
- 启动：`streamlit run src/web/app.py`

### Step 5：测试与验证

- 运行冒烟测试：`python -m pytest tests/ -v`
- 在 Web Demo 上用测试集图片手动验证

### Step 6：最终报告

- 按作业 16 部分填写 `reports/final-report.md`
- 包含实验结果对比表、训练曲线、误检/漏检分析

---

## 6. 质量标准（来自 `QUALITY_SCORE.md`）

| 指标 | 目标值 | 验证方法 |
|:-----|:------|:--------|
| mAP@50 | ≥ 0.70 | 验证集计算 |
| mAP@50-95 | ≥ 0.45 | 验证集计算 |
| Precision | ≥ 0.75 | 验证集计算 |
| Recall | ≥ 0.65 | 验证集计算 |
| 推理时间 | < 0.5s/图 (GPU) / < 3s/图 (CPU) | 计时测试 |
| 冒烟测试通过 | 5+/5 pass | pytest |
| Harness 文档一致性 | 11 个必需文件齐全且与项目一致 | 人工检查 |
| 错误记录 | ≥ 5 条真实错误及分析 | 人工检查 |

---

## 7. 常见错误及应对策略

| 错误场景 | 正确应对方式 |
|:---------|:-----------|
| `ultralytics` 下载预训练权重失败 | 设置镜像或手动下载 .pt 文件到本地 |
| 训练 OOM | 减小 batch size、减小 imgsz、使用更小模型 |
| 某个类别 AP 极低 | 检查该类标注数量、标注质量、是否被其他类混淆 |
| Web Demo 模型加载慢 | 改用轻量模型 (yolo11n)、缓存到 session_state |
| 检测到很多误检 | 提高置信度阈值、检查训练数据是否有相似物体 |

---

## 8. 最终交付物检查清单

- [x] 项目代码（`src/`, `tests/`, `requirements.txt`）可完整运行
- [x] 所有 Harness 文档齐全
- [ ] `experiments/` 下至少包含 3 组实验，每组有 `config.yaml` + `metrics.json` + `training_curves.png`
- [ ] 至少 5 条错误案例记录在 `tech-debt-tracker.md`
- [ ] `reports/final-report.md` 满足作业 16 个部分要求
- [ ] Web Demo 可正常运行并展示检测结果
- [ ] 冒烟测试全部通过
