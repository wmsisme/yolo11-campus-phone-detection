# 项目开发指导：基于 YOLO11 的手机使用检测系统

> 本文件是专为 AI Agent 编写的开发手册。
> **你的任务**：在遵守以下规则的前提下，协助开发者维护并改进一个**要实际上线使用的手机检测系统**。
> **核心原则**：先读规则，再行动；每次修改后更新进度；绝不伪造结果。

---

> ### ⚠️ 本项目首先是「**要上线的产品**」，作业只是它的一个侧面（2026-10-02 明确）
>
> 开发者原话：「**这不单单只是我的一个作业，而是一个用于上线的项目。项目的目的是要实现手机检测这个功能。**」
>
> 因此所有工作的**首要判据是「这个功能在真实使用中靠不靠得住」**，而不是"文档写了没、指标好不好看"：
>
> - **验收看真实场景表现**——曾出现"验证集 mAP@50=0.73 很好看，但上线后一用就把人脸判成手机"。
>   指标必须与用户能感知的效果一致（详见 `reports/final-report.md` 第 9/11 节）。
> - **交付物必须能真的跑起来**：线上 Demo <https://wmsisme.github.io/yolo11-campus-phone-detection/demo/> 是门面，
>   任何改动都要**回线复验**（资产可得、文案与模型口径一致、浏览器内推理结果与 Python 基准逐框一致）。
> - **对局限如实标注**——宁可在页面上写明"远距离小手机检不出"，也不要让人用错场景后才发现。
>
> ### 📌 当前实际口径（与下方"历史设计"不一致时以本段为准）
>
> | 项 | 当前事实 |
> |:--|:--|
> | 检测类别 | **`in_hand`（手持手机）/ `on_ear`（贴耳手机）**——「正在被使用的手机」；**不输出"人"这一类** |
> | 训练数据 | `phone_usage` 数据集（约 1.7 万张，本地保存不入库），**不是 Smart School v5** |
> | 生产权重 | `experiments/exp6_phone_usage_yolo11s/`（40 轮，干净数据） |
> | 部署阈值 | **conf = 0.50**（Precision 0.8251 / Recall 0.3520）；默认 0.25 时 P 0.6934 / R 0.3950 |
> | 线上形态 | 纯前端静态 Demo（`docs/demo/`，ONNX + onnxruntime-web）+ 本地 Streamlit（`src/web/app.py`） |
> | 已知限制 | 教室**远距离小手机**检不出（尺度/域差异；实测提高 imgsz 无效）；不圈出整个人 |
> | 变更原因 | 旧的重组数据集**标注污染**（人脸被标成 cellphone），已整体弃用——详见 `docs/exec-plans/tech-debt-tracker.md` #015/#016 |

---

## 1. 项目目标

开发一个**可上线的手机检测系统**，允许用户：

- **上传图片**，自动检测画面中**正在被使用的手机**（**手持 `in_hand` / 贴耳 `on_ear`**）。
  > 口径说明：数据集只标注了手机本体、**没有"人"的框**，故以「手机框」代表"此处有人在用手机"，
  > **不会圈出整个人**。若要"框住人"，需另找带人物标注的数据重训。
- **切换不同模型**（YOLO11n/s/m）对比检测效果。
- **查看检测报告**（检测框数量、类别分布、置信度统计）。
- **导出检测报告**为 Markdown 格式。
- **纯前端在线体验**：模型在浏览器内推理，**图片不上传服务器、无需 API Key**。

项目同时包含**完整模型训练**、**多组实验对比**、**错误分析**和**完整的 Harness 文档**，
并被作为课程项目的展示对象（`reports/final-report.md` 按 16 节组织）。

---

## 2. 技术路线

| 模块 | 技术选型 | 说明 |
|:-----|:---------|:-----|
| 目标检测 | YOLO11 (n/s/m 三尺度) | 当前生产权重为 **yolo11s**，在 `phone_usage` 数据集上训练，含数据增强 |
| 数据来源 | `phone_usage` 数据集（本地，不入库） | COCO 派生的手机使用场景标注（约 1.7 万张）；**旧的重组数据集已因标注污染弃用** |
| 前端界面 | Streamlit（本地） + 纯静态页（线上） | 三栏布局：上传/控制 → 检测画布 → 统计报告 |
| 后端推理 | Python + PyTorch + ultralytics（本地）／onnxruntime-web（浏览器） | 线上为**纯前端**，无需后端 |
| 实验追踪 | Markdown + metrics.json + 训练曲线图 | 每个实验的配置、指标、可视化 |

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

## 5. 当前生产链路（新会话照此上手）

> 与下方的"历史开发步骤"不同，**这一节是当前真实在用的链路**。

### Step A：环境

- `pip install -r requirements.txt`（核心：`ultralytics`, `torch`, `streamlit`, `opencv-python`, `pytest`, `onnxruntime`）

### Step B：本地跑起来（Streamlit）

- `streamlit run src/web/app.py` → <http://localhost:8501>
- 模型自动从 `experiments/` 扫描加载（当前会命中 `exp6_phone_usage_yolo11s`），
  **类别名以权重自带的 `names` 为准**（不要再硬编码——曾因此出现"模型框手机、界面标成人"）

### Step C：改模型 / 换权重后（**必做回线复验**）

```bash
# 1) 导出 ONNX（契约：imgsz 640 / opset 13 / 关闭内置 NMS）
python -m src.models.export_onnx --weights experiments/<exp>/best.pt \
    --out docs/demo/model/<name>.onnx
# 2) 重新生成示例图与自检基准（换口径时同时换 --preset）
python -m src.models.gen_demo_fixtures --preset phone-usage --model docs/demo/model/<name>.onnx
# 3) 同步前端类别与文案（app.js 的 CLASSES / MODELS / SAMPLE_FILES，index.html 的标签与说明）
# 4) 全量测试：确认关键用例是 PASSED 而不是 SKIPPED（写死旧模型名会静默 skip = 假通过）
python -m pytest tests/ -q -rs
# 5) 推送后回线复验：页面/JS/模型/示例图逐个 HTTP 200，且页面文案与新口径一致
```

### Step D：重新训练

```bash
python -m src.models.train --model yolo11s --dataset phone_usage \
    --epochs <N> --batch 4 --workers 2 --imgsz 640 --device 0 --exp_name <name>
```

> ⚠️ **两个已踩过的坑**：① 本机只有 15.6GB 内存，`batch 4 / workers 2` 是实测稳定值，
> 放大 worker 数会打爆页面文件（WinError 1455）并留下僵尸进程；
> ② **checkpoint 被 `strip_optimizer` 剥离后无法 `--resume`**（`epoch=-1`），
> 要支持续训须在训练中/收尾前另存未剥离的 ckpt。

### Step E：展示材料

- `reports/final-report.md` 按 16 节维护，**必须与当前口径和真实结论一致**（旧指标要标注作废，不能留着误导）

---

## 5b. 历史开发步骤（项目初期，保留作过程记录）

### Step 0：环境搭建

- 安装依赖：`pip install -r requirements.txt`
- 核心依赖：`ultralytics`, `torch`, `streamlit`, `opencv-python`, `pillow`, `matplotlib`, `pytest`

### Step 1：数据准备

- 运行 `python -m src.data.dataset` 将数据集组织为 YOLO 标准格式
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

- 运行测试：`python -m pytest tests/ -v`
- 在 Web Demo 上用真实图片手动验证

### Step 6：展示材料

- 按 16 部分填写 `reports/final-report.md`
- 包含实验结果对比表、训练曲线、误检/漏检分析

---

## 6. 质量标准

> **首要判据是"上线能不能用"**，而不是"指标好不好看"。历史教训：验证集 mAP@50 = 0.73 很漂亮，
> 但上线后**把人脸判成手机**（真实负样本误报 67.7%）——指标必须与用户能感知的效果一致。

| 指标 | 目标值 | 当前实测 | 说明 |
|:-----|:------|:--------|:-----|
| **真实负样本误报率**（conf≥0.5） | **< 5%** | **1.3%** ✅ | ← **最贴近用户痛点的指标**；旧模型为 67.7% |
| Precision（部署阈值 conf=0.50） | ≥ 0.80 | **0.8251** ✅ | 部署口径 |
| Precision（默认 conf=0.25） | ≥ 0.75 | 0.6934 ⚠️ | 保守口径下的权衡值（R 更高） |
| Recall（conf=0.50） | 记录并如实报告 | 0.3520 | 低召回是**有意的保守取舍**，已公开写明 |
| mAP@50 / mAP@50-95 | 记录并如实报告 | 0.3646 / 0.1743 | **不作为首要达标线**（易被标注质量扭曲） |
| 推理时间（浏览器，单张 640） | < 1s | 约 0.3s ✅ | 实测 |
| 全量测试 | 全绿 | **17 passed, 1 skipped** ✅ | 关键用例须真跑非 skip |
| 线上可访问性 | 资产全 200 | ✅ | 页面/JS/模型/示例图逐项核验 |
| Harness 文档一致性 | 与现状一致 | ✅ | 含 `AGENTS.md` 顶部"当前实际口径"表 |
| 错误记录 | ≥ 5 条 | **18 条** ✅ | `tech-debt-tracker.md` |

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

## 8. 交付与上线检查清单

**上线口径（首要判据）：**

- [x] **线上 Demo 可访问且真的能推理**：<https://wmsisme.github.io/yolo11-campus-phone-detection/demo/>
- [x] 线上资产逐项 HTTP 核验（页面 / app.js / selftest.json / ONNX 模型 / 示例图 全 200）
- [x] 页面文案与模型口径**一致**（类别名、指标、局限说明）
- [x] 浏览器内推理结果与 Python 参考实现**逐框一致**（自检 + 无头浏览器端到端测试）
- [x] 训练模型与训练数据集**不入库**（数据集本地保存；权重按需选择性入库）

**工程口径：**

- [x] 项目代码（`src/`, `tests/`, `requirements.txt`）可完整运行；`clone → pip install → 启动` 即可推理
- [x] 全量测试通过：`python -m pytest tests/ -q` → **17 passed, 1 skipped**
- [x] 所有 Harness 文档齐全且与项目现状一致
- [x] `experiments/` 下含多组实验，每组有 `config.yaml` + `metrics.json` + `training_curves.png`
- [x] 错误案例记录在 `tech-debt-tracker.md`（**18 条**，远超 5 条下限）
- [x] Web Demo（Streamlit 本地版 + 静态在线版）均可正常运行并展示检测结果

**展示口径（课程/面试用途）：**

- [x] `reports/final-report.md` 满足 16 个部分要求，且已同步到当前口径与真实结论
- [ ] 截取 Web Demo 运行截图插入报告
- [ ] （可选）录制演示视频
