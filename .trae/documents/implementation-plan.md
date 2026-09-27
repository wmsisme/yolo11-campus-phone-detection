# 校园图像语义搜索 + VQA 系统 — 实施计划

## 项目目标
开发一个 Web 应用，允许用户通过文字描述搜索校园图片，并对选中图片进行视觉问答。

## 实施步骤

### Step 0：环境搭建与 Harness 文档初始化
- 创建 `requirements.txt`（torch, transformers, open_clip_torch, peft, streamlit, Pillow, scikit-learn, faiss-cpu 等）
- 创建所有 Harness 文档骨架：
  - `ARCHITECTURE.md`
  - `README.md`
  - `docs/product-specs/index.md`
  - `docs/exec-plans/active/current-plan.md`
  - `docs/exec-plans/tech-debt-tracker.md`
  - `docs/QUALITY_SCORE.md`
  - `docs/RELIABILITY.md`
  - `docs/SECURITY.md`
  - `docs/references/reference-list.md`

### Step 1：数据准备
- 搜索/准备校园图像数据集（推荐 CUHK Campus Dataset，或从 COCO 筛选 + 自建小规模数据集）
- 实现 `src/data/dataset.py`：
  - 数据集下载（可脚本化）、训练/验证集划分（80/20）
  - CLIP 标准预处理（224×224）
  - 每张图片关联 1-3 条文本描述
- 创建数据目录结构

### Step 2：基线检索系统
- 实现 `src/models/clip_model.py`：加载预训练 CLIP (ViT-B/32)
- 实现 `src/retrieval/index.py`：
  - 提取所有图像特征并存储（npy + Faiss 索引）
  - 文本→特征→余弦相似度→Top-K 检索
- 运行验证集计算 Recall@5 和 mAP，结果存入 `experiments/exp0_baseline/`
- 实现 `src/utils/helpers.py`

### Step 3：LoRA 微调 CLIP
- 在 `src/models/clip_model.py` 中添加 LoRA 微调代码（r=8, alpha=16, InfoNCE 损失）
- 实验 1：10 epochs，保存模型，评估指标 → `experiments/exp1_lora_epoch10/`
- 实验 2：20 epochs（或不同学习率），评估指标 → `experiments/exp2_lora_epoch20/`
- 生成对比表格和可视化

### Step 4：VQA 模块集成
- 实现 `src/models/vqa_model.py`：
  - 加载 BLIP-2（优先本地运行，备选 API）
  - `answer_question(image, question)` 函数
  - 缓存与错误处理

### Step 5：Streamlit Web Demo
- 实现 `src/web/app.py`：
  - 左侧：文本搜索框 + 搜索按钮
  - 中间：检索结果图片网格（可点击选中）
  - 右侧：选中图片 + 问题输入 + 提问按钮 + 答案展示
  - 错误处理（模型加载失败、图片不存在等）

### Step 6：测试与验证
- 实现 `tests/test_smoke.py`：覆盖数据加载、检索函数、VQA 调用
- 手动测试 5 个典型查询样例

### Step 7：最终报告
- 撰写 `reports/final-report.md`（16 个部分，含截图、实验对比、错误分析）

---

## 文件结构总览

```
.
├── AGENT.md                           # 已有，AI Agent 开发指导
├── ARCHITECTURE.md                    # 待创建
├── README.md                          # 待创建
├── requirements.txt                   # 待创建
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                 # 数据集加载与预处理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_model.py              # CLIP 加载 + LoRA 微调
│   │   └── vqa_model.py               # BLIP-2 VQA 模型
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── index.py                   # 特征索引 + 检索
│   ├── web/
│   │   ├── __init__.py
│   │   └── app.py                     # Streamlit 主程序
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                 # 辅助函数
├── tests/
│   ├── __init__.py
│   └── test_smoke.py                  # 冒烟测试
├── docs/
│   ├── product-specs/
│   │   └── index.md
│   ├── exec-plans/
│   │   ├── active/
│   │   │   └── current-plan.md
│   │   └── tech-debt-tracker.md
│   ├── QUALITY_SCORE.md
│   ├── RELIABILITY.md
│   ├── SECURITY.md
│   └── references/
│       └── reference-list.md
├── experiments/
│   ├── exp0_baseline/
│   ├── exp1_lora_epoch10/
│   └── exp2_lora_epoch20/
└── reports/
    └── final-report.md
```

---

## 质量标准（目标）

| 指标 | 目标 |
|------|------|
| Recall@5 | ≥ 0.65（微调后比基线提高 ≥ 5%） |
| VQA 正确率 | 10题中 ≥ 70% |
| 检索响应 | < 2 秒 |
| VQA 响应 | < 5 秒 |
| 核心测试覆盖 | 数据加载 / 检索 / VQA 三种场景 |
