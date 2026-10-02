# 当前计划 (更新于 2026-06-15)

## 已完成

- [x] 项目分析：阅读作业文档，确定从 CLIP/BLIP-2 切换到 YOLO11
- [x] Step 0：备份旧项目到 `_archive/`，创建新目录结构
- [x] Step 1：数据准备 - dataset.py（smart school → YOLO 格式）
- [x] Step 2：训练脚本 - train.py（argparse 驱动，自动保存实验记录）
- [x] Step 3：推理模块 - detect.py（检测、绘图、报告生成）
- [x] Step 4：工具函数 - helpers.py + metrics.py
- [x] Step 5：Streamlit Web Demo - app.py（三栏布局）
- [x] Step 6：冒烟测试 - test_smoke.py（6 个测试类）
- [x] Step 7：Harness 文档重写（11 个文件全部完成）
- [x] requirements.txt 重写为 YOLO11 依赖
- [x] **原始数据集三组实验全部完成**：exp0 (yolo11n)、exp1 (yolo11s)、exp2 (yolo11m)
- [x] **多数据集重组**：合并 University-Outdoor、University ver 2/3、School v1 等 4 个 Roboflow 数据集
- [x] **重组数据集三组实验全部完成**：exp3 (yolo11n, 早停), exp4 (yolo11s), exp5 (yolo11m, 中断)
- [x] 模型下载问题修复：GitHub 损坏 → HF 镜像自动 fallback
- [x] Harness 文档进一步完善（README、final-report、current-plan、QUALITY_SCORE、tech-debt-tracker）

## 训练结果全量对比

### 原始 Smart School v5 数据集 (141 张, 仅 phone 类)

| 实验 | 模型 | Epochs | mAP@50 | mAP@50-95 | Precision | Recall | 最佳 epoch |
|------|------|--------|--------|-----------|-----------|--------|------------|
| exp0 | YOLO11n | 100 | 0.30 | 0.13 | 0.48 | 0.27 | 51 |
| exp1 | YOLO11s | 100 | 0.29 | 0.16 | 0.57 | 0.26 | 43 |
| exp2 | YOLO11m | 150 | 0.25 | 0.13 | 0.50 | 0.25 | — |

### 重组数据集 (合并 4 个数据集, 含 person/phone 等多类)

| 实验 | 模型 | Epochs | mAP@50 | mAP@50-95 | Precision | Recall | 最佳 epoch |
|------|------|--------|--------|-----------|-----------|--------|------------|
| exp3 | YOLO11n | 61 (早停) | **0.73** | **0.50** | 0.72 | 0.86 | 61 |
| exp4 | YOLO11s | 100 | **0.73** | **0.51** | 0.72 | 0.88 | — |
| exp5 | YOLO11m | 中断 | — | — | — | — | — |

### 关键结论

- **数据量是核心瓶颈**：从 141 张扩充到多数据集后，mAP@50 从 0.30 跃升到 0.73（+143%），达成质量目标
- **yolo11s 为最优方案**：mAP@50=0.73，Precision=0.72，Recall=0.88，均衡最佳
- **yolo11n 早停策略有效**：61 轮即收敛，结果与 yolo11s 100 轮持平

## 进行中

- [x] 文档完善（README、final-report、current-plan、QUALITY_SCORE）

## 🔄 2026-10-01~02 重训（exp6）：口径调整 + 换干净数据

起因：达铭在线上 Demo 实测报「经常把人脸检测为手机」。排查后确定**旧模型失败源于训练标注污染**
（详见 tech-debt #015/#016：22,879 张重组集全库含"人脸被标成 cellphone"，整体不可用）。
经他批准，把口径改为「**使用中的手机**」（`in_hand` 手持 / `on_ear` 贴耳），并用干净的
`手机数据集/phone_usage_split` 重训（本地数据、不入库）。

| 项 | 结果 |
|:--|:--|
| 训练 | `exp6_phone_usage_yolo11s`，**40 轮**（每轮 11.7 分钟） |
| 默认口径（conf=0.25） | mAP@50 **0.3646** / P 0.6934 / R 0.3950 |
| **达标口径（conf=0.50）** | **P = 0.8251 ✅** / R 0.3520 |
| **原始报障** | ✅ **已解决**：300 张真实无手机图上误报 **67.7% → 1.3%**（conf≥0.5），框尺度从"脸尺度 0.154"回到"手机尺度 0.017–0.023" |
| 遗留限制 | 教室远距离小手机（标注框边长中位 0.037）仍检不出——**尺度/域差异**，需补该尺度数据（tech-debt #017） |
| 未做 | 补到 80 轮：`last.pt` 被 strip 后无法 resume（#018），且末 20 轮 P 无上升趋势、预期仍达不到 0.80，故改用"阈值达标"路径 |

过程与踩坑全部留档：`progress-exp6.md`（进度/失败复盘）、`retrain-plan-exp6.md`（方案）、
tech-debt **#015~#018**、阈值扫描表 `runs/_diag/threshold_table.md`。

## 下一步（仓库化交付）

- [x] 复制优质模型权重到 `experiments/`（exp3 yolo11n、exp4 yolo11s 已入库；exp5 因 1 轮废权重不入库）
- [ ] 启动 Web Demo：`streamlit run src/web/app.py`（仓库开箱即用，无需先训练）
- [x] 运行全量测试：`python -m pytest tests/ -v` → **17 passed, 1 skipped**（含 ONNX 一致性与静态 Demo 端到端）
- [ ] 截取 Web Demo 运行截图，插入 final-report
- [ ] 录制演示视频（可选加分项）
- [ ] 提交压缩包 + 成绩记录表
- [ ] （新增）将 exp6 的干净口径模型接入 Web Demo / 静态 Demo（当前线上仍是旧 exp3 权重与旧两类口径）
- [ ] （新增）补"教室/远距离小目标"尺度数据后重训，以覆盖校园真实场景

## 2026-09-27 仓库化修复记录

本轮面向「上传 GitHub 作为可闭环仓库」做了一轮体检与修复，详见 `tech-debt-tracker.md` 错误 #006~#010：

| # | 问题 | 处理 |
|---|------|------|
| #006 | 冒烟测试把旧机器路径写进断言，必然失败 | 改为校验目录契约；未备数据集时跳过而非失败 |
| #007 | `data.yaml` 残留旧机器绝对路径（已乱码） | 抽出 `_write_data_yaml()`，每次刷新为当前机器路径 |
| #008 | Web Demo 回退 COCO 权重时强行改名 → 自行车被标成手机 | 返回 `(model, is_trained)`，仅训练权重改名；回退时明确告警 |
| #009 | `requirements.txt` 缺 `pandas`（metrics/app 均直接 import） | 补充 `pandas>=2.0.0` |
| #010 | 旧 CLIP+Faiss 残骸、无引用权重、tmp/、_archive/ | 全部移入回收站并写入 `.gitignore` |

同时：新增 `.gitignore`（锚定根目录规则，数据集/训练产物/预训练权重按约定不入库）、重写 `README.md`（含 clone→run 快速开始与真实实验数据）、同步 `QUALITY_SCORE.md` / `RELIABILITY.md` / `references`。

## 2026-09-30 静态在线 Demo（浏览器内推理）

给项目补上**公网可打开、点开就能用**的形态：`docs/demo/` 纯前端 Demo，发布在 GitHub Pages
（`main` 分支 `/docs` 目录）→ <https://wmsisme.github.io/yolo11-campus-phone-detection/demo/>。
动机是原来只有一个必须跑 Python 的 Streamlit 版，面试/分享场景下"打不开"。

- [x] ONNX 导出链路 `src/models/export_onnx.py`（imgsz=640 / opset=13 / 关闭内置 NMS；`--opset` 默认值已与产物对齐）
- [x] Python 参考实现 `src/models/onnx_infer.py`（letterbox → 解码 → 逐类别 NMS，作为前后端口径的规范）
- [x] 前端 `docs/demo/`：三栏界面（上传 / 画布 / 报告）、参数可调（置信度、IoU、类别过滤）、
      Markdown 报告导出、示例图一键体验、onnxruntime-web **本地化**（不依赖 CDN）
- [x] 自检机制：`?selftest=1` 用固定图片跑浏览器推理，与 Python 基准逐框比对（IoU ≥ 0.85）
- [x] 自动化：`tests/test_onnx_parity.py`（ONNX ↔ PyTorch 逐框一致）、
      `tests/test_webdemo.py`（无头 Edge 真跑页面 + CDP 探针 `tests/headless_probe.mjs`）
- [x] 用三次**故意破坏**证明自检有鉴别力（x 轴灰边 / y 轴灰边 / 类别错位 → 全部被抓到）
- [x] 技术债 #012~#014 记录：int8 量化不可用、`cv2.imread` 中文路径静默失败、示例图长宽比覆盖不足

**结论性取舍**：网页版只发布 **yolo11n fp32（10.1 MB）**——yolo11s 的 ONNX 有 36 MB（首屏代价过大），
int8 四种量化配置都会打坏分类头（检测数归零）。详见 `tech-debt-tracker.md` 错误 #012。

## 给下个 Agent 的提醒

- 6 组实验全部完成，重组数据集实验（exp3/exp4）结果优秀
- 推荐使用 exp4_yolo11s（mAP@50=0.726）或 exp3_yolo11n（0.729）作为 Web Demo 模型
- 仓库**不含数据集**（`.gitignore` 排除 `data_source/`、`data/`、`runs/`）：复现训练需先自行下载 Roboflow 数据集
- exp5_yolo11m 只训练了 1 轮即中断，其权重无参考价值；Web Demo 选 yolo11m 会加载到该废权重，已在 README 注明
- `data.yaml` 的 `path` 字段会被 `prepare_*` 自动刷新，**不要手工改**，也不要提交构建时的绝对路径
