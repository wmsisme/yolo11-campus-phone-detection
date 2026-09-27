# 技术债与已知错误

---

### 错误 #001 - 2026-06-06
- **现象**：旧项目 CLIP+BLIP-2 的 Recall@5 仅为 0.0065，远低于目标 0.65
- **原因**：数据集 original 是 YOLO 目标检测标注，类别为物体（刀、橡皮、头盔等），被强行用模板生成文本描述作为图文检索标注。任务与数据根本不匹配。
- **建议修复**：已重建为 YOLO11 目标检测项目，数据-任务天然匹配。
- **状态**：已通过项目重建解决

---

### 错误 #002 - 2026-06-06
- **现象**：smart school 数据集仅 141 张图片（train=96, val=31, test=14），小样本训练目标检测有挑战
- **原因**：Roboflow 公开数据集中"校园手机使用检测"场景的标注数据量有限
- **建议修复**：
  - 方案 A：利用 YOLO11 内置 Mosaic、HSV 抖动、翻转等数据增强
  - 方案 B：未来如有条件，可自行采集校园场景图片补充标注
- **状态**：已通过数据增强缓解，记录为已知限制

---

### 错误 #003 - 2026-06-06
- **现象**：`open_clip_torch` 和 `faiss` 等旧依赖在新项目中不再使用
- **原因**：技术路线从 CLIP+Faiss 检索切换为 YOLO11 检测
- **建议修复**：已重写 `requirements.txt`，移除不必要依赖
- **状态**：已修复

---

### 错误 #004 - 2026-06-06
- **现象**：加载 yolo11s.pt 和 yolo11m.pt 时报 `PytorchStreamReader failed reading zip archive: failed finding central directory`
- **原因**：GitHub release 下载在国内网络环境不稳定，18MB+ 大文件下载后 zip 损坏（文件大小正确但内容不完整）。yolo11n.pt（5.6MB）不受影响因为文件小。
- **建议修复**：
  - 方案 A（已实现）：train.py 自动检测损坏，从 HF 镜像 (hf-mirror.com) 重新下载
  - 方案 B：手动从 HuggingFace 下载后放到项目根目录
- **状态**：已修复（train.py 增加 HF 镜像自动 fallback）

---

### 错误 #005 - 2026-06-12
- **现象**：原始 Smart School v5 数据集仅 141 张图片且只含 phone 类，三轮训练后 mAP@50 最高仅 0.30，远低于 0.70 质量目标
- **原因**：单一 Roboflow 数据集的样本量和类别覆盖不足以训练高精度检测模型。141 张中的 96 张用于训练，数据增强无法弥补样本绝对数量不足。
- **建议修复**：
  - 方案 A（已实现）：从 Roboflow 下载 University-Outdoor、University ver 2/3、School v1 共 4 个数据集，通过 `merge_datasets.py` 合并类别映射，生成统一格式的重组数据集
  - 方案 B：自行采集校园场景图片并标注（时间成本高）
- **结果**：重组后 exp3 (yolo11n) mAP@50=0.73、exp4 (yolo11s) mAP@50=0.73，达到质量目标
- **状态**：已修复

---

### 错误 #006 - 2026-09-27
- **现象**：`python -m pytest tests/ -v` 报 1 failed（`TestImports::test_import_utils_helpers`），断言 `assert root.name == "期末作业" or "视觉ai" in str(root)` 失败
- **原因**：该断言把**旧机器的项目绝对路径**写进了测试。项目已迁移到 `D:\code_item\手机检测`，目录名不再匹配，测试必然失败；且断言本身对项目正确性毫无意义。
- **建议修复**：改为校验目录契约（`src/models/detect.py`、`requirements.txt` 是否存在），不绑定任何绝对路径。已修复。
- **状态**：已修复

---

### 错误 #007 - 2026-09-27
- **现象**：`data.yaml` 与 6 份实验 `config.yaml` 中的 `path` 字段指向不存在的旧路径（`D:\本科\作业\视觉ai\期末作业\...`），且因编码不匹配在文本中显示为乱码
- **原因**：`prepare_dataset()` / `prepare_reorganized_dataset()` 在“数据集已就绪”分支直接 `return`，跳过了 `data.yaml` 重写，使首次构建时写入的绝对路径永久滞留。仓库换机器/换盘后即为死路径。
- **建议修复**：抽出 `_write_data_yaml()`，两个准备函数**每次进入都刷新** `path` 为当前机器的 `resolve()` 结果。已修复并实测（`matches project root = True`）。
- **状态**：已修复

---

### 错误 #008 - 2026-09-27
- **现象**：Web Demo 中若某模型变体找不到训练权重，代码会强行把 `model.names` 改写成 `{0: "People using cellphone", 1: "cellphone"}`，导致 COCO 预训练权重把 person / bicycle 等类别**误标为手机**，界面还照常显示"检测到 N 个目标"与手机图例
- **原因**：原判断条件 `not model.names or len(model.names) > 2` 把“COCO 80 类预训练权重”也当成需要改名的对象；改名后类别语义与真实输出完全错位。
- **建议修复**：`load_detection_model()` 改为返回 `(model, is_trained)`；**仅训练权重**才强制类别名。回退通用权重时给出明确警告、界面显示"通用权重检出 N 个物体（类别为 COCO 80 类，与手机检测无关）"，不套用手机图例。另：`yolo11m` 会匹配到只训练 1 轮的 exp5 废权重，已在 README 注明。已修复。
- **状态**：已修复

---

### 错误 #009 - 2026-09-27
- **现象**：`requirements.txt` 缺少 `pandas`，但 `src/utils/metrics.py`（`plot_training_curves`）与 `src/web/app.py`（检测明细表）均直接 `import pandas`
- **原因**：依赖声明不完整。开发环境恰好装有 pandas（2.3.3）因而未暴露；在干净环境 `pip install -r requirements.txt` 后画训练曲线与渲染明细表会直接 `ModuleNotFoundError`。
- **建议修复**：`requirements.txt` 补充 `pandas>=2.0.0`。已修复。
- **状态**：已修复

---

### 错误 #010 - 2026-09-27
- **现象**：仓库内散落与当前技术路线无关的残骸——旧 CLIP+Faiss 检索路线产物（`data/features_*.npy`、`index_*.faiss`、`paths_*.json` 共 9 个文件）、无任何引用的权重（`yolo11s_new.pt`、`yolo26n.pt`）、`tmp/` 临时脚本、`_archive/` 旧项目
- **原因**：技术路线从 CLIP+Faiss 检索整体切换为 YOLO11 检测时，仅重建了 `src/`，磁盘残留未清理；`_archive/` 是刻意保留的备份。
- **建议修复**：全部移入回收站（非硬删）并写入 `.gitignore`。已修复。
- **状态**：已修复

---

> 后续训练和测试中发现的错误将继续追加到此文件。
