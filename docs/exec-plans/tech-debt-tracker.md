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

### 错误 #011 - 2026-09-27
- **现象**：运行 Web Demo 时 Streamlit 持续输出弃用告警 `Please replace use_container_width with width. use_container_width will be removed after 2025-12-31.`
- **原因**：`src/web/app.py` 有 7 处使用旧 API `use_container_width=True`（`st.button` ×2、`st.image` ×3、`st.dataframe` ×1、`st.download_button` ×1），该参数已被官方弃用并有明确移除时限。
- **建议修复**：全部替换为 `width="stretch"`；因新 API 需较新版本，`requirements.txt` 中 `streamlit` 下限由 `>=1.28.0` 提升为 `>=1.49.0`。已修复并复测（全路径 UI 验收 4/4 场景通过，告警消失）。
- **状态**：已修复

---

### 错误 #012 - 2026-09-30
- **现象**：为静态 Demo 压缩模型体积做 int8 量化，量化后的模型**能加载、能推理、不报错，但结果全错**——框回归正常，类别分数恒为 0.000，检测数从 N 变成 0。
- **原因**：YOLO11 的检测头（分类分支 + sigmoid + Concat）对激活量化极不友好。实测四种配置全部失效：`QuantFormat.QDQ/QOperator` × `per_channel=True/False`；且 `quantize_dynamic` 另一条路直接产 `ConvInteger`，ONNXRuntime 的 CPU EP 与 WASM EP 都没有该算子实现（`NOT_IMPLEMENTED`）。ONNX Runtime 官方对此类模型有明确提示：`Please consider to run pre-processing before quantization`（需要一个带预处理层的导出模型）。
- **建议修复**：**放弃 int8，静态 Demo 只发布 fp32 ONNX**（yolo11n 10.1 MB，对比：yolo11s 36.2 MB 也偏大，故网页版只放 nano）。`src/models/export_onnx.py` 的 `--int8` 现在带**自检**：量化后实跑几张小图比对检测数与类别分数上限，不合格就直接删掉产物并以返回码 3 退出——不允许"能加载但结果是错的"模型留在仓库里。
- **状态**：已知限制（量化在本模型上不可行，已用自检兜住）

---

### 错误 #013 - 2026-09-30
- **现象**：`cv2.imread()` 在含中文的路径上静默返回 `None`（`D:\code_item\手机检测\...`），报 `can't open/read file: check file path/integrity`，而 `Path.exists()` 为 True —— 很容易被误判成"数据集缺失/图片损坏"。
- **原因**：Windows 下 OpenCV 的 `imread` 走窄字符路径并且**不认 UTF-8 字节**。有意思的是：只要 `import ultralytics`，它会给 `cv2.imread` 打上 Unicode 路径补丁，于是"先 import ultralytics 的脚本能读、单独用 cv2 的脚本读不到"，排查时极易被误导。
- **建议修复**：新增 `src/utils/helpers.imread_unicode()`（`np.fromfile` + `cv2.imdecode`），新代码统一走它；导出脚本与示例图生成脚本已改用。
- **状态**：已修复

---

### 错误 #014 - 2026-09-30
- **现象**：静态 Demo 的浏览器自检一开始"全绿"，但把 `letterbox` 的灰边补偿 `- pad[0]` **故意删掉后自检依然 PASS** —— 也就是说那条断言当时没有鉴别力。
- **原因**：最初几张示例图全是 640×640 方形图（letterbox 不产生灰边，`pad` 恒为 0），那段代码根本没被执行；补了一张 640×400 横图后仍 PASS，因为**横图只让 `pad[1] ≠ 0`，被破坏的是 x 轴**。
- **建议修复**：示例图必须同时包含**横图（覆盖 pad[1]）与竖图（覆盖 pad[0]）**，并把这条约束写成 `src/models/gen_demo_fixtures.py` 的**自校验**（缺任一种就拒绝生成基准、返回码 5）。补上竖图后，同样的破坏立刻被自检抓住（minIoU 0.999 → 0）。
- **状态**：已修复
- **教训（可复用）**：自检的价值不看"是否 PASS"，而看"**改了该抓的东西它会不会红**"——新写的断言必须先用一次故意破坏证明它有牙齿。

---

---

### 错误 #015 - 2026-10-01
- **现象**：线上 Demo 实际使用时**频繁把人脸检测成手机**（用户报障原话：「检测效果不佳，经常出现把人脸检测为手机的情况」）。实测复现：40 张「标注中无手机」的验证图上，conf≥0.25 有 33 张冒出手机框（最高分 0.86），**conf≥0.70 仍有 25 张**；把误检框画出来一看，扣的全是**人脸**（安全帽脸、口罩脸、低头看手机时的人脸上半部）。
- **原因**：**训练标注本身把脸标成了手机**。合并后的 18,800 张训练图里，**42.9%（8,059 张）含至少一个"大手机框"**（边长 >35% 图幅）；train 共 24,993 个 `cellphone` 标注中，边长 >25% 的占 77.2%、>50% 的占 5.9%，而 `People using cellphone` 的中位边长仅 13.5%——**"手机"比"拿着手机的人"标注得还大**，语义倒挂。根因是 4 个 Roboflow 数据集对"手机"的定义不一致（有的标手机本体、有的标头部/手持区域），合并时**没有统一标注口径**，且合并脚本 `merge_datasets.py` 未入库、无法追溯。
- **关键影响**：val 集同源污染（25.1%，303/1730 张含大框），所以 **mAP@50=0.73 是虚高的**（照着坏标注考坏标注），**不能作为"检测效果好"的证据**。
- **已排除**：类别映射无错（data.yaml→权重内嵌 names→app.js→fixtures 全链条 0=使用手机的人、1=手机）；调阈值救不了（>0.85 仍有 3/40 漏网）；误检框与人物框 IoU 几乎全为 0，说明是**专属人脸触发模式**，而非"把人判成手机"。
- **建议修复**：① 逐数据集审计标出口径、定位污染源；② 清洗——建议**直接丢弃大框**而非重标（25k 个框重标成本过高，丢弃后仍余约 5.7k 个可用小框）；③ 清洗后重训 exp6（建议 yolo11s），**预期 mAP 明显下跌，那是正常且诚实的数字**；④ 把标注审计做成 `src/data/` 的训练前校验步骤，避免再次中毒。
- **状态**：待处理（已向开发者报告并给出四步路线，等待决定是否启动第①步审计）

---

> 后续训练和测试中发现的错误将继续追加到此文件。
