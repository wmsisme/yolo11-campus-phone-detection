# 静态在线 Demo（纯前端推理）

> 打开网页 → 传一张图 → **在你的浏览器里**跑 YOLO11 → 出检测框和报告。
> 没有后端、没有服务器、不需要 API Key，图片不会离开你的设备。

线上地址：**https://wmsisme.github.io/yolo11-campus-phone-detection/demo/**
（由本仓库 `main` 分支的 `/docs` 目录发布）

---

## 1. 它和 Streamlit 版是什么关系

| | 本静态 Demo（`docs/demo/`） | Streamlit 版（`src/web/app.py`） |
|:--|:--|:--|
| 运行位置 | 访客的浏览器（onnxruntime-web / WebAssembly） | 需要一台跑 Python 的机器 |
| 部署 | GitHub Pages 静态托管，零成本 | 需要服务器或本地启动 |
| 依赖 | 无 Python、无 API Key | ultralytics + torch + streamlit |
| 适合 | 分享链接给别人看、面试演示 | 批量检测、实验对比、报告导出 |

两者**共用同一个训练权重**：静态 Demo 用的 ONNX 由 `experiments/exp3_reorganized_yolo11n/best.pt`
导出，解码口径与 `src/models/onnx_infer.py` 完全一致（见下）。

---

## 2. 目录结构

```text
docs/demo/
├── index.html                  # 三栏界面：选择图片 / 画布 / 检测报告
├── app.js                      # 预处理 + 推理 + 解码 + NMS + 渲染（与 Python 参考实现对齐）
├── style.css
├── model/
│   └── phone-yolo11n.onnx      # 10.1 MB，fp32；导出方式见 src/models/export_onnx.py
├── samples/                    # 示例图（Roboflow Smart School v5，CC BY 4.0）
├── selftest.json               # 自检基准：Python 侧算出的检测框（由 gen_demo_fixtures.py 生成）
└── vendor/ort/                 # onnxruntime-web 本地副本（**不走 CDN**，避免网络依赖）
```

---

## 3. 本地怎么跑

浏览器对 `file://` 下的 `fetch()` 有限制，所以**不能直接双击 index.html**，要起一个静态服务：

```bash
# 在仓库根目录执行
python -m http.server 8080 --directory docs
# 然后打开 http://127.0.0.1:8080/demo/
```

调试用参数：

| URL | 作用 |
|:--|:--|
| `?sample=sample-classroom.jpg` | 直接加载某张示例图并跑一次推理 |
| `?selftest=1` | 用固定图片跑一遍，与 `selftest.json` 的 Python 基准逐框比对，结果写进 `#selftest-result` |

---

## 4. 怎么保证"浏览器算的和 Python 算的是一回事"

前端最容易出的问题不是"跑不起来"，而是**跑起来了但结果是错的**（坐标偏、类别错、NMS 口径不一致）。
所以这里做了三层对齐：

1. **同一套口径**：`app.js` 的 `letterboxToTensor / decode / nmsPerClass` 与
   `src/models/onnx_infer.py` 的 `letterbox / decode / nms` 一一对应（同样的填充值 114、
   同样的逐类别 NMS、同样的裁剪与排序）。
2. **基准值来自 Python**：`src/models/gen_demo_fixtures.py` 用 Python 参考实现在导出的 ONNX 上
   跑出真实检测框，写进 `selftest.json`；页面带 `?selftest=1` 时用同一批图跑浏览器推理并逐框比对
   （IoU ≥ 0.85），把 PASS/FAIL 写进 DOM。
3. **端到端自动化**：`tests/test_webdemo.py` 起本地服务 + 无头 Edge（`tests/headless_probe.mjs` 走 CDP）
   真跑一遍页面，断言自检 PASS、零 JS 报错、正常路径能出报告。

```bash
python -m pytest tests/test_webdemo.py tests/test_onnx_parity.py -v
```

> ⚠️ 自检必须"有牙齿"：示例图里**同时保留横图与竖图**，否则 letterbox 的 x/y 灰边补偿不会被覆盖。
> 这条约束写进了 `gen_demo_fixtures.py` 的自校验（缺一种就拒绝出基准）。踩坑经过见
> `docs/exec-plans/tech-debt-tracker.md` 错误 #014。

---

## 5. 重新生成产物

```bash
# 1) 从训练权重导出 ONNX（imgsz=640 / opset=13 / 关闭内置 NMS）
python -m src.models.export_onnx \
    --weights experiments/exp3_reorganized_yolo11n/best.pt \
    --out docs/demo/model/phone-yolo11n.onnx

# 2) 重新生成示例图与自检基准
python -m src.models.gen_demo_fixtures
```

> int8 量化版本**故意不发布**：实测四种量化配置都会打坏分类头（检测数归零），
> 详见 `docs/exec-plans/tech-debt-tracker.md` 错误 #012。`--int8` 现在会自检并在不合格时丢弃产物。

---

## 6. 数据、许可与隐私

- **隐私**：推理全部在浏览器本地完成，图片不会上传到任何服务器（页面源码里没有任何上传请求）。
- **示例图**：来自 Roboflow Universe 公开数据集 **Smart School v5**（CC BY 4.0），
  是教室/走廊的监控视角远景，画面中人物不可辨认；裁剪版本仅用于覆盖 letterbox 的不同长宽比。
- **模型**：由 [ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）训练得到；
  本页源码随仓库公开。
- **运行时**：onnxruntime-web 1.30.0（MIT），随包分发在 `vendor/ort/`。
