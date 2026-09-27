# 安全与数据边界

## 数据安全规则

- ✅ 允许：使用 `data_source/smart school.v5i.yolov11` 公开数据集训练
- ✅ 允许：用户自行上传校园场景图片进行检测
- ❌ 禁止：上传包含学生面部可识别信息的隐私图片
- ❌ 禁止：上传身份证、准考证等个人敏感信息的图片
- ❌ 禁止：将用户上传的图片用于任何其他目的

## API Key 与凭据

- ✅ 允许：本地运行模型，无需 API key
- ❌ 禁止：在代码中硬编码任何 API key、Token 或密码
- ❌ 禁止：将任何凭据提交到版本控制系统

## 工具边界（AI Agent 规则）

### 允许的操作
- 创建/修改 `src/` 下的源代码
- 创建/修改 `tests/` 下的测试文件
- 创建/修改 `experiments/` 下的实验记录
- 创建/修改 `docs/` 下的文档
- 创建/修改 `reports/` 下的报告
- 修改 `requirements.txt`

### 禁止的操作
- ❌ 删除 `data_source/` 下的任何原始数据文件
- ❌ 修改 `docs/product-specs/index.md` 中的核心验收标准（除非开发者明确要求）
- ❌ 伪造实验指标或检测结果
- ❌ 提交包含隐私信息的文件
- ❌ 访问项目目录之外的文件系统

## 学术诚信

- 所有实验指标必须基于实际训练结果，不得伪造
- 数据集来源必须明确标注（Roboflow Smart School v5）
- 引用他人工作（YOLO11、ultralytics）需在 `references/reference-list.md` 中注明
- AI Agent 辅助代码生成的过程需在最终报告中如实记录

## 数据来源声明

| 数据集 | 来源 | 许可证 | 用途 |
|--------|------|--------|------|
| Smart School v5 | Roboflow Universe | CC BY 4.0 | 校园手机使用检测模型训练 |
