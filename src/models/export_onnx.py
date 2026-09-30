"""把训练好的 YOLO11 权重导出为 ONNX，供纯前端（浏览器内）推理使用。

为什么需要它：`docs/demo/` 里的静态 Demo 不再依赖 Python 服务端，而是用
onnxruntime-web 在浏览器里直接跑模型。这就要求权重以 ONNX 格式随仓库分发，
且导出参数固定（imgsz / opset / NMS 关闭），否则前后端口径会不一致。

用法::

    # 导出 fp32（与 PyTorch 结果一致，体积约 15 MB）
    python -m src.models.export_onnx \
        --weights experiments/exp3_reorganized_yolo11n/best.pt \
        --out docs/demo/model/phone-yolo11n.onnx

    # 额外产出一份静态量化 int8 版本（体积约 1/4，用于弱网/移动端）
    # ⚠️ 2026-09-30 实测：**本项目的 YOLO11 模型量化后分类头失效**，脚本会自检并丢弃
    #    不合格产物（返回码 3），详见 docs/exec-plans/tech-debt-tracker.md。
    python -m src.models.export_onnx \
        --weights experiments/exp3_reorganized_yolo11n/best.pt \
        --out docs/demo/model/phone-yolo11n.onnx --int8 \
        --calib-dir data_source/reorganized_phone_dataset_yolo/reorganized_dataset/val/images

约定（前端 `docs/demo/app.js` 依赖这些约定）：
  * 输入：`images`，float32，形状 `[1, 3, imgsz, imgsz]`，RGB、/255、letterbox 填充
  * 输出：`output0`，形状 `[1, 4 + nc, 8400]`，行 0-3 为 cx,cy,w,h（imgsz 尺度），
    行 4.. 为各类别置信度（已过 sigmoid）
  * 模型内部**不做 NMS**（`nms=False`），NMS 由前端按 IoU 阈值执行

⚠️ int8 为什么必须走**静态**量化：`quantize_dynamic` 会把 Conv 拆成 `ConvInteger`，
而 ONNXRuntime 的 CPU EP 与 WASM EP 都没有该算子的实现（实测
`NOT_IMPLEMENTED: Could not find an implementation for ConvInteger`），导出的模型直接跑不起来。
静态量化产出的是 QDQ / QLinearConv，前后端都能执行，但需要一批校准图（默认取重组数据集的
验证集，只用于统计激活值范围，不参与训练）。

⚠️ 但静态量化在本模型上**依然不可用**：QDQ / QOperator × per-channel / per-layer 四种配置
全部把分类头打废（框回归正常、类别分数恒为 0 → 检测数 0/N）。所以 `--int8` 现在带**自检**：
跑不通就删掉产物并以返回码 3 退出，避免把一个"能加载但结果全错"的模型放进仓库。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def quantize_static_int8(src: Path, dst: Path, input_name: str, calib_dir: Path, imgsz: int, count: int) -> None:
    """静态量化（QDQ）——需要校准图片来统计激活值范围。

    ⚠️ 两个已知坑（2026-09-30 实测）：
    1. 量化器内部用 `onnx.load(path)` 读模型，**窄字符路径**在中文目录下会
       `FileNotFoundError`（例如 `D:\\code_item\\手机检测\\...`）。这里先把模型复制到
       临时 ASCII 目录再量化，最后拷回目标路径。
    2. 该配置量化出来的模型**分类头会失效**（类别分数恒为 0）——所以本函数只负责产出，
       是否可用由 `verify_quantized()` 判定，不合格的产物会被删除（见 `export()`）。
    """
    import shutil as _shutil
    import tempfile

    import cv2
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    from .onnx_infer import preprocess

    files = sorted(
        p for p in calib_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )[:count]
    if len(files) < 8:
        raise FileNotFoundError(f"校准图片不足（{len(files)} 张）：{calib_dir}")

    class LetterboxReader(CalibrationDataReader):
        def __init__(self) -> None:
            self.i = 0

        def get_next(self):
            if self.i >= len(files):
                return None
            img = cv2.imread(str(files[self.i]))
            if img is None:  # 中文路径兜底
                from src.utils.helpers import imread_unicode

                img = imread_unicode(files[self.i])
            self.i += 1
            if img is None:
                return self.get_next()
            tensor, _, _ = preprocess(img, imgsz)
            return {input_name: tensor}

        def rewind(self) -> None:
            self.i = 0

    with tempfile.TemporaryDirectory(prefix="onnx-quant-") as tmp:
        tmp_src = Path(tmp) / "model.onnx"
        tmp_dst = Path(tmp) / "model-int8.onnx"
        _shutil.copyfile(src, tmp_src)
        quantize_static(
            model_input=str(tmp_src),
            model_output=str(tmp_dst),
            calibration_data_reader=LetterboxReader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copyfile(tmp_dst, dst)


def verify_quantized(fp32_path: Path, int8_path: Path, images: list[Path], imgsz: int, max_images: int = 3) -> dict:
    """量化产物能不能用，用**检测结果**说话（而不是只看能不能加载）。

    判据：同一批图片上，量化模型的类别分数上限与检测数量不能塌成 0。
    """
    from src.utils.helpers import imread_unicode

    from .onnx_infer import OnnxDetector, preprocess

    names = ["People using cellphone", "cellphone"]
    a = OnnxDetector(fp32_path, names, imgsz)
    b = OnnxDetector(int8_path, names, imgsz)
    report = {"images": 0, "fp32_dets": 0, "int8_dets": 0, "fp32_score_max": 0.0, "int8_score_max": 0.0}
    for p in images[:max_images]:
        img = cv2.imread(str(p))
        if img is None:
            img = imread_unicode(p)
        if img is None:
            continue
        tensor, _, _ = preprocess(img, imgsz)
        y_fp32 = a.session.run(None, {a.input_name: tensor})[0]
        y_int8 = b.session.run(None, {b.input_name: tensor})[0]
        report["images"] += 1
        report["fp32_score_max"] = max(report["fp32_score_max"], float(y_fp32[0, 4:, :].max()))
        report["int8_score_max"] = max(report["int8_score_max"], float(y_int8[0, 4:, :].max()))
        report["fp32_dets"] += len(a.infer(img, 0.25, 0.7))
        report["int8_dets"] += len(b.infer(img, 0.25, 0.7))
    report["ok"] = bool(
        report["images"] > 0
        and report["fp32_dets"] > 0
        and report["int8_dets"] > 0
        and report["int8_score_max"] >= 0.5 * report["fp32_score_max"]
    )
    return report


def export(
    weights: Path,
    out: Path,
    imgsz: int,
    opset: int,
    int8: bool,
    calib_dir: Path,
    calib_count: int,
) -> dict:
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在：{weights}")

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=True,
            dynamic=False,
            nms=False,  # NMS 放前端做，保证阈值可调
            half=False,
            device="cpu",
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exported, out)

    info = {
        "weights": str(weights.relative_to(REPO_ROOT)) if weights.is_relative_to(REPO_ROOT) else str(weights),
        "onnx_fp32": str(out),
        "bytes_fp32": out.stat().st_size,
        "imgsz": imgsz,
        "opset": opset,
        "classes": [model.names[i] for i in sorted(model.names)],
        "nc": len(model.names),
    }

    if int8:
        out_int8 = out.with_name(out.stem + "-int8" + out.suffix)
        input_name = probe(Path(info["onnx_fp32"]), imgsz)["input_name"]
        quantize_static_int8(out, out_int8, input_name, calib_dir, imgsz, calib_count)
        verification = verify_quantized(out, out_int8, sorted(calib_dir.glob("*.jpg")), imgsz)
        info["int8_verification"] = verification
        if verification["ok"]:
            info["onnx_int8"] = str(out_int8)
            info["bytes_int8"] = out_int8.stat().st_size
            info["calib_images"] = min(calib_count, len(list(calib_dir.glob("*"))))
        else:
            # 不合格的量化模型不许留在仓库里——它能加载、能跑，但结果全是错的
            out_int8.unlink(missing_ok=True)
            info["int8_rejected"] = True

    return info


def probe(path: Path, imgsz: int) -> dict:
    """用 onnxruntime 跑一张全灰图，回报输入/输出形状，确认导出契约没变。

    跑不动（算子不被 EP 支持）时**不抛异常**，而是把错误原样报出来——
    这正是动态量化那条死路暴露问题的地方，必须让它可见。
    """
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        x = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        y = sess.run(None, {inp.name: x})[0]
        return {
            "input_name": inp.name,
            "input_shape": list(inp.shape),
            "output_shape": list(y.shape),
        }
    except Exception as exc:  # noqa: BLE001 - 目的是把不可运行的模型暴露出来
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="导出 YOLO11 → ONNX（前端推理用）")
    ap.add_argument("--weights", type=Path, required=True, help="训练权重 best.pt")
    ap.add_argument("--out", type=Path, required=True, help="输出 onnx 路径")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--int8", action="store_true", help="同时产出静态量化 int8 版本（需校准图）")
    ap.add_argument(
        "--calib-dir",
        type=Path,
        default=REPO_ROOT / "data_source/reorganized_phone_dataset_yolo/reorganized_dataset/val/images",
        help="int8 静态量化的校准图片目录",
    )
    ap.add_argument("--calib-count", type=int, default=64, help="使用的校准图片数量")
    args = ap.parse_args(argv)

    info = export(args.weights, args.out, args.imgsz, args.opset, args.int8, args.calib_dir, args.calib_count)

    print("导出完成：")
    print(f"  fp32 : {info['onnx_fp32']}  ({info['bytes_fp32'] / 1024 / 1024:.2f} MB)")
    print(f"  类别 : {info['classes']}")
    print(f"  契约 : {probe(Path(info['onnx_fp32']), args.imgsz)}")
    if "onnx_int8" in info:
        print(f"  int8 : {info['onnx_int8']}  ({info['bytes_int8'] / 1024 / 1024:.2f} MB, 校准 {info['calib_images']} 张)")
        print(f"  契约 : {probe(Path(info['onnx_int8']), args.imgsz)}")
        print(f"  校验 : {info['int8_verification']}")
    elif info.get("int8_rejected"):
        v = info["int8_verification"]
        print("  int8 : ❌ 已丢弃 —— 量化后分类头失效（检测数 "
              f"{v['int8_dets']}/{v['fp32_dets']}，类别分数上限 "
              f"{v['int8_score_max']:.3f}/{v['fp32_score_max']:.3f}）")
        print("         详见 docs/exec-plans/tech-debt-tracker.md「int8 量化不可用」")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
