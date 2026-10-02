"""Streamlit Web Demo：校园手机使用检测系统

运行方式: streamlit run src/web/app.py
"""

import io
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.detect import (
    detect_image,
    draw_boxes,
    generate_report,
    load_model,
    report_to_markdown,
)
from src.data.dataset import prepare_dataset
from src.utils.helpers import get_project_root, logger

# 页面配置
st.set_page_config(
    page_title="校园手机使用检测",
    page_icon="📱",
    layout="wide",
)


def init_session_state():
    """初始化 Streamlit session state"""
    defaults = {
        "model": None,
        "model_variant": "yolo11n",
        "model_loaded": False,
        "is_trained": False,
        "detections": [],
        "annotated_image": None,
        "report": None,
        "uploaded_image": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_detection_model(model_variant: str) -> tuple:
    """加载检测模型，优先使用训练好的实验模型，其次使用通用预训练

    Returns:
        (model, is_trained) —— is_trained 为 False 时表示退回通用 COCO 预训练权重，
        此时模型**不能**用于手机检测，调用方必须如实告知用户。
    """
    project_root = get_project_root()

    # 扫描 experiments/ 目录，找到包含该模型变体名的实验
    experiments_dir = project_root / "experiments"
    trained_path = None
    if experiments_dir.exists():
        for exp_dir in sorted(experiments_dir.iterdir(), reverse=True):
            if exp_dir.is_dir() and model_variant in exp_dir.name:
                best_pt = exp_dir / "best.pt"
                if best_pt.exists():
                    trained_path = str(best_pt.resolve())
                    logger.info(f"找到训练模型: {trained_path}")
                    break

    is_trained = trained_path is not None
    if not is_trained:
        st.warning(
            f"未找到 {model_variant} 的训练权重，已退回通用预训练权重（COCO 80 类）。"
            "该权重**不具备手机检测能力**，检测结果仅供流程演示。"
            "建议改用 yolo11n / yolo11s（仓库内含其训练权重）。"
        )
        trained_path = f"{model_variant}.pt"

    with st.spinner(f"正在加载模型 {model_variant}..."):
        model = load_model(trained_path)
        # 只有训练好的权重才需要处理类别名；通用 COCO 权重绝不能改写 names，
        # 否则会把 person/bicycle 误标成手机。
        #
        # ⚠️ 2026-10-02 修正：**不要硬编码类别名**。
        #   原先这里无条件写成 {0: "People using cellphone", 1: "cellphone"}，
        #   但项目已存在两套口径的权重：
        #     · 旧口径 exp3/exp4 → {0: People using cellphone, 1: cellphone}
        #     · 新口径 exp6（干净数据重训）→ {0: in_hand, 1: on_ear}
        #   扫描逻辑按"目录名含模型变体"匹配会选中 exp6（`exp6_phone_usage_yolo11s`
        #   含 `yolo11s`），此时硬改名字会导致**模型在框手机、界面标成"人"**的静默错位。
        #   正确做法：以权重自带的 names 为准，仅在缺失时兜底。
        if is_trained and hasattr(model, "names"):
            if not model.names:
                model.names = {0: "People using cellphone", 1: "cellphone"}
            else:
                logger.info(f"沿用权重自带类别名: {model.names}")
        return model, is_trained


def main():
    init_session_state()

    # 标题
    st.title("📱 校园手机使用检测系统")
    st.caption("基于 YOLO11 的校园场景手机使用行为检测 | 目标检测 + 统计报告")

    # ============ 左侧面板 ============
    with st.sidebar:
        st.header("🎛️ 控制面板")

        # 模型选择
        model_variant = st.selectbox(
            "模型选择",
            options=["yolo11n", "yolo11s", "yolo11m"],
            index=0,
            help="n=nano(最快), s=small(均衡), m=medium(最准)",
        )

        # 置信度阈值
        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="低于此置信度的检测结果将被过滤",
        )

        # 加载模型按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 加载模型", width="stretch"):
                if (model_variant != st.session_state.model_variant or
                        not st.session_state.model_loaded):
                    try:
                        model, is_trained = load_detection_model(model_variant)
                        st.session_state.model = model
                        st.session_state.model_variant = model_variant
                        st.session_state.model_loaded = True
                        st.session_state.is_trained = is_trained
                        if is_trained:
                            st.success(f"{model_variant} 加载成功!")
                    except Exception as e:
                        st.error(f"模型加载失败: {e}")
                        st.session_state.model_loaded = False

        # 上传图片
        st.divider()
        st.subheader("📁 上传图片")
        uploaded_file = st.file_uploader(
            "选择校园场景图片",
            type=["jpg", "jpeg", "png"],
            help="支持 JPG/PNG 格式",
        )

        # 开始检测按钮
        st.divider()
        detect_btn = st.button("🚀 开始检测", type="primary", width="stretch")

        # 显示模型状态
        if st.session_state.model_loaded:
            if st.session_state.get("is_trained", False):
                st.success(f"当前模型: {st.session_state.model_variant}（手机检测专用）")
            else:
                st.warning(f"当前模型: {st.session_state.model_variant}（通用权重，非手机专用）")
        else:
            st.info("请先点击「加载模型」")

    # ============ 中间主区域 ============
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📷 检测结果")

        if uploaded_file is not None:
            # 读取上传的图片
            image_bytes = uploaded_file.getvalue()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # 如果还没有模型加载且需要检测，自动加载
            if detect_btn:
                if not st.session_state.model_loaded:
                    try:
                        model, is_trained = load_detection_model(model_variant)
                        st.session_state.model = model
                        st.session_state.model_variant = model_variant
                        st.session_state.model_loaded = True
                        st.session_state.is_trained = is_trained
                    except Exception as e:
                        st.error(f"自动加载模型失败: {e}")

                if st.session_state.model_loaded:
                    with st.spinner("正在检测..."):
                        detections = detect_image(
                            st.session_state.model,
                            image_bgr,
                            conf_threshold=conf_threshold,
                        )
                        st.session_state.detections = detections

                        if detections:
                            annotated = draw_boxes(image_bgr, detections)
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.session_state.annotated_image = annotated_rgb

                            # 生成报告
                            report = generate_report(detections, uploaded_file.name)
                            st.session_state.report = report
                        else:
                            st.session_state.annotated_image = None
                            st.session_state.report = None

            # 显示图片
            if st.session_state.annotated_image is not None:
                st.image(st.session_state.annotated_image, width="stretch")
                if st.session_state.get("is_trained", False):
                    st.caption(f"检测到 {len(st.session_state.detections)} 个目标")

                    # 图例
                    st.markdown("""
                    <div style="display:flex; gap:20px; margin-top:10px;">
                        <span>🔴 <b>使用手机的人</b></span>
                        <span>🔵 <b>手机</b></span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # 通用 COCO 权重：类别不是手机，不能套用手机图例
                    st.caption(
                        f"通用权重检出 {len(st.session_state.detections)} 个物体"
                        "（类别为 COCO 80 类，与手机检测无关）"
                    )
            elif uploaded_file is not None and detect_btn:
                st.image(pil_image, width="stretch")
                st.info("未检测到任何目标。请尝试降低置信度阈值或更换图片。")
            elif uploaded_file is not None:
                st.image(pil_image, width="stretch")
                st.caption("点击「开始检测」进行分析")
        else:
            st.info("👈 请在左侧上传校园场景图片")

    # ============ 右侧面板 ============
    with col_right:
        st.subheader("📊 检测统计")

        if st.session_state.report is not None and st.session_state.detections:
            report = st.session_state.report

            # 总计
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("总检测数", report["total_detections"])
            with col_b:
                st.metric("平均置信度", f"{report['average_confidence']:.1%}")

            # 类别统计柱状图
            if report.get("class_counts"):
                st.divider()
                st.caption("各类别数量")

                counts = report["class_counts"]
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ["#FF4444" if "cellphone" in k.lower() and "people" not in k.lower() else "#4488FF" for k in counts.keys()]
                bars = ax.bar(counts.keys(), counts.values(), color=colors)
                for bar, val in zip(bars, counts.values()):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            str(val), ha="center", fontsize=12, fontweight="bold")
                ax.set_ylabel("数量")
                ax.set_ylim(0, max(counts.values()) + 2 if counts else 3)
                st.pyplot(fig)

            # 置信度分布
            st.divider()
            st.caption("各类别平均置信度")
            for cls_name, conf in report.get("class_avg_confidence", {}).items():
                st.metric(f"📌 {cls_name}", f"{conf:.2%}")

            # 检测明细表格
            st.divider()
            st.subheader("📋 检测明细")

            if report.get("detections"):
                rows = []
                for i, det in enumerate(report["detections"], 1):
                    rows.append({
                        "#": i,
                        "类别": det["class"],
                        "置信度": f"{det['confidence']:.2%}",
                        "面积": f"{det['area']:.0f}px²",
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", hide_index=True)

                # 导出报告按钮
                st.divider()
                md_report = report_to_markdown(report)
                st.download_button(
                    label="💾 导出检测报告 (Markdown)",
                    data=md_report,
                    file_name=f"检测报告_{uploaded_file.name.rsplit('.', 1)[0]}.md",
                    mime="text/markdown",
                    width="stretch",
                )

        elif uploaded_file is not None and detect_btn:
            st.info("未检测到目标，无法生成报告。")
        else:
            st.info("检测完成后，统计报告将在此处显示")

    # 底部说明
    st.divider()
    st.caption(
        "基于 YOLO11 目标检测 | 数据集: Roboflow Smart School + 多数据集重组 | "
        "模型: nano/small/medium 多尺度可切换"
    )


if __name__ == "__main__":
    main()
