"""YOLO11 训练入口脚本：命令行参数驱动，自动保存实验记录"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ultralytics import YOLO

from src.data.dataset import (
    prepare_dataset,
    prepare_phone_usage_dataset,
    prepare_reorganized_dataset,
)
from src.utils.helpers import get_project_root, logger, save_json
from src.utils.metrics import extract_metrics_from_results, plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="YOLO11 训练脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        help="YOLO11 模型变体 (default: yolo11n)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="批次大小 (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸 (default: 640)")
    parser.add_argument("--lr", type=float, default=0.01, help="初始学习率 (default: 0.01)")
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="实验名称，如 exp0_yolo11n_baseline",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练设备: 0(GPU), cpu, 或 0,1(多GPU) (default: 0)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help=("从 checkpoint 继续训练。可传路径（默认取 runs/detect/<exp_name>/weights/last.pt）。"
              "⚠️ 内部会传**绝对路径**给 ultralytics，避免它的 get_latest_run() 误续到别的实验。"
              "续训时**不要**再传 --epochs（会变成'再训 N 轮'）。"),
    )
    parser.add_argument(
        "--stop_after",
        type=int,
        default=0,
        help=("跑到第 N 轮后正常收尾停止（0=不启用）。用于把长训练分段跑："
              "第一阶段 --epochs 80 --stop_after 40，之后直接 --resume 补完剩下 40 轮。"
              "checkpoint 里记录的仍是 80 轮，所以续训能自动补到 80。"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="smart_school",
        choices=["smart_school", "reorganized", "phone_usage"],
        help=("数据集选择: smart_school(原始141张) / reorganized(重组集22879张，"
              "⚠️已证实标注污染、不可用) / phone_usage(手机使用状态 in_hand+on_ear，本地数据)"),
    )
    parser.add_argument(
        "--early_stop_map",
        type=float,
        default=0.0,
        help="mAP@50 早停阈值 (0=禁用, 例: 0.9 表示达到0.90即停止)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=("dataloader 进程数（默认 2）。⚠️ 本机 15.6GB 物理内存，ultralytics 默认的 8 个 "
              "worker 会因各自映射 torch 共享内存而把提交内存打爆——2026-10-01 实测报 "
              "WinError 1455（页面文件太小）导致训练在 epoch 1 崩溃。内存紧张时调小此值。"),
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="把图片缓存到内存/磁盘以减少 IO（会额外占内存，按需使用）",
    )
    args = parser.parse_args()

    project_root = get_project_root()

    # Step 1: 准备数据集
    logger.info("正在准备数据集...")
    if args.dataset == "reorganized":
        dataset_root = prepare_reorganized_dataset()
    elif args.dataset == "phone_usage":
        dataset_root = prepare_phone_usage_dataset()
    else:
        dataset_root = prepare_dataset()
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        logger.error(f"data.yaml 不存在: {data_yaml}")
        sys.exit(1)

    # 使用绝对路径确保 ultralytics 能找到
    data_yaml_abs = str(data_yaml.resolve())

    # Step 2: 设置实验目录
    exp_dir = project_root / "experiments" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"实验目录: {exp_dir}")

    # 保存实验配置
    config = {
        "experiment": args.exp_name,
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr,
        "device": args.device,
        "data_yaml": data_yaml_abs,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = exp_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"实验配置已保存: {config_path}")

    # Step 3: 加载模型（支持 GitHub 损坏自动从 HF 镜像下载）
    logger.info(f"正在加载模型: {args.model}.pt ...")
    model_path = Path(f"{args.model}.pt")

    def _download_from_hf_mirror(model_name: str, dest: Path) -> bool:
        """从 HuggingFace 镜像下载模型权重"""
        import requests
        url = f"https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/{model_name}.pt"
        try:
            logger.info(f"从 HF 镜像下载: {url}")
            r = requests.get(url, timeout=300)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(r.content)
                logger.info(f"HF 镜像下载完成: {dest.stat().st_size / 1024 / 1024:.1f}MB")
                return True
            else:
                logger.warning(f"HF 镜像返回 {r.status_code}")
        except Exception as e:
            logger.warning(f"HF 镜像下载异常: {e}")
        return False

    def _load_model_safe(model_name: str) -> YOLO:
        """安全加载模型，自动处理损坏文件和镜像回退"""
        mp = Path(f"{model_name}.pt")
        # 检查已有文件大小是否异常
        if mp.exists() and mp.stat().st_size < 1_000_000:
            logger.warning(f"模型文件异常小，删除: {mp}")
            mp.unlink()

        try:
            return YOLO(f"{model_name}.pt")
        except RuntimeError as e:
            err_msg = str(e)
            if "failed reading zip" in err_msg or "failed finding central" in err_msg:
                logger.warning(f"模型文件损坏 ({mp})，尝试重新下载...")
                if mp.exists():
                    mp.unlink()
                # 优先从 HF 镜像下载（国内更稳定）
                if _download_from_hf_mirror(model_name, mp):
                    return YOLO(str(mp))
                # 回退到 ultralytics 默认下载
                logger.info("HF 镜像失败，回退到默认下载源...")
                return YOLO(f"{model_name}.pt")
            raise

    # 断点续训：**必须显式指定本实验的 checkpoint**。
    #
    # ⚠️ 为什么不能直接传 `resume=True`：ultralytics 的 check_resume() 在 resume 不是路径时
    #    会调用 `get_latest_run()` —— 本机 runs/detect/ 下有 6 个老实验目录，
    #    它会挑到**别的实验**去续（实测机制如此）。所以这里自己解析出 last.pt 的**绝对路径**。
    #
    # 续训语义（读 ultralytics.resume_training 源码确认）：
    #     start_epoch = ckpt["epoch"] + 1
    #     assert 0 < start_epoch < self.epochs          # 已跑满则拒绝
    #     if self.epochs < start_epoch: self.epochs += ckpt["epoch"]   # 自动"再训 N 轮"
    #   ⇒ 想「跑完 40 轮后补到 80 轮」，**不要**在续训时传 --epochs 80（那会再训 80 轮 = 共 120）；
    #     直接 --resume 即可，它会从中断处补到 ckpt 里记录的 80 轮。
    #   ⇒ 而 checkpoint 里的 epochs 由**首次训练时**的 --epochs 决定：
    #     分两段跑就必须**首次就用最终总轮数**（本次 = 80），第一阶段用 --stop_after 截断。
    run_dir = project_root / "runs" / "detect" / args.exp_name
    last_pt = run_dir / "weights" / "last.pt"
    resume_path = None
    if args.resume:
        explicit = Path(args.resume) if isinstance(args.resume, str) else None
        if explicit and explicit.is_file():
            resume_path = explicit.resolve()
        elif last_pt.exists():
            resume_path = last_pt.resolve()
        else:
            logger.error(
                f"--resume 找不到可用 checkpoint：{last_pt} 不存在。\n"
                "  （不会退回到 get_latest_run()，以免误续到别的实验目录）"
            )
            sys.exit(2)
        # 打印断点信息，便于确认真从中断处继续
        try:
            import torch as _torch
            _ck = _torch.load(str(resume_path), map_location="cpu", weights_only=False)
            _done = int(_ck.get("epoch", -1)) + 1
            _target = int(_ck.get("train_args", {}).get("epochs")
                          or _ck.get("args", {}).get("epochs") or 0)
            logger.info(f"断点续训: {resume_path}")
            logger.info(f"  已完成 {_done} 轮，checkpoint 记录目标 {_target} 轮")
            if _target and _done >= _target:
                logger.info("  注意：该 checkpoint 已跑满目标轮数，续训只会重跑（请检查是否传了 --epochs）")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"读取断点信息失败（不影响续训）: {e}")

    if resume_path is not None:
        model = YOLO(str(resume_path))
    else:
        model = _load_model_safe(args.model)

    # 注册 mAP 早停回调
    if args.early_stop_map > 0:
        early_stop_triggered = [False]  # mutable wrapper for closure

        def on_fit_epoch_end(trainer):
            if hasattr(trainer, "metrics") and trainer.metrics:
                # 兼容不同 ultralytics 版本的 metrics key
                map50 = 0.0
                for key in ["metrics/mAP50(B)", "mAP50", "metrics/mAP50-95(B)"]:
                    v = trainer.metrics.get(key, None)
                    if v is not None and (key != "metrics/mAP50-95(B)" or map50 == 0):
                        map50 = max(map50, v if key != "metrics/mAP50-95(B)" else v * 1.5)
                # 如果上面都没匹配，遍历所有 key 找含 map50 的
                if map50 == 0:
                    for k, v in trainer.metrics.items():
                        if "map50" in str(k).lower() and "95" not in str(k).lower():
                            map50 = float(v)
                            break
                if map50 >= args.early_stop_map and not early_stop_triggered[0]:
                    early_stop_triggered[0] = True
                    trainer.stop_training = True
                    logger.info(
                        f"mAP@50={map50:.4f} >= {args.early_stop_map}, 提前停止训练!"
                    )

        def on_train_end(trainer):
            """早停后立即保存指标和模型"""
            if not early_stop_triggered[0]:
                return
            train_run_dir = Path(trainer.save_dir)
            # 保存当前指标
            metrics = extract_metrics_from_results(train_run_dir)
            metrics["experiment"] = args.exp_name
            metrics["model_variant"] = args.model
            metrics["epochs"] = trainer.epoch + 1  # 实际完成 epoch 数
            metrics["early_stopped"] = True
            save_json(metrics, exp_dir / "metrics.json")
            # 复制当前最佳模型
            best_pt = train_run_dir / "weights" / "best.pt"
            if best_pt.exists():
                shutil.copy2(best_pt, exp_dir / "best.pt")
                logger.info(f"最佳模型权重已复制: {exp_dir / 'best.pt'}")
            # 绘制训练曲线
            try:
                plot_training_curves(train_run_dir, exp_dir)
            except Exception as e:
                logger.warning(f"绘制训练曲线失败: {e}")

        # 需要同时监听 on_fit_epoch_end 和 on_train_epoch_end
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.add_callback("on_train_end", on_train_end)
        logger.info(f"早停机制已启用: mAP@50 >= {args.early_stop_map:.2f} 时停止")

    # 分段跑的截断：跑到第 stop_after 轮正常收尾（checkpoint 里仍是 args.epochs 轮）
    if args.stop_after > 0:
        if args.stop_after >= args.epochs and not resume_path:
            logger.warning(f"--stop_after({args.stop_after}) >= --epochs({args.epochs})，该设置无效")
        else:
            def _on_fit_epoch_end_seg(trainer):
                if int(getattr(trainer, "epoch", 0)) + 1 >= args.stop_after:
                    trainer.stop_training = True
                    logger.info(f"已到 --stop_after={args.stop_after} 轮，正常收尾停止；"
                                f"之后用 --resume 继续补完")

            model.add_callback("on_fit_epoch_end", _on_fit_epoch_end_seg)
            logger.info(f"分段模式：跑到第 {args.stop_after} 轮停止（目标总轮数 {args.epochs}）")

    logger.info(f"开始训练 (epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, "
                f"workers={args.workers}, resume={'是' if resume_path else '否'})...")
    results = model.train(
        data=data_yaml_abs,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=str(project_root / "runs" / "detect"),
        name=args.exp_name,
        exist_ok=True,
        # 显式传 checkpoint 绝对路径（True 会走 get_latest_run() 误续到别的实验）
        resume=str(resume_path) if resume_path else False,
        # 数据增强（YOLO11 默认为 True，显式设置确保启用）
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        translate=0.1,
    )

    # Step 4: 提取并保存指标
    train_run_dir = Path(results.save_dir)
    logger.info(f"训练结果目录: {train_run_dir}")

    metrics = extract_metrics_from_results(train_run_dir)
    logger.info(f"指标: mAP@50={metrics.get('mAP50', 'N/A')}, "
                f"mAP@50-95={metrics.get('mAP50-95', 'N/A')}")

    # 保存指标到实验目录
    metrics["experiment"] = args.exp_name
    metrics["model_variant"] = args.model
    metrics["epochs"] = args.epochs
    save_json(metrics, exp_dir / "metrics.json")

    # Step 5: 绘制训练曲线
    try:
        plot_training_curves(train_run_dir, exp_dir)
    except Exception as e:
        logger.warning(f"绘制训练曲线失败: {e}")

    # Step 6: 复制最佳模型权重
    best_pt = train_run_dir / "weights" / "best.pt"
    if best_pt.exists():
        dest_pt = exp_dir / "best.pt"
        shutil.copy2(best_pt, dest_pt)
        logger.info(f"最佳模型权重已复制: {dest_pt}")

    logger.info(f"训练完成! 实验: {args.exp_name}")
    logger.info(f"mAP@50 = {metrics.get('mAP50', 'N/A'):.4f}" if metrics.get('mAP50') else "mAP@50 数据不可用")


if __name__ == "__main__":
    main()
