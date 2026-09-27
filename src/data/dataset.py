"""数据集准备：从 data_source/smart school.v5i.yolov11/ 组织为 YOLO 标准格式"""

import shutil
from pathlib import Path

import yaml

from src.utils.helpers import get_project_root, logger


def _write_data_yaml(dest_root: Path) -> Path:
    """写入 data.yaml，并始终把 path 锚定为当前机器上的绝对真实路径

    YAML 里存的 path 是构建时机器的绝对路径，仓库换机器/换盘后即为死路径，
    因此每次进入准备流程都重写一次，保证克隆到任意路径都能直接训练。
    """
    data_yaml_path = dest_root / "data.yaml"
    data_yaml = {
        "path": str(dest_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": ["People using cellphone", "cellphone"],
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    return data_yaml_path


def prepare_dataset(force: bool = False) -> Path:
    """将 smart school 数据集复制到 data/phone_detection/ 并生成 data.yaml

    Args:
        force: 是否强制重新复制

    Returns:
        生成的数据集根目录路径 (data/phone_detection/)
    """
    project_root = get_project_root()
    source_root = project_root / "data_source" / "smart school.v5i.yolov11"
    dest_root = project_root / "data" / "phone_detection"

    if not source_root.exists():
        raise FileNotFoundError(f"数据源不存在: {source_root}")

    # 检查是否已经准备过（已就绪则只刷新 data.yaml 中的绝对路径）
    data_yaml_path = dest_root / "data.yaml"
    if data_yaml_path.exists() and not force:
        logger.info(f"数据集已就绪: {dest_root}")
        _write_data_yaml(dest_root)
        return dest_root

    # 映射: smart school 的 split 名 → YOLO 标准 split 名
    split_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    total_images = 0

    for src_split, dst_split in split_mapping.items():
        src_img_dir = source_root / src_split / "images"
        src_lbl_dir = source_root / src_split / "labels"
        dst_img_dir = dest_root / dst_split / "images"
        dst_lbl_dir = dest_root / dst_split / "labels"

        if not src_img_dir.exists():
            logger.warning(f"跳过 {src_split}: images 目录不存在")
            continue

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片
        for img_path in sorted(src_img_dir.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(img_path, dst_img_dir / img_path.name)
                total_images += 1

        # 复制标签
        if src_lbl_dir.exists():
            for lbl_path in sorted(src_lbl_dir.glob("*.txt")):
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

    # 生成 data.yaml（绝对路径按当前机器锚定）
    data_yaml_path = _write_data_yaml(dest_root)

    # 打印统计信息
    stats = {}
    for split in ["train", "val", "test"]:
        img_dir = dest_root / split / "images"
        lbl_dir = dest_root / split / "labels"
        if img_dir.exists():
            n_imgs = len(list(img_dir.glob("*")))
            n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            stats[split] = {"images": n_imgs, "labels": n_lbls}

    logger.info("=" * 50)
    logger.info("数据集准备完成!")
    logger.info(f"  数据源: {source_root}")
    logger.info(f"  目标:   {dest_root}")
    for split, s in stats.items():
        logger.info(f"  {split}: {s['images']} 张图片, {s['labels']} 个标注文件")
    logger.info(f"  类别: {['People using cellphone', 'cellphone']}")
    logger.info("=" * 50)

    return dest_root


def get_dataset_stats(dataset_root: Path) -> dict:
    """获取数据集统计信息"""
    dataset_root = Path(dataset_root)

    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    stats = {"classes": config["names"], "num_classes": config["nc"], "splits": {}}

    for split in ["train", "val", "test"]:
        img_dir = dataset_root / split / "images"
        if img_dir.exists():
            stats["splits"][split] = {
                "count": len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
            }

    return stats


def prepare_reorganized_dataset(force: bool = False) -> Path:
    """为重组数据集创建 data.yaml（数据已为 YOLO 格式，只需配置文件）

    Args:
        force: 是否强制重新生成 data.yaml

    Returns:
        数据集根目录路径
    """
    project_root = get_project_root()
    src = project_root / "data_source" / "reorganized_phone_dataset_yolo" / "reorganized_dataset"

    if not src.exists():
        raise FileNotFoundError(f"重组数据集不存在: {src}")

    data_yaml_path = src / "data.yaml"
    existed = data_yaml_path.exists()
    if existed and not force:
        logger.info(f"重组数据集 data.yaml 已存在，刷新绝对路径: {data_yaml_path}")
    _write_data_yaml(src)

    # 统计
    for split in ["train", "val", "test"]:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if img_dir.exists():
            n_imgs = len(list(img_dir.glob("*")))
            n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            logger.info(f"  {split}: {n_imgs} 张图片, {n_lbls} 个标注")

    logger.info(f"重组数据集 data.yaml 已生成: {data_yaml_path}")
    return src


if __name__ == "__main__":
    prepare_dataset()
    print(get_dataset_stats(get_project_root() / "data" / "phone_detection"))
