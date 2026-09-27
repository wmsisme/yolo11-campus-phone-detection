"""工具函数：项目根路径、日志、YAML/JSON 存取、计时器"""

import json
import logging
import time
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """返回项目根目录"""
    return Path(__file__).resolve().parent.parent.parent


def setup_logger(name: str = "phone_detection", level: int = logging.INFO) -> logging.Logger:
    """创建并返回配置好的 logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


logger = setup_logger()


def save_json(data: dict, path: Path) -> None:
    """保存字典为 JSON 文件，自动创建父目录"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON 已保存: {path}")


def load_json(path: Path) -> dict:
    """从 JSON 文件加载字典"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: dict, path: Path) -> None:
    """保存字典为 YAML 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"YAML 已保存: {path}")


def load_yaml(path: Path) -> dict:
    """从 YAML 文件加载字典"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Timer:
    """简单的计时器上下文管理器"""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            logger.info(f"{self.name} 耗时: {self.elapsed:.3f}s")
