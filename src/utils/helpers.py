"""工具函数：项目根路径、日志、YAML/JSON 存取、计时器"""

import json
import logging
import time
import yaml
import numpy as np
import cv2
from pathlib import Path


def get_project_root() -> Path:
    """返回项目根目录"""
    return Path(__file__).resolve().parent.parent.parent


def imread_unicode(path, flags: int = cv2.IMREAD_COLOR):
    """读取图片，**支持中文/非 ASCII 路径**。

    为什么不用 `cv2.imread`：Windows 下 OpenCV 走的是窄字符路径，遇到含中文的目录
    （例如本机的 `D:\\code_item\\手机检测\\...`）会直接返回 None，报
    `can't open/read file: check file path/integrity`，而同一路径 `Path.exists()` 为 True
    —— 特别容易被误判成"数据集缺失"。改用 `np.fromfile` + `cv2.imdecode` 绕开该限制。

    用法与 `cv2.imread` 一致，失败同样返回 None。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_unicode(path, image) -> bool:
    """写入图片，**支持中文/非 ASCII 路径**（`imread_unicode` 的写入版）。

    ⚠️ 比 `imread` 更阴的一点：`cv2.imwrite` 在中文路径下**会返回 True 却什么都没写**
    （2026-10-01 实测：`cv2.imwrite('runs/_diag/_t_中文.png', img)` 返回 True，紧接
    `os.path.exists()` 为 False）。也就是说**连返回值都不可信**，不事后 stat 根本发现不了。
    故这里用 `cv2.imencode` + `tofile` 绕开，并显式回报是否真的写成功。
    """
    ok, buf = cv2.imencode(Path(path).suffix or ".png", image)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
    except OSError:
        return False
    return Path(path).exists()


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
