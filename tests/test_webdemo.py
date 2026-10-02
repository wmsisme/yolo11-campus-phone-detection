"""静态 Demo 的端到端测试（浏览器内真推理，不是"文件存在性"检查）。

为什么要这么重：这个 Demo 的价值全在「打开就能真的跑出结果」——它没有后端，
所以过去那种"服务能起来就算过"的测法完全测不到它。本测试的做法是：
起一个本地静态服务（模拟 GitHub Pages 的目录结构）→ 用无头 Edge 打开页面 →
等页面里的自检跑完 → 断言检测框与 Python 基准逐框一致（IoU ≥ 0.85）。

判据来自 `docs/demo/selftest.json`，它由 `src/models/gen_demo_fixtures.py` 用
Python 参考实现（`src/models/onnx_infer.py`）生成——所以这条测试真正验证的是
**"浏览器算出来的和 Python 算出来的是同一件事"**。

缺少 Node / Edge / 模型文件时整体 skip（CI 上没有浏览器也不该红）。
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import pathlib
import shutil
import subprocess
import threading

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
DEMO_DIR = DOCS_DIR / "demo"
PROBE = REPO_ROOT / "tests" / "headless_probe.mjs"

EDGE_CANDIDATES = [
    os.environ.get("EDGE_PATH", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/microsoft-edge",
    "/usr/bin/google-chrome",
]


def _edge_path() -> str | None:
    for c in EDGE_CANDIDATES:
        if c and pathlib.Path(c).is_file():
            return c
    return None


def _have_node() -> bool:
    return shutil.which("node") is not None


requires_browser = pytest.mark.skipif(
    _edge_path() is None or not _have_node(),
    reason="需要 Node 与 Edge/Chrome 才能跑浏览器内推理测试",
)
requires_assets = pytest.mark.skipif(
    not (DEMO_DIR / "model" / "phone-usage-yolo11s.onnx").is_file()
    or not (DEMO_DIR / "selftest.json").is_file(),
    reason="Demo 产物缺失（先跑 src.models.export_onnx 与 src.models.gen_demo_fixtures）",
)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # 静音，别把测试输出淹了
        pass


@pytest.fixture(scope="module")
def demo_server():
    """把 docs/ 当站点根目录起服务，Demo 位于 /demo/ —— 与 GitHub Pages 的布局一致。"""
    handler = functools.partial(_QuietHandler, directory=str(DOCS_DIR))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/demo/"
    httpd.shutdown()


def _run_probe(url: str, wait_expr: str = "window.__selftest", timeout_ms: int = 300000) -> dict:
    env = dict(os.environ)
    edge = _edge_path()
    if edge:
        env["EDGE_PATH"] = edge
    proc = subprocess.run(
        ["node", str(PROBE), url, str(timeout_ms), wait_expr],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=timeout_ms / 1000 + 60,
    )
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip().startswith("{")]
    assert lines, f"探针没有输出 JSON：\nstdout={proc.stdout}\nstderr={proc.stderr}"
    report = json.loads(lines[-1])
    report["__exit__"] = proc.returncode
    return report


@requires_assets
def test_demo_assets_present():
    """产物齐不齐（离线可查的那部分）。"""
    must = [
        DEMO_DIR / "index.html",
        DEMO_DIR / "app.js",
        DEMO_DIR / "style.css",
        DEMO_DIR / "model" / "phone-usage-yolo11s.onnx",
        DEMO_DIR / "selftest.json",
        DEMO_DIR / "vendor" / "ort" / "ort.wasm.min.js",
        DEMO_DIR / "vendor" / "ort" / "ort-wasm-simd-threaded.wasm",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in must if not p.is_file()]
    assert not missing, f"Demo 缺少文件：{missing}"

    samples = sorted((DEMO_DIR / "samples").glob("*.jpg"))
    assert len(samples) >= 3, f"示例图不足：{[s.name for s in samples]}"
    for s in samples:
        assert s.stat().st_size > 10_000, f"{s.name} 太小，可能是坏图"

    spec = json.loads((DEMO_DIR / "selftest.json").read_text(encoding="utf-8"))
    assert spec["models"] and all(m["cases"] for m in spec["models"]), "selftest.json 里没有基准用例"
    # 示例图必须都在 selftest 基准里，否则页面上的图没有回归兜底
    baselined = {c["image"] for m in spec["models"] for c in m["cases"]}
    assert {s.name for s in samples} <= baselined, f"示例图未纳入基准：{ {s.name for s in samples} - baselined }"


@requires_browser
@requires_assets
def test_page_has_no_js_errors_and_assets_resolve(demo_server):
    """打开页面（不跑自检）应当零 JS 报错，且模型/自检数据可被取到。"""
    report = _run_probe(demo_server + "selftest.json", "1", timeout_ms=60000)
    assert report["__exit__"] == 0
    assert report["consoleErrors"] == [], f"页面有 JS 报错：{report['consoleErrors']}"


@requires_browser
@requires_assets
def test_browser_inference_matches_python_baseline(demo_server):
    """核心断言：浏览器内推理结果 == Python 基准（逐框 IoU ≥ 0.85，框数一致）。"""
    report = _run_probe(demo_server + "?selftest=1")
    assert report["consoleErrors"] == [], f"页面有 JS 报错：{report['consoleErrors']}"
    assert report["pass"] is True, f"自检未通过：{report['summary']}"
    assert report["__exit__"] == 0
    for case in report["details"]:
        assert case["got"], f"{case['image']} 在浏览器里一个框都没检测到"
        assert len(case["got"]) == len(case["want"]), f"{case['image']} 框数不一致"


@requires_browser
@requires_assets
def test_normal_path_renders_report(demo_server):
    """正常路径（点示例图 → 推理 → 渲染报告）：出结果、有报告、无报错。"""
    report = _run_probe(demo_server + "?sample=sample-in-hand.jpg", "window.__demo", timeout_ms=120000)
    assert report["consoleErrors"] == [], f"页面有 JS 报错：{report['consoleErrors']}"
    state = report.get("state") or {}
    assert state.get("ready") is True, f"推理未完成：{state}"
    assert state.get("detections", 0) > 0, f"这张示例图本该检出目标：{state}"
    dom = report.get("dom") or {}
    assert dom.get("canvasReady") is True, "画布没有进入 ready 状态"
    assert dom.get("reportLen", 0) > 200, "右侧报告没渲染出内容"
    assert "推理" in (dom.get("meta") or ""), f"画布下方没有耗时信息：{dom.get('meta')}"
