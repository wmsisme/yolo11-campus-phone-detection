#!/usr/bin/env node
/**
 * headless_probe.mjs —— 用无头 Edge + CDP 真跑一遍静态 Demo，等页面自检出结果。
 *
 * 为什么不用 `msedge --dump-dom`：DOM 在 load 事件后立刻 dump，而页面的自检要
 * 先下载 10 MB 模型、再跑 3 次 WebAssembly 推理（实测 1.9 秒就 dump 了，那时
 * 页面还停在"自检运行中…"）。所以这里用 DevTools Protocol 连上去**轮询**，
 * 直到 `window.__selftest` 被填充为止。
 *
 * 用法：
 *   node tests/headless_probe.mjs <url> [timeoutMs] [waitExpr] [--shot out.png]
 *     waitExpr 默认取 `window.__selftest`（页面自检）；正常路径可传
 *     `window.__demo`（每次推理完成后填充）。
 * 退出码：0 = 等到结果且 pass≠false，1 = 结果为失败，2 = 超时/环境错误
 * 输出：一行 JSON（含 pass / summary / lines / details / consoleErrors / dom 摘要）
 */
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const URL_ARG = process.argv[2];
const TIMEOUT_MS = Number(process.argv[3] || 240000);
const WAIT_EXPR = (process.argv[4] && !process.argv[4].startsWith('--'))
  ? process.argv[4]
  : 'window.__selftest';
const shotIdx = process.argv.indexOf('--shot');
const SHOT_PATH = shotIdx > 0 ? process.argv[shotIdx + 1] : null;
const sizeIdx = process.argv.indexOf('--size');
const VIEWPORT = sizeIdx > 0 ? process.argv[sizeIdx + 1] : null;   // 例如 1440x900
// 拍图前再等一会儿：`window.__demo` 填充的那一瞬，画面与报告可能还没渲染完，
// 直接截会拍到"结果已出但界面没画好"的中间态（做报告配图时踩过）。
const delayIdx = process.argv.indexOf('--delay');
const SHOT_DELAY_MS = delayIdx > 0 ? Number(process.argv[delayIdx + 1] || 1500) : 0;
const EDGE = process.env.EDGE_PATH
  || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

if (!URL_ARG) {
  console.error('usage: node headless_probe.mjs <url> [timeoutMs]');
  process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function launchEdge(port, profileDir) {
  return spawn(EDGE, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profileDir}`,
    'about:blank',
  ], { stdio: 'ignore' });
}

async function waitForEndpoint(port, deadline) {
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (res.ok) return true;
    } catch { /* 还没起来 */ }
    await sleep(300);
  }
  return false;
}

/** 极简 CDP 客户端：id 配对 + 事件收集。 */
class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.consoleErrors = [];
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
      } else if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
        this.consoleErrors.push(msg.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
      } else if (msg.method === 'Runtime.exceptionThrown') {
        const d = msg.params.exceptionDetails;
        this.consoleErrors.push('uncaught: ' + (d.exception?.description || d.text));
      }
    });
  }

  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout: ${method}`));
        }
      }, 120000);
    });
  }

  async evaluate(expression) {
    const r = await this.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    return r.result ? r.result.value : undefined;
  }
}

async function main() {
  const port = 9200 + Math.floor(Math.random() * 700);
  const profileDir = mkdtempSync(join(tmpdir(), 'demo-probe-'));
  const deadline = Date.now() + TIMEOUT_MS;
  const edge = launchEdge(port, profileDir);
  let ws;
  try {
    if (!(await waitForEndpoint(port, Date.now() + 20000))) {
      throw new Error('Edge DevTools 端口未就绪（Edge 是否安装？）');
    }
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page = targets.find((t) => t.type === 'page') || targets[0];
    if (!page) throw new Error('没有可用的页面 target');

    ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', reject, { once: true });
    });
    const cdp = new Cdp(ws);
    await cdp.send('Runtime.enable');
    await cdp.send('Page.enable');
    if (VIEWPORT) {
      const [w, h] = VIEWPORT.split('x').map(Number);
      await cdp.send('Emulation.setDeviceMetricsOverride', {
        width: w, height: h, deviceScaleFactor: 1, mobile: false,
      });
    }
    await cdp.send('Page.navigate', { url: URL_ARG });

    let result;
    while (Date.now() < deadline) {
      try {
        const raw = await cdp.evaluate(`${WAIT_EXPR} ? JSON.stringify(${WAIT_EXPR}) : ""`);
        if (raw) { result = JSON.parse(raw); break; }
      } catch { /* 导航过程中 evaluate 可能失败，重试 */ }
      await sleep(1000);
    }

    if (SHOT_PATH) {
      try {
        if (SHOT_DELAY_MS > 0) await sleep(SHOT_DELAY_MS);
        const shot = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
        writeFileSync(SHOT_PATH, Buffer.from(shot.data, 'base64'));
      } catch (e) {
        console.error('screenshot failed: ' + e.message);
      }
    }

    const dom = await cdp.evaluate(
      'JSON.stringify({title: document.title, box: (document.getElementById("selftest-result")||{}).textContent || "", meta: (document.getElementById("stageMeta")||{}).textContent || "", reportLen: ((document.getElementById("report")||{}).textContent || "").length, canvasReady: !!(document.getElementById("canvas")||{}).classList && document.getElementById("canvas").classList.contains("ready")})'
    ).catch(() => '{}');

    if (!result) {
      console.log(JSON.stringify({
        pass: false, timeout: true,
        summary: `TIMEOUT: ${WAIT_EXPR} 未在超时前填充`,
        consoleErrors: cdp.consoleErrors, dom: JSON.parse(dom || '{}'),
      }));
      return 2;
    }
    // 自检模式带 pass 字段；正常路径（window.__demo）只表示"推理已出结果"
    const ok = result.pass === undefined ? true : !!result.pass;
    console.log(JSON.stringify({
      pass: ok,
      summary: result.summary || (result.detections !== undefined
        ? `推理完成 ${result.detections} 个目标（${JSON.stringify(result.counts)}）` : ''),
      lines: result.lines || [],
      details: result.details || [],
      state: result,
      consoleErrors: cdp.consoleErrors,
      dom: JSON.parse(dom || '{}'),
    }));
    return ok ? 0 : 1;
  } finally {
    try { ws?.close(); } catch { /* ignore */ }
    edge.kill();
    await sleep(500);
    try { rmSync(profileDir, { recursive: true, force: true }); } catch { /* ignore */ }
  }
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.log(JSON.stringify({ pass: false, error: String(err && err.message || err) }));
    process.exit(2);
  });
