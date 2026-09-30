/* 校园手机使用检测 · 纯前端推理 Demo
 * ---------------------------------------------------------------------------
 * 这里没有后端：模型导出成 ONNX（见 src/models/export_onnx.py），
 * 由 onnxruntime-web（WebAssembly）在浏览器里执行。
 *
 * 前后端**同口径**是硬要求——本文件的 letterbox / 解码 / 按类别 NMS
 * 与 Python 参考实现 src/models/onnx_infer.py 逐步骤一一对应：
 *
 *   Python (onnx_infer.py)            JavaScript (本文件)
 *   --------------------------------  --------------------------------
 *   letterbox()                       letterboxToTensor()
 *   decode()  → cxcywh→xyxy           decode()
 *   nms() 逐类别贪心                  nmsPerClass()
 *
 * tests/test_webdemo.py 会用同一批图片把两侧结果逐框比对（IoU ≥ 0.9），
 * 页面上也可以带 ?selftest=1 自查（结果写入 #selftest-result）。
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------- 常量契约
  const IMGSZ = 640;      // 与导出时的 imgsz 一致
  const PAD_VALUE = 114;  // ultralytics letterbox 默认填充灰度
  const NUM_ANCHORS = 8400;
  const MAX_DET = 300;

  const CLASSES = [
    { id: 0, name: '使用手机的人', en: 'People using cellphone', color: '#1f6feb' },
    { id: 1, name: '手机', en: 'cellphone', color: '#e8590c' },
  ];

  // 模型清单：体积见仓库 docs/demo/model/，切换时按需下载（浏览器自动缓存）
  // 目前网页版只放 YOLO11n：yolo11s 的 ONNX 有 36 MB，公网首屏代价过大；
  // int8 静态量化会打坏这个模型的分类头（类别分数恒为 0，实测 4 种量化配置全部如此，
  // 详见 docs/exec-plans/tech-debt-tracker.md），所以不发行量化版本。
  const MODELS = [
    {
      id: 'n-fp32',
      label: 'YOLO11n',
      note: '10.1 MB · fp32',
      file: 'model/phone-yolo11n.onnx',
      metrics: 'mAP@50 0.729 · 早停于 61 epoch',
      recommended: true,
    },
  ];

  const SAMPLE_FILES = [
    'sample-classroom.jpg',
    'sample-classroom-2.jpg',
    'sample-corridor.jpg',
    'sample-classroom-wide.jpg',
    'sample-corridor-portrait.jpg',
  ];

  // ---------------------------------------------------------------- 全局状态
  const state = {
    session: null,        // ort.InferenceSession
    modelId: null,        // 当前模型 id
    modelCache: {},       // id → {session}
    image: null,          // HTMLImageElement
    imageName: '',
    raw: null,            // Float32Array：模型原始输出 (4+nc)*8400
    scale: 1,             // letterbox 缩放比
    pad: [0, 0],          // letterbox 灰边
    detections: [],       // 当前解码结果
    timings: {},
    enabled: [true, true],
    confThres: 0.25,
    iouThres: 0.7,
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    dropzone: $('dropzone'), fileInput: $('fileInput'), sampleList: $('sampleList'),
    confRange: $('confRange'), confVal: $('confVal'), iouRange: $('iouRange'), iouVal: $('iouVal'),
    cls0: $('cls0'), cls1: $('cls1'), modelList: $('modelList'), modelHint: $('modelHint'),
    stage: $('stage'), canvas: $('canvas'), emptyState: $('emptyState'), busy: $('busy'),
    busyText: $('busyText'), progressBar: $('progressBar'), stageMeta: $('stageMeta'),
    report: $('report'), btnMarkdown: $('btnMarkdown'), btnReset: $('btnReset'),
    runtimeInfo: $('runtimeInfo'), selftest: $('selftest-result'),
  };

  // ---------------------------------------------------------------- 工具函数
  function showBusy(text, ratio) {
    el.busy.classList.remove('hidden');
    el.busyText.textContent = text;
    if (typeof ratio === 'number') {
      el.progressBar.style.width = Math.round(Math.max(0, Math.min(1, ratio)) * 100) + '%';
    }
  }
  function hideBusy() {
    el.busy.classList.add('hidden');
    el.progressBar.style.width = '0%';
  }
  function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms'; }
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('图片加载失败: ' + src));
      img.src = src;
    });
  }

  // ---------------------------------------------------------------- 预处理
  /** 等比缩放 + 居中灰边，返回 {tensor, scale, pad}（对应 onnx_infer.letterbox/preprocess）。 */
  function letterboxToTensor(img) {
    const w = img.naturalWidth, h = img.naturalHeight;
    const scale = Math.min(IMGSZ / w, IMGSZ / h);
    const newW = Math.round(w * scale), newH = Math.round(h * scale);
    const pad = [(IMGSZ - newW) / 2, (IMGSZ - newH) / 2];

    const c = document.createElement('canvas');
    c.width = IMGSZ; c.height = IMGSZ;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = 'rgb(' + PAD_VALUE + ',' + PAD_VALUE + ',' + PAD_VALUE + ')';
    ctx.fillRect(0, 0, IMGSZ, IMGSZ);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, pad[0], pad[1], newW, newH);

    const px = ctx.getImageData(0, 0, IMGSZ, IMGSZ).data;
    const tensor = new Float32Array(3 * IMGSZ * IMGSZ);
    const area = IMGSZ * IMGSZ;
    for (let i = 0; i < area; i++) {
      const o = i * 4;
      tensor[i] = px[o] / 255;                 // R
      tensor[area + i] = px[o + 1] / 255;      // G
      tensor[area * 2 + i] = px[o + 2] / 255;  // B
    }
    return { tensor, scale, pad };
  }

  // ---------------------------------------------------------------- 后处理
  /** 按类别贪心 NMS，返回保留下来的下标（对应 onnx_infer.nms）。 */
  function nmsPerClass(boxes, scores, clsIds, iouThres) {
    const keep = [];
    const byClass = new Map();
    for (let i = 0; i < boxes.length; i++) {
      if (!byClass.has(clsIds[i])) byClass.set(clsIds[i], []);
      byClass.get(clsIds[i]).push(i);
    }
    for (const idxs of byClass.values()) {
      const order = idxs.slice().sort((a, b) => scores[b] - scores[a]);
      while (order.length) {
        const i = order.shift();
        keep.push(i);
        for (let k = order.length - 1; k >= 0; k--) {
          const j = order[k];
          const xx1 = Math.max(boxes[i][0], boxes[j][0]);
          const yy1 = Math.max(boxes[i][1], boxes[j][1]);
          const xx2 = Math.min(boxes[i][2], boxes[j][2]);
          const yy2 = Math.min(boxes[i][3], boxes[j][3]);
          const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
          const ai = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
          const aj = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]);
          const iou = inter / (ai + aj - inter + 1e-9);
          if (iou > iouThres) order.splice(k, 1);
        }
      }
    }
    return keep;
  }

  /** 原始输出 → 原图坐标系下的检测框（对应 onnx_infer.decode）。 */
  function decode(raw, scale, pad, imgW, imgH, confThres, iouThres, enabled) {
    const nc = CLASSES.length;
    const boxes = [], scores = [], clsIds = [];
    for (let i = 0; i < NUM_ANCHORS; i++) {
      let best = -1, bestScore = 0;
      for (let c = 0; c < nc; c++) {
        const s = raw[(4 + c) * NUM_ANCHORS + i];
        if (s > bestScore) { bestScore = s; best = c; }
      }
      if (best < 0 || bestScore < confThres) continue;
      if (enabled && enabled[best] === false) continue;
      const cx = raw[i], cy = raw[NUM_ANCHORS + i];
      const bw = raw[2 * NUM_ANCHORS + i], bh = raw[3 * NUM_ANCHORS + i];
      boxes.push([
        (cx - bw / 2 - pad[0]) / scale,
        (cy - bh / 2 - pad[1]) / scale,
        (cx + bw / 2 - pad[0]) / scale,
        (cy + bh / 2 - pad[1]) / scale,
      ]);
      scores.push(bestScore);
      clsIds.push(best);
    }
    if (!boxes.length) return [];

    const keep = nmsPerClass(boxes, scores, clsIds, iouThres);
    const dets = keep.map((i) => {
      const b = boxes[i];
      // 裁剪到图像范围（与 Python 侧 clip 一致）
      const xyxy = [
        Math.min(Math.max(b[0], 0), imgW),
        Math.min(Math.max(b[1], 0), imgH),
        Math.min(Math.max(b[2], 0), imgW),
        Math.min(Math.max(b[3], 0), imgH),
      ];
      return {
        cls: clsIds[i],
        clsName: CLASSES[clsIds[i]].name,
        clsColor: CLASSES[clsIds[i]].color,
        conf: scores[i],
        xyxy,
        areaRatio: ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (imgW * imgH),
      };
    });
    dets.sort((a, b) => b.conf - a.conf);
    return dets.slice(0, MAX_DET);
  }

  // ---------------------------------------------------------------- 模型加载
  /** 带进度条的模型下载（浏览器缓存命中时几乎瞬回）。 */
  async function fetchWithProgress(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('模型下载失败 HTTP ' + resp.status + '：' + url);
    const total = Number(resp.headers.get('content-length') || 0);
    if (!resp.body || !total) return new Uint8Array(await resp.arrayBuffer());
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received / total);
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    return buf;
  }

  async function ensureSession(model, onStatus) {
    if (state.modelCache[model.id]) return state.modelCache[model.id];
    onStatus('正在下载模型 ' + model.label + '（' + model.note + '）…', 0);
    const bytes = await fetchWithProgress(model.file, (r) => {
      el.busyText.textContent = '正在下载模型 ' + model.label + ' … ' + Math.round(r * 100) + '%';
      el.progressBar.style.width = Math.round(r * 92) + '%';
    });
    onStatus('正在初始化推理会话…', 0.95);
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    state.modelCache[model.id] = session;
    return session;
  }

  // ---------------------------------------------------------------- 推理主流程
  async function runInference(img, imageName) {
    const model = MODELS.find((m) => m.id === state.modelId) || MODELS[0];
    try {
      showBusy('准备中…', 0);
      const session = await ensureSession(model, (t, r) => showBusy(t, r));
      state.session = session;

      showBusy('预处理（letterbox + 归一化）…', 0.96);
      const t0 = performance.now();
      const { tensor, scale, pad } = letterboxToTensor(img);
      const tPre = performance.now() - t0;

      showBusy('模型推理中…（WebAssembly，单线程）', 0.97);
      const t1 = performance.now();
      const input = new ort.Tensor('float32', tensor, [1, 3, IMGSZ, IMGSZ]);
      const out = await session.run({ [session.inputNames[0]]: input });
      const raw = out[session.outputNames[0]].data;
      const tInf = performance.now() - t1;

      const t2 = performance.now();
      const dets = decode(raw, scale, pad, img.naturalWidth, img.naturalHeight,
        state.confThres, state.iouThres, state.enabled);
      const tDec = performance.now() - t2;

      state.image = img;
      state.imageName = imageName || 'image';
      state.raw = raw;
      state.scale = scale;
      state.pad = pad;
      state.detections = dets;
      state.timings = { pre: tPre, infer: tInf, decode: tDec };

      // 供自动化（tests/headless_probe.mjs）判断"这一轮推理真的出结果了"
      window.__demo = {
        ready: true,
        model: model.id,
        image: state.imageName,
        size: [img.naturalWidth, img.naturalHeight],
        detections: dets.length,
        counts: {
          [CLASSES[0].name]: dets.filter((d) => d.cls === 0).length,
          [CLASSES[1].name]: dets.filter((d) => d.cls === 1).length,
        },
        timings: { pre: Math.round(tPre), infer: Math.round(tInf), decode: Math.round(tDec) },
      };

      el.emptyState.style.display = 'none';
      el.canvas.classList.add('ready');
      draw();
      renderReport();
      el.btnMarkdown.disabled = false;
      el.btnReset.disabled = false;
      el.stageMeta.textContent =
        `${img.naturalWidth}×${img.naturalHeight} · 模型 ${model.label} · ` +
        `预处理 ${fmtMs(tPre)} · 推理 ${fmtMs(tInf)} · 后处理 ${fmtMs(tDec)}`;
      hideBusy();
      return { dets, timings: { pre: tPre, infer: tInf, decode: tDec } };
    } catch (err) {
      hideBusy();
      el.stageMeta.textContent = '出错了：' + err.message;
      console.error(err);
      throw err;
    }
  }

  /** 参数变化时不必重跑模型：原始输出已缓存，只重新解码。 */
  function reDecode() {
    if (!state.raw || !state.image) return;
    state.detections = decode(
      state.raw, state.scale, state.pad,
      state.image.naturalWidth, state.image.naturalHeight,
      state.confThres, state.iouThres, state.enabled
    );
    draw();
    renderReport();
  }

  // ---------------------------------------------------------------- 绘制
  function draw() {
    const img = state.image;
    if (!img) return;
    const canvas = el.canvas;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const lw = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 260));
    const fs = Math.max(12, Math.round(Math.min(canvas.width, canvas.height) / 38));
    ctx.lineWidth = lw;
    ctx.font = `600 ${fs}px "PingFang SC", "Microsoft YaHei", sans-serif`;
    ctx.textBaseline = 'top';

    state.detections.forEach((d, i) => {
      const [x1, y1, x2, y2] = d.xyxy;
      ctx.strokeStyle = d.clsColor;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const text = `${i + 1}. ${d.clsName} ${d.conf.toFixed(2)}`;
      const tw = ctx.measureText(text).width + lw * 3;
      const th = fs * 1.5;
      const ty = y1 - th < 0 ? y1 : y1 - th;
      ctx.fillStyle = d.clsColor;
      ctx.fillRect(x1, ty, tw, th);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, x1 + lw * 1.5, ty + fs * 0.24);
    });
  }

  // ---------------------------------------------------------------- 报告
  function renderReport() {
    const d = state.detections;
    const counts = [0, 0];
    d.forEach((x) => counts[x.cls]++);
    const t = state.timings;
    const model = MODELS.find((m) => m.id === state.modelId) || MODELS[0];

    if (!d.length) {
      el.report.innerHTML =
        `<p class="muted">这张图上没有超过阈值的目标。可以试试把<b>置信度阈值</b>调低，或换一张图。</p>` +
        timingHtml(t, model);
      return;
    }

    const cards = CLASSES.map((c) => `
      <div class="count-card">
        <div class="c-name"><i style="background:${c.color}"></i>${c.name}</div>
        <div class="c-num">${counts[c.id]}</div>
      </div>`).join('');

    // 置信度分布（5 档）
    const bins = [0, 0, 0, 0, 0];
    d.forEach((x) => { bins[Math.min(4, Math.floor((x.conf - 0.05) / 0.19))]++; });
    const maxBin = Math.max(1, ...bins);
    const labels = ['0.05–0.25', '0.25–0.45', '0.45–0.65', '0.65–0.85', '0.85–1.0'];
    const hist = bins.map((n, i) => `
      <div class="kv" style="display:flex;align-items:center;gap:6px">
        <span style="width:74px;font-variant-numeric:tabular-nums">${labels[i]}</span>
        <span style="flex:1;background:#eef2fa;border-radius:4px;height:9px;overflow:hidden">
          <span style="display:block;height:100%;width:${(n / maxBin) * 100}%;background:var(--ink-2)"></span>
        </span>
        <b style="width:18px;text-align:right">${n}</b>
      </div>`).join('');

    const rows = d.slice(0, 12).map((x, i) => `
      <tr>
        <td>${i + 1}</td>
        <td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${x.clsColor}"></span> ${x.clsName}</td>
        <td class="num">${x.conf.toFixed(3)}</td>
        <td class="num">${x.xyxy.map((v) => Math.round(v)).join(', ')}</td>
        <td class="num">${(x.areaRatio * 100).toFixed(1)}%</td>
      </tr>`).join('');

    el.report.innerHTML = `
      <div class="counts">${cards}</div>
      ${timingHtml(t, model)}
      <div style="margin-top:10px">
        <div class="kv" style="margin-bottom:4px"><b>置信度分布</b></div>
        ${hist}
      </div>
      <table class="det">
        <thead><tr><th>#</th><th>类别</th><th>置信度</th><th>位置 x1,y1,x2,y2</th><th>面积占比</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${d.length > 12 ? `<p class="muted">仅显示前 12 个框，完整明细见导出的 Markdown 报告（共 ${d.length} 个）。</p>` : ''}`;
  }

  function timingHtml(t, model) {
    if (!t || !t.infer) return '';
    return `<div class="kv">模型：<b>${model.label}</b>（${model.note}）</div>
      <div class="kv">预处理 <b>${fmtMs(t.pre)}</b> · 推理 <b>${fmtMs(t.infer)}</b> · 后处理 <b>${fmtMs(t.decode)}</b></div>
      <div class="kv">置信度阈值 <b>${state.confThres.toFixed(2)}</b> · NMS IoU <b>${state.iouThres.toFixed(2)}</b></div>`;
  }

  // ---------------------------------------------------------------- 导出报告
  function exportMarkdown() {
    const d = state.detections;
    const t = state.timings;
    const model = MODELS.find((m) => m.id === state.modelId) || MODELS[0];
    const counts = [0, 0];
    d.forEach((x) => counts[x.cls]++);
    const lines = [
      '# 校园手机使用检测报告', '',
      `- 图片：${state.imageName}（${state.image.naturalWidth}×${state.image.naturalHeight}）`,
      `- 模型：${model.label}（${model.note}）`,
      `- 参数：置信度阈值 ${state.confThres.toFixed(2)}，NMS IoU ${state.iouThres.toFixed(2)}`,
      `- 耗时：预处理 ${fmtMs(t.pre)} / 推理 ${fmtMs(t.infer)} / 后处理 ${fmtMs(t.decode)}`,
      `- 运行环境：浏览器内 ONNX Runtime Web（WebAssembly，单线程），图片未上传`,
      `- 生成时间：${new Date().toLocaleString('zh-CN')}`, '',
      '## 类别统计', '',
      '| 类别 | 数量 |', '|:--|--:|',
      ...CLASSES.map((c) => `| ${c.name}（${c.en}） | ${counts[c.id]} |`),
      `| **合计** | **${d.length}** |`, '',
      '## 检测明细', '',
      '| # | 类别 | 置信度 | x1 | y1 | x2 | y2 | 面积占比 |',
      '|--:|:--|--:|--:|--:|--:|--:|--:|',
      ...d.map((x, i) => `| ${i + 1} | ${x.clsName} | ${x.conf.toFixed(3)} | ` +
        x.xyxy.map((v) => Math.round(v)).join(' | ') + ` | ${(x.areaRatio * 100).toFixed(1)}% |`),
      '', '> 由「基于 YOLO11 的校园手机使用检测系统」在线 Demo 生成（纯前端推理）。',
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = '检测报告-' + state.imageName.replace(/\.[^.]+$/, '') + '.md';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ---------------------------------------------------------------- UI 绑定
  function buildModels() {
    el.modelList.innerHTML = MODELS.map((m) => `
      <label class="model-opt" data-id="${m.id}">
        <span>
          <span class="m-name">${m.label}${m.recommended ? ' ⭐' : ''}</span><br>
          <span class="m-meta">${m.note} · ${m.metrics}</span>
        </span>
        <input type="radio" name="model" value="${m.id}"${m.recommended ? ' checked' : ''} hidden>
      </label>`).join('');
    state.modelId = (MODELS.find((m) => m.recommended) || MODELS[0]).id;
    el.modelList.querySelectorAll('.model-opt').forEach((node) => {
      node.addEventListener('click', async () => {
        const id = node.dataset.id;
        if (id === state.modelId) return;
        state.modelId = id;
        el.modelList.querySelectorAll('.model-opt').forEach((n) => n.classList.toggle('active', n === node));
        node.querySelector('input').checked = true;
        if (state.image) await runInference(state.image, state.imageName);
      });
      node.classList.toggle('active', node.dataset.id === state.modelId);
    });
  }

  function buildSamples() {
    el.sampleList.innerHTML = SAMPLE_FILES
      .map((f) => `<img src="samples/${f}" alt="${f}" data-file="${f}" loading="lazy">`).join('');
    el.sampleList.querySelectorAll('img').forEach((node) => {
      node.addEventListener('click', async () => {
        el.sampleList.querySelectorAll('img').forEach((n) => n.classList.remove('active'));
        node.classList.add('active');
        const img = await loadImage('samples/' + node.dataset.file);
        await runInference(img, node.dataset.file);
      });
    });
  }

  function bindControls() {
    el.dropzone.addEventListener('click', () => el.fileInput.click());
    el.dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.fileInput.click(); }
    });
    ['dragenter', 'dragover'].forEach((ev) => el.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); el.dropzone.classList.add('over');
    }));
    ['dragleave', 'drop'].forEach((ev) => el.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); el.dropzone.classList.remove('over');
    }));
    el.dropzone.addEventListener('drop', (e) => {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) openFile(f);
    });
    el.fileInput.addEventListener('change', () => {
      if (el.fileInput.files[0]) openFile(el.fileInput.files[0]);
    });

    el.confRange.addEventListener('input', () => {
      state.confThres = Number(el.confRange.value);
      el.confVal.textContent = state.confThres.toFixed(2);
      reDecode();
    });
    el.iouRange.addEventListener('input', () => {
      state.iouThres = Number(el.iouRange.value);
      el.iouVal.textContent = state.iouThres.toFixed(2);
      reDecode();
    });
    el.cls0.addEventListener('change', () => { state.enabled[0] = el.cls0.checked; reDecode(); });
    el.cls1.addEventListener('change', () => { state.enabled[1] = el.cls1.checked; reDecode(); });
    el.btnMarkdown.addEventListener('click', exportMarkdown);
    el.btnReset.addEventListener('click', () => {
      state.image = null; state.raw = null; state.detections = [];
      el.canvas.classList.remove('ready');
      el.emptyState.style.display = '';
      el.stageMeta.textContent = '';
      el.btnMarkdown.disabled = true; el.btnReset.disabled = true;
      el.report.innerHTML = '<p class="muted">检测后这里会显示类别计数、置信度分布与每个框的明细。</p>';
      el.sampleList.querySelectorAll('img').forEach((n) => n.classList.remove('active'));
      el.fileInput.value = '';
    });
  }

  async function openFile(file) {
    if (!/^image\//.test(file.type)) {
      el.stageMeta.textContent = '请选择图片文件（JPG / PNG）。';
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImage(url);
      await runInference(img, file.name);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // ---------------------------------------------------------------- 自检模式
  /** ?selftest=1：用固定图片跑一遍，与 Python 参考结果逐框比对，结果写入 #selftest-result。 */
  async function runSelfTest() {
    const box = el.selftest;
    box.hidden = false;
    box.className = 'selftest show';
    box.textContent = '自检运行中…';
    const lines = [];
    const details = [];
    let pass = true;
    try {
      const spec = await (await fetch('selftest.json')).json();
      for (const modelSpec of spec.models) {
        const model = MODELS.find((m) => m.id === modelSpec.id);
        state.modelId = model.id;
        const session = await ensureSession(model, () => {});
        for (const cs of modelSpec.cases) {
          const img = await loadImage('samples/' + cs.image);
          state.session = session;
          const t0 = performance.now();
          const { tensor, scale, pad } = letterboxToTensor(img);
          const out = await session.run({
            [session.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, IMGSZ, IMGSZ]),
          });
          const raw = out[session.outputNames[0]].data;
          const ms = performance.now() - t0;
          const dets = decode(raw, scale, pad, img.naturalWidth, img.naturalHeight,
            cs.conf, cs.iou, [true, true]);

          const got = dets.length, want = cs.boxes.length;
          const ious = cs.boxes.map((b) => {
            let best = 0;
            for (const d of dets) {
              if (d.cls !== b.cls_id) continue;   // fixture 里的字段是 cls_id（见 onnx_infer.Detection.to_dict）
              const xx1 = Math.max(b.xyxy[0], d.xyxy[0]), yy1 = Math.max(b.xyxy[1], d.xyxy[1]);
              const xx2 = Math.min(b.xyxy[2], d.xyxy[2]), yy2 = Math.min(b.xyxy[3], d.xyxy[3]);
              const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
              const aa = (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]);
              const ab = (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]);
              best = Math.max(best, inter / (aa + ab - inter + 1e-9));
            }
            return best;
          });
          const minIou = ious.length ? Math.min(...ious) : 1;
          const ok = got === want && minIou >= spec.iouTolerance;
          if (!ok) pass = false;
          lines.push(`${ok ? 'PASS' : 'FAIL'} [${model.id}] ${cs.image}: ` +
            `boxes ${got}/${want}, minIoU ${minIou.toFixed(3)}, ${Math.round(ms)} ms`);
          details.push({
            model: model.id,
            image: cs.image,
            size: [img.naturalWidth, img.naturalHeight],
            got: dets.map((d) => ({
              cls: d.cls, conf: +d.conf.toFixed(3),
              xyxy: d.xyxy.map((v) => Math.round(v)),
            })),
            want: cs.boxes.map((b) => ({
              cls: b.cls_id, conf: b.conf,
              xyxy: b.xyxy.map((v) => Math.round(v)),
            })),
          });
        }
      }
    } catch (err) {
      pass = false;
      lines.push('ERROR ' + err.message);
    }
    const summary = (pass ? 'SELFTEST PASS' : 'SELFTEST FAIL') + ' | ' + lines.join(' | ');
    box.className = 'selftest show ' + (pass ? 'pass' : 'fail');
    box.textContent = summary;
    box.dataset.result = pass ? 'pass' : 'fail';
    window.__selftest = { pass, summary, lines, details };
    document.title = (pass ? '✅ ' : '❌ ') + 'selftest · ' + document.title;
  }

  // ---------------------------------------------------------------- 启动
  async function boot() {
    // 用**绝对 URL**：ORT 内部对 wasm 产物走动态 import，相对写法会被当成裸模块名解析失败
    // （实测报 "Failed to resolve module specifier 'vendor/ort/ort-wasm-simd-threaded.jsep.mjs'"）。
    // 另外只引入 ort.wasm.min.js（wasm-only 包）——全量包会额外去拉 WebGPU 版的 27 MB jsep 产物。
    ort.env.wasm.wasmPaths = new URL('vendor/ort/', location.href).href;
    ort.env.wasm.numThreads = 1;   // 静态托管没有 SharedArrayBuffer，必须单线程
    ort.env.wasm.simd = true;
    ort.env.logLevel = 'error';
    el.runtimeInfo.textContent = 'onnxruntime-web ' + (ort.env.versions ? ort.env.versions.web : '') +
      ' · WebAssembly 单线程';

    buildModels();
    buildSamples();
    bindControls();

    const params = new URLSearchParams(location.search);
    if (params.get('selftest') === '1') {
      await runSelfTest();
    } else if (params.get('sample')) {
      const img = await loadImage('samples/' + params.get('sample'));
      await runInference(img, params.get('sample'));
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
