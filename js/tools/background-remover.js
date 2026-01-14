(() => {
  'use strict';
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const TOOL_ID = 'background-remover';

  // This tool supports two removal modes:
  // - AI matting: high-quality mask (good for hair/fur) using an ISNet/RMBG-style model in the browser.
  // - Legacy color key: fast solid-background removal that preserves the old behavior as a fallback.
  //
  // Key best-practice change vs. the original implementation:
  // - We keep the *original* image for final export and only generate a smaller processing copy to compute the mask.
  //   The mask is then scaled up and applied at export time, preserving the original resolution.

  const form = $('#bgtool-form');
  const fileInput = $('#bgtool-file');
  const dropzone = $('#bgtool-dropzone');
  const fileList = $('#bgtool-filelist');
  const resultsEl = $('#bgtool-results');
  const pickButtons = $$('[data-bgtool="pick"]');

  const countEl = $('#bgtool-count');
  const totalEl = $('#bgtool-total');

  const methodSelect = $('#bgtool-method');
  const aiSettings = $('#bgtool-ai-settings');
  const colorkeySettings = $('#bgtool-colorkey-settings');
  const deviceSelect = $('#bgtool-device');
  const processingSelect = $('#bgtool-processing');
  const progressWrap = $('#bgtool-progress');
  const progressFill = $('#bgtool-progress-fill');
  const progressLabel = $('#bgtool-progress-label');

  const thresholdWrap = $('#bgtool-threshold-wrap');
  const thresholdInput = $('#bgtool-threshold');
  const thresholdValue = $('#bgtool-threshold-value');
  const featherInput = $('#bgtool-feather');
  const featherValue = $('#bgtool-feather-value');

  const formatSelect = $('#bgtool-format');
  const bgWrap = $('#bgtool-bg-wrap');
  const bgInput = $('#bgtool-bg');
  const bgLabel = $('#bgtool-bg-label');

  const colorInput = $('#bgtool-color');
  const colorLabel = $('#bgtool-color-label');
  const toleranceInput = $('#bgtool-tolerance');
  const toleranceValue = $('#bgtool-tolerance-value');

  const canvas = $('#bgtool-canvas');
  const overlay = $('#bgtool-overlay');
  const statusEl = $('#bgtool-status');
  const dimLabel = $('#bgtool-dim');
  const maskLabel = $('#bgtool-removed');
  const selectedLabel = $('#bgtool-selected');
  const viewButtons = $$('[data-bgtool-view]');

  const refineEnabledInput = $('#bgtool-refine-enabled');
  const brushModeSelect = $('#bgtool-brush-mode');
  const brushSizeInput = $('#bgtool-brush-size');
  const brushSizeValue = $('#bgtool-brush-size-value');
  const clearEditsBtn = $('#bgtool-clear-edits');

  const downloadSelectedBtn = $('#bgtool-download-selected');
  const downloadAllBtn = $('#bgtool-download-all');
  const resetBtn = $('#bgtool-reset');

  if (!form || !fileInput || !dropzone || !canvas || !resultsEl) return;

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const LIMITS = Object.freeze({
    maxFiles: 20,
    // We allow large images, but handle oversize uploads gracefully.
    maxFileBytes: 40 * 1024 * 1024,
    maxPixels: 80_000_000,
    recommendedPixels: 50_000_000,
    // Canvas maximums vary by browser; 16k is a conservative, common cap.
    maxCanvasDim: 16384,
  });

  const BG_REMOVAL_IMPORT_URL = 'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.7.0/+esm';
  let bgRemovalPromise = null;

  const state = {
    runId: 0,
    working: false,
    jobs: [],
    activeJobId: null,
    view: 'cutout',
    maskType: 'alpha',
    thresholdPct: 50,
    featherPx: 0,
    outputFormat: 'image/png',
    bgColor: '#ffffff',
  };

  const active = {
    job: null,
    // Processing-size sources used for preview and brush edits.
    sourceCanvas: document.createElement('canvas'),
    sourceCtx: null,
    maskCanvas: document.createElement('canvas'),
    maskCtx: null,
    derivedMaskCanvas: document.createElement('canvas'),
    derivedMaskCtx: null,
    dirtyMask: false,
    drawing: false,
    rafPending: false,
  };

  active.sourceCtx = active.sourceCanvas.getContext('2d', { alpha: false });
  active.maskCtx = active.maskCanvas.getContext('2d', { alpha: true });
  active.derivedMaskCtx = active.derivedMaskCanvas.getContext('2d', { alpha: true, willReadFrequently: true });

  const previewCtx = canvas.getContext('2d', { alpha: true });
  if (!previewCtx) return;

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const formatBytes = (bytes) => {
    const size = Math.max(0, Number(bytes) || 0);
    if (size < 1024) return `${size} B`;
    const units = ['KB', 'MB', 'GB'];
    let val = size / 1024;
    let unit = units[0];
    for (let i = 1; i < units.length && val >= 1024; i++) {
      val /= 1024;
      unit = units[i];
    }
    return `${val.toFixed(val >= 10 ? 0 : 1)} ${unit}`;
  };

  const getErrorMessage = (err) => {
    if (!err) return '';
    if (typeof err === 'string') return err;
    if (err instanceof Error) return err.message || '';
    return String(err?.message || err || '');
  };

  const formatAiInitHint = (message) => {
    const msg = String(message || '');
    const looksLikeCsp = /content security policy/i.test(msg) || /violates the following content security policy/i.test(msg);
    const looksLikeBlobImport = /failed to fetch dynamically imported module/i.test(msg) && /blob:/i.test(msg);
    const looksLikeBackend = /no available backend found/i.test(msg);
    if (looksLikeCsp || looksLikeBlobImport) {
      return "AI init blocked (likely CSP). Ensure `script-src blob:` + `worker-src blob:` + `'unsafe-eval'` + `'wasm-unsafe-eval'` are allowed, then hard-refresh. You can also switch to Solid color (legacy).";
    }
    if (looksLikeBackend) {
      return 'AI backend is unavailable. Try hard-refreshing, or use Solid color (legacy).';
    }
    return '';
  };

  const normalizeHex = (hex) => {
    const value = String(hex || '').trim();
    if (!value) return '#000000';
    if (value.startsWith('#') && value.length === 7) return value.toUpperCase();
    if (!value.startsWith('#') && value.length === 6) return `#${value}`.toUpperCase();
    return value.toUpperCase();
  };

  const hexToRgb = (hex) => {
    const cleaned = String(hex || '').replace('#', '').trim();
    const num = parseInt(cleaned || '000000', 16);
    return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
  };

  const fileNameBase = (name) => String(name || 'image').replace(/\.[a-z0-9]+$/i, '');
  const fileNameSafe = (name) => String(name || 'image').replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '') || 'image';

  const extForFormat = (format) => {
    switch (format) {
      case 'image/jpeg': return 'jpg';
      case 'image/webp': return 'webp';
      case 'image/tiff': return 'tif';
      case 'image/png':
      default: return 'png';
    }
  };

  const showOverlay = (msg) => {
    if (!overlay) return;
    overlay.textContent = msg || 'Click or drop photos to start';
    overlay.style.display = 'flex';
  };

  const hideOverlay = () => {
    if (overlay) overlay.style.display = 'none';
  };

  const setStatus = (msg) => {
    if (!statusEl) return;
    statusEl.textContent = String(msg || '');
  };

  const showProgress = (label, ratio) => {
    if (!progressWrap || !progressFill || !progressLabel) return;
    progressWrap.hidden = false;
    progressLabel.textContent = label || 'Loading…';
    const pct = Number.isFinite(ratio) ? clamp(ratio, 0, 1) * 100 : 0;
    progressFill.style.width = `${pct.toFixed(1)}%`;
  };

  const hideProgress = () => {
    if (!progressWrap || !progressFill || !progressLabel) return;
    progressWrap.hidden = true;
    progressFill.style.width = '0%';
    progressLabel.textContent = '';
  };

  const loadBgRemoval = async () => {
    if (!bgRemovalPromise) {
      // Lazy-load the AI library so the tool remains fast to open.
      bgRemovalPromise = import(BG_REMOVAL_IMPORT_URL);
    }
    return await bgRemovalPromise;
  };

  const createJob = (file) => ({
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    file,
    name: file?.name || 'image',
    bytes: file?.size || 0,
    status: 'queued',
    message: '',
    approved: true,
    original: { width: 0, height: 0, pixels: 0 },
    processing: { width: 0, height: 0, maxDim: 0, method: '' },
    // Results are stored as blobs + object URLs (small enough to keep for batch previews).
    // We keep an immutable "base" mask so users can revert brush edits.
    baseMaskBlob: null,
    baseCutoutBlob: null,
    maskBlob: null,
    maskUrl: '',
    cutoutBlob: null,
    cutoutUrl: '',
  });

  const revokeJobUrls = (job) => {
    if (!job) return;
    if (job.maskUrl) URL.revokeObjectURL(job.maskUrl);
    if (job.cutoutUrl) URL.revokeObjectURL(job.cutoutUrl);
    job.maskUrl = '';
    job.cutoutUrl = '';
  };

  const updateSummary = () => {
    const totalFiles = state.jobs.length;
    const totalBytes = state.jobs.reduce((sum, j) => sum + (j.bytes || 0), 0);
    if (countEl) countEl.textContent = `${totalFiles} file${totalFiles === 1 ? '' : 's'}`;
    if (totalEl) totalEl.textContent = `Total: ${formatBytes(totalBytes)}`;
  };

  const updateMethodVisibility = () => {
    const method = String(methodSelect?.value || 'ai-best');
    const aiOn = method.startsWith('ai-');
    if (aiSettings) aiSettings.hidden = !aiOn;
    if (colorkeySettings) colorkeySettings.hidden = aiOn;
    markSessionDirty();
  };

  const updateMaskControls = () => {
    const maskType = state.maskType;
    if (thresholdWrap) thresholdWrap.hidden = maskType !== 'binary';
    if (thresholdValue) thresholdValue.textContent = `${state.thresholdPct}%`;
    if (featherValue) featherValue.textContent = `${state.featherPx} px`;
  };

  const updateOutputControls = () => {
    const fmt = String(formatSelect?.value || 'image/png');
    state.outputFormat = fmt;
    const needsBg = fmt === 'image/jpeg' || fmt === 'image/webp-solid';
    if (bgWrap) bgWrap.hidden = !needsBg;
    markSessionDirty();
  };

  const updateRefineControls = () => {
    const canEdit = !!active.job && active.job.status === 'ready';
    const enabled = !!(refineEnabledInput && refineEnabledInput.checked);
    if (refineEnabledInput) refineEnabledInput.disabled = !canEdit;
    if (brushModeSelect) brushModeSelect.disabled = !canEdit || !enabled;
    if (brushSizeInput) brushSizeInput.disabled = !canEdit || !enabled;
    if (clearEditsBtn) clearEditsBtn.disabled = !canEdit;
    canvas.style.cursor = (canEdit && enabled && state.view === 'cutout') ? 'crosshair' : 'default';
  };

  const updateActionButtons = () => {
    const activeReady = !!active.job && active.job.status === 'ready';
    if (downloadSelectedBtn) downloadSelectedBtn.disabled = !activeReady;
    const anyApproved = state.jobs.some((j) => j.status === 'ready' && j.approved);
    if (downloadAllBtn) downloadAllBtn.disabled = !anyApproved;
    viewButtons.forEach((btn) => {
      btn.disabled = !activeReady;
      btn.classList.toggle('is-active', activeReady && btn.dataset.bgtoolView === state.view);
    });
  };

  const renderFileList = () => {
    if (!fileList) return;
    const rows = state.jobs.map((job) => {
      const status = job.status;
      const metaParts = [];
      if (job.original.width && job.original.height) metaParts.push(`${job.original.width}×${job.original.height}`);
      metaParts.push(formatBytes(job.bytes));
      const meta = metaParts.join(' · ');
      const disabled = status === 'processing' ? 'disabled' : '';
      const statusLabel = status === 'queued'
        ? 'Queued'
        : status === 'processing'
          ? 'Processing…'
          : status === 'ready'
            ? 'Ready'
            : 'Error';
      const message = job.message ? `<p class="bgtool-item-message">${job.message}</p>` : '';
      return `
        <li class="bgtool-item ${status === 'error' ? 'is-error' : ''}" data-bgtool-id="${job.id}">
          <div class="bgtool-item-main">
            <div class="bgtool-item-title">${job.name}</div>
            <div class="bgtool-item-meta">${meta}</div>
            ${message}
          </div>
          <div class="bgtool-item-actions">
            <span class="bgtool-pill">${statusLabel}</span>
            <button type="button" class="btn-ghost" data-bgtool-remove="${job.id}" ${disabled}>Remove</button>
          </div>
        </li>
      `;
    });
    fileList.innerHTML = rows.join('');
  };

  const renderResults = () => {
    const cards = state.jobs
      .filter((job) => job.status === 'ready' || job.status === 'error')
      .map((job) => {
        const selected = job.id === state.activeJobId;
        const thumb = job.cutoutUrl
          ? `<img class="bgtool-result-thumb" src="${job.cutoutUrl}" alt="Cutout preview for ${job.name}">`
          : `<div class="bgtool-result-thumb bgtool-result-thumb-empty" aria-hidden="true"></div>`;
        const rerun = `<button type="button" class="btn-secondary bgtool-result-rerun" data-bgtool-rerun="${job.id}">Run again</button>`;
        const dims = job.original.width && job.original.height ? `${job.original.width}×${job.original.height}` : '—';
        const status = job.status === 'error' ? `<span class="bgtool-pill bgtool-pill-error">Error</span>` : `<span class="bgtool-pill bgtool-pill-ok">Ready</span>`;
        const checked = job.approved ? 'checked' : '';
        const message = job.message ? `<p class="bgtool-result-message">${job.message}</p>` : '';
        return `
          <article class="bgtool-result ${selected ? 'is-selected' : ''}" data-bgtool-select="${job.id}">
            <div class="bgtool-result-media">
              ${thumb}
              <div class="bgtool-result-media-actions">
                ${rerun}
              </div>
            </div>
            <div class="bgtool-result-body">
              <div class="bgtool-result-head">
                <div>
                  <h3 class="bgtool-result-title">${job.name}</h3>
                  <div class="bgtool-result-meta">${dims} · ${formatBytes(job.bytes)}</div>
                </div>
                ${status}
              </div>
              ${message}
              <label class="bgtool-option bgtool-approve">
                <input type="checkbox" data-bgtool-approve="${job.id}" ${checked} ${job.status !== 'ready' ? 'disabled' : ''}>
                Approve for download
              </label>
            </div>
          </article>
        `;
      });

    resultsEl.innerHTML = cards.join('') || '<p class="bgtool-empty">Processed results will appear here.</p>';
  };

  const readImageBitmap = async (file) => {
    // Attempt to respect EXIF orientation without requiring any additional libraries.
    // (Not all browsers support the imageOrientation option; we fall back safely.)
    const blob = file instanceof Blob ? file : new Blob([await file.arrayBuffer()]);
    try {
      return await createImageBitmap(blob, { imageOrientation: 'from-image' });
    } catch {
      return await createImageBitmap(blob);
    }
  };

  const drawContain = (ctx, bitmap, targetW, targetH) => {
    const sw = bitmap.width;
    const sh = bitmap.height;
    const scale = Math.min(targetW / sw, targetH / sh);
    const dw = Math.max(1, Math.round(sw * scale));
    const dh = Math.max(1, Math.round(sh * scale));
    const dx = Math.round((targetW - dw) / 2);
    const dy = Math.round((targetH - dh) / 2);
    ctx.clearRect(0, 0, targetW, targetH);
    ctx.drawImage(bitmap, dx, dy, dw, dh);
    return { drawW: dw, drawH: dh, drawX: dx, drawY: dy, scale };
  };

  const toInt = (value, fallback) => {
    const num = Number.parseInt(String(value), 10);
    return Number.isFinite(num) ? num : fallback;
  };

  const computeProcessingSize = (width, height, maxDim) => {
    const safeMax = clamp(toInt(maxDim, 1024), 512, 4096);
    const scale = Math.min(1, safeMax / Math.max(width, height));
    return {
      width: Math.max(1, Math.round(width * scale)),
      height: Math.max(1, Math.round(height * scale)),
      maxDim: safeMax,
    };
  };

  const canvasToBlob = (srcCanvas, mime, quality) => new Promise((resolve, reject) => {
    try {
      srcCanvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error('Unable to export image (browser blocked).'));
          return;
        }
        resolve(blob);
      }, mime, quality);
    } catch (err) {
      reject(err);
    }
  });

  const deriveMaskCanvas = (baseMaskCanvas, { targetW, targetH, maskType, thresholdPct, featherPx, blurScale }) => {
    if (!active.derivedMaskCtx) return baseMaskCanvas;
    const w = targetW || baseMaskCanvas.width;
    const h = targetH || baseMaskCanvas.height;

    active.derivedMaskCanvas.width = w;
    active.derivedMaskCanvas.height = h;
    const ctx = active.derivedMaskCtx;
    ctx.clearRect(0, 0, w, h);

    // Feather: blur the mask in *output pixels*. For preview, we scale the blur down.
    const blurPx = clamp((Number(featherPx) || 0) * (Number(blurScale) || 1), 0, 40);
    ctx.filter = blurPx > 0 ? `blur(${blurPx}px)` : 'none';
    ctx.drawImage(baseMaskCanvas, 0, 0, w, h);
    ctx.filter = 'none';

    if (maskType === 'binary') {
      // Create a binary mask from the alpha matte.
      const threshold = clamp(Number(thresholdPct) || 50, 1, 99) / 100;
      const img = ctx.getImageData(0, 0, w, h);
      const data = img.data;
      for (let i = 0; i < data.length; i += 4) {
        const a = data[i + 3] / 255;
        data[i] = 255;
        data[i + 1] = 255;
        data[i + 2] = 255;
        data[i + 3] = a >= threshold ? 255 : 0;
      }
      ctx.putImageData(img, 0, 0);
    }

    return active.derivedMaskCanvas;
  };

  const renderActivePreview = () => {
    if (!active.job || active.job.status !== 'ready') {
      previewCtx.clearRect(0, 0, canvas.width, canvas.height);
      showOverlay('Click or drop photos to start');
      updateActionButtons();
      updateRefineControls();
      return;
    }

    const job = active.job;
    hideOverlay();

    const w = active.sourceCanvas.width;
    const h = active.sourceCanvas.height;
    canvas.width = w;
    canvas.height = h;
    previewCtx.clearRect(0, 0, w, h);

    const originalW = job.original.width || 0;
    const originalH = job.original.height || 0;
    if (dimLabel && originalW && originalH) dimLabel.textContent = `Size: ${originalW} × ${originalH}`;
    if (selectedLabel) selectedLabel.textContent = `Selected: ${job.name}`;

    // Preview blur should be scaled down vs. the final export, because the preview canvas is smaller.
    const exportScale = Math.max(1, (originalW || w) / w, (originalH || h) / h);
    const previewBlurScale = 1 / exportScale;
    const maskForPreview = deriveMaskCanvas(active.maskCanvas, {
      targetW: w,
      targetH: h,
      maskType: state.maskType,
      thresholdPct: state.thresholdPct,
      featherPx: state.featherPx,
      blurScale: previewBlurScale,
    });

    if (state.view === 'original') {
      previewCtx.drawImage(active.sourceCanvas, 0, 0);
      if (maskLabel) maskLabel.textContent = 'Mask: (hidden)';
      updateRefineControls();
      updateActionButtons();
      return;
    }

    if (state.view === 'mask') {
      previewCtx.fillStyle = '#0B0F14';
      previewCtx.fillRect(0, 0, w, h);
      previewCtx.drawImage(maskForPreview, 0, 0);
      if (maskLabel) maskLabel.textContent = `Mask: ${state.maskType === 'binary' ? 'binary' : 'alpha matte'}`;
      updateRefineControls();
      updateActionButtons();
      return;
    }

    // Cutout view: apply the (possibly edited) mask to the processing-size source.
    previewCtx.drawImage(active.sourceCanvas, 0, 0);
    previewCtx.globalCompositeOperation = 'destination-in';
    previewCtx.drawImage(maskForPreview, 0, 0);
    previewCtx.globalCompositeOperation = 'source-over';
    if (maskLabel) maskLabel.textContent = `Mask: ${state.maskType === 'binary' ? 'binary' : 'alpha matte'} · Feather ${state.featherPx}px`;

    updateRefineControls();
    updateActionButtons();
  };

  const schedulePreviewRender = () => {
    if (active.rafPending) return;
    active.rafPending = true;
    window.requestAnimationFrame(() => {
      active.rafPending = false;
      renderActivePreview();
    });
  };

  const commitActiveMask = async () => {
    if (!active.job || !active.dirtyMask) return;
    const job = active.job;
    const newMaskBlob = await canvasToBlob(active.maskCanvas, 'image/png');
    if (job.maskBlob && job.maskUrl) URL.revokeObjectURL(job.maskUrl);
    job.maskBlob = newMaskBlob;
    job.maskUrl = URL.createObjectURL(newMaskBlob);

    // Refresh the stored cutout thumbnail so batch previews match the edited mask.
    try {
      const cutoutCanvas = document.createElement('canvas');
      cutoutCanvas.width = job.processing.width;
      cutoutCanvas.height = job.processing.height;
      const cutoutCtx = cutoutCanvas.getContext('2d', { alpha: true });
      if (cutoutCtx) {
        cutoutCtx.drawImage(active.sourceCanvas, 0, 0);
        cutoutCtx.globalCompositeOperation = 'destination-in';
        cutoutCtx.drawImage(active.maskCanvas, 0, 0);
        cutoutCtx.globalCompositeOperation = 'source-over';
        const newCutoutBlob = await canvasToBlob(cutoutCanvas, 'image/png');
        if (job.cutoutBlob && job.cutoutUrl) URL.revokeObjectURL(job.cutoutUrl);
        job.cutoutBlob = newCutoutBlob;
        job.cutoutUrl = URL.createObjectURL(newCutoutBlob);
      }
    } catch {}

    active.dirtyMask = false;
    markSessionDirty();
  };

  const loadJobIntoActive = async (job, currentRunId) => {
    active.job = job;
    active.dirtyMask = false;
    active.drawing = false;

    if (!job || job.status !== 'ready') {
      schedulePreviewRender();
      return;
    }

    // Build a processing-size source canvas for previews and edits.
    const bitmap = await readImageBitmap(job.file);
    if (currentRunId !== state.runId) {
      if (bitmap && typeof bitmap.close === 'function') bitmap.close();
      return;
    }

    active.sourceCanvas.width = job.processing.width;
    active.sourceCanvas.height = job.processing.height;
    drawContain(active.sourceCtx, bitmap, job.processing.width, job.processing.height);

    if (bitmap && typeof bitmap.close === 'function') bitmap.close();

    // Load the latest saved mask (including edits) into an editable canvas.
    active.maskCanvas.width = job.processing.width;
    active.maskCanvas.height = job.processing.height;
    active.maskCtx.clearRect(0, 0, job.processing.width, job.processing.height);
    if (job.maskBlob) {
      const maskBitmap = await createImageBitmap(job.maskBlob);
      if (currentRunId !== state.runId) {
        if (maskBitmap && typeof maskBitmap.close === 'function') maskBitmap.close();
        return;
      }
      active.maskCtx.drawImage(maskBitmap, 0, 0, job.processing.width, job.processing.height);
      if (maskBitmap && typeof maskBitmap.close === 'function') maskBitmap.close();
    }

    schedulePreviewRender();
  };

  const selectJob = async (jobId) => {
    await commitActiveMask();
    state.activeJobId = jobId;
    const job = state.jobs.find((j) => j.id === jobId) || null;
    if (!job) return;
    const currentRunId = state.runId;
    await loadJobIntoActive(job, currentRunId);
    markSessionDirty();
  };

  const ensureCanvasLimits = (width, height, label) => {
    const maxDim = LIMITS.maxCanvasDim;
    if (width > maxDim || height > maxDim) {
      throw new Error(`${label || 'Image'} is too large for this browser (max dimension ≈ ${maxDim}px).`);
    }
  };

  const buildExportCanvas = async (job, { bgFillCss, exportMaskType }) => {
    const bitmap = await readImageBitmap(job.file);
    const w = bitmap.width;
    const h = bitmap.height;
    ensureCanvasLimits(w, h, 'Export');

    // Rebuild an export mask at the *original* resolution by scaling up the edited mask.
    // This preserves original resolution while keeping mask generation efficient.
    const maskScale = Math.max(1, w / job.processing.width, h / job.processing.height);
    const exportMaskCanvas = document.createElement('canvas');
    exportMaskCanvas.width = w;
    exportMaskCanvas.height = h;
    const exportMaskCtx = exportMaskCanvas.getContext('2d', { alpha: true });
    if (!exportMaskCtx) throw new Error('Unable to create export canvas.');

    exportMaskCtx.clearRect(0, 0, w, h);

    if (exportMaskType === 'binary') {
      // Avoid thresholding at full resolution (can be huge memory for 50MP images).
      // Instead, threshold at processing resolution, then scale up with nearest-neighbor
      // so the export stays binary without allocating a giant ImageData buffer.
      const binaryBaseMask = deriveMaskCanvas(active.maskCanvas, {
        targetW: job.processing.width,
        targetH: job.processing.height,
        maskType: 'binary',
        thresholdPct: state.thresholdPct,
        featherPx: state.featherPx,
        blurScale: 1 / maskScale,
      });
      exportMaskCtx.imageSmoothingEnabled = false;
      exportMaskCtx.drawImage(binaryBaseMask, 0, 0, w, h);
      exportMaskCtx.imageSmoothingEnabled = true;
    } else {
      const maskForExport = deriveMaskCanvas(active.maskCanvas, {
        targetW: w,
        targetH: h,
        maskType: 'alpha',
        thresholdPct: state.thresholdPct,
        featherPx: state.featherPx,
        blurScale: 1,
      });
      exportMaskCtx.drawImage(maskForExport, 0, 0, w, h);
    }

    // Create cutout via canvas compositing (no pixel loops) so large images stay manageable.
    const cutoutCanvas = document.createElement('canvas');
    cutoutCanvas.width = w;
    cutoutCanvas.height = h;
    const cutoutCtx = cutoutCanvas.getContext('2d', { alpha: true });
    if (!cutoutCtx) throw new Error('Unable to create export canvas.');
    cutoutCtx.drawImage(bitmap, 0, 0);
    cutoutCtx.globalCompositeOperation = 'destination-in';
    cutoutCtx.drawImage(exportMaskCanvas, 0, 0);
    cutoutCtx.globalCompositeOperation = 'source-over';

    // Optional solid background fill (for JPEG / solid WebP).
    if (bgFillCss) {
      const outCanvas = document.createElement('canvas');
      outCanvas.width = w;
      outCanvas.height = h;
      const outCtx = outCanvas.getContext('2d', { alpha: true });
      if (!outCtx) throw new Error('Unable to create export canvas.');
      outCtx.fillStyle = bgFillCss;
      outCtx.fillRect(0, 0, w, h);
      outCtx.drawImage(cutoutCanvas, 0, 0);
      if (bitmap && typeof bitmap.close === 'function') bitmap.close();
      return outCanvas;
    }

    if (bitmap && typeof bitmap.close === 'function') bitmap.close();
    return cutoutCanvas;
  };

  const ensureUtif = async () => {
    if (window.UTIF && typeof window.UTIF.encodeImage === 'function') return window.UTIF;

    // Lazy-load UTIF only when the user selects TIFF export to keep the tool lightweight.
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/utif@3.1.0/UTIF.min.js';
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load TIFF encoder.'));
      document.head.appendChild(script);
    });

    if (!window.UTIF || typeof window.UTIF.encodeImage !== 'function') {
      throw new Error('TIFF encoder is unavailable.');
    }
    return window.UTIF;
  };

  const exportBlobForJob = async (job) => {
    const fmt = state.outputFormat;
    const wantsSolid = fmt === 'image/jpeg' || fmt === 'image/webp-solid';
    const bgFillCss = wantsSolid ? normalizeHex(bgInput?.value || state.bgColor) : '';
    const mime = fmt === 'image/webp-solid' ? 'image/webp' : fmt;

    // Ensure the active mask is saved before exporting (so brush edits are included).
    if (active.job && active.job.id === job.id) await commitActiveMask();

    // If exporting a job that's not currently active, load its mask into the active editor.
    // This keeps mask derivation consistent for both preview and exports.
    if (!active.job || active.job.id !== job.id) {
      await loadJobIntoActive(job, state.runId);
    }

    const exportCanvas = await buildExportCanvas(job, {
      bgFillCss,
      exportMaskType: state.maskType,
    });

    if (mime === 'image/tiff') {
      const UTIF = await ensureUtif();
      const ctx = exportCanvas.getContext('2d', { alpha: true, willReadFrequently: true });
      if (!ctx) throw new Error('Unable to create TIFF image.');
      const img = ctx.getImageData(0, 0, exportCanvas.width, exportCanvas.height);
      const tiff = UTIF.encodeImage(img.data.buffer, exportCanvas.width, exportCanvas.height);
      return new Blob([tiff], { type: 'image/tiff' });
    }

    const quality = (mime === 'image/jpeg' || mime === 'image/webp') ? 0.92 : undefined;
    return await canvasToBlob(exportCanvas, mime, quality);
  };

  const triggerDownload = (blob, filename) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 2000);
  };

  const downloadSelected = async () => {
    if (!active.job || active.job.status !== 'ready') return;
    const job = active.job;
    setStatus('Preparing download…');
    try {
      const blob = await exportBlobForJob(job);
      const ext = extForFormat(state.outputFormat === 'image/webp-solid' ? 'image/webp' : state.outputFormat);
      const name = `${fileNameSafe(fileNameBase(job.name))}-cutout.${ext}`;
      triggerDownload(blob, name);
      setStatus('Download started.');
    } catch (err) {
      setStatus(err?.message || 'Unable to export that file.');
    } finally {
      updateActionButtons();
    }
  };

  const downloadApproved = async () => {
    const jobs = state.jobs.filter((j) => j.status === 'ready' && j.approved);
    if (!jobs.length) return;
    setStatus(`Preparing ${jobs.length} download${jobs.length === 1 ? '' : 's'}…`);
    state.working = true;
    updateActionButtons();
    try {
      for (let i = 0; i < jobs.length; i++) {
        const blob = await exportBlobForJob(jobs[i]);
        const ext = extForFormat(state.outputFormat === 'image/webp-solid' ? 'image/webp' : state.outputFormat);
        const name = `${fileNameSafe(fileNameBase(jobs[i].name))}-cutout.${ext}`;
        triggerDownload(blob, name);
        await new Promise((r) => setTimeout(r, 180));
      }
      setStatus('Download started. If prompted, allow multiple downloads.');
    } catch (err) {
      setStatus(err?.message || 'Unable to export one of the approved images.');
    } finally {
      state.working = false;
      updateActionButtons();
    }
  };

  const processJobLegacy = async (job, bitmap, currentRunId) => {
    const target = hexToRgb(colorInput?.value || '#ffffff');
    const tolerance = clamp(toInt(toleranceInput?.value, 24), 0, 255);

    const maxDim = toInt(processingSelect?.value, 1024);
    const processing = computeProcessingSize(bitmap.width, bitmap.height, maxDim);
    job.processing.width = processing.width;
    job.processing.height = processing.height;
    job.processing.maxDim = processing.maxDim;
    job.processing.method = 'colorkey';

    active.sourceCanvas.width = processing.width;
    active.sourceCanvas.height = processing.height;
    drawContain(active.sourceCtx, bitmap, processing.width, processing.height);

    // Create a binary mask by measuring distance to the target background color.
    const img = active.sourceCtx.getImageData(0, 0, processing.width, processing.height);
    const data = img.data;
    const mask = document.createElement('canvas');
    mask.width = processing.width;
    mask.height = processing.height;
    const maskCtx = mask.getContext('2d', { alpha: true });
    if (!maskCtx) throw new Error('Unable to create mask.');
    const out = maskCtx.createImageData(processing.width, processing.height);
    const outData = out.data;

    let removed = 0;
    for (let i = 0; i < data.length; i += 4) {
      const dr = data[i] - target.r;
      const dg = data[i + 1] - target.g;
      const db = data[i + 2] - target.b;
      const dist = Math.sqrt(dr * dr + dg * dg + db * db);
      const keep = dist > tolerance;
      outData[i] = 255;
      outData[i + 1] = 255;
      outData[i + 2] = 255;
      outData[i + 3] = keep ? 255 : 0;
      if (!keep) removed++;
    }
    maskCtx.putImageData(out, 0, 0);

    if (currentRunId !== state.runId) return;

    const maskBlob = await canvasToBlob(mask, 'image/png');
    job.baseMaskBlob = maskBlob;
    job.maskBlob = maskBlob;
    job.maskUrl = URL.createObjectURL(maskBlob);

    // Store a quick cutout preview.
    const cutoutCanvas = document.createElement('canvas');
    cutoutCanvas.width = processing.width;
    cutoutCanvas.height = processing.height;
    const cutoutCtx = cutoutCanvas.getContext('2d', { alpha: true });
    if (!cutoutCtx) throw new Error('Unable to create cutout.');
    cutoutCtx.drawImage(active.sourceCanvas, 0, 0);
    cutoutCtx.globalCompositeOperation = 'destination-in';
    cutoutCtx.drawImage(mask, 0, 0);
    cutoutCtx.globalCompositeOperation = 'source-over';
    const cutoutBlob = await canvasToBlob(cutoutCanvas, 'image/png');
    job.baseCutoutBlob = cutoutBlob;
    job.cutoutBlob = cutoutBlob;
    job.cutoutUrl = URL.createObjectURL(cutoutBlob);

    job.message = `Legacy mode removed ~${removed.toLocaleString('en-US')} pixels (tolerance ${tolerance}).`;
  };

  const processJobAi = async (job, bitmap, currentRunId) => {
    const maxDim = toInt(processingSelect?.value, 1024);
    const processing = computeProcessingSize(bitmap.width, bitmap.height, maxDim);
    job.processing.width = processing.width;
    job.processing.height = processing.height;
    job.processing.maxDim = processing.maxDim;
    job.processing.method = String(methodSelect?.value || 'ai-best');

    // Build a processing-size canvas to keep memory stable for very large uploads.
    const workCanvas = document.createElement('canvas');
    workCanvas.width = processing.width;
    workCanvas.height = processing.height;
    const workCtx = workCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
    if (!workCtx) throw new Error('Unable to create working canvas.');
    drawContain(workCtx, bitmap, processing.width, processing.height);

    const imageData = workCtx.getImageData(0, 0, processing.width, processing.height);
    // @imgly/background-removal expects a Blob / ArrayBuffer (or URL) and will convert it internally to an ndarray.
    // Passing ImageData directly breaks because it lacks `.shape`/`.data`.
    const rgbaBlob = new Blob([imageData.data], {
      type: `image/x-rgba8;width=${processing.width};height=${processing.height}`,
    });

    const lib = await loadBgRemoval();
    const device = 'cpu';
    const model = String(methodSelect?.value || 'ai-best') === 'ai-fast' ? 'isnet_quint8' : 'isnet_fp16';

    const config = {
      device,
      model,
      // Default publicPath uses staticimgly.com; make it explicit so it's easy to override/host later.
      publicPath: 'https://staticimgly.com/@imgly/background-removal-data/${PACKAGE_VERSION}/dist/',
      output: { format: 'image/png', quality: 0.92 },
      progress: (key, current, total) => {
        if (currentRunId !== state.runId) return;
        const safeTotal = Math.max(1, Number(total) || 1);
        const ratio = clamp((Number(current) || 0) / safeTotal, 0, 1);
        const label = key.startsWith('compute:')
          ? `Processing… (${key.replace('compute:', '')})`
          : `Downloading ${key}…`;
        showProgress(label, ratio);
      },
    };

    let maskBlob;
    try {
      maskBlob = await lib.segmentForeground(rgbaBlob, config);
    } finally {
      if (currentRunId === state.runId) hideProgress();
    }
    if (currentRunId !== state.runId) return;

    if (!(maskBlob instanceof Blob)) {
      throw new Error('AI model did not return a mask blob.');
    }

    job.maskBlob = maskBlob;
    job.maskUrl = URL.createObjectURL(maskBlob);

    // Create a quick cutout preview at processing size.
    const maskBitmap = await createImageBitmap(maskBlob);
    if (currentRunId !== state.runId) {
      if (maskBitmap && typeof maskBitmap.close === 'function') maskBitmap.close();
      return;
    }

    const cutoutCanvas = document.createElement('canvas');
    cutoutCanvas.width = processing.width;
    cutoutCanvas.height = processing.height;
    const cutoutCtx = cutoutCanvas.getContext('2d', { alpha: true });
    if (!cutoutCtx) throw new Error('Unable to create cutout.');
    cutoutCtx.drawImage(workCanvas, 0, 0);
    cutoutCtx.globalCompositeOperation = 'destination-in';
    cutoutCtx.drawImage(maskBitmap, 0, 0);
    cutoutCtx.globalCompositeOperation = 'source-over';
    if (maskBitmap && typeof maskBitmap.close === 'function') maskBitmap.close();

    const cutoutBlob = await canvasToBlob(cutoutCanvas, 'image/png');
    job.baseCutoutBlob = cutoutBlob;
    job.cutoutBlob = cutoutBlob;
    job.cutoutUrl = URL.createObjectURL(cutoutBlob);

    job.message = `AI processed at ${processing.maxDim}px (CPU).`;
  };

  const processJob = async (job, currentRunId) => {
    if (job.bytes > LIMITS.maxFileBytes) {
      job.status = 'error';
      job.message = `File is too large (${formatBytes(job.bytes)}). Try compressing or resizing first.`;
      return;
    }

    let bitmap;
    try {
      bitmap = await readImageBitmap(job.file);
      if (currentRunId !== state.runId) {
        if (bitmap && typeof bitmap.close === 'function') bitmap.close();
        return;
      }
      job.original.width = bitmap.width;
      job.original.height = bitmap.height;
      job.original.pixels = bitmap.width * bitmap.height;
      if (job.original.pixels > LIMITS.maxPixels) {
        job.status = 'error';
        job.message = `Image is too large (${Math.round(job.original.pixels / 1_000_000)} MP). Try a smaller version.`;
        return;
      }

      if (job.original.pixels > LIMITS.recommendedPixels) {
        job.message = `Large image (${Math.round(job.original.pixels / 1_000_000)} MP). Processing may take longer.`;
      }

      const method = String(methodSelect?.value || 'ai-best');
      if (method === 'colorkey') {
        await processJobLegacy(job, bitmap, currentRunId);
      } else {
        await processJobAi(job, bitmap, currentRunId);
      }

      job.status = 'ready';
    } catch (err) {
      job.status = 'error';
      const raw = getErrorMessage(err) || 'Unable to process that file.';
      const aiHint = formatAiInitHint(raw);
      job.message = aiHint ? `${aiHint} (${raw})` : raw;
      hideProgress();
    } finally {
      if (bitmap && typeof bitmap.close === 'function') bitmap.close();
    }
  };

  const processQueue = async () => {
    if (state.working) return;
    state.working = true;
    const currentRunId = state.runId;
    updateActionButtons();
    try {
      for (;;) {
        if (currentRunId !== state.runId) return;
        const next = state.jobs.find((j) => j.status === 'queued');
        if (!next) break;
        next.status = 'processing';
        renderFileList();
        renderResults();
        updateSummary();
        setStatus(`Processing ${next.name}…`);
        await processJob(next, currentRunId);
        renderFileList();
        renderResults();
        updateSummary();
        await new Promise((r) => window.requestAnimationFrame(r));

        // Auto-select the first ready result so users see something immediately.
        if (!state.activeJobId && next.status === 'ready') {
          await selectJob(next.id);
        }
      }
      setStatus(state.jobs.some((j) => j.status === 'ready') ? 'Done. Review results below.' : 'Add photos to begin.');
    } finally {
      hideProgress();
      state.working = false;
      updateActionButtons();
      updateRefineControls();
    }
  };

  const addFiles = (files) => {
    const list = Array.from(files || []).filter((f) => f && /^image\//.test(f.type));
    if (!list.length) return;

    const remaining = Math.max(0, LIMITS.maxFiles - state.jobs.length);
    const toAdd = list.slice(0, remaining);
    const skipped = list.length - toAdd.length;

    toAdd.forEach((file) => state.jobs.push(createJob(file)));
    if (skipped > 0) setStatus(`Added ${toAdd.length} file(s). Skipped ${skipped} due to the ${LIMITS.maxFiles} file limit.`);
    markSessionDirty();

    updateSummary();
    renderFileList();
    renderResults();
    updateActionButtons();

    if (state.jobs.length) hideOverlay();
    processQueue().catch(() => {});
  };

  const clearAll = async () => {
    await commitActiveMask();
    state.runId += 1;
    state.jobs.forEach(revokeJobUrls);
    state.jobs = [];
    state.activeJobId = null;
    active.job = null;
    active.dirtyMask = false;
    active.sourceCanvas.width = 1;
    active.sourceCanvas.height = 1;
    active.maskCanvas.width = 1;
    active.maskCanvas.height = 1;
    previewCtx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = 1;
    canvas.height = 1;
    showOverlay('Click or drop photos to start');
    if (dimLabel) dimLabel.textContent = 'Size: N/A';
    if (maskLabel) maskLabel.textContent = 'Mask: N/A';
    if (selectedLabel) selectedLabel.textContent = 'Selected: None';
    setStatus('Cleared.');
    updateSummary();
    renderFileList();
    renderResults();
    updateActionButtons();
    updateRefineControls();
    markSessionDirty();
  };

  const removeJob = async (jobId) => {
    if (state.working) return;
    await commitActiveMask();
    const idx = state.jobs.findIndex((j) => j.id === jobId);
    if (idx === -1) return;
    const [job] = state.jobs.splice(idx, 1);
    revokeJobUrls(job);
    if (state.activeJobId === jobId) {
      state.activeJobId = null;
      active.job = null;
      schedulePreviewRender();
    }
    updateSummary();
    renderFileList();
    renderResults();
    updateActionButtons();
    updateRefineControls();
    markSessionDirty();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-hover');
    const files = e.dataTransfer?.files;
    if (!files || !files.length) return;
    addFiles(files);
  };

  const updateBgLabel = () => {
    const hex = normalizeHex(bgInput?.value || state.bgColor);
    state.bgColor = hex;
    if (bgLabel) bgLabel.textContent = hex;
  };

  const updateLegacyLabels = () => {
    if (colorLabel) colorLabel.textContent = normalizeHex(colorInput?.value || '#ffffff');
    if (toleranceValue) toleranceValue.textContent = String(toInt(toleranceInput?.value, 24));
  };

  const applyBrush = (x, y) => {
    if (!active.maskCtx) return;
    const enabled = !!(refineEnabledInput && refineEnabledInput.checked);
    if (!enabled) return;
    if (!active.job || active.job.status !== 'ready') return;
    if (state.view !== 'cutout') return;

    const mode = String(brushModeSelect?.value || 'erase');
    const radius = clamp(toInt(brushSizeInput?.value, 24), 4, 240);

    // Soft edge brush with radial gradient improves cleanup around fine structures.
    const ctx = active.maskCtx;
    ctx.save();
    ctx.globalCompositeOperation = mode === 'erase' ? 'destination-out' : 'source-over';
    const grad = ctx.createRadialGradient(x, y, 0, x, y, radius);
    if (mode === 'erase') {
      grad.addColorStop(0, 'rgba(0,0,0,0.95)');
      grad.addColorStop(1, 'rgba(0,0,0,0)');
    } else {
      grad.addColorStop(0, 'rgba(255,255,255,0.95)');
      grad.addColorStop(1, 'rgba(255,255,255,0)');
    }
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    active.dirtyMask = true;
    markSessionDirty();
    schedulePreviewRender();
  };

  const canvasPoint = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);
    return { x: clamp(x, 0, canvas.width), y: clamp(y, 0, canvas.height) };
  };

  const sampleLegacyColor = (event) => {
    if (!active.job || active.job.status !== 'ready') return false;
    if (String(methodSelect?.value || 'ai-best') !== 'colorkey') return false;
    if (refineEnabledInput && refineEnabledInput.checked) return false;

    const point = canvasPoint(event);
    // Sample from the underlying source image (not the cutout), so transparency/checkerboards
    // do not affect the picked color.
    const img = active.sourceCtx.getImageData(Math.floor(point.x), Math.floor(point.y), 1, 1);
    const r = img.data[0];
    const g = img.data[1];
    const b = img.data[2];
    const hex = normalizeHex(((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1));
    if (colorInput) colorInput.value = hex.toLowerCase();
    if (colorLabel) colorLabel.textContent = hex;
    setStatus(`Sampled ${hex}. Reprocess to apply.`);
    markSessionDirty();
    return true;
  };

  const reprocessSelected = async () => {
    if (!active.job) return;
    if (active.job.status !== 'ready' && active.job.status !== 'error') return;
    await commitActiveMask();
    revokeJobUrls(active.job);
    active.job.status = 'queued';
    active.job.message = '';
    active.job.baseMaskBlob = null;
    active.job.baseCutoutBlob = null;
    active.job.maskBlob = null;
    active.job.cutoutBlob = null;
    active.job.maskUrl = '';
    active.job.cutoutUrl = '';
    renderFileList();
    renderResults();
    updateActionButtons();
    processQueue().catch(() => {});
  };

  const clearEdits = async () => {
    if (!active.job || active.job.status !== 'ready') return;
    if (!active.job.baseMaskBlob) return;
    const maskBitmap = await createImageBitmap(active.job.baseMaskBlob);
    active.maskCtx.clearRect(0, 0, active.maskCanvas.width, active.maskCanvas.height);
    active.maskCtx.drawImage(maskBitmap, 0, 0, active.maskCanvas.width, active.maskCanvas.height);
    if (maskBitmap && typeof maskBitmap.close === 'function') maskBitmap.close();
    active.dirtyMask = true;
    schedulePreviewRender();
    markSessionDirty();
  };

  // --- Event wiring ---

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-hover');
  });
  dropzone.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-hover');
  });
  dropzone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-hover');
  });
  dropzone.addEventListener('drop', handleDrop);

  overlay?.addEventListener('click', () => fileInput.click());
  overlay?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  pickButtons.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      fileInput.click();
    });
  });

  fileInput.addEventListener('change', () => {
    addFiles(fileInput.files);
    fileInput.value = '';
  });

  fileList?.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-bgtool-remove]');
    if (!btn) return;
    removeJob(btn.dataset.bgtoolRemove);
  });

  resultsEl.addEventListener('click', (e) => {
    const rerunBtn = e.target.closest('[data-bgtool-rerun]');
    if (rerunBtn) {
      e.preventDefault();
      e.stopPropagation();
      const id = rerunBtn.dataset.bgtoolRerun;
      if (id) {
        selectJob(id)
          .then(() => reprocessSelected())
          .catch(() => {});
      }
      return;
    }

    const approve = e.target.closest('[data-bgtool-approve]');
    if (approve) {
      const id = approve.dataset.bgtoolApprove;
      const job = state.jobs.find((j) => j.id === id);
      if (job) {
        job.approved = !!approve.checked;
        updateActionButtons();
        markSessionDirty();
      }
      return;
    }

    const card = e.target.closest('[data-bgtool-select]');
    if (!card) return;
    const id = card.dataset.bgtoolSelect;
    if (!id) return;
    selectJob(id).catch(() => {});
  });

  downloadSelectedBtn?.addEventListener('click', downloadSelected);
  downloadAllBtn?.addEventListener('click', downloadApproved);
  resetBtn?.addEventListener('click', clearAll);
  clearEditsBtn?.addEventListener('click', clearEdits);

  methodSelect?.addEventListener('change', () => {
    updateMethodVisibility();
    // Changing method affects mask generation, so prompt users by re-queueing the selected job if desired.
    if (active.job) reprocessSelected().catch(() => {});
  });
  deviceSelect?.addEventListener('change', () => {
    markSessionDirty();
    if (active.job && String(methodSelect?.value || '').startsWith('ai-')) reprocessSelected().catch(() => {});
  });
  processingSelect?.addEventListener('change', () => {
    markSessionDirty();
    if (active.job) reprocessSelected().catch(() => {});
  });

  $$('input[name="bgtool-mask-type"]').forEach((input) => {
    input.addEventListener('change', () => {
      state.maskType = input.value === 'binary' ? 'binary' : 'alpha';
      updateMaskControls();
      schedulePreviewRender();
      markSessionDirty();
    });
  });

  thresholdInput?.addEventListener('input', () => {
    state.thresholdPct = clamp(toInt(thresholdInput.value, 50), 1, 99);
    updateMaskControls();
    schedulePreviewRender();
    markSessionDirty();
  });
  featherInput?.addEventListener('input', () => {
    state.featherPx = clamp(toInt(featherInput.value, 0), 0, 40);
    updateMaskControls();
    schedulePreviewRender();
    markSessionDirty();
  });

  formatSelect?.addEventListener('change', updateOutputControls);
  bgInput?.addEventListener('input', () => {
    updateBgLabel();
    schedulePreviewRender();
    markSessionDirty();
  });

  colorInput?.addEventListener('input', () => {
    updateLegacyLabels();
    markSessionDirty();
  });
  toleranceInput?.addEventListener('input', () => {
    updateLegacyLabels();
    markSessionDirty();
  });

  viewButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      state.view = btn.dataset.bgtoolView || 'cutout';
      schedulePreviewRender();
      markSessionDirty();
    });
  });

  refineEnabledInput?.addEventListener('change', () => {
    updateRefineControls();
    markSessionDirty();
  });
  brushSizeInput?.addEventListener('input', () => {
    const size = clamp(toInt(brushSizeInput.value, 24), 6, 120);
    if (brushSizeValue) brushSizeValue.textContent = `${size} px`;
    markSessionDirty();
  });
  brushModeSelect?.addEventListener('change', markSessionDirty);

  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    if (sampleLegacyColor(event)) return;
    const enabled = !!(refineEnabledInput && refineEnabledInput.checked);
    if (!enabled) return;
    if (!active.job || active.job.status !== 'ready') return;
    if (state.view !== 'cutout') return;
    canvas.setPointerCapture(event.pointerId);
    active.drawing = true;
    const point = canvasPoint(event);
    applyBrush(point.x, point.y);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!active.drawing) return;
    const point = canvasPoint(event);
    applyBrush(point.x, point.y);
  });
  canvas.addEventListener('pointerup', async (event) => {
    if (!active.drawing) return;
    active.drawing = false;
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch {}
    await commitActiveMask();
    renderResults();
    updateActionButtons();
  });
  canvas.addEventListener('pointercancel', () => {
    active.drawing = false;
  });

  form.addEventListener('submit', (e) => e.preventDefault());

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (!detail || detail.toolId !== TOOL_ID) return;
    const payload = detail.payload;
    if (!payload || typeof payload !== 'object') return;

    const inputs = {
      Method: String(methodSelect?.value || '').trim(),
      Device: String(deviceSelect?.value || '').trim(),
      'Processing size': String(processingSelect?.value || '').trim(),
      'Mask type': state.maskType,
      Threshold: `${state.thresholdPct}%`,
      Feather: `${state.featherPx}px`,
      Format: String(formatSelect?.value || '').trim(),
    };

    const files = state.jobs.map((j) => j.name).slice(0, 8);
    if (files.length) inputs.Files = files.join(', ') + (state.jobs.length > files.length ? ` (+${state.jobs.length - files.length} more)` : '');

    payload.inputs = inputs;

    const summaryParts = [];
    if (active.job) summaryParts.push(active.job.name);
    summaryParts.push(`${state.jobs.filter((j) => j.status === 'ready').length}/${state.jobs.length} ready`);
    payload.outputSummary = summaryParts.join(' · ') || 'Background removed';

    // Store a small preview of the current canvas output.
    try {
      const maxDim = 420;
      const srcW = canvas.width || 0;
      const srcH = canvas.height || 0;
      if (srcW && srcH) {
        const scale = Math.min(1, maxDim / Math.max(srcW, srcH));
        const w = Math.max(1, Math.round(srcW * scale));
        const h = Math.max(1, Math.round(srcH * scale));
        const preview = document.createElement('canvas');
        preview.width = w;
        preview.height = h;
        const pctx = preview.getContext('2d', { alpha: true });
        if (pctx) {
          pctx.drawImage(canvas, 0, 0, w, h);
          const dataUrl = preview.toDataURL('image/png');
          if (dataUrl.startsWith('data:image/png')) {
            payload.output = { kind: 'image', mime: 'image/png', dataUrl, width: w, height: h };
          }
        }
      }
    } catch {}
  });

  // Initial UI state.
  showOverlay('Click or drop photos to start');
  updateSummary();
  updateMethodVisibility();
  updateLegacyLabels();
  updateBgLabel();
  updateMaskControls();
  updateOutputControls();
  updateActionButtons();
  updateRefineControls();
})();
