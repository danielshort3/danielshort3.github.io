(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const fileInput = $('#imgopt-file');
  const dropzone = $('#imgopt-dropzone');
  const fileList = $('#imgopt-filelist');
  const countEl = $('#imgopt-count');
  const totalEl = $('#imgopt-total');

  const form = $('#imgopt-form');
  const formatSelect = $('#imgopt-format');
  const qualityWrap = $('#imgopt-quality-wrap');
  const qualityInput = $('#imgopt-quality');
  const qualityValue = $('#imgopt-quality-value');

  const flattenWrap = $('#imgopt-flatten-wrap');
  const flattenInput = $('#imgopt-flatten');
  const flattenColor = $('#imgopt-flatten-color');
  const flattenColorLabel = $('#imgopt-flatten-color-label');

  const resizeMode = $('#imgopt-resize-mode');
  const dimsWrap = $('#imgopt-dims-wrap');
  const widthInput = $('#imgopt-width');
  const heightInput = $('#imgopt-height');
  const scaleWrap = $('#imgopt-scale-wrap');
  const scaleInput = $('#imgopt-scale');
  const keepAspectInput = $('#imgopt-keep-aspect');
  const noUpscaleInput = $('#imgopt-no-upscale');

  const responsiveInput = $('#imgopt-responsive');
  const responsiveWrap = $('#imgopt-responsive-wrap');
  const responsiveWidthsInput = $('#imgopt-responsive-widths');

  const suffixInput = $('#imgopt-suffix');
  const clearBtn = $('#imgopt-clear');
  const statusEl = $('#imgopt-status');
  const resultsEl = $('#imgopt-results');
  const downloadAllBtn = $('#imgopt-download-all');
  const processBtn = $('#imgopt-process');

  if (!fileInput || !dropzone || !fileList || !form || !formatSelect || !resultsEl) return;

  const TOOL_ID = 'image-optimizer';
  const MAX_SAVED_OUTPUT_LINES = 120;
  const MEBIBYTE = 1024 * 1024;
  const IMAGE_LIMITS = Object.freeze({
    maxFiles: 20,
    maxFileBytes: 25 * MEBIBYTE,
    maxInputBytes: 150 * MEBIBYTE,
    maxDimension: 12000,
    maxDecodedPixelsPerFile: 40_000_000,
    maxDecodedPixelsTotal: 120_000_000,
    maxEstimatedOutputBytes: 256 * MEBIBYTE,
    maxActualOutputBytes: 200 * MEBIBYTE
  });
  const PREVIEW_MAX_SIDE = 96;
  const IMAGE_HEADER_SCAN_BYTES = 4 * MEBIBYTE;
  const SUPPORTED_INPUT_TYPES = new Set([
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/webp',
    'image/avif'
  ]);

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const dispatchToolRunEvent = (eventName, detail = {}) => {
    try {
      document.dispatchEvent(new CustomEvent(eventName, {
        detail: { toolId: TOOL_ID, ...detail }
      }));
    } catch {}
  };

  const state = {
    items: [],
    outputs: [],
    working: false,
    cancelRequested: false,
    operationId: 0,
    selectionVersion: 0,
    metadataQueue: Promise.resolve(),
    nextId: 1,
    supports: {}
  };

  const updateLayoutState = () => {
    const nextState = state.working
      ? 'working'
      : state.outputs.length
        ? 'results'
        : state.items.length
          ? 'ready'
          : 'empty';
    if (document.body) document.body.dataset.toolsState = nextState;
  };

  const formatBytes = (bytes) => {
    const n = Number(bytes || 0);
    if (!Number.isFinite(n) || n <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const idx = Math.min(units.length - 1, Math.floor(Math.log10(n) / 3));
    const val = n / Math.pow(1024, idx);
    const precision = val >= 100 ? 0 : val >= 10 ? 1 : 2;
    return `${val.toFixed(precision)} ${units[idx]}`;
  };

  const describeSelectionLimits = () => `${IMAGE_LIMITS.maxFiles} files, ${formatBytes(IMAGE_LIMITS.maxFileBytes)} each, ${formatBytes(IMAGE_LIMITS.maxInputBytes)} total`;

  const validateImageDimensions = (width, height, filename = 'Image') => {
    const safeWidth = Math.trunc(Number(width));
    const safeHeight = Math.trunc(Number(height));
    if (!Number.isFinite(safeWidth) || !Number.isFinite(safeHeight) || safeWidth < 1 || safeHeight < 1) {
      throw new Error(`${filename} has invalid dimensions.`);
    }
    if (safeWidth > IMAGE_LIMITS.maxDimension || safeHeight > IMAGE_LIMITS.maxDimension) {
      throw new Error(`${filename} is ${safeWidth} x ${safeHeight}; each side must be ${IMAGE_LIMITS.maxDimension.toLocaleString()} px or less.`);
    }
    const pixels = safeWidth * safeHeight;
    if (pixels > IMAGE_LIMITS.maxDecodedPixelsPerFile) {
      throw new Error(`${filename} has ${(pixels / 1_000_000).toFixed(1)} MP; the limit is ${(IMAGE_LIMITS.maxDecodedPixelsPerFile / 1_000_000).toFixed(0)} MP per image.`);
    }
    return { width: safeWidth, height: safeHeight, pixels };
  };

  const createCancellationError = () => {
    const error = new Error('Optimization cancelled. Selected images were kept.');
    error.name = 'AbortError';
    return error;
  };

  const throwIfCancelled = (operationId) => {
    if (state.cancelRequested || operationId !== state.operationId) {
      throw createCancellationError();
    }
  };

  const clampInt = (value, min, max) => {
    const n = Math.trunc(Number(value));
    if (!Number.isFinite(n)) return null;
    return Math.min(max, Math.max(min, n));
  };

  const safeText = (str) => String(str || '').replace(/\s+/g, ' ').trim();

  const normalizeSuffix = (suffix) => {
    const cleaned = safeText(suffix)
      .replace(/[^\w.-]+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-+|-+$/g, '');
    if (!cleaned) return '';
    return cleaned.startsWith('-') ? cleaned : `-${cleaned}`;
  };

  const splitName = (filename) => {
    const name = String(filename || 'image');
    const dot = name.lastIndexOf('.');
    if (dot <= 0) return { stem: name, ext: '' };
    return { stem: name.slice(0, dot), ext: name.slice(dot) };
  };

  const sanitizeStem = (stem) => {
    const cleaned = safeText(stem)
      .replace(/[^\w.-]+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-+|-+$/g, '');
    return cleaned || 'image';
  };

  const extForMime = (mime) => {
    switch (mime) {
      case 'image/jpeg': return '.jpg';
      case 'image/png': return '.png';
      case 'image/webp': return '.webp';
      case 'image/avif': return '.avif';
      default: return '.png';
    }
  };

  const labelForMime = (mime) => {
    switch (mime) {
      case 'image/jpeg': return 'JPEG';
      case 'image/png': return 'PNG';
      case 'image/webp': return 'WebP';
      case 'image/avif': return 'AVIF';
      default: return safeText(mime).toUpperCase() || 'IMAGE';
    }
  };

  const supportsEncoding = (mime) => {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;
      const data = canvas.toDataURL(mime);
      return typeof data === 'string' && data.startsWith(`data:${mime}`);
    } catch {
      return false;
    }
  };

  const initEncodingSupport = () => {
    state.supports['image/jpeg'] = supportsEncoding('image/jpeg');
    state.supports['image/png'] = supportsEncoding('image/png');
    state.supports['image/webp'] = supportsEncoding('image/webp');
    state.supports['image/avif'] = supportsEncoding('image/avif');

    const opts = $$('#imgopt-format option');
    opts.forEach((opt) => {
      const value = opt.getAttribute('value');
      if (!value || value === 'keep') return;
      if (!state.supports[value]) {
        opt.disabled = true;
        opt.textContent = opt.textContent.replace(/\s*\(.*?\)\s*$/, '').trim() + ' (unsupported)';
      }
    });
  };

  const updateControlsVisibility = () => {
    const selected = formatSelect.value;
    const lossy = selected === 'keep' || selected === 'image/jpeg' || selected === 'image/webp' || selected === 'image/avif';
    if (qualityWrap) qualityWrap.hidden = !lossy;

    const showFlatten = selected === 'image/jpeg';
    if (flattenWrap) flattenWrap.hidden = !showFlatten;
    if (flattenInput) {
      if (showFlatten) {
        flattenInput.checked = true;
        flattenInput.disabled = true;
      } else {
        flattenInput.disabled = false;
      }
    }

    const mode = resizeMode?.value || 'none';
    const usingResponsive = Boolean(responsiveInput?.checked);

    if (responsiveWrap) responsiveWrap.hidden = !usingResponsive;

    if (dimsWrap) dimsWrap.hidden = usingResponsive || mode === 'none' || mode === 'scale';
    if (scaleWrap) scaleWrap.hidden = usingResponsive || mode !== 'scale';

    if (resizeMode) resizeMode.disabled = usingResponsive;
    if (keepAspectInput) {
      if (usingResponsive) {
        keepAspectInput.checked = true;
        keepAspectInput.disabled = true;
      } else {
        keepAspectInput.disabled = false;
      }
    }
  };

  const updateQualityLabel = () => {
    if (!qualityValue || !qualityInput) return;
    qualityValue.textContent = String(qualityInput.value || '');
  };

  const updateFlattenLabel = () => {
    if (!flattenColor || !flattenColorLabel) return;
    flattenColorLabel.textContent = (flattenColor.value || '#000000').toLowerCase();
  };

  const setStatus = (msg) => {
    if (statusEl) statusEl.textContent = msg || '';
  };

  const setWorking = (working) => {
    state.working = Boolean(working);
    if (processBtn) processBtn.disabled = state.working || state.items.length === 0;
    if (clearBtn) {
      clearBtn.disabled = false;
      clearBtn.textContent = state.working ? 'Cancel' : 'Clear';
      clearBtn.setAttribute('aria-label', state.working ? 'Cancel optimization' : 'Clear selected images');
    }
    if (downloadAllBtn) downloadAllBtn.disabled = state.working || state.outputs.length === 0;
    if (fileInput) fileInput.disabled = state.working;
    if (formatSelect) formatSelect.disabled = state.working;
    if (resizeMode) resizeMode.disabled = state.working || Boolean(responsiveInput?.checked);
    if (qualityInput) qualityInput.disabled = state.working;
    if (flattenInput) flattenInput.disabled = state.working;
    if (flattenColor) flattenColor.disabled = state.working;
    if (widthInput) widthInput.disabled = state.working;
    if (heightInput) heightInput.disabled = state.working;
    if (scaleInput) scaleInput.disabled = state.working;
    if (keepAspectInput) keepAspectInput.disabled = state.working || Boolean(responsiveInput?.checked);
    if (noUpscaleInput) noUpscaleInput.disabled = state.working;
    if (responsiveInput) responsiveInput.disabled = state.working;
    if (responsiveWidthsInput) responsiveWidthsInput.disabled = state.working;
    if (suffixInput) suffixInput.disabled = state.working;
    updateLayoutState();
  };

  const revokeOutputs = () => {
    state.outputs.forEach((out) => {
      if (out.url) URL.revokeObjectURL(out.url);
    });
    state.outputs = [];
    if (resultsEl) resultsEl.innerHTML = '';
    if (downloadAllBtn) downloadAllBtn.disabled = true;
    updateLayoutState();
    markSessionDirty();
  };

  const clearAll = () => {
    if (state.working) return;
    state.selectionVersion += 1;
    state.items.forEach((it) => {
      it.removed = true;
      if (it.previewUrl) URL.revokeObjectURL(it.previewUrl);
    });
    state.items = [];
    state.nextId = 1;
    fileList.innerHTML = '';
    updateSummary();
    revokeOutputs();
    setStatus('Add images, choose settings, then click Optimize.');
    fileInput.value = '';
    markSessionDirty();
  };

  const updateSummary = () => {
    const totalBytes = state.items.reduce((sum, it) => sum + (it.file?.size || 0), 0);
    if (countEl) countEl.textContent = `${state.items.length} ${state.items.length === 1 ? 'image' : 'images'}`;
    if (totalEl) totalEl.textContent = `Total: ${state.items.length ? formatBytes(totalBytes) : '0 B'}`;
    if (processBtn) processBtn.disabled = state.working || state.items.length === 0;
    updateLayoutState();
  };

  const decodeBitmap = async (file) => {
    if (typeof createImageBitmap === 'function') {
      try {
        return await createImageBitmap(file, { imageOrientation: 'from-image' });
      } catch {
        return await createImageBitmap(file);
      }
    }

    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.decoding = 'async';
      img.src = url;
      if (img.decode) {
        await img.decode();
      } else {
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
        });
      }
      return img;
    } finally {
      URL.revokeObjectURL(url);
    }
  };

  const readFourCc = (view, offset) => {
    if (!view || offset < 0 || offset + 4 > view.byteLength) return '';
    return String.fromCharCode(
      view.getUint8(offset),
      view.getUint8(offset + 1),
      view.getUint8(offset + 2),
      view.getUint8(offset + 3)
    );
  };

  const readUint24Le = (view, offset) => {
    if (!view || offset < 0 || offset + 3 > view.byteLength) return 0;
    return view.getUint8(offset) |
      (view.getUint8(offset + 1) << 8) |
      (view.getUint8(offset + 2) << 16);
  };

  const parsePngDimensions = (view) => {
    if (view.byteLength < 24 || view.getUint32(0) !== 0x89504e47 || readFourCc(view, 12) !== 'IHDR') return null;
    return { width: view.getUint32(16), height: view.getUint32(20) };
  };

  const parseJpegDimensions = (view) => {
    if (view.byteLength < 4 || view.getUint16(0) !== 0xffd8) return null;
    const sofMarkers = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);
    let offset = 2;
    while (offset + 4 <= view.byteLength) {
      if (view.getUint8(offset) !== 0xff) {
        offset += 1;
        continue;
      }
      while (offset < view.byteLength && view.getUint8(offset) === 0xff) offset += 1;
      if (offset >= view.byteLength) break;
      const marker = view.getUint8(offset);
      offset += 1;
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) continue;
      if (offset + 2 > view.byteLength) break;
      const length = view.getUint16(offset);
      if (length < 2 || offset + length > view.byteLength) break;
      if (sofMarkers.has(marker) && length >= 7) {
        return { width: view.getUint16(offset + 5), height: view.getUint16(offset + 3) };
      }
      offset += length;
    }
    return null;
  };

  const parseWebpDimensions = (view) => {
    if (view.byteLength < 30 || readFourCc(view, 0) !== 'RIFF' || readFourCc(view, 8) !== 'WEBP') return null;
    let offset = 12;
    while (offset + 8 <= view.byteLength) {
      const chunkType = readFourCc(view, offset);
      const chunkSize = view.getUint32(offset + 4, true);
      const dataOffset = offset + 8;
      if (chunkType === 'VP8X' && chunkSize >= 10 && dataOffset + 10 <= view.byteLength) {
        return {
          width: readUint24Le(view, dataOffset + 4) + 1,
          height: readUint24Le(view, dataOffset + 7) + 1
        };
      }
      if (chunkType === 'VP8 ' && chunkSize >= 10 && dataOffset + 10 <= view.byteLength &&
        view.getUint8(dataOffset + 3) === 0x9d && view.getUint8(dataOffset + 4) === 0x01 && view.getUint8(dataOffset + 5) === 0x2a) {
        return {
          width: view.getUint16(dataOffset + 6, true) & 0x3fff,
          height: view.getUint16(dataOffset + 8, true) & 0x3fff
        };
      }
      if (chunkType === 'VP8L' && chunkSize >= 5 && dataOffset + 5 <= view.byteLength && view.getUint8(dataOffset) === 0x2f) {
        const bits = view.getUint32(dataOffset + 1, true);
        return { width: (bits & 0x3fff) + 1, height: ((bits >>> 14) & 0x3fff) + 1 };
      }
      const nextOffset = dataOffset + chunkSize + (chunkSize % 2);
      if (nextOffset <= offset || nextOffset > view.byteLength) break;
      offset = nextOffset;
    }
    return null;
  };

  const parseAvifDimensions = (view) => {
    if (view.byteLength < 32 || readFourCc(view, 4) !== 'ftyp') return null;
    for (let offset = 4; offset + 16 <= view.byteLength; offset += 1) {
      if (readFourCc(view, offset) !== 'ispe' || offset < 4) continue;
      const boxSize = view.getUint32(offset - 4);
      if (boxSize < 20 || offset - 4 + boxSize > view.byteLength) continue;
      return { width: view.getUint32(offset + 8), height: view.getUint32(offset + 12) };
    }
    return null;
  };

  const readImageHeaderDimensions = async (file) => {
    const bytes = await file.slice(0, Math.min(file.size, IMAGE_HEADER_SCAN_BYTES)).arrayBuffer();
    const view = new DataView(bytes);
    const type = String(file.type || '').toLowerCase();
    const extension = String(file.name || '').toLowerCase().match(/\.([a-z0-9]+)$/)?.[1] || '';
    if (type === 'image/png' || extension === 'png') return parsePngDimensions(view);
    if (type === 'image/jpeg' || type === 'image/jpg' || extension === 'jpg' || extension === 'jpeg') return parseJpegDimensions(view);
    if (type === 'image/webp' || extension === 'webp') return parseWebpDimensions(view);
    if (type === 'image/avif' || extension === 'avif') return parseAvifDimensions(view);
    return null;
  };

  const createPreviewUrl = async (decoded) => {
    const sourceWidth = decoded?.width || decoded?.naturalWidth || 0;
    const sourceHeight = decoded?.height || decoded?.naturalHeight || 0;
    if (!sourceWidth || !sourceHeight) return null;
    const scale = Math.min(1, PREVIEW_MAX_SIDE / sourceWidth, PREVIEW_MAX_SIDE / sourceHeight);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(sourceWidth * scale));
    canvas.height = Math.max(1, Math.round(sourceHeight * scale));
    try {
      const ctx = canvas.getContext('2d', { alpha: true });
      if (!ctx) return null;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'medium';
      ctx.drawImage(decoded, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, 'image/png');
      });
      return blob ? URL.createObjectURL(blob) : null;
    } finally {
      canvas.width = 1;
      canvas.height = 1;
    }
  };

  const getItemById = (id) => state.items.find((it) => String(it.id) === String(id));

  const removeItem = (id) => {
    const idx = state.items.findIndex((it) => String(it.id) === String(id));
    if (idx < 0) return;
    const [removed] = state.items.splice(idx, 1);
    if (removed) removed.removed = true;
    if (removed?.previewUrl) URL.revokeObjectURL(removed.previewUrl);
    renderFileList();
    revokeOutputs();
    updateSummary();
    setStatus('Selection updated. Click Optimize to regenerate outputs.');
  };

  const renderFileList = () => {
    fileList.innerHTML = '';
    if (!state.items.length) return;

    const frag = document.createDocumentFragment();
    state.items.forEach((it) => {
      const li = document.createElement('li');
      li.className = 'imgopt-file';
      li.dataset.id = String(it.id);

      const thumb = document.createElement('img');
      thumb.className = 'imgopt-thumb';
      thumb.alt = '';
      thumb.decoding = 'async';
      thumb.loading = 'lazy';
      thumb.hidden = !it.previewUrl;
      if (it.previewUrl) thumb.src = it.previewUrl;

      const meta = document.createElement('div');
      meta.className = 'imgopt-file-meta';

      const name = document.createElement('p');
      name.className = 'imgopt-file-name';
      name.textContent = it.file.name;

      const sub = document.createElement('p');
      sub.className = 'imgopt-file-sub';
      const dims = it.width && it.height ? `${it.width} × ${it.height}` : 'N/A';
      const typeLabel = it.file.type ? labelForMime(it.file.type) : 'Image';
      sub.textContent = `${typeLabel} · ${dims} · ${formatBytes(it.file.size)}`;

      meta.appendChild(name);
      meta.appendChild(sub);

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn-icon imgopt-remove';
      btn.setAttribute('aria-label', `Remove ${it.file.name}`);
      btn.dataset.imgoptRemove = String(it.id);
      btn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"></path></svg>';

      li.appendChild(thumb);
      li.appendChild(meta);
      li.appendChild(btn);
      frag.appendChild(li);
    });
    fileList.appendChild(frag);
  };

  const hydrateItemMetadata = async (item) => {
    if (!item || item.width || item.removed || !state.items.includes(item)) return;
    let decoded = null;
    try {
      const headerDimensions = await readImageHeaderDimensions(item.file);
      if (!headerDimensions) {
        throw new Error(`${item.file.name} dimensions could not be verified safely before decoding.`);
      }
      validateImageDimensions(headerDimensions.width, headerDimensions.height, item.file.name);
      decoded = await decodeBitmap(item.file);
      if (item.removed || item.selectionVersion !== state.selectionVersion || !state.items.includes(item)) return;
      const dimensions = validateImageDimensions(
        decoded.width || decoded.naturalWidth,
        decoded.height || decoded.naturalHeight,
        item.file.name
      );
      const otherPixels = state.items.reduce((sum, candidate) => {
        if (candidate === item || !candidate.width || !candidate.height) return sum;
        return sum + (candidate.width * candidate.height);
      }, 0);
      if (otherPixels + dimensions.pixels > IMAGE_LIMITS.maxDecodedPixelsTotal) {
        throw new Error(`${item.file.name} would take the selection above the ${(IMAGE_LIMITS.maxDecodedPixelsTotal / 1_000_000).toFixed(0)} MP decoded-pixel limit.`);
      }
      let previewUrl = null;
      try {
        previewUrl = await createPreviewUrl(decoded);
      } catch (_) {}
      if (item.removed || item.selectionVersion !== state.selectionVersion || !state.items.includes(item)) {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        return;
      }
      item.width = dimensions.width;
      item.height = dimensions.height;
      item.previewUrl = previewUrl;
    } finally {
      if (decoded && typeof decoded.close === 'function') decoded.close();
    }
  };

  const addFiles = (list) => {
    const files = Array.from(list || []);
    const accepted = files.filter((file) => {
      if (!file) return false;
      const mimeType = String(file.type || '').toLowerCase();
      if (SUPPORTED_INPUT_TYPES.has(mimeType)) return true;
      return !mimeType && /\.(?:avif|jpe?g|png|webp)$/i.test(String(file.name || ''));
    });
    if (!accepted.length) {
      setStatus('No supported image files were detected.');
      return;
    }

    const rejected = files
      .filter((file) => file && !accepted.includes(file))
      .map((file) => `${file.name}: use PNG, JPEG, WebP, or AVIF`);
    const queued = [];
    let selectedBytes = state.items.reduce((sum, item) => sum + (item.file?.size || 0), 0);
    const selectionVersion = state.selectionVersion;
    accepted.forEach((file) => {
      if (state.items.length >= IMAGE_LIMITS.maxFiles) {
        rejected.push(`${file.name}: only ${IMAGE_LIMITS.maxFiles} files can be selected`);
        return;
      }
      if (file.size > IMAGE_LIMITS.maxFileBytes) {
        rejected.push(`${file.name}: ${formatBytes(file.size)} exceeds ${formatBytes(IMAGE_LIMITS.maxFileBytes)}`);
        return;
      }
      if (selectedBytes + file.size > IMAGE_LIMITS.maxInputBytes) {
        rejected.push(`${file.name}: selection would exceed ${formatBytes(IMAGE_LIMITS.maxInputBytes)} total`);
        return;
      }
      selectedBytes += file.size;
      const id = state.nextId++;
      const item = {
        id,
        file,
        previewUrl: null,
        width: null,
        height: null,
        removed: false,
        selectionVersion: state.selectionVersion
      };
      state.items.push(item);
      queued.push(item);
    });

    renderFileList();
    updateSummary();
    revokeOutputs();
    if (!queued.length) {
      setStatus(`No images added. ${rejected.slice(0, 2).join(' ') || `Limits: ${describeSelectionLimits()}.`}`);
      return;
    }

    setStatus(`Checking ${queued.length} ${queued.length === 1 ? 'image' : 'images'} safely... Limits: ${describeSelectionLimits()}.`);
    state.metadataQueue = state.metadataQueue.then(async () => {
      const dimensionRejections = [];
      for (const item of queued) {
        if (item.removed || item.selectionVersion !== state.selectionVersion || !state.items.includes(item)) continue;
        try {
          await hydrateItemMetadata(item);
        } catch (error) {
          item.removed = true;
          const index = state.items.indexOf(item);
          if (index >= 0) state.items.splice(index, 1);
          dimensionRejections.push(error?.message || `${item.file.name} could not be decoded.`);
        }
      }
      if (selectionVersion !== state.selectionVersion) return;
      renderFileList();
      updateSummary();
      const allRejections = rejected.concat(dimensionRejections);
      const readyCount = queued.filter((item) => !item.removed && item.width && item.height && state.items.includes(item)).length;
      if (allRejections.length) {
        setStatus(`${readyCount} added; ${allRejections.length} rejected. ${allRejections.slice(0, 2).join(' ')}`);
      } else {
        setStatus(`${readyCount} ${readyCount === 1 ? 'image is' : 'images are'} ready. Click Optimize to generate downloads.`);
      }
    }).catch(() => {
      setStatus('Image checks failed. Remove the affected files and try again.');
    });
  };

  const parseResponsiveWidths = (raw) => {
    const parts = String(raw || '')
      .split(/[, ]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    const widths = parts
      .map((p) => clampInt(p, 16, 20000))
      .filter((n) => n !== null);
    const unique = Array.from(new Set(widths)).sort((a, b) => a - b);
    return unique;
  };

  const inferOutputMime = (requested, inputMime) => {
    if (requested && requested !== 'keep') return requested;
    const m = inputMime || '';
    if (state.supports[m]) return m;
    return 'image/png';
  };

  const getQuality = () => {
    const n = clampInt(qualityInput?.value, 1, 100);
    const pct = n === null ? 82 : n;
    return Math.min(1, Math.max(0.01, pct / 100));
  };

  const computeTargetSize = ({ srcW, srcH, mode, maxW, maxH, scalePct, keepAspect, noUpscale }) => {
    let w = srcW;
    let h = srcH;
    const safeW = clampInt(maxW, 1, 20000);
    const safeH = clampInt(maxH, 1, 20000);
    const safeScale = clampInt(scalePct, 1, 500);
    const aspect = srcW > 0 ? (srcH / srcW) : 1;

    switch (mode) {
      case 'scale': {
        const s = (safeScale === null ? 100 : safeScale) / 100;
        w = Math.round(srcW * s);
        h = Math.round(srcH * s);
        break;
      }
      case 'maxWidth': {
        const targetW = safeW === null ? srcW : safeW;
        if (keepAspect) {
          w = targetW;
          h = Math.round(targetW * aspect);
        } else {
          w = targetW;
          h = srcH;
        }
        break;
      }
      case 'maxHeight': {
        const targetH = safeH === null ? srcH : safeH;
        if (keepAspect) {
          h = targetH;
          w = Math.round(targetH / aspect);
        } else {
          h = targetH;
          w = srcW;
        }
        break;
      }
      case 'fit': {
        const boundW = safeW === null ? srcW : safeW;
        const boundH = safeH === null ? srcH : safeH;
        if (keepAspect) {
          const scale = Math.min(boundW / srcW, boundH / srcH);
          w = Math.round(srcW * scale);
          h = Math.round(srcH * scale);
        } else {
          w = boundW;
          h = boundH;
        }
        break;
      }
      case 'exact': {
        w = safeW === null ? srcW : safeW;
        h = safeH === null ? srcH : safeH;
        break;
      }
      case 'none':
      default:
        w = srcW;
        h = srcH;
        break;
    }

    w = Math.max(1, Math.round(w));
    h = Math.max(1, Math.round(h));

    if (noUpscale) {
      const scale = Math.min(1, srcW / w, srcH / h);
      w = Math.max(1, Math.round(w * scale));
      h = Math.max(1, Math.round(h * scale));
    }

    return { width: w, height: h };
  };

  const buildOutputVariants = ({
    srcW,
    srcH,
    responsiveEnabled,
    responsiveWidths,
    noUpscale,
    mode,
    maxW,
    maxH,
    scalePct,
    keepAspect
  }) => {
    const variants = [];
    if (responsiveEnabled && responsiveWidths.length) {
      responsiveWidths.forEach((width) => {
        if (noUpscale && width > srcW) return;
        const height = Math.max(1, Math.round((srcH / srcW) * width));
        variants.push({ width, height, variantWidth: width });
      });
    } else {
      const { width, height } = computeTargetSize({
        srcW,
        srcH,
        mode,
        maxW,
        maxH,
        scalePct,
        keepAspect,
        noUpscale
      });
      variants.push({ width, height, variantWidth: null });
    }
    return variants;
  };

  const buildOutputPlan = (settings) => {
    const variantsById = new Map();
    let estimatedBytes = 0;
    state.items.forEach((item) => {
      const source = validateImageDimensions(item.width, item.height, item.file.name);
      const variants = buildOutputVariants({
        srcW: source.width,
        srcH: source.height,
        ...settings
      });
      if (!variants.length) {
        throw new Error('No output sizes were generated. Try disabling "Prevent upscaling" or adjust widths.');
      }
      variants.forEach((variant) => {
        const output = validateImageDimensions(variant.width, variant.height, `${item.file.name} output`);
        estimatedBytes += output.pixels * 4;
      });
      variantsById.set(item.id, variants);
    });
    if (estimatedBytes > IMAGE_LIMITS.maxEstimatedOutputBytes) {
      throw new Error(`These settings need about ${formatBytes(estimatedBytes)} of output canvas memory; reduce image sizes or responsive widths to stay under ${formatBytes(IMAGE_LIMITS.maxEstimatedOutputBytes)}.`);
    }
    return { variantsById, estimatedBytes };
  };

  const canvasToBlob = (canvas, mime, quality) => new Promise((resolve, reject) => {
    try {
      canvas.toBlob((blob) => {
        if (!blob) reject(new Error('Unable to encode image'));
        else resolve(blob);
      }, mime, quality);
    } catch (err) {
      reject(err);
    }
  });

  const buildOutputName = ({ inputName, suffix, mime, variantWidth }) => {
    const { stem } = splitName(inputName);
    const base = sanitizeStem(stem);
    const ext = extForMime(mime);
    const widthPart = variantWidth ? `-${variantWidth}w` : '';
    return `${base}${suffix}${widthPart}${ext}`;
  };

  const makeUniqueNamer = () => {
    const counts = new Map();
    return (name) => {
      const key = String(name || '').toLowerCase();
      const current = counts.get(key) || 0;
      counts.set(key, current + 1);
      if (current === 0) return name;
      const dot = name.lastIndexOf('.');
      if (dot <= 0) return `${name}-${current + 1}`;
      return `${name.slice(0, dot)}-${current + 1}${name.slice(dot)}`;
    };
  };

  const renderOutputs = ({ responsiveEnabled }) => {
    resultsEl.innerHTML = '';
    if (!state.outputs.length) {
      setStatus('No outputs yet. Add images and click Optimize.');
      if (downloadAllBtn) downloadAllBtn.disabled = true;
      return;
    }

    const byInput = new Map();
    state.outputs.forEach((out) => {
      const key = String(out.inputId);
      const list = byInput.get(key) || [];
      list.push(out);
      byInput.set(key, list);
    });

    const frag = document.createDocumentFragment();
    byInput.forEach((outs, key) => {
      const item = getItemById(key);
      const group = document.createElement('section');
      group.className = 'imgopt-group';

      const head = document.createElement('div');
      head.className = 'imgopt-group-head';

      const titleWrap = document.createElement('div');
      const h3 = document.createElement('h3');
      h3.className = 'imgopt-group-title';
      h3.textContent = item?.file?.name || 'Image';
      const sub = document.createElement('p');
      sub.className = 'imgopt-group-sub';
      const dims = item?.width && item?.height ? `${item.width} × ${item.height}` : 'N/A';
      const origSize = item?.file?.size ? formatBytes(item.file.size) : 'N/A';
      sub.textContent = `Original: ${dims} · ${origSize}`;
      titleWrap.appendChild(h3);
      titleWrap.appendChild(sub);

      const summary = document.createElement('p');
      summary.className = 'imgopt-group-summary';
      const totalOut = outs.reduce((sum, o) => sum + (o.blob?.size || 0), 0);
      summary.textContent = `${outs.length} output${outs.length === 1 ? '' : 's'} · ${formatBytes(totalOut)}`;

      head.appendChild(titleWrap);
      head.appendChild(summary);
      group.appendChild(head);

      outs.sort((a, b) => (a.variantWidth || 0) - (b.variantWidth || 0));
      const ul = document.createElement('ul');
      ul.className = 'imgopt-output-list';
      ul.setAttribute('aria-label', 'Optimized outputs');

      outs.forEach((out) => {
        const li = document.createElement('li');
        li.className = 'imgopt-output';

        const meta = document.createElement('div');
        meta.className = 'imgopt-output-meta';

        const name = document.createElement('p');
        name.className = 'imgopt-output-name';
        name.textContent = out.name;

        const details = document.createElement('p');
        details.className = 'imgopt-output-sub';
        details.textContent = `${labelForMime(out.mime)} · ${out.width} × ${out.height} · ${formatBytes(out.blob.size)}`;

        meta.appendChild(name);
        meta.appendChild(details);

        const a = document.createElement('a');
        a.className = 'btn-secondary imgopt-download';
        a.href = out.url;
        a.download = out.name;
        a.textContent = 'Download';

        li.appendChild(meta);
        li.appendChild(a);
        ul.appendChild(li);
      });

      group.appendChild(ul);

      if (responsiveEnabled && outs.length > 1 && outs.every((o) => o.variantWidth)) {
        const srcset = outs
          .map((o) => `${o.name} ${o.variantWidth}w`)
          .join(', ');
        const srcsetWrap = document.createElement('div');
        srcsetWrap.className = 'imgopt-srcset';

        const row = document.createElement('div');
        row.className = 'imgopt-srcset-head';

        const label = document.createElement('p');
        label.className = 'imgopt-srcset-label';
        label.textContent = 'Srcset snippet';

        const copyBtn = document.createElement('button');
        copyBtn.type = 'button';
        copyBtn.className = 'btn-ghost imgopt-copy';
        copyBtn.textContent = 'Copy';

        const ta = document.createElement('textarea');
        ta.className = 'imgopt-srcset-text';
        ta.readOnly = true;
        ta.rows = 2;
        ta.value = srcset;

        copyBtn.addEventListener('click', async () => {
          try {
            if (navigator.clipboard?.writeText) {
              await navigator.clipboard.writeText(srcset);
            } else {
              ta.focus();
              ta.select();
              document.execCommand('copy');
            }
            copyBtn.textContent = 'Copied';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 900);
          } catch {
            setStatus('Unable to copy. Select the text and copy manually.');
          }
        });

        row.appendChild(label);
        row.appendChild(copyBtn);
        srcsetWrap.appendChild(row);
        srcsetWrap.appendChild(ta);
        group.appendChild(srcsetWrap);
      }

      frag.appendChild(group);
    });

    resultsEl.appendChild(frag);
    if (downloadAllBtn) downloadAllBtn.disabled = false;
    markSessionDirty();
  };

  const processAll = async () => {
    if (state.working) return;
    if (!state.items.length) {
      setStatus('Add at least one image first.');
      dropzone.focus();
      dispatchToolRunEvent('tools:run-error', { errorType: 'validation' });
      return;
    }

    revokeOutputs();
    state.cancelRequested = false;
    const operationId = state.operationId + 1;
    state.operationId = operationId;
    setWorking(true);
    setStatus('Finishing image checks...');

    try {
      await state.metadataQueue;
      throwIfCancelled(operationId);
      if (!state.items.length) {
        const error = new Error('No images passed the safety checks. Add smaller files and try again.');
        error.errorType = 'validation';
        throw error;
      }

      const suffix = normalizeSuffix(suffixInput?.value ?? '-optimized');
      const requestedMime = formatSelect.value;
      const keepAspect = Boolean(keepAspectInput?.checked);
      const noUpscale = Boolean(noUpscaleInput?.checked);
      const mode = resizeMode?.value || 'none';
      const maxW = widthInput?.value;
      const maxH = heightInput?.value;
      const scalePct = scaleInput?.value;
      const responsiveEnabled = Boolean(responsiveInput?.checked);
      const responsiveWidths = responsiveEnabled ? parseResponsiveWidths(responsiveWidthsInput?.value) : [];
      const chosenMime = requestedMime === 'keep' ? null : requestedMime;
      const selectedMime = chosenMime || 'keep';
      if (selectedMime !== 'keep' && !state.supports[selectedMime]) {
        const error = new Error(`${labelForMime(selectedMime)} export is not supported in this browser.`);
        error.errorType = 'unsupported';
        throw error;
      }

      const quality = getQuality();
      const flattenHex = (flattenColor?.value || '#0d1117').toLowerCase();
      const outputPlan = buildOutputPlan({
        responsiveEnabled,
        responsiveWidths,
        noUpscale,
        mode,
        maxW,
        maxH,
        scalePct,
        keepAspect
      });
      const makeUnique = makeUniqueNamer();
      let actualOutputBytes = 0;
      setStatus(`Output plan approved: about ${formatBytes(outputPlan.estimatedBytes)} of canvas memory, with saved files capped at ${formatBytes(IMAGE_LIMITS.maxActualOutputBytes)}. Starting...`);

      for (let idx = 0; idx < state.items.length; idx++) {
        throwIfCancelled(operationId);
        const item = state.items[idx];
        setStatus(`Optimizing ${idx + 1} / ${state.items.length}: ${item.file.name}`);

        let decoded = null;
        try {
          decoded = await decodeBitmap(item.file);
          throwIfCancelled(operationId);
          const source = validateImageDimensions(
            decoded.width || decoded.naturalWidth,
            decoded.height || decoded.naturalHeight,
            item.file.name
          );
          item.width = source.width;
          item.height = source.height;

          const outputMime = inferOutputMime(requestedMime, item.file.type);
          if (!state.supports[outputMime]) {
            const error = new Error(`Encoding not supported: ${outputMime}`);
            error.errorType = 'unsupported';
            throw error;
          }

          const variants = outputPlan.variantsById.get(item.id) || [];
          for (const variant of variants) {
            throwIfCancelled(operationId);
            const canvas = document.createElement('canvas');
            canvas.width = variant.width;
            canvas.height = variant.height;
            try {
              const ctx = canvas.getContext('2d', { alpha: true });
              if (!ctx) throw new Error('This browser could not allocate an output canvas. Try smaller dimensions.');
              ctx.imageSmoothingEnabled = true;
              ctx.imageSmoothingQuality = 'high';

              if (outputMime === 'image/jpeg') {
                ctx.fillStyle = flattenHex;
                ctx.fillRect(0, 0, canvas.width, canvas.height);
              }

              ctx.drawImage(decoded, 0, 0, canvas.width, canvas.height);
              const outBlob = await canvasToBlob(canvas, outputMime, quality);
              throwIfCancelled(operationId);
              if (actualOutputBytes + outBlob.size > IMAGE_LIMITS.maxActualOutputBytes) {
                throw new Error(`Encoded files would exceed the ${formatBytes(IMAGE_LIMITS.maxActualOutputBytes)} output limit. Reduce dimensions, quality, or responsive widths.`);
              }
              actualOutputBytes += outBlob.size;
              const outUrl = URL.createObjectURL(outBlob);
              const rawName = buildOutputName({
                inputName: item.file.name,
                suffix,
                mime: outputMime,
                variantWidth: variant.variantWidth
              });
              const outName = makeUnique(rawName);

              state.outputs.push({
                inputId: item.id,
                blob: outBlob,
                url: outUrl,
                name: outName,
                mime: outputMime,
                width: variant.width,
                height: variant.height,
                variantWidth: variant.variantWidth
              });
            } finally {
              canvas.width = 1;
              canvas.height = 1;
            }
          }
        } finally {
          if (decoded && typeof decoded.close === 'function') decoded.close();
        }

        await new Promise((r) => window.requestAnimationFrame(r));
      }

      renderOutputs({ responsiveEnabled });
      setStatus(`Done. Generated ${state.outputs.length} optimized ${state.outputs.length === 1 ? 'image' : 'images'} (${formatBytes(actualOutputBytes)} total).`);
      dispatchToolRunEvent('tools:run-complete', {
        resultBucket: state.outputs.length === 1 ? 'single_output' : 'multiple_outputs'
      });
    } catch (err) {
      revokeOutputs();
      setStatus(err?.message || 'Unable to optimize images. Please try different files or settings.');
      dispatchToolRunEvent('tools:run-error', {
        errorType: err?.errorType || (err?.name === 'AbortError' ? 'cancelled' : 'processing')
      });
    } finally {
      state.cancelRequested = false;
      setWorking(false);
    }
  };

  const triggerDownload = (out) => {
    const a = document.createElement('a');
    a.href = out.url;
    a.download = out.name;
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const downloadAll = async () => {
    if (state.working || !state.outputs.length) return;
    setWorking(true);
    try {
      setStatus(`Downloading ${state.outputs.length} file${state.outputs.length === 1 ? '' : 's'}…`);
      for (let i = 0; i < state.outputs.length; i++) {
        triggerDownload(state.outputs[i]);
        await new Promise((r) => setTimeout(r, 160));
      }
      setStatus('Download started. If prompted, allow multiple downloads.');
    } finally {
      setWorking(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-hover');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-hover');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-hover');
    const files = e.dataTransfer?.files;
    if (!files || !files.length) return;
    addFiles(files);
  };

  initEncodingSupport();
  updateControlsVisibility();
  updateQualityLabel();
  updateFlattenLabel();
  updateSummary();
  updateLayoutState();

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  dropzone.addEventListener('dragover', handleDrag);
  dropzone.addEventListener('dragenter', handleDrag);
  dropzone.addEventListener('dragleave', handleDragLeave);
  dropzone.addEventListener('drop', handleDrop);

  fileInput.addEventListener('change', () => {
    addFiles(fileInput.files);
    fileInput.value = '';
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    processAll();
  });

  fileList.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-imgopt-remove]');
    if (!btn || state.working) return;
    removeItem(btn.dataset.imgoptRemove);
  });

  clearBtn?.addEventListener('click', () => {
    if (state.working) {
      state.cancelRequested = true;
      clearBtn.disabled = true;
      setStatus('Cancelling after the current browser operation finishes...');
      return;
    }
    clearAll();
  });
  downloadAllBtn?.addEventListener('click', downloadAll);

  formatSelect.addEventListener('change', updateControlsVisibility);
  resizeMode?.addEventListener('change', updateControlsVisibility);
  responsiveInput?.addEventListener('change', updateControlsVisibility);
  qualityInput?.addEventListener('input', updateQualityLabel);
  flattenColor?.addEventListener('input', updateFlattenLabel);

  $$('[data-imgopt-pick]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      fileInput.click();
    });
  });

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (!detail || detail.toolId !== TOOL_ID) return;

    const payload = detail.payload;
    if (!payload || typeof payload !== 'object') return;

    const safeNames = state.items
      .map((it) => String(it?.file?.name || '').trim())
      .filter(Boolean);
    const namePreview = safeNames.slice(0, 6).join(', ');
    const nameSuffix = safeNames.length > 6 ? ` …+${safeNames.length - 6}` : '';
    const imagesLabel = safeNames.length
      ? `${safeNames.length} file${safeNames.length === 1 ? '' : 's'} (${namePreview}${nameSuffix})`
      : 'No files selected';

    const formatLabel = (() => {
      const value = String(formatSelect?.value || '').trim();
      if (!value || value === 'keep') return 'Keep original';
      return labelForMime(value);
    })();

    const mode = String(resizeMode?.value || 'none').trim() || 'none';
    const resizeLabel = (() => {
      if (responsiveInput?.checked) {
        const widths = String(responsiveWidthsInput?.value || '').trim();
        return widths ? `Responsive widths (${widths})` : 'Responsive widths';
      }
      if (mode === 'none') return 'None';
      if (mode === 'scale') return `Scale ${scaleInput?.value || ''}%`;
      const parts = [];
      if (widthInput?.value) parts.push(`W ${widthInput.value}px`);
      if (heightInput?.value) parts.push(`H ${heightInput.value}px`);
      return parts.length ? parts.join(' · ') : 'Resize';
    })();

    const quality = qualityInput?.value ? `Q ${qualityInput.value}` : '';

    payload.inputs = {
      Images: imagesLabel,
      Format: formatLabel,
      Resize: resizeLabel,
      ...(quality ? { Quality: quality } : {})
    };

    const totalBytes = state.outputs.reduce((sum, out) => sum + (out?.blob?.size || 0), 0);
    const outputSummary = state.outputs.length
      ? `${state.outputs.length} output${state.outputs.length === 1 ? '' : 's'} · ${formatBytes(totalBytes)}`
      : safeNames.length
        ? `${safeNames.length} image${safeNames.length === 1 ? '' : 's'} selected`
        : 'No outputs yet';

    payload.outputSummary = outputSummary;

    if (!state.outputs.length) return;

    const rows = state.outputs
      .slice()
      .sort((a, b) => String(a.name || '').localeCompare(String(b.name || '')))
      .map((out) => `${out.name} · ${labelForMime(out.mime)} · ${out.width} × ${out.height} · ${formatBytes(out.blob.size)}`);

    const clipped = rows.slice(0, MAX_SAVED_OUTPUT_LINES);
    if (rows.length > MAX_SAVED_OUTPUT_LINES) {
      clipped.push(`…and ${rows.length - MAX_SAVED_OUTPUT_LINES} more`);
    }

    payload.output = {
      kind: 'text',
      summary: outputSummary,
      text: clipped.join('\n')
    };
  });
})();
