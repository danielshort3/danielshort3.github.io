(() => {
  'use strict';
  const $ = (sel) => document.querySelector(sel);
  const form = $('#bgtool-form');
  const fileInput = $('#bgtool-file');
  const colorInput = $('#bgtool-color');
  const colorLabel = $('#bgtool-color-label');
  const toleranceInput = $('#bgtool-tolerance');
  const toleranceValue = $('#bgtool-tolerance-value');
  const canvas = $('#bgtool-canvas');
  const overlay = $('#bgtool-overlay');
  const dimLabel = $('#bgtool-dim');
  const removedLabel = $('#bgtool-removed');
  const status = $('#bgtool-status');
  const downloadBtn = $('#bgtool-download');
  const resetBtn = $('#bgtool-reset');
  const toggleBtn = $('#bgtool-toggle-original');

  if (!form || !fileInput || !colorInput || !toleranceInput || !canvas) return;

  const TOOL_ID = 'background-remover';

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const ctx = canvas.getContext('2d');
  let imageBitmap = null;
  let originalImageData = null;
  let processedImageData = null;
  let showingOriginal = false;

  const updateToleranceLabel = () => {
    if (toleranceValue) toleranceValue.textContent = toleranceInput.value;
  };

  const hexToRgb = (hex) => {
    const cleaned = hex.replace('#', '');
    const num = parseInt(cleaned, 16);
    return {
      r: (num >> 16) & 255,
      g: (num >> 8) & 255,
      b: num & 255
    };
  };

  const rgbToHex = (r, g, b) => {
    const toHex = (v) => v.toString(16).padStart(2, '0');
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
  };

  const setColorLabel = (hex) => {
    colorLabel.textContent = hex.toLowerCase();
  };

  const readFile = async (file) => {
    const buffer = await file.arrayBuffer();
    const blob = new Blob([buffer]);
    return await createImageBitmap(blob);
  };

  const resizeCanvas = (bitmap) => {
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
  };

  const showOverlay = (msg) => {
    if (!overlay) return;
    overlay.textContent = msg || 'Upload an image to begin';
    overlay.style.display = 'flex';
  };

  const hideOverlay = () => {
    if (overlay) overlay.style.display = 'none';
  };

  const renderImage = (bitmap) => {
    resizeCanvas(bitmap);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bitmap, 0, 0);
    originalImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    processedImageData = null;
    showingOriginal = false;
    dimLabel.textContent = `Size: ${bitmap.width} × ${bitmap.height}`;
    removedLabel.textContent = 'Removed pixels: 0';
    downloadBtn.disabled = false;
    toggleBtn && (toggleBtn.disabled = false);
    hideOverlay();
  };

  const drawCurrent = () => {
    if (!originalImageData) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      showOverlay('Upload an image to begin');
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const data = (showingOriginal || !processedImageData) ? originalImageData : processedImageData;
    ctx.putImageData(data, 0, 0);
    hideOverlay();
  };

  const applyRemoval = () => {
    if (!originalImageData) return;
    const tolerance = parseInt(toleranceInput.value, 10) || 0;
    const { r: tr, g: tg, b: tb } = hexToRgb(colorInput.value || '#ffffff');
    const data = new Uint8ClampedArray(originalImageData.data);
    let removed = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const dist = Math.sqrt(
        Math.pow(r - tr, 2) +
        Math.pow(g - tg, 2) +
        Math.pow(b - tb, 2)
      );
      if (dist <= tolerance) {
        data[i + 3] = 0; // alpha -> transparent
        removed++;
      }
    }
    processedImageData = new ImageData(data, originalImageData.width, originalImageData.height);
    showingOriginal = false;
    drawCurrent();
    removedLabel.textContent = `Removed pixels: ${removed.toLocaleString('en-US')}`;
    status.textContent = `Applied removal with tolerance ${tolerance}.`;
  };

  const download = () => {
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = 'background-removed.png';
    a.click();
  };

  const reset = () => {
    fileInput.value = '';
    colorInput.value = '#ffffff';
    setColorLabel('#ffffff');
    toleranceInput.value = '24';
    updateToleranceLabel();
    imageBitmap = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    showOverlay('Upload an image to begin');
    dimLabel.textContent = 'Size: N/A';
    removedLabel.textContent = 'Removed pixels: N/A';
    status.textContent = '';
    downloadBtn.disabled = true;
    toggleBtn && (toggleBtn.disabled = true);
    processedImageData = null;
    originalImageData = null;
    showingOriginal = false;
    markSessionDirty();
  };

  const handleFileChange = async () => {
    const [file] = fileInput.files || [];
    if (!file) {
      reset();
      return;
    }
    if (!/^image\//.test(file.type)) {
      status.textContent = 'Please choose a PNG or JPG image.';
      return;
    }
    try {
      showOverlay('Loading image...');
      imageBitmap = await readFile(file);
      renderImage(imageBitmap);
      status.textContent = 'Image loaded. Click the image to sample or adjust tolerance.';
      applyRemoval();
    } catch (err) {
      status.textContent = 'Unable to read that file. Please try another image.';
      reset();
    }
  };

  const handleCanvasClick = (event) => {
    if (!originalImageData) {
      fileInput.click();
      status.textContent = 'Choose an image to start.';
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) * (canvas.width / rect.width));
    const y = Math.floor((event.clientY - rect.top) * (canvas.height / rect.height));
    const idx = (y * canvas.width + x) * 4;
    const data = originalImageData.data;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    const hex = rgbToHex(r, g, b);
    colorInput.value = hex;
    setColorLabel(hex);
    status.textContent = `Sampled ${hex} from the image.`;
    applyRemoval();
    markSessionDirty();
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (!originalImageData) {
      status.textContent = 'Upload an image first.';
      return;
    }
    applyRemoval();
  });

  fileInput.addEventListener('change', handleFileChange);
  colorInput.addEventListener('input', () => {
    setColorLabel(colorInput.value);
    if (originalImageData) applyRemoval();
  });
  toleranceInput.addEventListener('input', () => {
    updateToleranceLabel();
    if (originalImageData) applyRemoval();
  });
  resetBtn?.addEventListener('click', reset);
  canvas.addEventListener('click', handleCanvasClick);
  canvas.addEventListener('mouseenter', () => {
    canvas.style.cursor = 'crosshair';
  });
  canvas.addEventListener('mouseleave', () => {
    canvas.style.cursor = 'default';
  });
  canvas.addEventListener('dragover', (e) => {
    e.preventDefault();
    canvas.parentElement?.classList.add('drag-hover');
  });
  canvas.addEventListener('dragleave', (e) => {
    e.preventDefault();
    canvas.parentElement?.classList.remove('drag-hover');
  });
  canvas.addEventListener('drop', (e) => {
    e.preventDefault();
    canvas.parentElement?.classList.remove('drag-hover');
    const files = e.dataTransfer?.files;
    if (!files || !files.length) return;
    fileInput.files = files;
    handleFileChange();
  });
  overlay?.addEventListener('click', () => fileInput.click());
  overlay?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });
  overlay?.addEventListener('dragover', (e) => {
    e.preventDefault();
    canvas.parentElement?.classList.add('drag-hover');
  });
  overlay?.addEventListener('dragleave', (e) => {
    e.preventDefault();
    canvas.parentElement?.classList.remove('drag-hover');
  });
  overlay?.addEventListener('drop', (e) => {
    e.preventDefault();
    canvas.parentElement?.classList.remove('drag-hover');
    const files = e.dataTransfer?.files;
    if (!files || !files.length) return;
    fileInput.files = files;
    handleFileChange();
  });
  downloadBtn?.addEventListener('click', download);
  toggleBtn?.addEventListener('click', () => {
    if (!originalImageData) return;
    showingOriginal = !showingOriginal;
    drawCurrent();
    toggleBtn.textContent = showingOriginal ? 'Show transparent preview' : 'Show original';
    status.textContent = showingOriginal ? 'Viewing original image.' : 'Viewing transparent preview.';
    markSessionDirty();
  });

  updateToleranceLabel();
  setColorLabel(colorInput.value);

  const captureSummary = () => {
    const parts = [
      String(status?.textContent || '').trim(),
      String(removedLabel?.textContent || '').trim(),
      String(dimLabel?.textContent || '').trim(),
    ].filter(Boolean);
    return parts.join(' · ');
  };

  const buildPreviewDataUrl = (maxDim = 420) => {
    try {
      const srcW = canvas.width || 0;
      const srcH = canvas.height || 0;
      if (!srcW || !srcH) return null;
      const scale = Math.min(1, maxDim / Math.max(srcW, srcH));
      const w = Math.max(1, Math.round(srcW * scale));
      const h = Math.max(1, Math.round(srcH * scale));
      const preview = document.createElement('canvas');
      preview.width = w;
      preview.height = h;
      const pctx = preview.getContext('2d', { alpha: true });
      if (!pctx) return null;
      pctx.drawImage(canvas, 0, 0, w, h);
      const dataUrl = preview.toDataURL('image/png');
      if (!dataUrl.startsWith('data:image/png')) return null;
      return { dataUrl, width: w, height: h };
    } catch {
      return null;
    }
  };

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const file = fileInput.files?.[0];
    payload.inputs = {
      File: file?.name || 'No file selected',
      Color: String(colorInput.value || '').trim(),
      Tolerance: String(toleranceInput.value || '').trim()
    };

    const summary = captureSummary();
    payload.outputSummary = summary || 'Background removed';

    const preview = buildPreviewDataUrl(420);
    if (preview?.dataUrl) {
      payload.output = {
        kind: 'image',
        mime: 'image/png',
        dataUrl: preview.dataUrl,
        width: preview.width,
        height: preview.height,
        summary
      };
    }
  });
})();
