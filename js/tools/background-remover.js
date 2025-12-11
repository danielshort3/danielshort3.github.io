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
  const dropBtn = $('#bgtool-drop');
  const toggleBtn = $('#bgtool-toggle-original');

  if (!form || !fileInput || !colorInput || !toleranceInput || !canvas) return;

  const ctx = canvas.getContext('2d');
  let imageBitmap = null;
  let currentImageData = null;
  let originalImageData = null;
  let processedImageData = null;
  let showingOriginal = false;
  let pendingSample = false;

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
    currentImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    originalImageData = currentImageData;
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
    currentImageData = null;
    imageBitmap = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    showOverlay('Upload an image to begin');
    dimLabel.textContent = 'Size: —';
    removedLabel.textContent = 'Removed pixels: —';
    status.textContent = '';
    downloadBtn.disabled = true;
    toggleBtn && (toggleBtn.disabled = true);
    processedImageData = null;
    originalImageData = null;
    showingOriginal = false;
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
      status.textContent = 'Image loaded. Adjust color and tolerance, then apply.';
      applyRemoval();
    } catch (err) {
      status.textContent = 'Unable to read that file. Please try another image.';
      reset();
    }
  };

  const handleCanvasClick = (event) => {
    if (!originalImageData) return;
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
    pendingSample = false;
    canvas.style.cursor = 'default';
    status.textContent = `Sampled ${hex} from the image.`;
    applyRemoval();
  };

  const triggerFileSelect = () => {
    fileInput.click();
  };

  const handleDrop = async (event) => {
    event.preventDefault();
    dropBtn?.classList.remove('bgtool-drop-hover');
    const files = event.dataTransfer?.files;
    if (!files || !files.length) return;
    fileInput.files = files;
    handleFileChange();
  };

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (!currentImageData) {
      status.textContent = 'Upload an image first.';
      return;
    }
    applyRemoval();
  });

  fileInput.addEventListener('change', handleFileChange);
  colorInput.addEventListener('input', () => setColorLabel(colorInput.value));
  toleranceInput.addEventListener('input', updateToleranceLabel);
  dropBtn?.addEventListener('click', triggerFileSelect);
  dropBtn?.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropBtn.classList.add('bgtool-drop-hover');
  });
  dropBtn?.addEventListener('dragleave', () => dropBtn.classList.remove('bgtool-drop-hover'));
  dropBtn?.addEventListener('drop', handleDrop);
  resetBtn?.addEventListener('click', reset);
  canvas.addEventListener('click', handleCanvasClick);
  canvas.addEventListener('mouseenter', () => {
    if (currentImageData) canvas.style.cursor = 'crosshair';
  });
  canvas.addEventListener('mouseleave', () => {
    canvas.style.cursor = 'default';
    pendingSample = false;
  });
  downloadBtn?.addEventListener('click', download);
  toggleBtn?.addEventListener('click', () => {
    if (!originalImageData) return;
    showingOriginal = !showingOriginal;
    drawCurrent();
    toggleBtn.textContent = showingOriginal ? 'Show transparent preview' : 'Show original';
    status.textContent = showingOriginal ? 'Viewing original image.' : 'Viewing transparent preview.';
  });

  updateToleranceLabel();
  setColorLabel(colorInput.value);
})();
