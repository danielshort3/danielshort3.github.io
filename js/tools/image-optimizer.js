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

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const state = {
    items: [],
    outputs: [],
    working: false,
    nextId: 1,
    supports: {}
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
    if (processBtn) processBtn.disabled = state.working;
    if (clearBtn) clearBtn.disabled = state.working;
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
  };

  const revokeOutputs = () => {
    state.outputs.forEach((out) => {
      if (out.url) URL.revokeObjectURL(out.url);
    });
    state.outputs = [];
    if (resultsEl) resultsEl.innerHTML = '';
    if (downloadAllBtn) downloadAllBtn.disabled = true;
    markSessionDirty();
  };

  const clearAll = () => {
    if (state.working) return;
    state.items.forEach((it) => {
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

  const getItemById = (id) => state.items.find((it) => String(it.id) === String(id));

  const removeItem = (id) => {
    const idx = state.items.findIndex((it) => String(it.id) === String(id));
    if (idx < 0) return;
    const [removed] = state.items.splice(idx, 1);
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
      thumb.src = it.previewUrl;

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
    if (!item || item.width) return;
    try {
      const decoded = await decodeBitmap(item.file);
      item.width = decoded.width || decoded.naturalWidth || null;
      item.height = decoded.height || decoded.naturalHeight || null;
      if (decoded && typeof decoded.close === 'function') decoded.close();
      renderFileList();
      updateSummary();
    } catch {
      // Ignore decode errors; user will see failures on processing.
    }
  };

  const addFiles = (list) => {
    const files = Array.from(list || []);
    const accepted = files.filter((f) => f && /^image\//.test(f.type || 'image/'));
    if (!accepted.length) {
      setStatus('No supported image files were detected.');
      return;
    }

    accepted.forEach((file) => {
      const id = state.nextId++;
      const previewUrl = URL.createObjectURL(file);
      const item = { id, file, previewUrl, width: null, height: null };
      state.items.push(item);
      hydrateItemMetadata(item);
    });

    renderFileList();
    updateSummary();
    revokeOutputs();
    setStatus(`${accepted.length} ${accepted.length === 1 ? 'image added' : 'images added'}. Click Optimize to generate downloads.`);
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
      return;
    }

    revokeOutputs();
    setWorking(true);
    setStatus('Preparing…');

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
      setStatus(`${labelForMime(selectedMime)} export is not supported in this browser.`);
      setWorking(false);
      return;
    }

    const quality = getQuality();
    const flattenHex = (flattenColor?.value || '#0d1117').toLowerCase();

    const makeUnique = makeUniqueNamer();

    try {
      for (let idx = 0; idx < state.items.length; idx++) {
        const item = state.items[idx];
        setStatus(`Optimizing ${idx + 1} / ${state.items.length}: ${item.file.name}`);

        const decoded = await decodeBitmap(item.file);
        const srcW = decoded.width || decoded.naturalWidth;
        const srcH = decoded.height || decoded.naturalHeight;

        item.width = item.width || srcW;
        item.height = item.height || srcH;

        const outputMime = inferOutputMime(requestedMime, item.file.type);
        if (!state.supports[outputMime]) {
          throw new Error(`Encoding not supported: ${outputMime}`);
        }

        const variants = [];
        if (responsiveEnabled && responsiveWidths.length) {
          responsiveWidths.forEach((w) => {
            if (noUpscale && w > srcW) return;
            const h = Math.max(1, Math.round((srcH / srcW) * w));
            variants.push({ width: w, height: h, variantWidth: w });
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

        if (!variants.length) {
          throw new Error('No output sizes were generated. Try disabling “Prevent upscaling” or adjust widths.');
        }

        for (const v of variants) {
          const canvas = document.createElement('canvas');
          canvas.width = v.width;
          canvas.height = v.height;
          const ctx = canvas.getContext('2d', { alpha: true });
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = 'high';

          if (outputMime === 'image/jpeg') {
            ctx.fillStyle = flattenHex;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
          }

          ctx.drawImage(decoded, 0, 0, canvas.width, canvas.height);

          const outBlob = await canvasToBlob(canvas, outputMime, quality);
          const outUrl = URL.createObjectURL(outBlob);
          const rawName = buildOutputName({
            inputName: item.file.name,
            suffix,
            mime: outputMime,
            variantWidth: v.variantWidth
          });
          const outName = makeUnique(rawName);

          state.outputs.push({
            inputId: item.id,
            blob: outBlob,
            url: outUrl,
            name: outName,
            mime: outputMime,
            width: canvas.width,
            height: canvas.height,
            variantWidth: v.variantWidth
          });
        }

        if (decoded && typeof decoded.close === 'function') decoded.close();
        await new Promise((r) => window.requestAnimationFrame(r));
      }

      renderOutputs({ responsiveEnabled });
      setStatus(`Done. Generated ${state.outputs.length} optimized ${state.outputs.length === 1 ? 'image' : 'images'}.`);
    } catch (err) {
      revokeOutputs();
      setStatus(err?.message || 'Unable to optimize images. Please try different files or settings.');
    } finally {
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

  clearBtn?.addEventListener('click', clearAll);
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
