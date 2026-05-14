(() => {
  'use strict';

  const TOOL_ID = 'whisper-transcribe-monitor';
  const API_BASE = '/api/tools/transcribe';
  const DEFAULT_CONFIG = {
    service: 'Amazon Transcribe',
    region: 'us-east-2',
    languageCode: 'en-US',
    pricePerSecond: 0.0004,
    pricePerMinute: 0.024,
    minDurationSeconds: 15,
    minBillableSeconds: 15,
    maxFilesPerRun: 10,
    maxFileBytes: 500 * 1024 * 1024,
    maxTotalCostUsd: 10,
    supportedFormats: ['amr', 'flac', 'm4a', 'mp3', 'mp4', 'ogg', 'wav', 'webm']
  };
  const VIDEO_FORMATS = new Set(['mp4', 'webm']);
  const POLL_INTERVAL_MS = 5000;
  const DURATION_TIMEOUT_MS = 10000;

  const $id = (id) => document.getElementById(id);

  const authPillEl = $id('transcribe-auth-pill');
  const authStatusEl = $id('transcribe-auth-status');
  const signInBtn = $id('transcribe-sign-in');
  const serviceEl = $id('transcribe-stat-service');
  const priceEl = $id('transcribe-stat-price');
  const minimumEl = $id('transcribe-stat-minimum');
  const acceptedEl = $id('transcribe-stat-accepted');
  const formEl = $id('transcribe-form');
  const fileEl = $id('transcribe-files');
  const summaryEl = $id('transcribe-summary');
  const tableWrapEl = $id('transcribe-table-wrap');
  const tableBodyEl = $id('transcribe-file-rows');
  const totalEl = $id('transcribe-total');
  const approveEl = $id('transcribe-approve');
  const startBtn = $id('transcribe-start');
  const cancelBtn = $id('transcribe-cancel');
  const resetBtn = $id('transcribe-reset');
  const runStatusEl = $id('transcribe-run-status');
  const progressWrapEl = $id('transcribe-progress-wrap');
  const progressLabelEl = $id('transcribe-progress-label');
  const progressBarEl = $id('transcribe-progress');
  const resultsEl = $id('transcribe-results');

  if (!formEl || !fileEl || !startBtn || !tableBodyEl) return;

  const state = {
    config: { ...DEFAULT_CONFIG },
    files: [],
    busy: false,
    canceled: false,
    activeXhr: null,
    activeController: null,
    analyzing: false
  };

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const cleanText = (value) => String(value || '').replace(/\s+/g, ' ').trim();

  const setText = (el, value) => {
    if (el) el.textContent = value || '';
  };

  const setStatus = (el, message, tone) => {
    if (!el) return;
    el.textContent = message || '';
    el.dataset.tone = tone || '';
  };

  const setAuthPill = (stateName, label) => {
    if (!authPillEl) return;
    authPillEl.dataset.state = stateName || '';
    authPillEl.textContent = label || '';
  };

  const formatNumber = (value, digits = 2) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return '--';
    return num.toFixed(digits).replace(/\.00$/, '');
  };

  const formatBytes = (bytes) => {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return '--';
    const mb = value / (1024 * 1024);
    if (mb >= 1024) return `${formatNumber(mb / 1024, 2)} GB`;
    if (mb >= 1) return `${formatNumber(mb, 2)} MB`;
    return `${formatNumber(value / 1024, 1)} KB`;
  };

  const formatClock = (totalSeconds) => {
    const seconds = Math.max(0, Math.round(Number(totalSeconds) || 0));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    const pad = (value) => String(value).padStart(2, '0');
    if (hours > 0) return `${hours}:${pad(minutes)}:${pad(secs)}`;
    return `${pad(minutes)}:${pad(secs)}`;
  };

  const formatUsd = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num) || num <= 0) return '$0.00';
    const digits = num < 0.01 ? 4 : 2;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    }).format(num);
  };

  const getExtension = (name) => {
    const match = String(name || '').toLowerCase().match(/\.([a-z0-9]+)$/);
    return match ? match[1] : '';
  };

  const safeDownloadName = (name) => {
    const base = String(name || 'transcript').replace(/\.[^.]+$/, '') || 'transcript';
    return `${base.replace(/[^a-zA-Z0-9._-]+/g, '_') || 'transcript'}-transcript.txt`;
  };

  const supportedFormats = () => new Set(
    Array.isArray(state.config.supportedFormats)
      ? state.config.supportedFormats.map((item) => String(item).toLowerCase())
      : DEFAULT_CONFIG.supportedFormats
  );

  const billableSeconds = (durationSeconds) => {
    const duration = Number(durationSeconds);
    if (!Number.isFinite(duration) || duration <= 0) return 0;
    return Math.max(Number(state.config.minBillableSeconds) || 15, Math.ceil(duration));
  };

  const estimatedCost = (durationSeconds) => {
    const cost = billableSeconds(durationSeconds) * Number(state.config.pricePerSecond || DEFAULT_CONFIG.pricePerSecond);
    return Number(cost.toFixed(6));
  };

  const acceptedFiles = () => state.files.filter((item) => item.status !== 'skipped');

  const completedFiles = () => state.files.filter((item) => item.status === 'complete');

  const estimatedTotal = () => acceptedFiles().reduce((sum, item) => sum + Number(item.estimatedCostUsd || 0), 0);

  const finalTotal = () => completedFiles().reduce((sum, item) => sum + Number(item.costUsd || item.estimatedCostUsd || 0), 0);

  const authIsReady = () => {
    const authApi = window.ToolsAuth;
    if (!authApi || !authApi.getAuth || !authApi.authIsValid) return false;
    return authApi.authIsValid(authApi.getAuth());
  };

  const updateLayoutState = () => {
    const hasResults = state.files.some((item) => item.transcript || item.status === 'complete' || item.status === 'failed');
    const hasFiles = state.files.length > 0;
    const nextState = state.busy
      ? 'working'
      : hasResults
        ? 'results'
        : hasFiles
          ? 'ready'
          : 'empty';
    if (document.body) document.body.dataset.toolsState = nextState;
  };

  const setBusy = (busy) => {
    state.busy = Boolean(busy);
    if (fileEl) fileEl.disabled = state.busy || state.analyzing;
    if (resetBtn) resetBtn.disabled = state.busy || state.analyzing;
    if (cancelBtn) cancelBtn.disabled = !state.busy;
    if (startBtn) {
      startBtn.classList.toggle('is-busy', state.busy);
    }
    updateControls();
    updateLayoutState();
  };

  const updateProgress = ({ stateName = 'hidden', ratio = 0, label = '' } = {}) => {
    if (!progressWrapEl || !progressBarEl) return;
    progressWrapEl.dataset.state = stateName;
    if (stateName === 'hidden') {
      progressBarEl.value = 0;
      setText(progressLabelEl, 'Progress');
      return;
    }
    const safeRatio = Math.min(1, Math.max(0, Number(ratio) || 0));
    progressBarEl.value = safeRatio;
    setText(progressLabelEl, label || 'Progress');
  };

  const updateStats = () => {
    setText(serviceEl, state.config.service || 'Amazon Transcribe');
    setText(priceEl, `${formatUsd(state.config.pricePerMinute)} / min`);
    setText(minimumEl, `${Number(state.config.minDurationSeconds) || 15} sec`);
    setText(acceptedEl, String(acceptedFiles().length));
  };

  const updateAuthUi = () => {
    if (!window.ToolsAuth) {
      setAuthPill('warning', 'Loading auth');
      setStatus(authStatusEl, 'Loading tools account sign-in.', 'warning');
      if (signInBtn) signInBtn.disabled = true;
      updateControls();
      return;
    }

    const authed = authIsReady();
    if (authed) {
      const user = window.ToolsAuth.getUser ? window.ToolsAuth.getUser(window.ToolsAuth.getAuth()) : {};
      const label = cleanText(user.email || user.name) || 'Signed in';
      setAuthPill('ok', 'Signed in');
      setStatus(authStatusEl, `${label} can start transcription jobs.`, 'success');
      if (signInBtn) signInBtn.disabled = true;
    } else {
      setAuthPill('err', 'Sign in required');
      setStatus(authStatusEl, 'Sign in before uploading files or starting paid transcription jobs.', 'warning');
      if (signInBtn) signInBtn.disabled = false;
    }
    updateControls();
  };

  const updateSummary = () => {
    const totalCount = state.files.length;
    const readyCount = acceptedFiles().length;
    const skippedCount = state.files.filter((item) => item.status === 'skipped').length;
    const completedCount = completedFiles().length;
    const failedCount = state.files.filter((item) => item.status === 'failed').length;
    const totalCost = estimatedTotal();
    const runCost = finalTotal();

    if (!totalCount) {
      setStatus(summaryEl, 'No files selected.', '');
      setText(totalEl, 'Estimated total: $0.00');
      return;
    }

    const parts = [`${readyCount} ready`, `${skippedCount} skipped`];
    if (completedCount || failedCount) parts.push(`${completedCount} complete`, `${failedCount} failed`);
    setStatus(summaryEl, `${parts.join(' · ')}. Estimated total ${formatUsd(totalCost)}.`, skippedCount ? 'warning' : 'success');
    setText(totalEl, completedCount ? `Completed run total: ${formatUsd(runCost)} · Estimated total: ${formatUsd(totalCost)}` : `Estimated total: ${formatUsd(totalCost)}`);
  };

  const statusLabel = (item) => {
    if (item.status === 'checking') return 'Checking';
    if (item.status === 'skipped') return `Skipped: ${item.skipReason || 'Not eligible'}`;
    if (item.status === 'ready') return 'Ready';
    if (item.status === 'presigning') return 'Preparing upload';
    if (item.status === 'uploading') return `Uploading ${Math.round((Number(item.progress) || 0) * 100)}%`;
    if (item.status === 'starting') return 'Starting job';
    if (item.status === 'transcribing') return 'Transcribing';
    if (item.status === 'complete') return 'Complete';
    if (item.status === 'failed') return `Failed: ${item.error || 'Transcription failed'}`;
    if (item.status === 'canceled') return 'Canceled';
    return cleanText(item.status) || 'Pending';
  };

  const rowTone = (item) => {
    if (item.status === 'skipped') return 'warning';
    if (item.status === 'failed') return 'error';
    if (item.status === 'complete') return 'success';
    if (item.status === 'uploading' || item.status === 'transcribing') return 'active';
    return '';
  };

  const renderTable = () => {
    if (tableWrapEl) tableWrapEl.hidden = state.files.length === 0;
    tableBodyEl.innerHTML = state.files.map((item) => `
      <tr data-tone="${escapeHtml(rowTone(item))}">
        <td>
          <span class="transcribe-file-name">${escapeHtml(item.name)}</span>
          <span class="transcribe-file-meta">${escapeHtml(formatBytes(item.bytes))}${item.extension ? ` · ${escapeHtml(item.extension.toUpperCase())}` : ''}</span>
        </td>
        <td>${item.durationSeconds ? escapeHtml(formatClock(item.durationSeconds)) : '--'}</td>
        <td>${item.billableSeconds ? `${escapeHtml(String(item.billableSeconds))} sec` : '--'}</td>
        <td>${escapeHtml(formatUsd(item.estimatedCostUsd || 0))}</td>
        <td>${escapeHtml(statusLabel(item))}</td>
      </tr>
    `).join('');
    updateStats();
    updateSummary();
    updateControls();
    updateLayoutState();
  };

  const renderResults = () => {
    if (!resultsEl) return;
    const resultItems = state.files.filter((item) => item.transcript || item.status === 'complete' || item.status === 'failed');
    if (!resultItems.length) {
      resultsEl.innerHTML = '<p class="transcribe-empty">Completed transcripts will appear here.</p>';
      return;
    }
    resultsEl.innerHTML = resultItems.map((item) => {
      const transcript = String(item.transcript || '').trim();
      const status = item.status === 'complete'
        ? `Cost: ${formatUsd(item.costUsd || item.estimatedCostUsd || 0)} · ${item.billableSeconds || 0} billable sec`
        : escapeHtml(item.error || 'Transcription failed.');
      return `
        <article class="transcribe-result" data-status="${escapeHtml(item.status)}" data-id="${escapeHtml(item.id)}">
          <div class="transcribe-result-header">
            <div>
              <h3>${escapeHtml(item.name)}</h3>
              <p>${status}</p>
            </div>
            <div class="transcribe-result-actions">
              <button type="button" class="btn-secondary" data-transcribe-action="copy" data-id="${escapeHtml(item.id)}" ${transcript ? '' : 'disabled'}>Copy</button>
              <button type="button" class="btn-secondary" data-transcribe-action="download" data-id="${escapeHtml(item.id)}" ${transcript ? '' : 'disabled'}>Download</button>
            </div>
          </div>
          <textarea readonly>${escapeHtml(transcript)}</textarea>
        </article>
      `;
    }).join('');
  };

  const updateControls = () => {
    const readyCount = acceptedFiles().filter((item) => !['complete', 'failed'].includes(item.status)).length;
    const approved = Boolean(approveEl && approveEl.checked);
    const canStart = !state.busy && !state.analyzing && authIsReady() && approved && readyCount > 0;
    if (approveEl) approveEl.disabled = state.busy || state.analyzing || readyCount === 0;
    if (startBtn) startBtn.disabled = !canStart;
  };

  const readJson = async (res) => {
    let data = null;
    try {
      data = await res.json();
    } catch {
      data = null;
    }
    if (!res.ok || data?.ok === false) {
      const err = new Error(data?.error || data?.message || `Request failed (${res.status}).`);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  };

  const authFetchJson = async (url, options = {}) => {
    if (!window.ToolsAuth || !window.ToolsAuth.fetchWithAuth) {
      throw new Error('Sign in before starting transcription jobs.');
    }
    const res = await window.ToolsAuth.fetchWithAuth(url, options);
    return readJson(res);
  };

  const loadConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/config`, { method: 'GET' });
      const data = await readJson(res);
      state.config = { ...DEFAULT_CONFIG, ...data };
      updateStats();
      updateSummary();
    } catch (err) {
      setStatus(runStatusEl, err?.message || 'Transcribe configuration is unavailable.', 'warning');
      state.config = { ...DEFAULT_CONFIG };
      updateStats();
    }
  };

  const probeDuration = (file, extension) => new Promise((resolve) => {
    if (!file || !window.URL || typeof window.URL.createObjectURL !== 'function') {
      resolve(null);
      return;
    }

    const src = window.URL.createObjectURL(file);
    const tag = VIDEO_FORMATS.has(extension) || String(file.type || '').toLowerCase().startsWith('video/')
      ? 'video'
      : 'audio';
    const el = document.createElement(tag);
    let done = false;
    let timeoutId = null;

    const finish = (value) => {
      if (done) return;
      done = true;
      if (timeoutId) window.clearTimeout(timeoutId);
      try {
        el.removeAttribute('src');
        el.load();
      } catch {}
      try {
        window.URL.revokeObjectURL(src);
      } catch {}
      const duration = Number(value);
      resolve(Number.isFinite(duration) && duration > 0 ? duration : null);
    };

    timeoutId = window.setTimeout(() => finish(null), DURATION_TIMEOUT_MS);
    el.preload = 'metadata';
    el.muted = true;
    el.playsInline = true;
    el.onloadedmetadata = () => finish(el.duration);
    el.onerror = () => finish(null);
    el.src = src;
  });

  const analyzeSelectedFiles = async () => {
    const files = Array.from(fileEl.files || []);
    state.analyzing = true;
    state.files = files.map((file, index) => ({
      id: `${Date.now()}-${index}-${Math.random().toString(36).slice(2)}`,
      file,
      name: file.name || `file-${index + 1}`,
      extension: getExtension(file.name || ''),
      contentType: file.type || 'application/octet-stream',
      bytes: Number(file.size) || 0,
      durationSeconds: null,
      billableSeconds: 0,
      estimatedCostUsd: 0,
      costUsd: 0,
      progress: 0,
      status: 'checking',
      skipReason: '',
      transcript: ''
    }));
    if (approveEl) approveEl.checked = false;
    renderTable();
    renderResults();
    setStatus(runStatusEl, files.length ? 'Checking file durations...' : '', '');

    const formats = supportedFormats();
    let acceptedCost = 0;
    let acceptedCount = 0;

    for (let i = 0; i < state.files.length; i += 1) {
      const item = state.files[i];
      if (i >= Number(state.config.maxFilesPerRun || DEFAULT_CONFIG.maxFilesPerRun)) {
        item.status = 'skipped';
        item.skipReason = `Run limit is ${state.config.maxFilesPerRun} files.`;
        renderTable();
        continue;
      }
      if (!formats.has(item.extension)) {
        item.status = 'skipped';
        item.skipReason = 'Unsupported file type.';
        renderTable();
        continue;
      }
      if (!item.bytes) {
        item.status = 'skipped';
        item.skipReason = 'Empty file.';
        renderTable();
        continue;
      }
      if (item.bytes > Number(state.config.maxFileBytes || DEFAULT_CONFIG.maxFileBytes)) {
        item.status = 'skipped';
        item.skipReason = `File exceeds ${formatBytes(state.config.maxFileBytes)}.`;
        renderTable();
        continue;
      }

      item.durationSeconds = await probeDuration(item.file, item.extension);
      if (!Number.isFinite(Number(item.durationSeconds))) {
        item.status = 'skipped';
        item.skipReason = 'Unable to read duration before upload.';
        renderTable();
        continue;
      }
      if (item.durationSeconds < Number(state.config.minDurationSeconds || 15)) {
        item.status = 'skipped';
        item.skipReason = `Under ${state.config.minDurationSeconds || 15} seconds.`;
        renderTable();
        continue;
      }

      item.billableSeconds = billableSeconds(item.durationSeconds);
      item.estimatedCostUsd = estimatedCost(item.durationSeconds);
      if (acceptedCost + item.estimatedCostUsd > Number(state.config.maxTotalCostUsd || DEFAULT_CONFIG.maxTotalCostUsd)) {
        item.status = 'skipped';
        item.skipReason = `Total estimate cap is ${formatUsd(state.config.maxTotalCostUsd)}.`;
        renderTable();
        continue;
      }

      item.status = 'ready';
      acceptedCost += item.estimatedCostUsd;
      acceptedCount += 1;
      renderTable();
    }

    state.analyzing = false;
    setStatus(
      runStatusEl,
      acceptedCount ? `Ready to transcribe ${acceptedCount} file${acceptedCount === 1 ? '' : 's'}.` : 'No selected files are eligible for transcription.',
      acceptedCount ? 'success' : 'warning'
    );
    renderTable();
    markSessionDirty();
  };

  const abortActive = () => {
    try {
      if (state.activeXhr && state.activeXhr.readyState !== 4) state.activeXhr.abort();
    } catch {}
    try {
      if (state.activeController) state.activeController.abort();
    } catch {}
    state.activeXhr = null;
    state.activeController = null;
  };

  const uploadFile = (item, uploadUrl, headers = {}) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.activeXhr = xhr;
    xhr.open('PUT', uploadUrl, true);
    Object.entries(headers || {}).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      try {
        xhr.setRequestHeader(key, String(value));
      } catch {}
    });
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      item.progress = event.total > 0 ? event.loaded / event.total : 0;
      renderTable();
      updateProgress({
        stateName: 'visible',
        ratio: item.progress,
        label: `Uploading ${item.name} (${Math.round(item.progress * 100)}%)`
      });
    };
    xhr.onload = () => {
      state.activeXhr = null;
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
        return;
      }
      reject(new Error(`Upload failed (${xhr.status}).`));
    };
    xhr.onerror = () => {
      state.activeXhr = null;
      reject(new Error('Upload failed.'));
    };
    xhr.onabort = () => {
      state.activeXhr = null;
      reject(new Error('Upload canceled.'));
    };
    xhr.send(item.file);
  });

  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const pollRun = async (item, runToken) => {
    while (!state.canceled) {
      const controller = new AbortController();
      state.activeController = controller;
      let data;
      try {
        data = await authFetchJson(`${API_BASE}/status?run=${encodeURIComponent(runToken)}`, {
          method: 'GET',
          signal: controller.signal
        });
      } finally {
        if (state.activeController === controller) state.activeController = null;
      }

      const status = String(data.status || '').toUpperCase();
      if (status === 'COMPLETED') {
        item.status = 'complete';
        item.transcript = String(data.transcript || '').trim();
        item.costUsd = Number(data.costUsd || item.estimatedCostUsd || 0);
        item.billableSeconds = Number(data.billableSeconds || item.billableSeconds || 0);
        renderTable();
        renderResults();
        markSessionDirty();
        return;
      }
      if (status === 'FAILED') {
        item.status = 'failed';
        item.error = data.error || 'Transcription failed.';
        item.costUsd = Number(data.costUsd || item.estimatedCostUsd || 0);
        renderTable();
        renderResults();
        markSessionDirty();
        return;
      }

      item.status = 'transcribing';
      renderTable();
      updateProgress({ stateName: 'visible', ratio: 1, label: `Transcribing ${item.name}...` });
      await sleep(POLL_INTERVAL_MS);
    }
    throw new Error('Canceled.');
  };

  const runFile = async (item) => {
    item.status = 'presigning';
    item.error = '';
    item.progress = 0;
    renderTable();
    updateProgress({ stateName: 'visible', ratio: 0, label: `Preparing ${item.name}...` });

    const controller = new AbortController();
    state.activeController = controller;
    let presign;
    try {
      presign = await authFetchJson(`${API_BASE}/presign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: item.name,
          contentType: item.contentType,
          bytes: item.bytes,
          durationSeconds: item.durationSeconds
        }),
        signal: controller.signal
      });
    } finally {
      if (state.activeController === controller) state.activeController = null;
    }

    if (state.canceled) throw new Error('Canceled.');

    item.status = 'uploading';
    renderTable();
    await uploadFile(item, presign.uploadUrl, presign.headers || {});
    if (state.canceled) throw new Error('Canceled.');

    item.status = 'starting';
    renderTable();
    updateProgress({ stateName: 'visible', ratio: 1, label: `Starting ${item.name}...` });

    const startController = new AbortController();
    state.activeController = startController;
    let start;
    try {
      start = await authFetchJson(`${API_BASE}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ quoteToken: presign.quoteToken }),
        signal: startController.signal
      });
    } finally {
      if (state.activeController === startController) state.activeController = null;
    }

    item.status = 'transcribing';
    item.runToken = start.runToken;
    renderTable();
    await pollRun(item, start.runToken);
  };

  const runBatch = async () => {
    if (!authIsReady()) {
      setStatus(runStatusEl, 'Sign in before starting transcription jobs.', 'warning');
      updateAuthUi();
      return;
    }

    const queue = acceptedFiles().filter((item) => !['complete', 'failed'].includes(item.status));
    if (!queue.length) {
      setStatus(runStatusEl, 'No eligible files to transcribe.', 'warning');
      return;
    }
    if (!approveEl || !approveEl.checked) {
      setStatus(runStatusEl, 'Review and approve the estimated charge before starting.', 'warning');
      return;
    }

    state.canceled = false;
    setBusy(true);
    setStatus(runStatusEl, `Starting ${queue.length} transcription job${queue.length === 1 ? '' : 's'}...`, '');
    updateProgress({ stateName: 'visible', ratio: 0, label: 'Starting batch...' });

    for (let i = 0; i < queue.length; i += 1) {
      const item = queue[i];
      if (state.canceled) break;
      try {
        setStatus(runStatusEl, `Processing ${i + 1} of ${queue.length}: ${item.name}`, '');
        await runFile(item);
      } catch (err) {
        if (state.canceled || err?.name === 'AbortError' || err?.message === 'Canceled.') {
          item.status = 'canceled';
          item.error = 'Canceled.';
        } else {
          item.status = 'failed';
          item.error = err?.message || 'Transcription failed.';
        }
        renderTable();
        renderResults();
        markSessionDirty();
      }
    }

    const completed = completedFiles().length;
    const failed = state.files.filter((item) => item.status === 'failed').length;
    const canceled = state.files.filter((item) => item.status === 'canceled').length;
    const message = state.canceled
      ? `Canceled. ${completed} complete, ${failed} failed, ${canceled} canceled.`
      : `Done. ${completed} complete, ${failed} failed.`;
    setStatus(runStatusEl, message, failed || canceled ? 'warning' : 'success');
    setBusy(false);
    updateProgress({ stateName: 'hidden' });
    renderTable();
    renderResults();
  };

  const reset = () => {
    state.canceled = true;
    abortActive();
    state.files = [];
    if (fileEl) fileEl.value = '';
    if (approveEl) approveEl.checked = false;
    setBusy(false);
    setStatus(runStatusEl, '', '');
    updateProgress({ stateName: 'hidden' });
    renderTable();
    renderResults();
    markSessionDirty();
  };

  if (signInBtn) {
    signInBtn.addEventListener('click', () => {
      if (window.ToolsAuth && window.ToolsAuth.signIn) {
        window.ToolsAuth.signIn({ returnTo: `${window.location.pathname}${window.location.search}${window.location.hash}` });
      }
    });
  }

  fileEl.addEventListener('change', () => {
    analyzeSelectedFiles();
  });

  if (approveEl) {
    approveEl.addEventListener('change', updateControls);
  }

  formEl.addEventListener('submit', (event) => {
    event.preventDefault();
    runBatch();
  });

  if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
      state.canceled = true;
      abortActive();
      setStatus(runStatusEl, 'Canceling. Already submitted AWS jobs may still incur cost.', 'warning');
      setBusy(false);
      updateProgress({ stateName: 'hidden' });
    });
  }

  if (resetBtn) resetBtn.addEventListener('click', reset);

  if (resultsEl) {
    resultsEl.addEventListener('click', async (event) => {
      const actionBtn = event.target.closest('[data-transcribe-action]');
      if (!actionBtn) return;
      const action = actionBtn.getAttribute('data-transcribe-action');
      const id = actionBtn.getAttribute('data-id');
      const item = state.files.find((entry) => entry.id === id);
      const transcript = String(item?.transcript || '').trim();
      if (!item || !transcript) return;
      if (action === 'copy') {
        try {
          await navigator.clipboard.writeText(transcript);
          setStatus(runStatusEl, `Copied transcript for ${item.name}.`, 'success');
        } catch {
          setStatus(runStatusEl, 'Copy failed. Select the text manually.', 'error');
        }
      }
      if (action === 'download') {
        const blob = new Blob([transcript], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = safeDownloadName(item.name);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    });
  }

  document.addEventListener('tools:auth-changed', updateAuthUi);

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const accepted = acceptedFiles();
    const skipped = state.files.filter((item) => item.status === 'skipped');
    const completed = completedFiles();
    const failed = state.files.filter((item) => item.status === 'failed');
    payload.inputs = {
      Files: `${state.files.length} selected`,
      Accepted: String(accepted.length),
      Skipped: String(skipped.length),
      'Estimated total': formatUsd(estimatedTotal())
    };
    payload.outputSummary = completed.length || failed.length
      ? `${completed.length} complete · ${failed.length} failed · ${formatUsd(finalTotal())} run cost`
      : 'No transcripts saved in session history.';
  });

  loadConfig().finally(() => {
    updateAuthUi();
    renderTable();
    renderResults();
  });
  window.setTimeout(updateAuthUi, 250);
  window.setTimeout(updateAuthUi, 1000);
})();
