(() => {
  'use strict';

  const DEFAULT_ENDPOINT = 'https://coxbbervgzwhm5tu53dutxwfca0vxdkg.lambda-url.us-east-2.on.aws/';
  const STORAGE_ENDPOINT = 'tool.whisperTranscribe.endpoint';
  const STORAGE_LIMIT = 'tool.whisperTranscribe.concurrencyLimit';

  const $id = (id) => document.getElementById(id);
  const endpointEl = $id('whispermon-endpoint');
  const limitEl = $id('whispermon-limit');
  const checkBtn = $id('whispermon-check');
  const saveBtn = $id('whispermon-save');
  const endpointStatusEl = $id('whispermon-endpoint-status');
  const fileEl = $id('whispermon-file');
  const audioMetaEl = $id('whispermon-audio-meta');

  const runnerForm = $id('whispermon-runner');
  const totalEl = $id('whispermon-total');
  const parallelEl = $id('whispermon-parallel');
  const pathEl = $id('whispermon-path');
  const startBtn = $id('whispermon-start');
  const stopBtn = $id('whispermon-stop');
  const resetBtn = $id('whispermon-reset');
  const runStatusEl = $id('whispermon-run-status');

  const inflightEl = $id('whispermon-inflight');
  const headroomEl = $id('whispermon-headroom');
  const successEl = $id('whispermon-success');
  const throttlesEl = $id('whispermon-throttles');
  const errorsEl = $id('whispermon-errors');
  const p95El = $id('whispermon-p95');
  const rpsEl = $id('whispermon-rps');
  const capacityEl = $id('whispermon-capacity');
  const meterFillEl = $id('whispermon-meter-fill');
  const meterMetaEl = $id('whispermon-meter-meta');
  const meterRpsFillEl = $id('whispermon-meter-rps-fill');
  const meterRpsMetaEl = $id('whispermon-meter-rps-meta');
  const logEl = $id('whispermon-log');

  if (!endpointEl || !limitEl || !runnerForm || !fileEl) return;

  const safeGet = (key) => {
    if (!key) return null;
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  };

  const safeSet = (key, value) => {
    if (!key) return;
    try {
      localStorage.setItem(key, value);
    } catch {
      // Ignore storage errors.
    }
  };

  const normalizeBase = (url) => {
    if (!url) return '';
    const raw = String(url).trim();
    if (!raw) return '';
    return raw.endsWith('/') ? raw : `${raw}/`;
  };

  const joinUrl = (base, path = '') => {
    const left = String(base || '').trim();
    const right = String(path || '').trim();
    if (!left) return right;
    if (!right) return left;
    if (left.endsWith('/') && right.startsWith('/')) return left + right.slice(1);
    if (!left.endsWith('/') && !right.startsWith('/')) return `${left}/${right}`;
    return left + right;
  };

  const setStatus = (el, message, tone) => {
    if (!el) return;
    el.textContent = message || '';
    el.dataset.tone = tone || '';
  };

  const clampInt = (value, min, max, fallback) => {
    const parsed = Number.parseInt(String(value || ''), 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  const formatMs = (ms) => {
    if (!Number.isFinite(ms) || ms < 0) return '—';
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  };

  const formatNumber = (value, digits = 2) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return '—';
    return num.toFixed(digits).replace(/\.00$/, '');
  };

  const percentile = (arr, pct) => {
    const list = Array.isArray(arr) ? arr.filter(n => Number.isFinite(n)).slice() : [];
    if (!list.length) return null;
    list.sort((a, b) => a - b);
    const p = Math.min(100, Math.max(0, pct));
    const idx = Math.floor((p / 100) * (list.length - 1));
    return list[idx];
  };

  const requestJson = async (url, options = {}) => {
    const res = await fetch(url, options);
    const text = await res.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
    return {
      ok: res.ok,
      status: res.status,
      statusText: res.statusText,
      data,
      text
    };
  };

  const loadSavedConfig = () => {
    const storedEndpoint = safeGet(STORAGE_ENDPOINT);
    const storedLimit = safeGet(STORAGE_LIMIT);
    endpointEl.value = storedEndpoint || DEFAULT_ENDPOINT;
    if (storedLimit) {
      const num = clampInt(storedLimit, 1, 200, 10);
      limitEl.value = String(num);
    }
  };

  const saveConfig = () => {
    const endpoint = normalizeBase(endpointEl.value || DEFAULT_ENDPOINT);
    const limit = clampInt(limitEl.value, 1, 200, 10);
    safeSet(STORAGE_ENDPOINT, endpoint);
    safeSet(STORAGE_LIMIT, String(limit));
    setStatus(endpointStatusEl, 'Saved.', 'success');
  };

  let audioState = {
    filename: '',
    mime: 'audio/wav',
    bytes: 0,
    b64: ''
  };

  const fileToBase64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== 'string') {
        reject(new Error('Unable to read file.'));
        return;
      }
      const parts = result.split(',', 2);
      const b64 = parts.length === 2 ? parts[1] : '';
      if (!b64) {
        reject(new Error('Unable to read file.'));
        return;
      }
      resolve(b64);
    };
    reader.onerror = () => reject(new Error('Unable to read file.'));
    reader.readAsDataURL(file);
  });

  const onAudioChange = async () => {
    const file = fileEl.files && fileEl.files[0];
    if (!file) {
      audioState = { filename: '', mime: 'audio/wav', bytes: 0, b64: '' };
      if (audioMetaEl) audioMetaEl.textContent = 'No audio loaded.';
      return;
    }
    const filename = String(file.name || '').trim() || 'audio.wav';
    const mime = String(file.type || 'audio/wav').trim() || 'audio/wav';
    const bytes = Number(file.size) || 0;

    if (audioMetaEl) {
      audioMetaEl.textContent = 'Loading audio…';
      audioMetaEl.dataset.tone = '';
    }

    try {
      const b64 = await fileToBase64(file);
      audioState = { filename, mime, bytes, b64 };
      if (audioMetaEl) {
        audioMetaEl.textContent = `${filename} • ${formatNumber(bytes / 1024, 1)} KB`;
        audioMetaEl.dataset.tone = 'success';
      }
    } catch (err) {
      audioState = { filename, mime, bytes, b64: '' };
      if (audioMetaEl) {
        audioMetaEl.textContent = `Error: ${err?.message || 'Unable to read file.'}`;
        audioMetaEl.dataset.tone = 'error';
      }
    }
  };

  const checkStatus = async () => {
    setStatus(endpointStatusEl, 'Checking…', '');
    const base = normalizeBase(endpointEl.value || DEFAULT_ENDPOINT);
    if (!base) {
      setStatus(endpointStatusEl, 'Enter a Function URL.', 'error');
      return;
    }
    try {
      const res = await requestJson(base, { method: 'GET' });
      if (!res.ok) {
        const msg = res.data?.error || res.data?.message || res.text || `${res.status} ${res.statusText}`;
        setStatus(endpointStatusEl, `Error: ${msg}`, 'error');
        return;
      }
      const model = res.data?.model || res.data?.model_id || res.data?.modelId || 'unknown';
      const maxSec = res.data?.max_audio_seconds ?? res.data?.maxAudioSeconds;
      const detail = Number.isFinite(Number(maxSec)) ? `max ${maxSec}s` : '';
      setStatus(endpointStatusEl, `OK • ${model}${detail ? ` • ${detail}` : ''}`, 'success');
    } catch (err) {
      setStatus(endpointStatusEl, `Error: ${err?.message || 'Request failed.'}`, 'error');
    }
  };

  let runState = null;
  let pendingRender = false;

  const clearLog = () => {
    if (!logEl) return;
    logEl.innerHTML = '';
  };

  const pushLog = (line, tone) => {
    if (!logEl) return;
    const row = document.createElement('div');
    row.className = 'whispermon-log-row';
    row.dataset.tone = tone || '';
    row.textContent = line;
    logEl.prepend(row);

    const rows = logEl.querySelectorAll('.whispermon-log-row');
    if (rows.length > 80) {
      for (let i = 80; i < rows.length; i += 1) rows[i].remove();
    }
  };

  const resetStats = () => {
    runState = null;
    setStatus(runStatusEl, '', '');
    if (startBtn) startBtn.classList.remove('is-busy');
    if (stopBtn) stopBtn.disabled = true;
    if (startBtn) startBtn.disabled = false;
    clearLog();
    scheduleRender();
  };

  const scheduleRender = () => {
    if (pendingRender) return;
    pendingRender = true;
    window.requestAnimationFrame(() => {
      pendingRender = false;
      render();
    });
  };

  const render = () => {
    const limit = clampInt(limitEl?.value, 1, 200, 10);
    const inflight = runState ? runState.inflight : 0;
    const headroom = Math.max(0, limit - inflight);

    if (inflightEl) inflightEl.textContent = String(inflight);
    if (headroomEl) headroomEl.textContent = String(headroom);
    if (successEl) successEl.textContent = String(runState ? runState.success : 0);
    if (throttlesEl) throttlesEl.textContent = String(runState ? runState.throttles : 0);
    if (errorsEl) errorsEl.textContent = String(runState ? runState.errors : 0);

    const now = performance.now();
    const elapsedSec = runState ? Math.max(0, (now - runState.startedAt) / 1000) : 0;
    const completed = runState ? runState.completed : 0;
    const rps = elapsedSec > 0 ? (completed / elapsedSec) : null;
    if (rpsEl) rpsEl.textContent = rps === null ? '—' : formatNumber(rps, 2);

    const p95 = runState ? percentile(runState.successDurations, 95) : null;
    if (p95El) p95El.textContent = p95 ? formatMs(p95) : '—';

    const capacity = p95 ? (limit / (p95 / 1000)) : null;
    if (capacityEl) capacityEl.textContent = capacity === null ? '—' : formatNumber(capacity, 2);

    const ratio = limit > 0 ? Math.min(1, inflight / limit) : 0;
    if (meterFillEl) meterFillEl.style.width = `${Math.round(ratio * 100)}%`;
    if (meterFillEl) {
      if (ratio >= 0.9) meterFillEl.dataset.tone = 'danger';
      else if (ratio >= 0.7) meterFillEl.dataset.tone = 'warning';
      else meterFillEl.dataset.tone = '';
    }
    if (meterMetaEl) meterMetaEl.textContent = `${inflight} / ${limit}`;

    const rpsRatio = (capacity !== null && rps !== null && capacity > 0) ? Math.min(1, rps / capacity) : 0;
    if (meterRpsFillEl) meterRpsFillEl.style.width = `${Math.round(rpsRatio * 100)}%`;
    if (meterRpsFillEl) {
      if (rpsRatio >= 0.9) meterRpsFillEl.dataset.tone = 'danger';
      else if (rpsRatio >= 0.7) meterRpsFillEl.dataset.tone = 'warning';
      else meterRpsFillEl.dataset.tone = '';
    }
    if (meterRpsMetaEl) {
      if (capacity === null || rps === null || capacity <= 0) meterRpsMetaEl.textContent = '—';
      else meterRpsMetaEl.textContent = `${formatNumber(rps, 2)} / ${formatNumber(capacity, 2)} req/s`;
    }
  };

  const startRun = async ({ total, parallel, path }) => {
    const base = normalizeBase(endpointEl.value || DEFAULT_ENDPOINT);
    if (!base) {
      setStatus(runStatusEl, 'Enter a Function URL.', 'error');
      return;
    }
    if (!audioState.b64) {
      setStatus(runStatusEl, 'Choose a WAV file first.', 'error');
      return;
    }

    const limit = clampInt(limitEl.value, 1, 200, 10);
    const totalRequests = clampInt(total, 1, 500, 12);
    const parallelRequests = clampInt(parallel, 1, 50, 12);
    const safeParallel = Math.max(1, parallelRequests);
    const url = joinUrl(base, path || '/');

    saveConfig();

    runState = {
      startedAt: performance.now(),
      url,
      total: totalRequests,
      parallel: safeParallel,
      limit,
      nextIndex: 0,
      inflight: 0,
      completed: 0,
      success: 0,
      throttles: 0,
      errors: 0,
      canceled: 0,
      successDurations: [],
      aborters: new Set(),
      stopRequested: false
    };

    clearLog();
    setStatus(runStatusEl, 'Running…', '');
    if (startBtn) startBtn.classList.add('is-busy');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;

    const payload = JSON.stringify({
      audio_b64: audioState.b64,
      mime_type: audioState.mime
    });

    const pump = async () => {
      if (!runState || runState.stopRequested) return;
      while (runState.inflight < runState.parallel && runState.nextIndex < runState.total && !runState.stopRequested) {
        const idx = runState.nextIndex;
        runState.nextIndex += 1;
        runState.inflight += 1;
        scheduleRender();
        runRequest(idx, url, payload);
      }
    };

    const finishIfDone = () => {
      if (!runState) return;
      const done = runState.completed + runState.canceled >= runState.total;
      if (!done) return;
      const summary = `Done • success ${runState.success}, throttles ${runState.throttles}, errors ${runState.errors}`;
      setStatus(runStatusEl, summary, runState.errors ? 'warning' : 'success');
      if (startBtn) startBtn.classList.remove('is-busy');
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = true;
    };

    const runRequest = async (idx, requestUrl, body) => {
      const controller = new AbortController();
      runState.aborters.add(controller);
      const started = performance.now();
      try {
        const res = await fetch(requestUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
          signal: controller.signal
        });
        const text = await res.text();
        const duration = performance.now() - started;
        runState.completed += 1;
        runState.inflight = Math.max(0, runState.inflight - 1);
        runState.aborters.delete(controller);

        if (res.status === 200) {
          runState.success += 1;
          runState.successDurations.push(duration);
          pushLog(`#${idx + 1} 200 ${formatMs(duration)}`, 'success');
        } else if (res.status === 429) {
          runState.throttles += 1;
          pushLog(`#${idx + 1} 429 ${formatMs(duration)}`, 'warning');
        } else {
          runState.errors += 1;
          let message = '';
          if (text) {
            try {
              const parsed = JSON.parse(text);
              message = parsed?.error || parsed?.message || '';
            } catch {
              message = '';
            }
          }
          const suffix = message ? ` • ${message}` : '';
          pushLog(`#${idx + 1} ${res.status} ${formatMs(duration)}${suffix}`, 'error');
        }
        scheduleRender();
        if (runState.stopRequested) finishIfDone();
        else {
          finishIfDone();
          pump();
        }
      } catch (err) {
        const duration = performance.now() - started;
        runState.inflight = Math.max(0, runState.inflight - 1);
        runState.aborters.delete(controller);

        if (err && err.name === 'AbortError') {
          runState.canceled += 1;
          pushLog(`#${idx + 1} canceled ${formatMs(duration)}`, 'muted');
        } else {
          runState.completed += 1;
          runState.errors += 1;
          pushLog(`#${idx + 1} error ${formatMs(duration)}`, 'error');
        }
        scheduleRender();
        if (runState.stopRequested) finishIfDone();
        else {
          finishIfDone();
          pump();
        }
      }
    };

    await pump();
  };

  const stopRun = () => {
    if (!runState) return;
    runState.stopRequested = true;
    setStatus(runStatusEl, 'Stopping…', 'warning');
    runState.aborters.forEach((controller) => controller.abort());
    runState.aborters.clear();
    if (stopBtn) stopBtn.disabled = true;
    if (startBtn) startBtn.classList.remove('is-busy');
    if (startBtn) startBtn.disabled = false;
    scheduleRender();
  };

  const onSubmit = (e) => {
    e.preventDefault();
    if (runState && !runState.stopRequested) return;
    const total = clampInt(totalEl?.value, 1, 500, 12);
    const parallel = clampInt(parallelEl?.value, 1, 50, 12);
    const path = String(pathEl?.value || '/').trim() || '/';
    startRun({ total, parallel, path });
  };

  loadSavedConfig();
  render();

  if (fileEl) fileEl.addEventListener('change', onAudioChange);
  if (checkBtn) checkBtn.addEventListener('click', checkStatus);
  if (saveBtn) saveBtn.addEventListener('click', saveConfig);
  if (runnerForm) runnerForm.addEventListener('submit', onSubmit);
  if (stopBtn) stopBtn.addEventListener('click', stopRun);
  if (resetBtn) resetBtn.addEventListener('click', resetStats);
})();
