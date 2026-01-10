(() => {
  'use strict';

  const ENDPOINT_BASE = 'https://coxbbervgzwhm5tu53dutxwfca0vxdkg.lambda-url.us-east-2.on.aws/';

  const STORAGE_WARMUP_AT = 'tool.whisperTranscribe.lastWarmupAt';
  const STORAGE_WARMUP_BASE = 'tool.whisperTranscribe.lastWarmupBase';

  const WARMUP_MIN_INTERVAL_MS = 10 * 60 * 1000;
  const WARMUP_AUDIO_MS = 250;

  const $id = (id) => document.getElementById(id);

  const healthPillEl = $id('whispermon-health-pill');
  const endpointStatusEl = $id('whispermon-endpoint-status');
  const checkBtn = $id('whispermon-check');

  const statModelEl = $id('whispermon-stat-model');
  const statDirectLimitEl = $id('whispermon-stat-direct-limit');
  const statUploadLimitEl = $id('whispermon-stat-upload-limit');
  const statPartMaxEl = $id('whispermon-stat-part-max');

  const formEl = $id('whispermon-form');
  const fileEl = $id('whispermon-file');
  const audioMetaEl = $id('whispermon-audio-meta');
  const startBtn = $id('whispermon-start');
  const cancelBtn = $id('whispermon-cancel');
  const resetBtn = $id('whispermon-reset');
  const runStatusEl = $id('whispermon-run-status');

  const progressWrapEl = $id('whispermon-upload-progress-wrap');
  const progressLabelEl = $id('whispermon-upload-progress-label');
  const progressBarEl = $id('whispermon-upload-progress');

  const partMinutesEl = $id('whispermon-part-minutes');
  const partMinutesMetaEl = $id('whispermon-part-minutes-meta');

  const directMeterFillEl = $id('whispermon-direct-meter-fill');
  const directMeterMetaEl = $id('whispermon-direct-meter-meta');
  const uploadMeterFillEl = $id('whispermon-upload-meter-fill');
  const uploadMeterMetaEl = $id('whispermon-upload-meter-meta');

  const transcriptEl = $id('whispermon-transcript');
  const copyTranscriptBtn = $id('whispermon-copy-transcript');
  const transcriptStatusEl = $id('whispermon-transcript-status');

  if (!formEl || !fileEl || !startBtn) return;

  const STORAGE_PART_MINUTES = 'tool.whisperTranscribe.partMinutes';

  let serverReady = false;
  let serverInfo = {
    model: 'unknown',
    modelLoaded: false,
    maxAudioSeconds: null,
    maxUploadBytes: null,
    maxDirectBytes: null,
    uploadsEnabled: false,
    maxPartMinutes: 30,
    defaultPartMinutes: 30
  };

  let audioState = {
    filename: '',
    mime: 'application/octet-stream',
    bytes: 0,
    file: null,
    durationSeconds: null
  };

  let activeRequest = null;
  let durationProbeSeq = 0;

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

  const parseTimestamp = (value) => {
    const parsed = Number.parseInt(String(value || ''), 10);
    if (!Number.isFinite(parsed) || parsed <= 0) return null;
    return parsed;
  };

  const normalizeBase = (url) => {
    const raw = String(url || '').trim();
    if (!raw) return '';
    return raw.endsWith('/') ? raw : `${raw}/`;
  };

  const joinUrl = (base, path = '') => {
    const left = normalizeBase(base);
    const right = String(path || '').trim();
    if (!left) return right;
    if (!right) return left;
    if (right.startsWith('/')) return `${left}${right.slice(1)}`;
    return `${left}${right}`;
  };

  const setStatus = (el, message, tone) => {
    if (!el) return;
    el.textContent = message || '';
    el.dataset.tone = tone || '';
  };

  const setHealth = (state, label) => {
    if (!healthPillEl) return;
    healthPillEl.dataset.state = state || '';
    healthPillEl.textContent = label || '';
  };

  const setTranscript = (text) => {
    const value = (text || '').trim();
    if (transcriptEl) transcriptEl.value = value;
    if (copyTranscriptBtn) copyTranscriptBtn.disabled = !value;
  };

  const clearTranscript = () => {
    setTranscript('');
    setStatus(transcriptStatusEl, '', '');
  };

  const formatNumber = (value, digits = 2) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return '—';
    return num.toFixed(digits).replace(/\.00$/, '');
  };

  const clampInt = (value, min, max, fallback) => {
    const parsed = Number.parseInt(String(value || ''), 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  const formatBytes = (bytes) => {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return '—';
    const mb = value / (1024 * 1024);
    if (mb >= 1) return `${formatNumber(mb, 2)} MB`;
    return `${formatNumber(value / 1024, 1)} KB`;
  };

  const setStat = (el, value) => {
    if (!el) return;
    el.textContent = (value || '').trim() || '—';
  };

  const createSilentWavBlob = (durationMs) => {
    const sampleRate = 16000;
    const channels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const ms = Math.max(1, Number(durationMs) || 1);
    const samples = Math.max(1, Math.round(sampleRate * (ms / 1000)));
    const dataBytes = samples * channels * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(buffer);

    const writeAscii = (offset, value) => {
      for (let i = 0; i < value.length; i += 1) {
        view.setUint8(offset + i, value.charCodeAt(i) & 0xff);
      }
    };

    writeAscii(0, 'RIFF');
    view.setUint32(4, 36 + dataBytes, true);
    writeAscii(8, 'WAVE');
    writeAscii(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * channels * bytesPerSample, true);
    view.setUint16(32, channels * bytesPerSample, true);
    view.setUint16(34, bitsPerSample, true);
    writeAscii(36, 'data');
    view.setUint32(40, dataBytes, true);

    return new Blob([buffer], { type: 'audio/wav' });
  };

  const currentPartMinutes = () => {
    const max = clampInt(serverInfo.maxPartMinutes, 1, 60, 30);
    const defaultMinutes = clampInt(serverInfo.defaultPartMinutes, 1, max, 30);
    const minMinutes = 5;
    const sliderMin = Math.min(minMinutes, max);
    return clampInt(partMinutesEl?.value, sliderMin, max, defaultMinutes);
  };

  const updatePartMinutesMeta = () => {
    const minutes = currentPartMinutes();
    if (partMinutesMetaEl) {
      partMinutesMetaEl.textContent = `Splits long media into ${minutes}-minute parts for transcription.`;
    }
  };

  const setBusy = (busy) => {
    const isBusy = Boolean(busy);
    if (startBtn) {
      if (isBusy) startBtn.classList.add('is-busy');
      else startBtn.classList.remove('is-busy');
      startBtn.disabled = isBusy || !serverReady;
    }
    if (cancelBtn) cancelBtn.disabled = !isBusy;
    if (resetBtn) resetBtn.disabled = isBusy;
    if (fileEl) fileEl.disabled = isBusy;
    if (checkBtn) checkBtn.disabled = isBusy;
  };

  const setUploadProgress = ({ state, ratio, label }) => {
    if (!progressWrapEl || !progressBarEl) return;
    const nextState = state || 'hidden';
    progressWrapEl.dataset.state = nextState;
    if (nextState === 'hidden') {
      progressBarEl.value = 0;
      if (progressLabelEl) progressLabelEl.textContent = 'Upload progress';
      return;
    }

    const safeRatio = Math.min(1, Math.max(0, Number(ratio) || 0));
    progressBarEl.value = safeRatio;
    if (progressLabelEl) progressLabelEl.textContent = label || 'Upload progress';
  };

  const abortActive = () => {
    if (!activeRequest) return;
    const { xhr, controller } = activeRequest;
    try {
      if (xhr && xhr.readyState !== 4) xhr.abort();
    } catch {
      // Ignore.
    }
    try {
      if (controller) controller.abort();
    } catch {
      // Ignore.
    }
    activeRequest = null;
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

  const shouldWarmUp = ({ base, modelLoaded }) => {
    const last = parseTimestamp(safeGet(STORAGE_WARMUP_AT));
    const lastBase = normalizeBase(safeGet(STORAGE_WARMUP_BASE));
    if (!base) return true;
    if (!modelLoaded) return true;
    if (!last || !lastBase || lastBase !== base) return true;
    return Date.now() - last >= WARMUP_MIN_INTERVAL_MS;
  };

  const setServerReadyState = (ready) => {
    serverReady = Boolean(ready);
    if (startBtn && !startBtn.classList.contains('is-busy')) {
      startBtn.disabled = !serverReady;
    }
  };

  const setMeter = ({ fillEl, metaEl, ratio, tone, meta }) => {
    const safeRatio = Math.min(1, Math.max(0, Number(ratio) || 0));
    if (fillEl) {
      fillEl.style.width = `${safeRatio * 100}%`;
      if (tone) fillEl.dataset.tone = tone;
      else delete fillEl.dataset.tone;
    }
    if (metaEl) metaEl.textContent = meta || '';
  };

  const probeMediaDuration = async (file) => {
    if (!file || !window.URL || typeof window.URL.createObjectURL !== 'function') return null;
    durationProbeSeq += 1;
    const seq = durationProbeSeq;
    const src = window.URL.createObjectURL(file);
    const tag = String(file.type || '').toLowerCase().startsWith('video') ? 'video' : 'audio';
    const el = document.createElement(tag);
    el.preload = 'metadata';
    el.muted = true;
    el.playsInline = true;

    return await new Promise((resolve) => {
      const cleanup = () => {
        try {
          el.removeAttribute('src');
          el.load();
        } catch {
          // Ignore.
        }
        try {
          window.URL.revokeObjectURL(src);
        } catch {
          // Ignore.
        }
      };

      const finish = (value) => {
        cleanup();
        if (seq !== durationProbeSeq) return resolve(null);
        const num = Number(value);
        if (!Number.isFinite(num) || num <= 0) return resolve(null);
        return resolve(num);
      };

      el.onloadedmetadata = () => finish(el.duration);
      el.onerror = () => finish(null);
      el.src = src;
    });
  };

  const estimateParts = () => {
    const duration = Number(audioState.durationSeconds);
    if (!Number.isFinite(duration) || duration <= 0) return null;
    const partSec = currentPartMinutes() * 60;
    if (!Number.isFinite(partSec) || partSec <= 0) return null;
    return Math.max(1, Math.ceil(duration / partSec));
  };

  const updateAudioMeta = () => {
    if (!audioMetaEl) return;
    if (!audioState.file) {
      audioMetaEl.textContent = 'No media loaded.';
      audioMetaEl.dataset.tone = '';
      setMeter({
        fillEl: directMeterFillEl,
        metaEl: directMeterMetaEl,
        ratio: 0,
        tone: '',
        meta: 'Select a file to compare against the direct request limit.'
      });
      setMeter({
        fillEl: uploadMeterFillEl,
        metaEl: uploadMeterMetaEl,
        ratio: 0,
        tone: '',
        meta: 'Select a file to compare against the upload limit.'
      });
      return;
    }

    const limitBytes = serverInfo.maxUploadBytes;
    const tooLarge = Number.isFinite(Number(limitBytes)) && limitBytes > 0 && audioState.bytes > limitBytes;
    const limitLabel = Number.isFinite(Number(limitBytes)) && limitBytes > 0 ? formatBytes(limitBytes) : '—';
    const typeLabel = audioState.mime && audioState.mime !== 'application/octet-stream' ? audioState.mime : 'unknown type';
    const durationLabel = Number.isFinite(Number(audioState.durationSeconds)) ? formatClock(audioState.durationSeconds) : '';
    const partCount = estimateParts();
    const minutes = currentPartMinutes();
    const partCountLabel = Number.isFinite(Number(partCount)) ? `${partCount} part${partCount === 1 ? '' : 's'} @ ${minutes}m` : '';
    const extra = [durationLabel, partCountLabel].filter(Boolean);
    const base = `${audioState.filename} • ${formatBytes(audioState.bytes)} • ${typeLabel}${extra.length ? ` • ${extra.join(' • ')}` : ''}`;
    audioMetaEl.textContent = tooLarge ? `${base} • exceeds ${limitLabel} limit` : base;
    audioMetaEl.dataset.tone = tooLarge ? 'error' : 'success';

    const directLimit = serverInfo.maxDirectBytes;
    const hasDirectLimit = Number.isFinite(Number(directLimit)) && Number(directLimit) > 0;
    const directRatio = hasDirectLimit ? (audioState.bytes / Number(directLimit)) : 0;
    const directTooLarge = hasDirectLimit && audioState.bytes > Number(directLimit);
    const uploadsEnabled = Boolean(serverInfo.uploadsEnabled);
    const directTone = directTooLarge ? (uploadsEnabled ? 'warning' : 'danger') : '';
    const directRemaining = hasDirectLimit ? (Number(directLimit) - audioState.bytes) : 0;
    const directOver = hasDirectLimit ? (audioState.bytes - Number(directLimit)) : 0;
    const directRemainingLabel = directRemaining > 0 ? `${formatBytes(directRemaining)} remaining` : '';
    const directOverLabel = directOver > 0 ? `${formatBytes(directOver)} over` : '';
    const directMeta = hasDirectLimit
      ? `${formatBytes(audioState.bytes)} / ${formatBytes(directLimit)}${directTooLarge ? (uploadsEnabled ? ` (${directOverLabel}, will use S3 upload)` : ` (${directOverLabel})`) : (directRemainingLabel ? ` (${directRemainingLabel})` : '')}`
      : 'Direct limit unavailable.';
    setMeter({
      fillEl: directMeterFillEl,
      metaEl: directMeterMetaEl,
      ratio: directRatio,
      tone: directTone,
      meta: directMeta
    });

    const uploadLimit = serverInfo.maxUploadBytes;
    const hasUploadLimit = Number.isFinite(Number(uploadLimit)) && Number(uploadLimit) > 0;
    const uploadRatio = hasUploadLimit ? (audioState.bytes / Number(uploadLimit)) : 0;
    const uploadTooLarge = hasUploadLimit && audioState.bytes > Number(uploadLimit);
    const uploadTone = uploadTooLarge ? 'danger' : (uploadRatio >= 0.9 ? 'warning' : '');
    const uploadRemaining = hasUploadLimit ? (Number(uploadLimit) - audioState.bytes) : 0;
    const uploadOver = hasUploadLimit ? (audioState.bytes - Number(uploadLimit)) : 0;
    const uploadRemainingLabel = uploadRemaining > 0 ? `${formatBytes(uploadRemaining)} remaining` : '';
    const uploadOverLabel = uploadOver > 0 ? `${formatBytes(uploadOver)} over` : '';
    const uploadMeta = hasUploadLimit
      ? `${formatBytes(audioState.bytes)} / ${formatBytes(uploadLimit)}${uploadTooLarge ? ` (${uploadOverLabel})` : (uploadRemainingLabel ? ` (${uploadRemainingLabel})` : '')}`
      : 'Upload limit unavailable.';
    setMeter({
      fillEl: uploadMeterFillEl,
      metaEl: uploadMeterMetaEl,
      ratio: uploadRatio,
      tone: uploadTone,
      meta: uploadMeta
    });
  };

  const onAudioChange = () => {
    const file = fileEl.files && fileEl.files[0];
    if (!file) {
      audioState = { filename: '', mime: 'application/octet-stream', bytes: 0, file: null, durationSeconds: null };
      clearTranscript();
      updateAudioMeta();
      return;
    }
    audioState = {
      filename: String(file.name || '').trim() || 'media',
      mime: String(file.type || 'application/octet-stream').trim() || 'application/octet-stream',
      bytes: Number(file.size) || 0,
      file,
      durationSeconds: null
    };
    clearTranscript();
    updateAudioMeta();
    probeMediaDuration(file).then((duration) => {
      if (audioState.file !== file) return;
      audioState.durationSeconds = duration;
      updateAudioMeta();
    });
  };

  const checkStatus = async () => {
    setServerReadyState(false);
    setHealth('warming', 'Warming up');
    setStatus(endpointStatusEl, 'Connecting to AWS Lambda…', '');

    const base = normalizeBase(ENDPOINT_BASE);
    if (!base) {
      setHealth('err', 'No endpoint');
      setStatus(endpointStatusEl, 'No endpoint configured.', 'error');
      return { ok: false, base };
    }

    try {
      const res = await requestJson(base, { method: 'GET' });
      if (!res.ok) {
        setHealth('err', 'Unavailable');
        setStatus(endpointStatusEl, 'Unable to reach AWS Lambda.', 'error');
        return { ok: false, base, error: res.text || `${res.status} ${res.statusText}` };
      }

      const model = res.data?.model || res.data?.model_id || res.data?.modelId || 'unknown';
      const modelLoaded = Boolean(res.data?.model_loaded ?? res.data?.modelLoaded);
      const maxSec = res.data?.max_audio_seconds ?? res.data?.maxAudioSeconds;
      const maxBytes = res.data?.max_audio_bytes ?? res.data?.maxAudioBytes;

      const maxUploadBytes = res.data?.max_upload_bytes ?? res.data?.maxUploadBytes ?? maxBytes;
      const maxDirectBytes = res.data?.max_direct_upload_bytes ?? res.data?.maxDirectUploadBytes ?? maxBytes;
      const uploadsEnabled = Boolean(res.data?.uploads_enabled ?? res.data?.uploadsEnabled);
      const maxPartMinutes = res.data?.max_part_minutes ?? res.data?.maxPartMinutes;
      const defaultPartMinutes = res.data?.default_part_minutes ?? res.data?.defaultPartMinutes;

      serverInfo = {
        model,
        modelLoaded,
        maxAudioSeconds: Number.isFinite(Number(maxSec)) ? Number(maxSec) : null,
        maxUploadBytes: Number.isFinite(Number(maxUploadBytes)) ? Number(maxUploadBytes) : null,
        maxDirectBytes: Number.isFinite(Number(maxDirectBytes)) ? Number(maxDirectBytes) : null,
        uploadsEnabled,
        maxPartMinutes: Number.isFinite(Number(maxPartMinutes)) ? Number(maxPartMinutes) : 30,
        defaultPartMinutes: Number.isFinite(Number(defaultPartMinutes)) ? Number(defaultPartMinutes) : 30
      };

      setStat(statModelEl, serverInfo.model);
      setStat(statDirectLimitEl, serverInfo.maxDirectBytes ? formatBytes(serverInfo.maxDirectBytes) : '');
      setStat(statUploadLimitEl, serverInfo.maxUploadBytes ? formatBytes(serverInfo.maxUploadBytes) : '');
      setStat(statPartMaxEl, `${clampInt(serverInfo.maxPartMinutes, 1, 60, 30)} min`);

      const parts = ['OK', model].filter(Boolean);
      if (serverInfo.maxAudioSeconds) parts.push(`max ${serverInfo.maxAudioSeconds}s`);
      if (serverInfo.maxUploadBytes) parts.push(`max ${formatBytes(serverInfo.maxUploadBytes)}`);
      if (!modelLoaded) parts.push('cold');
      setStatus(endpointStatusEl, parts.join(' • '), 'success');
      updateAudioMeta();
      if (partMinutesEl) {
        const maxMinutes = clampInt(serverInfo.maxPartMinutes, 1, 60, 30);
        const defaultMinutes = clampInt(serverInfo.defaultPartMinutes, 1, maxMinutes, 30);
        const stored = safeGet(STORAGE_PART_MINUTES);
        const minMinutes = 5;
        const sliderMin = Math.min(minMinutes, maxMinutes);
        partMinutesEl.min = String(sliderMin);
        partMinutesEl.max = String(maxMinutes);
        const nextValue = clampInt(partMinutesEl.value || stored, sliderMin, maxMinutes, defaultMinutes);
        partMinutesEl.value = String(nextValue);
      }
      updatePartMinutesMeta();

      return {
        ok: true,
        base,
        modelLoaded
      };
    } catch (err) {
      setHealth('err', 'Unavailable');
      setStatus(endpointStatusEl, 'Unable to reach AWS Lambda.', 'error');
      return { ok: false, base: ENDPOINT_BASE, error: err?.message || 'Request failed.' };
    }
  };

  const warmUpServer = async ({ base, modelLoaded }) => {
    const normalized = normalizeBase(base);
    if (!normalized) return;
    if (!shouldWarmUp({ base: normalized, modelLoaded })) {
      setHealth('ok', 'Ready');
      setStatus(endpointStatusEl, 'Ready.', 'success');
      setServerReadyState(true);
      return;
    }

    setServerReadyState(false);
    setHealth('warming', 'Warming up');
    setStatus(endpointStatusEl, 'Warming up…', 'warning');

    const url = joinUrl(normalized, '/transcribe');
    const body = createSilentWavBlob(WARMUP_AUDIO_MS);
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body
      });
      if (!res.ok) throw new Error('Warm-up failed.');

      safeSet(STORAGE_WARMUP_AT, String(Date.now()));
      safeSet(STORAGE_WARMUP_BASE, normalized);
      setHealth('ok', 'Ready');
      setStatus(endpointStatusEl, 'Ready.', 'success');
      setServerReadyState(true);
    } catch {
      setHealth('err', 'Unavailable');
      setStatus(endpointStatusEl, 'Warm-up failed.', 'error');
      setServerReadyState(false);
    }
  };

  const checkAndWarm = async () => {
    const status = await checkStatus();
    if (status && status.ok) await warmUpServer(status);
  };

  const extractPresign = (data) => {
    if (!data || typeof data !== 'object') return null;
    const url = data.upload_url || data.uploadUrl || data.url;
    const fields = data.fields;
    const key = data.key || data.object_key || data.objectKey;
    if (!url || !fields || !key) return null;
    return { url, fields, key };
  };

  const presignUpload = async (file) => {
    const controller = new AbortController();
    activeRequest = { xhr: null, controller };
    try {
      const payload = {
        filename: String(file?.name || '').trim() || 'media',
        content_type: String(file?.type || 'application/octet-stream').trim() || 'application/octet-stream',
        bytes: Number(file?.size) || 0
      };
      const res = await requestJson(joinUrl(ENDPOINT_BASE, '/presign'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      if (!res.ok) return { ok: false, error: res.data?.error || res.text || 'Presign failed.' };
      const presign = extractPresign(res.data);
      if (!presign) return { ok: false, error: 'Invalid presign response.' };
      return { ok: true, ...presign };
    } catch (err) {
      if (err && err.name === 'AbortError') return { ok: false, aborted: true };
      return { ok: false, error: err?.message || 'Presign failed.' };
    } finally {
      if (activeRequest?.controller === controller) activeRequest = null;
    }
  };

  const uploadViaPresignedPost = ({ url, fields, file }) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    activeRequest = { xhr, controller: null };
    xhr.open('POST', url, true);
    xhr.responseType = 'text';

    xhr.upload.onprogress = (evt) => {
      if (!evt || !evt.lengthComputable) return;
      const ratio = evt.total > 0 ? (evt.loaded / evt.total) : 0;
      setUploadProgress({
        state: 'uploading',
        ratio,
        label: `Uploading ${formatBytes(evt.loaded)} / ${formatBytes(evt.total)}`
      });
    };

    xhr.onload = () => {
      activeRequest = null;
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
        return;
      }
      reject(new Error(`Upload failed (${xhr.status}).`));
    };

    xhr.onerror = () => {
      activeRequest = null;
      reject(new Error('Upload failed.'));
    };

    xhr.onabort = () => {
      activeRequest = null;
      reject(new Error('Upload canceled.'));
    };

    const form = new FormData();
    const entries = Object.entries(fields || {});
    for (let i = 0; i < entries.length; i += 1) {
      const [key, value] = entries[i];
      if (value === undefined || value === null) continue;
      form.append(key, String(value));
    }
    form.append('file', file, file.name || 'media');
    xhr.send(form);
  });

  const uploadDirectToLambda = (file) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    activeRequest = { xhr, controller: null };
    const minutes = currentPartMinutes();
    const url = new URL(joinUrl(ENDPOINT_BASE, '/transcribe'));
    url.searchParams.set('part_minutes', String(minutes));
    xhr.open('POST', url.toString(), true);
    xhr.responseType = 'text';

    const contentType = String(file?.type || 'application/octet-stream').trim() || 'application/octet-stream';
    try {
      xhr.setRequestHeader('Content-Type', contentType);
    } catch {
      // Ignore header failures.
    }

    xhr.upload.onprogress = (evt) => {
      if (!evt || !evt.lengthComputable) return;
      const ratio = evt.total > 0 ? (evt.loaded / evt.total) : 0;
      setUploadProgress({
        state: 'uploading',
        ratio,
        label: `Uploading ${formatBytes(evt.loaded)} / ${formatBytes(evt.total)}`
      });
    };

    xhr.onload = () => {
      activeRequest = null;
      const text = xhr.responseText || '';
      let data = null;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = null;
        }
      }
      resolve({ ok: xhr.status >= 200 && xhr.status < 300, status: xhr.status, data, text });
    };

    xhr.onerror = () => {
      activeRequest = null;
      reject(new Error('Upload failed.'));
    };

    xhr.onabort = () => {
      activeRequest = null;
      reject(new Error('Upload canceled.'));
    };

    xhr.send(file);
  });

  const transcribeS3Key = async ({ key, partMinutes }) => {
    const controller = new AbortController();
    activeRequest = { xhr: null, controller };
    try {
      const url = new URL(joinUrl(ENDPOINT_BASE, '/transcribe-s3'));
      url.searchParams.set('part_minutes', String(partMinutes));
      const res = await requestJson(url.toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key }),
        signal: controller.signal
      });
      if (!res.ok) {
        const message = res.data?.error || res.data?.message || res.text || 'Transcription failed.';
        return { ok: false, status: res.status, error: message };
      }
      return { ok: true, status: res.status, data: res.data };
    } catch (err) {
      if (err && err.name === 'AbortError') return { ok: false, aborted: true };
      return { ok: false, error: err?.message || 'Transcription failed.' };
    } finally {
      if (activeRequest?.controller === controller) activeRequest = null;
    }
  };

  const formatClock = (totalSeconds) => {
    const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    const pad = (value) => String(value).padStart(2, '0');
    if (hours > 0) return `${hours}:${pad(minutes)}:${pad(secs)}`;
    return `${pad(minutes)}:${pad(secs)}`;
  };

  const formatParts = (parts) => {
    const list = Array.isArray(parts) ? parts : [];
    const blocks = [];
    for (let i = 0; i < list.length; i += 1) {
      const part = list[i] || {};
      const idx = Number.isFinite(Number(part.index)) ? Number(part.index) : (i + 1);
      const startSec = Number(part.start_sec ?? part.startSec) || 0;
      const endSec = Number(part.end_sec ?? part.endSec) || 0;
      const transcript = String(part.transcript || '').trim();
      const header = `Part ${idx} (${formatClock(startSec)}–${formatClock(endSec)})`;
      blocks.push(`${header}\n${transcript}`.trim());
    }
    return blocks.filter(Boolean).join('\n\n');
  };

  const applyTranscriptResponse = (data) => {
    const parts = data?.parts;
    if (Array.isArray(parts) && parts.length) {
      const formatted = formatParts(parts);
      if (formatted) setTranscript(formatted);
      const label = parts.length === 1 ? 'Transcript ready.' : `Transcript ready (${parts.length} parts).`;
      setStatus(transcriptStatusEl, label, formatted ? 'success' : 'warning');
      if (formatted && transcriptEl && typeof transcriptEl.scrollIntoView === 'function') {
        transcriptEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      return;
    }
    const transcript = data?.transcript;
    if (typeof transcript === 'string') {
      setTranscript(transcript);
      setStatus(transcriptStatusEl, transcript.trim() ? 'Transcript ready.' : 'Transcript returned empty.', transcript.trim() ? 'success' : 'warning');
      if (transcript.trim() && transcriptEl && typeof transcriptEl.scrollIntoView === 'function') {
        transcriptEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  const transcribeSelectedFile = async () => {
    if (!serverReady) {
      setStatus(runStatusEl, 'Server warming up...', 'warning');
      return;
    }
    if (!audioState.file) {
      setStatus(runStatusEl, 'Choose an audio or video file first.', 'error');
      return;
    }

    const maxBytes = serverInfo.maxUploadBytes;
    if (Number.isFinite(Number(maxBytes)) && maxBytes > 0 && audioState.bytes > maxBytes) {
      setStatus(runStatusEl, `File exceeds upload limit (${formatBytes(maxBytes)}).`, 'error');
      return;
    }

    clearTranscript();
    abortActive();
    setBusy(true);
    const minutes = currentPartMinutes();
    safeSet(STORAGE_PART_MINUTES, String(minutes));
    setUploadProgress({ state: 'uploading', ratio: 0, label: 'Starting upload…' });
    setStatus(runStatusEl, 'Preparing upload…', '');

    const file = audioState.file;
    const base = normalizeBase(ENDPOINT_BASE);
    const canUseUploads = Boolean(serverInfo.uploadsEnabled);
    const shouldUseUploads = canUseUploads && (
      !serverInfo.maxDirectBytes
      || !Number.isFinite(Number(serverInfo.maxDirectBytes))
      || audioState.bytes > serverInfo.maxDirectBytes
    );

    try {
      if (shouldUseUploads) {
        const presign = await presignUpload(file);
        if (!presign.ok) {
          if (presign.aborted) throw new Error('Canceled.');
          throw new Error(presign.error || 'Upload setup failed.');
        }

        setStatus(runStatusEl, 'Uploading…', '');
        await uploadViaPresignedPost({ url: presign.url, fields: presign.fields, file });
        setUploadProgress({ state: 'uploading', ratio: 1, label: 'Upload complete.' });
        setStatus(runStatusEl, 'Transcribing…', '');

        const result = await transcribeS3Key({ key: presign.key, partMinutes: minutes });
        if (!result.ok) {
          if (result.aborted) throw new Error('Canceled.');
          throw new Error(result.error || 'Transcription failed.');
        }
        applyTranscriptResponse(result.data);
        setStatus(runStatusEl, 'Done.', 'success');
        return;
      }

      setStatus(runStatusEl, 'Uploading…', '');
      const direct = await uploadDirectToLambda(file);
      setUploadProgress({ state: 'uploading', ratio: 1, label: 'Upload complete.' });
      if (!direct.ok) {
        const message = direct.data?.error || direct.data?.message || direct.text || `Request failed (${direct.status}).`;
        throw new Error(message);
      }
      applyTranscriptResponse(direct.data);
      setStatus(runStatusEl, 'Done.', 'success');
    } catch (err) {
      const message = err?.message || 'Request failed.';
      const tone = message === 'Canceled.' ? 'warning' : 'error';
      setStatus(runStatusEl, message, tone);
    } finally {
      setBusy(false);
      setUploadProgress({ state: 'hidden' });
    }
  };

  const resetForm = () => {
    abortActive();
    setBusy(false);
    setServerReadyState(serverReady);
    if (fileEl) fileEl.value = '';
    audioState = { filename: '', mime: 'application/octet-stream', bytes: 0, file: null, durationSeconds: null };
    setStatus(runStatusEl, '', '');
    setUploadProgress({ state: 'hidden' });
    clearTranscript();
    updateAudioMeta();
  };

  if (fileEl) fileEl.addEventListener('change', onAudioChange);
  if (checkBtn) checkBtn.addEventListener('click', checkAndWarm);
  if (partMinutesEl) {
    partMinutesEl.addEventListener('input', () => {
      updatePartMinutesMeta();
      safeSet(STORAGE_PART_MINUTES, String(currentPartMinutes()));
      updateAudioMeta();
    });
  }
  if (formEl) {
    formEl.addEventListener('submit', (e) => {
      e.preventDefault();
      transcribeSelectedFile();
    });
  }
  if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
      abortActive();
      setStatus(runStatusEl, 'Canceled.', 'warning');
      setBusy(false);
      setUploadProgress({ state: 'hidden' });
    });
  }
  if (resetBtn) resetBtn.addEventListener('click', resetForm);
  if (copyTranscriptBtn) {
    copyTranscriptBtn.addEventListener('click', async () => {
      const text = (transcriptEl?.value || '').trim();
      if (!text) {
        setStatus(transcriptStatusEl, 'Nothing to copy yet.', 'warning');
        return;
      }
      try {
        await navigator.clipboard.writeText(text);
        setStatus(transcriptStatusEl, 'Transcript copied.', 'success');
      } catch {
        setStatus(transcriptStatusEl, 'Copy failed. Please copy manually.', 'error');
      }
    });
  }

  setBusy(false);
  setServerReadyState(false);
  updateAudioMeta();
  updatePartMinutesMeta();

  window.setTimeout(() => {
    checkAndWarm();
  }, 0);
})();
