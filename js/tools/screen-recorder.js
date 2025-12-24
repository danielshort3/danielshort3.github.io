(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const el = {
    source: $('[data-screenrec="source"]'),
    regionModes: $$('[data-screenrec="region-mode"]'),
    selectRegion: $('[data-screenrec="select-region"]'),
    clearRegion: $('[data-screenrec="clear-region"]'),
    regionHelp: $('[data-screenrec="region-help"]'),
    format: $('[data-screenrec="format"]'),
    formatHelp: $('[data-screenrec="format-help"]'),
    startCapture: $('[data-screenrec="start-capture"]'),
    stopCapture: $('[data-screenrec="stop-capture"]'),
    video: $('[data-screenrec="video"]'),
    overlay: $('[data-screenrec="overlay"]'),
    overlayLabel: $('[data-screenrec="overlay-label"]'),
    selection: $('[data-screenrec="selection"]'),
    placeholder: $('[data-screenrec="placeholder"]'),
    status: $('[data-screenrec="status"]'),
    captureMeta: $('[data-screenrec="capture-meta"]'),
    regionMeta: $('[data-screenrec="region-meta"]'),
    timer: $('[data-screenrec="timer"]'),
    startRecord: $('[data-screenrec="start-record"]'),
    pauseRecord: $('[data-screenrec="pause-record"]'),
    stopRecord: $('[data-screenrec="stop-record"]'),
    download: $('[data-screenrec="download"]'),
    downloadNote: $('[data-screenrec="download-note"]')
  };

  if (!el.source || !el.startCapture || !el.video || !el.startRecord) return;

  const supportsCapture = Boolean(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
  const supportsRecorder = typeof window.MediaRecorder !== 'undefined';
  const supportsCanvasCapture = Boolean(window.HTMLCanvasElement && HTMLCanvasElement.prototype && HTMLCanvasElement.prototype.captureStream);

  const state = {
    stream: null,
    recorder: null,
    outputStream: null,
    chunks: [],
    recording: false,
    paused: false,
    captureActive: false,
    selecting: false,
    cropMode: 'full',
    cropRegion: null,
    drawId: null,
    timerId: null,
    timerStart: 0,
    timerPausedAt: 0,
    totalPausedMs: 0,
    downloadUrl: null,
    stopReason: ''
  };

  const MIME_TYPE_OPTIONS = [
    { label: 'MP4 (H.264)', mimeType: 'video/mp4;codecs=avc1.42E01E' },
    { label: 'MP4 (default)', mimeType: 'video/mp4' },
    { label: 'WebM (VP9)', mimeType: 'video/webm;codecs=vp9' },
    { label: 'WebM (VP8)', mimeType: 'video/webm;codecs=vp8' },
    { label: 'WebM (default)', mimeType: 'video/webm' }
  ];

  const PREFERRED_MIME_TYPES = [
    'video/mp4;codecs=avc1.42E01E',
    'video/mp4',
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm'
  ];

  const setStatus = (text, tone) => {
    if (!el.status) return;
    el.status.textContent = text;
    if (tone) {
      el.status.dataset.tone = tone;
    } else {
      delete el.status.dataset.tone;
    }
  };

  const formatDuration = (ms) => {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) {
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  };

  const formatBytes = (bytes) => {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const idx = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
    const value = bytes / (1024 ** idx);
    return `${value.toFixed(value >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`;
  };

  const extensionFromMime = (mimeType) => {
    if (!mimeType) return 'webm';
    if (mimeType.includes('mp4')) return 'mp4';
    if (mimeType.includes('webm')) return 'webm';
    return 'webm';
  };

  const buildFilename = (ext) => {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    return `screen-recording-${stamp}.${ext}`;
  };

  const getSupportedFormats = () => {
    if (!supportsRecorder || !MediaRecorder.isTypeSupported) return [];
    return MIME_TYPE_OPTIONS.filter((opt) => MediaRecorder.isTypeSupported(opt.mimeType));
  };

  const getPreferredMimeType = () => {
    if (!supportsRecorder || !MediaRecorder.isTypeSupported) return '';
    for (const type of PREFERRED_MIME_TYPES) {
      if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return '';
  };

  const setFormatOptions = () => {
    if (!el.format) return;
    el.format.innerHTML = '';
    const autoOpt = document.createElement('option');
    autoOpt.value = 'auto';
    autoOpt.textContent = 'Auto (best supported)';
    el.format.appendChild(autoOpt);

    const supported = getSupportedFormats();
    supported.forEach((opt) => {
      const option = document.createElement('option');
      option.value = opt.mimeType;
      option.textContent = opt.label;
      el.format.appendChild(option);
    });

    el.format.value = 'auto';

    if (el.formatHelp) {
      if (!supported.length) {
        el.formatHelp.textContent = 'No explicit formats detected. The browser default will be used.';
      } else {
        const labels = supported.map((opt) => opt.label).join(', ');
        const mp4Supported = supported.some((opt) => opt.mimeType.includes('mp4'));
        const mp4Note = mp4Supported ? 'MP4 is supported in this browser.' : 'MP4 recording is not supported here.';
        el.formatHelp.textContent = `Supported: ${labels}. ${mp4Note}`;
      }
    }
  };

  const getSelectedMimeType = () => {
    if (!el.format) return '';
    if (el.format.value && el.format.value !== 'auto') return el.format.value;
    return getPreferredMimeType();
  };

  const setDownload = (blob, mimeType) => {
    if (!el.download) return;
    if (state.downloadUrl) {
      URL.revokeObjectURL(state.downloadUrl);
    }
    const ext = extensionFromMime(mimeType);
    const name = buildFilename(ext);
    const url = URL.createObjectURL(blob);
    state.downloadUrl = url;
    el.download.href = url;
    el.download.download = name;
    el.download.hidden = false;
    if (el.downloadNote) {
      el.downloadNote.textContent = `Ready: ${name} (${formatBytes(blob.size)}).`;
      el.downloadNote.hidden = false;
    }
  };

  const clearDownload = () => {
    if (state.downloadUrl) {
      URL.revokeObjectURL(state.downloadUrl);
    }
    state.downloadUrl = null;
    if (el.download) {
      el.download.hidden = true;
      el.download.href = '#';
      el.download.removeAttribute('download');
    }
    if (el.downloadNote) {
      el.downloadNote.hidden = true;
      el.downloadNote.textContent = '';
    }
  };

  const stopTracks = (stream) => {
    if (!stream) return;
    stream.getTracks().forEach((track) => track.stop());
  };

  const updateCaptureMeta = () => {
    if (!el.captureMeta) return;
    if (!state.stream) {
      el.captureMeta.textContent = 'Not started';
      return;
    }
    const track = state.stream.getVideoTracks()[0];
    const settings = track && track.getSettings ? track.getSettings() : {};
    const parts = [];
    if (settings.displaySurface) {
      const surface = settings.displaySurface;
      const label = surface === 'monitor'
        ? 'Monitor'
        : surface === 'window'
          ? 'Window'
          : surface === 'browser'
            ? 'Browser tab'
            : surface;
      parts.push(label);
    }
    const width = settings.width || el.video.videoWidth;
    const height = settings.height || el.video.videoHeight;
    if (width && height) {
      parts.push(`${width}x${height}`);
    }
    if (settings.frameRate) {
      parts.push(`${Math.round(settings.frameRate)} fps`);
    }
    el.captureMeta.textContent = parts.length ? parts.join(' | ') : 'Capture active';
  };

  const updateRegionMeta = () => {
    if (!el.regionMeta) return;
    if (state.cropMode === 'custom') {
      if (state.cropRegion) {
        const { width, height } = state.cropRegion;
        el.regionMeta.textContent = `Custom ${Math.round(width)}x${Math.round(height)}`;
      } else {
        el.regionMeta.textContent = 'Custom region not set';
      }
    } else {
      el.regionMeta.textContent = 'Full capture';
    }
  };

  const updateTimer = () => {
    if (!el.timer) return;
    if (!state.recording) return;
    const now = performance.now();
    const endTime = state.timerPausedAt || now;
    const elapsed = endTime - state.timerStart - state.totalPausedMs;
    el.timer.textContent = formatDuration(elapsed);
  };

  const startTimer = () => {
    state.timerStart = performance.now();
    state.timerPausedAt = 0;
    state.totalPausedMs = 0;
    if (el.timer) el.timer.textContent = '00:00';
    if (state.timerId) clearInterval(state.timerId);
    state.timerId = setInterval(updateTimer, 250);
  };

  const pauseTimer = () => {
    state.timerPausedAt = performance.now();
  };

  const resumeTimer = () => {
    if (!state.timerPausedAt) return;
    state.totalPausedMs += performance.now() - state.timerPausedAt;
    state.timerPausedAt = 0;
  };

  const stopTimer = () => {
    if (state.timerId) {
      clearInterval(state.timerId);
      state.timerId = null;
    }
  };

  const resetTimer = () => {
    stopTimer();
    if (el.timer) el.timer.textContent = '00:00';
  };

  const updateButtons = () => {
    const videoReady = Boolean(el.video && el.video.videoWidth);
    const canRecord = state.captureActive && !state.recording && (state.cropMode === 'full' || state.cropRegion);
    const canSelect = state.captureActive && state.cropMode === 'custom' && !state.recording && videoReady;
    if (el.startCapture) el.startCapture.disabled = !supportsCapture || !supportsRecorder || state.captureActive;
    if (el.stopCapture) el.stopCapture.disabled = !state.captureActive;
    if (el.selectRegion) el.selectRegion.disabled = !canSelect;
    if (el.clearRegion) el.clearRegion.disabled = !state.cropRegion || state.recording;
    if (el.startRecord) el.startRecord.disabled = !canRecord;
    if (el.pauseRecord) el.pauseRecord.disabled = !state.recording;
    if (el.stopRecord) el.stopRecord.disabled = !state.recording;
    if (el.pauseRecord) el.pauseRecord.textContent = state.paused ? 'Resume' : 'Pause';
  };

  const clearSelection = () => {
    state.cropRegion = null;
    if (el.selection) {
      el.selection.hidden = true;
      el.selection.style.width = '0px';
      el.selection.style.height = '0px';
    }
    updateRegionMeta();
    updateButtons();
  };

  const setCropMode = (mode) => {
    if (mode === 'custom' && !supportsCanvasCapture) {
      state.cropMode = 'full';
      updateRegionMeta();
      updateButtons();
      setStatus('Custom region capture is not supported in this browser.', 'warn');
      return;
    }
    state.cropMode = mode;
    if (mode === 'full') {
      clearSelection();
      if (el.regionHelp) {
        el.regionHelp.textContent = 'Custom region is cropped locally after capture.';
      }
    } else if (el.regionHelp) {
      el.regionHelp.textContent = 'Click Select region, then drag on the preview.';
    }
    updateRegionMeta();
    updateButtons();
  };

  const getDisplayMetrics = () => {
    const rect = el.video.getBoundingClientRect();
    const videoWidth = el.video.videoWidth || rect.width;
    const videoHeight = el.video.videoHeight || rect.height;
    const elementWidth = rect.width;
    const elementHeight = rect.height;
    if (!videoWidth || !videoHeight) {
      return {
        rect,
        videoWidth: elementWidth,
        videoHeight: elementHeight,
        displayWidth: elementWidth,
        displayHeight: elementHeight,
        offsetX: 0,
        offsetY: 0
      };
    }
    const videoAspect = videoWidth / videoHeight;
    const elementAspect = elementWidth / elementHeight;
    let displayWidth = elementWidth;
    let displayHeight = elementHeight;
    let offsetX = 0;
    let offsetY = 0;
    if (videoAspect > elementAspect) {
      displayWidth = elementWidth;
      displayHeight = elementWidth / videoAspect;
      offsetY = (elementHeight - displayHeight) / 2;
    } else {
      displayHeight = elementHeight;
      displayWidth = elementHeight * videoAspect;
      offsetX = (elementWidth - displayWidth) / 2;
    }
    return { rect, videoWidth, videoHeight, displayWidth, displayHeight, offsetX, offsetY };
  };

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const normalizePoint = (clientX, clientY, metrics) => {
    const x = clientX - metrics.rect.left;
    const y = clientY - metrics.rect.top;
    const minX = metrics.offsetX;
    const minY = metrics.offsetY;
    const maxX = metrics.offsetX + metrics.displayWidth;
    const maxY = metrics.offsetY + metrics.displayHeight;
    return {
      x: clamp(x, minX, maxX),
      y: clamp(y, minY, maxY),
      minX,
      minY,
      maxX,
      maxY
    };
  };

  const showSelection = (x, y, width, height) => {
    if (!el.selection) return;
    el.selection.hidden = false;
    el.selection.style.left = `${x}px`;
    el.selection.style.top = `${y}px`;
    el.selection.style.width = `${width}px`;
    el.selection.style.height = `${height}px`;
  };

  const selectionState = {
    active: false,
    startX: 0,
    startY: 0,
    metrics: null
  };

  const startRegionSelection = () => {
    if (!state.captureActive || state.recording) return;
    state.selecting = true;
    if (el.overlay) el.overlay.classList.add('is-active');
    if (el.overlayLabel) el.overlayLabel.hidden = false;
    if (el.selection) el.selection.hidden = true;
    setStatus('Drag to select a region.', 'pending');
  };

  const endRegionSelection = () => {
    state.selecting = false;
    if (el.overlay) el.overlay.classList.remove('is-active');
    if (el.overlayLabel) el.overlayLabel.hidden = true;
  };

  const handlePointerDown = (event) => {
    if (!state.selecting || !el.video) return;
    const metrics = getDisplayMetrics();
    const point = normalizePoint(event.clientX, event.clientY, metrics);
    selectionState.active = true;
    selectionState.startX = point.x;
    selectionState.startY = point.y;
    selectionState.metrics = metrics;
    showSelection(point.x, point.y, 0, 0);
    el.overlay?.setPointerCapture?.(event.pointerId);
  };

  const handlePointerMove = (event) => {
    if (!selectionState.active) return;
    const metrics = selectionState.metrics || getDisplayMetrics();
    const point = normalizePoint(event.clientX, event.clientY, metrics);
    const startX = selectionState.startX;
    const startY = selectionState.startY;
    const x = Math.min(startX, point.x);
    const y = Math.min(startY, point.y);
    const width = Math.abs(point.x - startX);
    const height = Math.abs(point.y - startY);
    showSelection(x, y, width, height);
  };

  const handlePointerUp = (event) => {
    if (!selectionState.active) return;
    const metrics = selectionState.metrics || getDisplayMetrics();
    const point = normalizePoint(event.clientX, event.clientY, metrics);
    const startX = selectionState.startX;
    const startY = selectionState.startY;
    const x = Math.min(startX, point.x);
    const y = Math.min(startY, point.y);
    const width = Math.abs(point.x - startX);
    const height = Math.abs(point.y - startY);
    selectionState.active = false;

    const minSize = 40;
    if (width < minSize || height < minSize) {
      clearSelection();
      setStatus('Region too small. Drag a larger area.', 'warn');
      endRegionSelection();
      return;
    }

    const scaleX = metrics.videoWidth / metrics.displayWidth;
    const scaleY = metrics.videoHeight / metrics.displayHeight;
    const cropX = (x - metrics.offsetX) * scaleX;
    const cropY = (y - metrics.offsetY) * scaleY;
    const cropWidth = width * scaleX;
    const cropHeight = height * scaleY;

    state.cropRegion = {
      x: Math.max(0, cropX),
      y: Math.max(0, cropY),
      width: Math.max(1, cropWidth),
      height: Math.max(1, cropHeight)
    };

    showSelection(x, y, width, height);
    setStatus('Region selected. Ready to record.', 'ready');
    endRegionSelection();
    updateRegionMeta();
    updateButtons();
  };

  const handlePointerCancel = () => {
    if (!selectionState.active) return;
    selectionState.active = false;
    clearSelection();
    endRegionSelection();
  };

  const buildConstraints = (surface) => {
    const video = {
      frameRate: { ideal: 30, max: 60 },
      cursor: 'motion'
    };
    if (surface && surface !== 'any') {
      video.displaySurface = surface;
    }
    return {
      video,
      audio: false,
      selfBrowserSurface: 'exclude',
      surfaceSwitching: 'include',
      monitorTypeSurfaces: 'include'
    };
  };

  const startCapture = async () => {
    if (!supportsCapture || !supportsRecorder || state.captureActive) return;
    clearDownload();
    setStatus('Requesting capture permission...', 'pending');

    let stream = null;
    const surface = el.source ? el.source.value : 'any';
    try {
      stream = await navigator.mediaDevices.getDisplayMedia(buildConstraints(surface));
    } catch (err) {
      if (surface !== 'any') {
        try {
          stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
        } catch (fallbackErr) {
          setStatus('Capture request failed. Check permissions.', 'error');
          return;
        }
      } else {
        setStatus('Capture request failed. Check permissions.', 'error');
        return;
      }
    }

    if (!stream) {
      setStatus('Capture did not start.', 'error');
      return;
    }

    stream.getAudioTracks().forEach((track) => stream.removeTrack(track));
    state.stream = stream;
    state.captureActive = true;
    state.stopReason = '';
    el.video.srcObject = stream;
    el.video.play().catch(() => {});
    if (el.placeholder) el.placeholder.hidden = true;
    updateCaptureMeta();
    updateRegionMeta();
    updateButtons();
    resetTimer();

    const track = stream.getVideoTracks()[0];
    track.addEventListener('ended', () => {
      stopCapture('Capture ended by the browser.');
    }, { once: true });

    setStatus('Capture ready. Select a region if needed, then start recording.', 'ready');
  };

  const finalizeStopCapture = () => {
    stopTracks(state.stream);
    state.stream = null;
    state.captureActive = false;
    state.recording = false;
    state.paused = false;
    state.recorder = null;
    state.outputStream = null;
    stopTimer();
    if (el.video) el.video.srcObject = null;
    if (el.placeholder) el.placeholder.hidden = false;
    clearSelection();
    updateCaptureMeta();
    updateRegionMeta();
    updateButtons();
    const message = state.stopReason || (state.downloadUrl ? 'Capture stopped. Download ready.' : 'Capture stopped.');
    setStatus(message, 'idle');
    state.stopReason = '';
  };

  const stopCapture = (reason) => {
    if (!state.captureActive) return;
    if (reason) state.stopReason = reason;
    if (state.recording && state.recorder && state.recorder.state !== 'inactive') {
      state.recorder.addEventListener('stop', finalizeStopCapture, { once: true });
      state.recorder.stop();
      return;
    }
    finalizeStopCapture();
  };

  const stopDrawLoop = () => {
    if (state.drawId) {
      cancelAnimationFrame(state.drawId);
      state.drawId = null;
    }
  };

  const startDrawLoop = () => {
    if (!state.cropRegion || !state.recording || !state.outputStream) return;
    const { x, y, width, height } = state.cropRegion;
    const canvas = state.outputStream.canvas;
    const ctx = state.outputStream.ctx;
    if (!canvas || !ctx) return;
    canvas.width = Math.round(width);
    canvas.height = Math.round(height);
    const fps = 30;
    const frameDuration = 1000 / fps;
    let lastFrame = performance.now();

    const draw = (now) => {
      if (!state.recording || state.paused) return;
      if (now - lastFrame >= frameDuration) {
        ctx.drawImage(el.video, x, y, width, height, 0, 0, canvas.width, canvas.height);
        lastFrame = now;
      }
      state.drawId = requestAnimationFrame(draw);
    };

    state.drawId = requestAnimationFrame(draw);
  };

  const buildOutputStream = () => {
    if (state.cropMode !== 'custom' || !state.cropRegion || !supportsCanvasCapture) return state.stream;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { alpha: false });
    canvas.width = Math.round(state.cropRegion.width);
    canvas.height = Math.round(state.cropRegion.height);
    const fps = 30;
    const canvasStream = canvas.captureStream(fps);
    canvasStream.canvas = canvas;
    canvasStream.ctx = ctx;
    return canvasStream;
  };

  const startRecording = () => {
    if (!state.captureActive || state.recording) return;
    if (state.cropMode === 'custom' && !state.cropRegion) {
      setStatus('Select a region before recording.', 'warn');
      return;
    }
    if (state.cropMode === 'custom' && !supportsCanvasCapture) {
      setStatus('Custom region capture is not supported in this browser.', 'error');
      return;
    }

    clearDownload();
    const mimeType = getSelectedMimeType();
    const outputStream = buildOutputStream();
    if (state.cropMode === 'custom' && state.cropRegion && outputStream.canvas && outputStream.ctx) {
      const { x, y, width, height } = state.cropRegion;
      outputStream.ctx.drawImage(el.video, x, y, width, height, 0, 0, outputStream.canvas.width, outputStream.canvas.height);
    }
    const options = {};
    if (mimeType) options.mimeType = mimeType;

    let recorder;
    try {
      recorder = new MediaRecorder(outputStream, options);
    } catch (err) {
      if (options.mimeType) {
        try {
          recorder = new MediaRecorder(outputStream);
          setStatus('Selected format not supported. Using browser default.', 'warn');
        } catch (fallbackErr) {
          setStatus('Recording failed to initialize in this browser.', 'error');
          return;
        }
      } else {
        setStatus('Recording failed to initialize in this browser.', 'error');
        return;
      }
    }

    state.recorder = recorder;
    state.outputStream = outputStream !== state.stream ? outputStream : null;
    state.chunks = [];
    state.recording = true;
    state.paused = false;
    resetTimer();
    startTimer();

    recorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) {
        state.chunks.push(event.data);
      }
    });

    recorder.addEventListener('stop', () => {
      stopDrawLoop();
      state.recording = false;
      state.paused = false;
      stopTimer();
      const mime = recorder.mimeType || mimeType || 'video/webm';
      if (state.chunks.length) {
        const blob = new Blob(state.chunks, { type: mime });
        setDownload(blob, mime);
        setStatus('Recording stopped. Download ready.', 'ready');
        state.chunks = [];
      } else {
        setStatus('Recording stopped. No data captured.', 'warn');
      }
      if (state.outputStream && state.outputStream !== state.stream) {
        stopTracks(state.outputStream);
        state.outputStream = null;
      }
      updateButtons();
    });

    recorder.start(200);
    if (state.cropMode === 'custom') {
      startDrawLoop();
    }
    setStatus('Recording...', 'recording');
    updateButtons();
  };

  const togglePause = () => {
    if (!state.recorder || !state.recording) return;
    if (state.recorder.state === 'recording') {
      state.recorder.pause();
      state.paused = true;
      pauseTimer();
      stopDrawLoop();
      setStatus('Recording paused.', 'warn');
    } else if (state.recorder.state === 'paused') {
      state.recorder.resume();
      state.paused = false;
      resumeTimer();
      if (state.cropMode === 'custom') {
        startDrawLoop();
      }
      setStatus('Recording...', 'recording');
    }
    updateButtons();
  };

  const stopRecording = () => {
    if (!state.recorder || !state.recording) return;
    if (state.recorder.state !== 'inactive') {
      state.recorder.stop();
    }
  };

  const init = () => {
    if (!supportsCapture || !supportsRecorder) {
      setStatus('Screen recording is not supported in this browser.', 'error');
      if (el.formatHelp) {
        el.formatHelp.textContent = 'Upgrade to a modern browser with Screen Capture and MediaRecorder support.';
      }
      updateButtons();
      return;
    }

    setFormatOptions();
    updateButtons();
    updateRegionMeta();

    if (!supportsCanvasCapture) {
      el.regionModes.forEach((input) => {
        if (input.value === 'custom') {
          input.disabled = true;
          input.closest('label')?.classList.add('is-disabled');
        }
      });
      if (el.regionHelp) {
        el.regionHelp.textContent = 'Custom region capture is not supported in this browser.';
      }
    }

    el.startCapture?.addEventListener('click', startCapture);
    el.stopCapture?.addEventListener('click', () => stopCapture());
    el.startRecord?.addEventListener('click', startRecording);
    el.pauseRecord?.addEventListener('click', togglePause);
    el.stopRecord?.addEventListener('click', stopRecording);
    el.selectRegion?.addEventListener('click', startRegionSelection);
    el.clearRegion?.addEventListener('click', () => {
      clearSelection();
      setStatus('Region cleared.', 'ready');
    });

    el.regionModes.forEach((input) => {
      input.addEventListener('change', () => {
        if (input.checked) {
          setCropMode(input.value);
        }
      });
    });

    el.overlay?.addEventListener('pointerdown', handlePointerDown);
    el.overlay?.addEventListener('pointermove', handlePointerMove);
    el.overlay?.addEventListener('pointerup', handlePointerUp);
    el.overlay?.addEventListener('pointercancel', handlePointerCancel);
    el.overlay?.addEventListener('pointerleave', handlePointerCancel);

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && state.selecting) {
        handlePointerCancel();
        setStatus('Region selection cancelled.', 'warn');
      }
    });

    el.video?.addEventListener('loadedmetadata', () => {
      updateCaptureMeta();
      updateButtons();
    });
  };

  init();
})();
