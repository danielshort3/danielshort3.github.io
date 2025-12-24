(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);

  const el = {
    format: $('[data-screenrec="format"]'),
    formatHelp: $('[data-screenrec="format-help"]'),
    audioToggle: $('[data-screenrec="audio-toggle"]'),
    audioHelp: $('[data-screenrec="audio-help"]'),
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
    testRecord: $('[data-screenrec="test-record"]'),
    editPanel: $('[data-screenrec="edit-panel"]'),
    trimStart: $('[data-screenrec="trim-start"]'),
    trimEnd: $('[data-screenrec="trim-end"]'),
    trimStartLabel: $('[data-screenrec="trim-start-label"]'),
    trimEndLabel: $('[data-screenrec="trim-end-label"]'),
    trimLength: $('[data-screenrec="trim-length"]'),
    selectCrop: $('[data-screenrec="select-crop"]'),
    clearCrop: $('[data-screenrec="clear-crop"]'),
    exportBtn: $('[data-screenrec="export"]'),
    download: $('[data-screenrec="download"]'),
    downloadNote: $('[data-screenrec="download-note"]')
  };

  if (!el.startCapture || !el.video || !el.startRecord) return;

  const supportsCapture = Boolean(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
  const supportsRecorder = typeof window.MediaRecorder !== 'undefined';
  const supportsCanvasCapture = Boolean(window.HTMLCanvasElement && HTMLCanvasElement.prototype && HTMLCanvasElement.prototype.captureStream);

  const state = {
    stream: null,
    recorder: null,
    chunks: [],
    recording: false,
    paused: false,
    captureActive: false,
    testing: false,
    selecting: false,
    cropRegion: null,
    timerId: null,
    timerStart: 0,
    timerPausedAt: 0,
    totalPausedMs: 0,
    downloadUrl: null,
    recordedBlob: null,
    recordedUrl: null,
    recordedDuration: 0,
    recordedHasAudio: false,
    trimStart: 0,
    trimEnd: 0,
    exporting: false,
    stopReason: ''
  };

  const MIN_TRIM_GAP = 0.2;
  const DEFAULT_EXPORT_FPS = 30;
  const TEST_DURATION_MS = 5000;

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

  const streamHasAudio = (stream) => Boolean(stream && stream.getAudioTracks && stream.getAudioTracks().length);

  const updateCaptureMeta = () => {
    if (!el.captureMeta) return;
    if (!state.stream) {
      el.captureMeta.textContent = state.recordedUrl ? 'Recorded clip' : 'Not started';
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
    const hasAudio = streamHasAudio(state.stream);
    parts.push(hasAudio ? 'Audio on' : 'Audio off');
    el.captureMeta.textContent = parts.length ? parts.join(' | ') : 'Capture active';
  };

  const updateCropMeta = () => {
    if (!el.regionMeta) return;
    if (state.cropRegion) {
      el.regionMeta.textContent = `Crop ${Math.round(state.cropRegion.width)}x${Math.round(state.cropRegion.height)}`;
    } else {
      el.regionMeta.textContent = 'Full frame';
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

  const resetTimer = (valueMs) => {
    stopTimer();
    if (el.timer) {
      el.timer.textContent = valueMs ? formatDuration(valueMs) : '00:00';
    }
  };

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const updateTrimUI = (source) => {
    if (!el.trimStart || !el.trimEnd) return;
    const duration = Math.max(0, state.recordedDuration || 0);
    let start = parseFloat(el.trimStart.value);
    let end = parseFloat(el.trimEnd.value);
    if (Number.isNaN(start)) start = 0;
    if (Number.isNaN(end)) end = duration;
    start = clamp(start, 0, duration);
    end = clamp(end, 0, duration);
    if (end - start < MIN_TRIM_GAP) {
      if (source === 'start') {
        start = Math.max(0, end - MIN_TRIM_GAP);
      } else {
        end = Math.min(duration, start + MIN_TRIM_GAP);
      }
    }
    state.trimStart = start;
    state.trimEnd = end;
    el.trimStart.value = start;
    el.trimEnd.value = end;
    if (el.trimStartLabel) el.trimStartLabel.textContent = formatDuration(start * 1000);
    if (el.trimEndLabel) el.trimEndLabel.textContent = formatDuration(end * 1000);
    if (el.trimLength) el.trimLength.textContent = formatDuration(Math.max(0, (end - start) * 1000));
  };

  const initTrimControls = () => {
    if (!el.trimStart || !el.trimEnd) return;
    el.trimStart.addEventListener('input', () => {
      updateTrimUI('start');
      updateButtons();
    });
    el.trimEnd.addEventListener('input', () => {
      updateTrimUI('end');
      updateButtons();
    });
  };

  const resetRecordedClip = () => {
    if (state.recordedUrl) {
      URL.revokeObjectURL(state.recordedUrl);
    }
    state.recordedUrl = null;
    state.recordedBlob = null;
    state.recordedDuration = 0;
    state.recordedHasAudio = false;
    state.trimStart = 0;
    state.trimEnd = 0;
    clearDownload();
    clearCrop();
    if (el.editPanel) el.editPanel.hidden = true;
    if (el.trimStart) {
      el.trimStart.max = '0';
      el.trimStart.value = '0';
    }
    if (el.trimEnd) {
      el.trimEnd.max = '0';
      el.trimEnd.value = '0';
    }
    if (el.trimStartLabel) el.trimStartLabel.textContent = '00:00';
    if (el.trimEndLabel) el.trimEndLabel.textContent = '00:00';
    if (el.trimLength) el.trimLength.textContent = '00:00';
    updateCaptureMeta();
  };

  const setLivePreview = () => {
    if (!state.stream) return;
    el.video.src = '';
    el.video.srcObject = state.stream;
    el.video.controls = false;
    el.video.loop = false;
    el.video.muted = true;
    el.video.play().catch(() => {});
    if (el.placeholder) el.placeholder.hidden = true;
  };

  const setRecordedClip = (blob) => {
    if (!blob) return;
    resetTimer();
    if (state.recordedUrl) {
      URL.revokeObjectURL(state.recordedUrl);
    }
    state.recordedBlob = blob;
    state.recordedUrl = URL.createObjectURL(blob);
    el.video.srcObject = null;
    el.video.src = state.recordedUrl;
    el.video.controls = true;
    el.video.loop = true;
    el.video.muted = false;
    el.video.play().catch(() => {});
    if (el.placeholder) el.placeholder.hidden = true;

    const handleMetadata = () => {
      state.recordedDuration = Number.isFinite(el.video.duration) ? el.video.duration : 0;
      if (el.trimStart) el.trimStart.max = String(state.recordedDuration);
      if (el.trimEnd) el.trimEnd.max = String(state.recordedDuration);
      if (el.trimStart) el.trimStart.value = '0';
      if (el.trimEnd) el.trimEnd.value = String(state.recordedDuration);
      updateTrimUI('end');
      resetTimer(state.recordedDuration * 1000);
      updateCropMeta();
      updateButtons();
    };
    if (el.video.readyState >= 1) {
      handleMetadata();
    } else {
      el.video.addEventListener('loadedmetadata', handleMetadata, { once: true });
    }

    if (el.editPanel) el.editPanel.hidden = false;
    setStatus('Clip ready to edit.', 'ready');
  };

  const showOverlay = (active) => {
    if (el.overlay) {
      el.overlay.classList.toggle('is-active', active);
    }
    if (el.overlayLabel) {
      el.overlayLabel.hidden = !active;
    }
  };

  const clearSelection = () => {
    if (el.selection) {
      el.selection.hidden = true;
      el.selection.style.width = '0px';
      el.selection.style.height = '0px';
    }
  };

  const clearCrop = () => {
    state.cropRegion = null;
    clearSelection();
    updateCropMeta();
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

  const startCropSelection = () => {
    if (!state.recordedUrl || state.recording || state.exporting) return;
    state.selecting = true;
    showOverlay(true);
    setStatus('Drag to select a crop region.', 'pending');
  };

  const endCropSelection = () => {
    state.selecting = false;
    showOverlay(false);
  };

  const handlePointerDown = (event) => {
    if (!state.selecting) return;
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
      clearCrop();
      setStatus('Crop region too small. Drag a larger area.', 'warn');
      endCropSelection();
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
    setStatus('Crop region set.', 'ready');
    endCropSelection();
    updateCropMeta();
    updateButtons();
  };

  const handlePointerCancel = () => {
    if (!selectionState.active) return;
    selectionState.active = false;
    clearCrop();
    endCropSelection();
  };

  const updateButtons = () => {
    const hasRecording = Boolean(state.recordedUrl);
    const trimOk = hasRecording && (state.trimEnd - state.trimStart) >= MIN_TRIM_GAP;
    const busy = state.recording || state.exporting || state.testing;
    if (el.startCapture) {
      el.startCapture.disabled = !supportsCapture || !supportsRecorder || state.captureActive || busy;
    }
    if (el.stopCapture) {
      el.stopCapture.disabled = !state.captureActive || busy;
    }
    if (el.startRecord) {
      el.startRecord.disabled = !state.captureActive || busy;
    }
    if (el.pauseRecord) {
      el.pauseRecord.disabled = !state.recording || state.exporting || state.testing;
      el.pauseRecord.textContent = state.paused ? 'Resume' : 'Pause';
    }
    if (el.stopRecord) {
      el.stopRecord.disabled = !state.recording || state.exporting || state.testing;
    }
    if (el.testRecord) {
      el.testRecord.disabled = !state.captureActive || state.recording || state.exporting || state.testing;
    }
    if (el.selectCrop) {
      el.selectCrop.disabled = !hasRecording || !supportsCanvasCapture || state.exporting;
    }
    if (el.clearCrop) {
      el.clearCrop.disabled = !state.cropRegion || state.exporting;
    }
    if (el.exportBtn) {
      el.exportBtn.disabled = !hasRecording || !supportsCanvasCapture || state.exporting || !trimOk;
    }
    if (el.audioToggle) {
      el.audioToggle.disabled = !supportsCapture || state.captureActive || state.recording || state.exporting || state.testing;
    }
  };

  const startCapture = async () => {
    if (!supportsCapture || !supportsRecorder || state.captureActive || state.recording || state.testing) return;
    setStatus('Requesting capture permission...', 'pending');

    const includeAudio = Boolean(el.audioToggle?.checked);
    const constraints = { video: true, audio: includeAudio };
    let stream = null;
    let audioFallback = false;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia(constraints);
    } catch (_) {
      if (includeAudio) {
        try {
          stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
          audioFallback = true;
        } catch (err) {
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

    if (!includeAudio) {
      stream.getAudioTracks().forEach((track) => {
        stream.removeTrack(track);
        track.stop();
      });
    }

    clearDownload();
    resetRecordedClip();
    state.stream = stream;
    state.captureActive = true;
    state.stopReason = '';
    setLivePreview();
    updateCaptureMeta();
    updateCropMeta();
    updateButtons();
    resetTimer();

    const track = stream.getVideoTracks()[0];
    track.addEventListener('ended', () => {
      if (state.recording) {
        state.stopReason = 'Capture ended by the browser.';
        stopRecording();
      } else if (state.testing) {
        state.stopReason = 'Capture ended by the browser.';
        state.testing = false;
        updateButtons();
        stopCapture('Capture ended by the browser.');
      } else {
        stopCapture('Capture ended by the browser.');
      }
    }, { once: true });

    if (audioFallback) {
      setStatus('Audio capture is not available. Capturing without audio.', 'warn');
    } else {
      setStatus('Capture ready. Start recording when you are ready.', 'ready');
    }
  };

  const stopCapture = (reason) => {
    if (!state.captureActive) return;
    stopTracks(state.stream);
    state.stream = null;
    state.captureActive = false;
    updateCaptureMeta();
    updateButtons();
    if (!state.recordedUrl) {
      el.video.srcObject = null;
      if (el.placeholder) el.placeholder.hidden = false;
    }
    resetTimer(state.recordedDuration ? state.recordedDuration * 1000 : 0);
    const message = reason || state.stopReason || (state.recordedUrl ? 'Capture stopped. Clip ready to edit.' : 'Capture stopped.');
    setStatus(message, state.recordedUrl ? 'ready' : 'idle');
    state.stopReason = '';
  };

  const createRecorder = (stream, mimeType) => {
    const options = {};
    if (mimeType) options.mimeType = mimeType;
    let recorder;
    let finalMime = mimeType;
    try {
      recorder = new MediaRecorder(stream, options);
    } catch (err) {
      if (mimeType) {
        recorder = new MediaRecorder(stream);
        finalMime = '';
        setStatus('Selected format not supported. Using browser default.', 'warn');
      } else {
        throw err;
      }
    }
    return { recorder, mimeType: finalMime };
  };

  const startRecording = () => {
    if (!state.captureActive || state.recording || state.testing || !state.stream) return;
    clearDownload();
    resetRecordedClip();
    clearCrop();

    const mimeType = getSelectedMimeType();
    let recorderInfo;
    try {
      recorderInfo = createRecorder(state.stream, mimeType);
    } catch (_) {
      setStatus('Recording failed to initialize in this browser.', 'error');
      return;
    }

    state.recorder = recorderInfo.recorder;
    state.chunks = [];
    state.recording = true;
    state.paused = false;
    startTimer();

    state.recorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) {
        state.chunks.push(event.data);
      }
    });

    state.recorder.addEventListener('stop', () => {
      state.recording = false;
      state.paused = false;
      stopTimer();
      const mime = state.recorder.mimeType || recorderInfo.mimeType || 'video/webm';
      if (state.chunks.length) {
        const blob = new Blob(state.chunks, { type: mime });
        state.chunks = [];
        state.recordedHasAudio = streamHasAudio(state.stream);
        setRecordedClip(blob);
      } else {
        setStatus('Recording stopped. No data captured.', 'warn');
      }
      stopCapture();
      updateButtons();
    }, { once: true });

    state.recorder.start(200);
    setStatus('Recording...', 'recording');
    updateButtons();
  };

  const togglePause = () => {
    if (!state.recorder || !state.recording) return;
    if (state.recorder.state === 'recording') {
      state.recorder.pause();
      state.paused = true;
      pauseTimer();
      setStatus('Recording paused.', 'warn');
    } else if (state.recorder.state === 'paused') {
      state.recorder.resume();
      state.paused = false;
      resumeTimer();
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

  const playTestClip = (blob) => {
    if (!blob) {
      setStatus('Test recording failed.', 'warn');
      return;
    }
    const url = URL.createObjectURL(blob);
    el.video.srcObject = null;
    el.video.src = url;
    el.video.controls = true;
    el.video.loop = false;
    el.video.muted = false;
    el.video.play().catch(() => {});

    const restore = () => {
      URL.revokeObjectURL(url);
      if (state.captureActive && state.stream) {
        setLivePreview();
        setStatus('Test complete. Live preview restored.', 'ready');
        updateCaptureMeta();
      } else {
        setStatus('Test complete.', 'ready');
      }
      updateButtons();
    };

    el.video.addEventListener('ended', restore, { once: true });
  };

  const recordTestClip = () => {
    if (!state.captureActive || state.recording || state.exporting || state.testing || !state.stream) return;
    state.testing = true;
    updateButtons();
    setStatus('Recording test clip...', 'pending');

    const mimeType = getSelectedMimeType();
    let recorderInfo;
    try {
      recorderInfo = createRecorder(state.stream, mimeType);
    } catch (_) {
      state.testing = false;
      updateButtons();
      setStatus('Test recording failed to initialize.', 'error');
      return;
    }

    const testRecorder = recorderInfo.recorder;
    const chunks = [];

    testRecorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    });

    const stopTimerId = setTimeout(() => {
      if (testRecorder.state !== 'inactive') {
        testRecorder.stop();
      }
    }, TEST_DURATION_MS);

    testRecorder.addEventListener('stop', () => {
      clearTimeout(stopTimerId);
      state.testing = false;
      updateButtons();
      const mime = testRecorder.mimeType || recorderInfo.mimeType || 'video/webm';
      if (chunks.length) {
        const blob = new Blob(chunks, { type: mime });
        setStatus('Playing test clip...', 'ready');
        playTestClip(blob);
      } else {
        setStatus('Test recording produced no data.', 'warn');
      }
    }, { once: true });

    try {
      testRecorder.start(200);
    } catch (err) {
      clearTimeout(stopTimerId);
      state.testing = false;
      updateButtons();
      setStatus('Test recording failed to start.', 'error');
    }
  };

  const waitForEvent = (target, event, readyCheck) => new Promise((resolve) => {
    if (readyCheck && readyCheck()) {
      resolve();
      return;
    }
    const handler = () => {
      target.removeEventListener(event, handler);
      resolve();
    };
    target.addEventListener(event, handler);
  });

  const exportClip = async () => {
    if (!state.recordedUrl || !supportsCanvasCapture || state.exporting) return;
    if ((state.trimEnd - state.trimStart) < MIN_TRIM_GAP) {
      setStatus('Trim range is too short.', 'warn');
      return;
    }

    state.exporting = true;
    updateButtons();
    clearDownload();
    setStatus('Exporting clip...', 'pending');

    const exportVideo = document.createElement('video');
    exportVideo.src = state.recordedUrl;
    exportVideo.muted = !state.recordedHasAudio;
    exportVideo.playsInline = true;
    exportVideo.preload = 'auto';

    let didCleanup = false;
    let cleanup = () => {};

    try {
      await waitForEvent(exportVideo, 'loadedmetadata', () => exportVideo.readyState >= 1);
      const duration = Number.isFinite(exportVideo.duration) ? exportVideo.duration : state.recordedDuration;
      const start = clamp(state.trimStart, 0, duration);
      const end = clamp(state.trimEnd, 0, duration);

      if (end - start < MIN_TRIM_GAP) {
        setStatus('Trim range is too short.', 'warn');
        state.exporting = false;
        updateButtons();
        return;
      }

      exportVideo.currentTime = start;
      await waitForEvent(exportVideo, 'seeked', () => Math.abs(exportVideo.currentTime - start) < 0.01);

      const sourceWidth = exportVideo.videoWidth || 1;
      const sourceHeight = exportVideo.videoHeight || 1;
      const crop = state.cropRegion;
      const sx = crop ? crop.x : 0;
      const sy = crop ? crop.y : 0;
      const sw = crop ? crop.width : sourceWidth;
      const sh = crop ? crop.height : sourceHeight;

      const canvas = document.createElement('canvas');
      canvas.width = Math.round(sw);
      canvas.height = Math.round(sh);
      const ctx = canvas.getContext('2d', { alpha: false });
      const canvasStream = canvas.captureStream(DEFAULT_EXPORT_FPS);
      const outputStream = new MediaStream();
      const videoTrack = canvasStream.getVideoTracks()[0];
      if (videoTrack) outputStream.addTrack(videoTrack);

      let audioContext = null;
      let audioTracks = [];
      if (state.recordedHasAudio && (window.AudioContext || window.webkitAudioContext)) {
        try {
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          audioContext = new AudioCtx();
          const source = audioContext.createMediaElementSource(exportVideo);
          const destination = audioContext.createMediaStreamDestination();
          source.connect(destination);
          if (audioContext.state === 'suspended') {
            await audioContext.resume();
          }
          audioTracks = destination.stream.getAudioTracks();
          audioTracks.forEach((track) => outputStream.addTrack(track));
        } catch (_) {
          audioTracks = [];
        }
      }

      if (state.recordedHasAudio && !audioTracks.length) {
        setStatus('Exporting clip without audio (browser limitation).', 'warn');
      }

      cleanup = () => {
        if (didCleanup) return;
        didCleanup = true;
        stopTracks(canvasStream);
        audioTracks.forEach((track) => track.stop());
        if (audioContext) {
          audioContext.close().catch(() => {});
        }
      };

      const mimeType = getSelectedMimeType();
      const { recorder, mimeType: recorderMime } = createRecorder(outputStream, mimeType);
      const chunks = [];

      recorder.addEventListener('dataavailable', (event) => {
        if (event.data && event.data.size > 0) {
          chunks.push(event.data);
        }
      });

      const stopPromise = new Promise((resolve) => {
        recorder.addEventListener('stop', () => {
          cleanup();
          resolve();
        }, { once: true });
      });

      recorder.start(200);

      const drawFrame = () => {
        if (exportVideo.paused || exportVideo.ended) return;
        ctx.drawImage(exportVideo, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
        requestAnimationFrame(drawFrame);
      };

      const monitor = () => {
        if (exportVideo.currentTime >= (end - 0.03) || exportVideo.ended) {
          exportVideo.pause();
          recorder.stop();
          return;
        }
        requestAnimationFrame(monitor);
      };

      await exportVideo.play();
      drawFrame();
      monitor();
      await stopPromise;

      if (chunks.length) {
        const mime = recorder.mimeType || recorderMime || 'video/webm';
        const blob = new Blob(chunks, { type: mime });
        setDownload(blob, mime);
        setStatus('Export ready. Download below.', 'ready');
      } else {
        setStatus('Export failed to produce data.', 'error');
      }
    } catch (err) {
      setStatus('Export failed. Try again.', 'error');
    } finally {
      state.exporting = false;
      cleanup();
      updateButtons();
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
    if (!supportsCanvasCapture && el.formatHelp) {
      el.formatHelp.textContent += ' Trim and crop require canvas capture support.';
    }
    updateButtons();
    updateCropMeta();
    initTrimControls();

    el.startCapture?.addEventListener('click', startCapture);
    el.stopCapture?.addEventListener('click', () => stopCapture());
    el.startRecord?.addEventListener('click', startRecording);
    el.pauseRecord?.addEventListener('click', togglePause);
    el.stopRecord?.addEventListener('click', stopRecording);
    el.testRecord?.addEventListener('click', recordTestClip);
    el.selectCrop?.addEventListener('click', startCropSelection);
    el.clearCrop?.addEventListener('click', () => {
      clearCrop();
      setStatus('Crop cleared.', 'ready');
    });
    el.exportBtn?.addEventListener('click', exportClip);
    el.download?.addEventListener('click', (event) => {
      if (!state.downloadUrl) {
        event.preventDefault();
        setStatus('Export a clip before downloading.', 'warn');
      }
    });

    el.overlay?.addEventListener('pointerdown', handlePointerDown);
    el.overlay?.addEventListener('pointermove', handlePointerMove);
    el.overlay?.addEventListener('pointerup', handlePointerUp);
    el.overlay?.addEventListener('pointercancel', handlePointerCancel);
    el.overlay?.addEventListener('pointerleave', handlePointerCancel);

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && state.selecting) {
        handlePointerCancel();
        setStatus('Crop selection cancelled.', 'warn');
      }
    });

    el.video?.addEventListener('loadedmetadata', updateCaptureMeta);
  };

  init();
})();
