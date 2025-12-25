(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const el = {
    formatOptions: $('[data-screenrec="format-options"]'),
    formatHelp: $('[data-screenrec="format-help"]'),
    fpsSelect: $('[data-screenrec="fps-select"]'),
    audioToggle: $('[data-screenrec="audio-toggle"]'),
    audioHelp: $('[data-screenrec="audio-help"]'),
    audioLevel: $('[data-screenrec="audio-level"]'),
    audioLevelValue: $('[data-screenrec="audio-level-value"]'),
    audioLevelHelp: $('[data-screenrec="audio-level-help"]'),
    audioMeter: $('[data-screenrec="audio-meter"]'),
    audioMeterFill: $('[data-screenrec="audio-meter-fill"]'),
    startCapture: $('[data-screenrec="start-capture"]'),
    stopCapture: $('[data-screenrec="stop-capture"]'),
    captureActions: $('[data-screenrec="capture-actions"]'),
    testCapture: $('[data-screenrec="test-capture"]'),
    cropToggle: $('[data-screenrec="crop-toggle"]'),
    cropOverlay: $('[data-screenrec="crop-overlay"]'),
    cropSelection: $('[data-screenrec="crop-selection"]'),
    grid: $('[data-screenrec="grid"]'),
    controlsPanel: $('[data-screenrec="controls-panel"]'),
    previewPanel: $('[data-screenrec="preview-panel"]'),
    stage: $('[data-screenrec="stage"]'),
    video: $('[data-screenrec="video"]'),
    placeholder: $('[data-screenrec="placeholder"]'),
    status: $('[data-screenrec="status"]'),
    captureMeta: $('[data-screenrec="capture-meta"]'),
    timer: $('[data-screenrec="timer"]'),
    startRecord: $('[data-screenrec="start-record"]'),
    pauseRecord: $('[data-screenrec="pause-record"]'),
    stopRecord: $('[data-screenrec="stop-record"]'),
    downloadLinks: $$('[data-screenrec="download"]'),
    downloadNotes: $$('[data-screenrec="download-note"]'),
    downloadPanels: $$('[data-screenrec="download-panel"]')
  };

  if (!el.startCapture || !el.video || !el.startRecord) return;

  const supportsCapture = Boolean(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
  const supportsRecorder = typeof window.MediaRecorder !== 'undefined';

  const state = {
    stream: null,
    recorder: null,
    chunks: [],
    recording: false,
    paused: false,
    captureActive: false,
    timerId: null,
    timerStart: 0,
    timerPausedAt: 0,
    totalPausedMs: 0,
    downloadUrl: null,
    recordedBlob: null,
    recordedUrl: null,
    recordedDuration: 0,
    recordedHasAudio: false,
    recordedMimeType: '',
    stopReason: '',
    recordingStream: null,
    recordingCleanup: null,
    testMode: false,
    testTimerId: null,
    cropRegion: null,
    cropSelecting: false,
    cropStart: null,
    cropCurrent: null,
    cropDragging: false,
    cropDragOffset: null,
    cropAnimationId: null,
    audioContext: null,
    audioMeterContext: null,
    audioMeterAnalyser: null,
    audioMeterSource: null,
    audioMeterData: null,
    audioMeterId: null,
    audioMeterLevel: 0
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
    if (!el.formatOptions) return;
    el.formatOptions.innerHTML = '';

    const supported = getSupportedFormats();
    const options = [
      { label: 'Auto (best supported)', mimeType: 'auto' },
      ...supported
    ];

    options.forEach((opt, index) => {
      const label = document.createElement('label');
      label.className = 'screenrec-radio';

      const input = document.createElement('input');
      input.type = 'radio';
      input.name = 'screenrec-format';
      input.value = opt.mimeType;
      input.checked = index === 0;

      const text = document.createElement('span');
      text.textContent = opt.label || opt.mimeType;

      label.appendChild(input);
      label.appendChild(text);
      el.formatOptions.appendChild(label);
    });

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
    if (!el.formatOptions) return '';
    const selected = el.formatOptions.querySelector('input[name="screenrec-format"]:checked');
    if (!selected || selected.value === 'auto') return getPreferredMimeType();
    return selected.value;
  };

  const setDownload = (blob, mimeType) => {
    if (!el.downloadLinks.length) return;
    if (state.downloadUrl) {
      URL.revokeObjectURL(state.downloadUrl);
    }
    const ext = extensionFromMime(mimeType);
    const name = buildFilename(ext);
    const url = URL.createObjectURL(blob);
    state.downloadUrl = url;
    el.downloadPanels.forEach((panel) => {
      panel.hidden = false;
    });
    el.downloadLinks.forEach((link) => {
      link.href = url;
      link.download = name;
      link.hidden = false;
    });
    el.downloadNotes.forEach((note) => {
      note.textContent = `Ready: ${name} (${formatBytes(blob.size)}).`;
      note.hidden = false;
    });
  };

  const clearDownload = () => {
    if (state.downloadUrl) {
      URL.revokeObjectURL(state.downloadUrl);
    }
    state.downloadUrl = null;
    el.downloadPanels.forEach((panel) => {
      panel.hidden = true;
    });
    el.downloadLinks.forEach((link) => {
      link.hidden = true;
      link.href = '#';
      link.removeAttribute('download');
    });
    el.downloadNotes.forEach((note) => {
      note.hidden = true;
      note.textContent = '';
    });
  };

  const getSelectedFps = () => {
    if (!el.fpsSelect) return 0;
    const value = el.fpsSelect.value;
    if (value === 'auto') return 0;
    const fps = Number(value);
    return Number.isFinite(fps) && fps > 0 ? fps : 0;
  };

  const getAudioGain = () => {
    if (!el.audioLevel) return 1;
    const value = Number(el.audioLevel.value);
    if (!Number.isFinite(value)) return 1;
    return Math.max(0, value) / 100;
  };

  const updateAudioLevelValue = () => {
    if (!el.audioLevel || !el.audioLevelValue) return;
    const value = Number(el.audioLevel.value);
    const safeValue = Number.isFinite(value) ? value : 100;
    el.audioLevelValue.textContent = `${Math.round(safeValue)}%`;
  };

  const setAudioMeterLevel = (value) => {
    if (!el.audioMeter || !el.audioMeterFill) return;
    const clamped = Math.max(0, Math.min(1, value));
    el.audioMeterFill.style.transform = `scaleX(${clamped})`;
    el.audioMeter.setAttribute('aria-valuenow', `${Math.round(clamped * 100)}`);
  };

  const updateAudioMeterState = () => {
    if (!el.audioMeter) return;
    const active = state.captureActive && streamHasAudio(state.stream);
    if (active) {
      el.audioMeter.dataset.active = 'true';
    } else {
      delete el.audioMeter.dataset.active;
      setAudioMeterLevel(0);
    }
  };

  const updatePlaceholder = () => {
    if (!el.placeholder) return;
    const hasMediaSource = Boolean(el.video && (el.video.srcObject || el.video.currentSrc));
    const hasSource = Boolean(state.captureActive || state.recordedUrl || hasMediaSource);
    const showPlaceholder = !hasSource;
    el.placeholder.hidden = !showPlaceholder;
  };

  const hidePlaceholder = () => {
    if (!el.placeholder) return;
    el.placeholder.hidden = true;
  };

  const handleVideoReady = () => {
    if (!el.video) return;
    if (el.video.srcObject || el.video.currentSrc) {
      hidePlaceholder();
    }
  };

  const setView = () => {
    if (el.controlsPanel) el.controlsPanel.hidden = false;
    if (el.previewPanel) el.previewPanel.hidden = false;
    if (el.grid) {
      el.grid.dataset.view = state.captureActive || state.recordedUrl ? 'preview' : 'controls';
    }
    updatePlaceholder();
  };

  const getVideoFrameRect = () => {
    if (!el.stage || !el.video) return null;
    const stageRect = el.stage.getBoundingClientRect();
    const track = state.stream ? state.stream.getVideoTracks()[0] : null;
    const settings = track && track.getSettings ? track.getSettings() : {};
    const videoWidth = el.video.videoWidth || settings.width;
    const videoHeight = el.video.videoHeight || settings.height;
    if (!videoWidth || !videoHeight || !stageRect.width || !stageRect.height) return null;
    const stageRatio = stageRect.width / stageRect.height;
    const videoRatio = videoWidth / videoHeight;
    let displayWidth = stageRect.width;
    let displayHeight = stageRect.height;
    let offsetX = 0;
    let offsetY = 0;
    if (videoRatio > stageRatio) {
      displayWidth = stageRect.width;
      displayHeight = stageRect.width / videoRatio;
      offsetY = (stageRect.height - displayHeight) / 2;
    } else {
      displayHeight = stageRect.height;
      displayWidth = stageRect.height * videoRatio;
      offsetX = (stageRect.width - displayWidth) / 2;
    }
    return {
      stageRect,
      videoWidth,
      videoHeight,
      displayWidth,
      displayHeight,
      offsetX,
      offsetY
    };
  };

  const getCropBoxFromRegion = (frame) => {
    if (!state.cropRegion || !frame) return null;
    return {
      x: frame.offsetX + state.cropRegion.x * frame.displayWidth,
      y: frame.offsetY + state.cropRegion.y * frame.displayHeight,
      width: state.cropRegion.width * frame.displayWidth,
      height: state.cropRegion.height * frame.displayHeight
    };
  };

  const updateCropSelectionBox = (box) => {
    if (!el.cropSelection) return;
    if (!box) {
      el.cropSelection.hidden = true;
      return;
    }
    el.cropSelection.hidden = false;
    el.cropSelection.style.left = `${Math.round(box.x)}px`;
    el.cropSelection.style.top = `${Math.round(box.y)}px`;
    el.cropSelection.style.width = `${Math.round(box.width)}px`;
    el.cropSelection.style.height = `${Math.round(box.height)}px`;
  };

  const syncCropSelection = () => {
    if (!state.cropRegion) {
      updateCropSelectionBox(null);
      return;
    }
    const frame = getVideoFrameRect();
    if (!frame) return;
    updateCropSelectionBox(getCropBoxFromRegion(frame));
  };

  const updateCropOverlay = () => {
    if (!el.cropOverlay) return;
    const showOverlay = state.cropSelecting || state.cropRegion;
    el.cropOverlay.hidden = !showOverlay;
    if (state.cropSelecting) {
      el.cropOverlay.dataset.mode = 'select';
    } else if (state.cropRegion) {
      el.cropOverlay.dataset.mode = 'drag';
    } else {
      delete el.cropOverlay.dataset.mode;
    }
    if (!showOverlay) {
      updateCropSelectionBox(null);
      return;
    }
    if (!state.cropSelecting) {
      syncCropSelection();
    }
  };

  const clearCrop = () => {
    state.cropRegion = null;
    state.cropSelecting = false;
    state.cropStart = null;
    state.cropCurrent = null;
    state.cropDragging = false;
    state.cropDragOffset = null;
    updateCropOverlay();
    updateButtons();
  };

  const stopAudioMeter = () => {
    if (state.audioMeterId) {
      cancelAnimationFrame(state.audioMeterId);
      state.audioMeterId = null;
    }
    if (state.audioMeterSource) {
      try {
        state.audioMeterSource.disconnect();
      } catch (_) {}
    }
    if (state.audioMeterAnalyser) {
      try {
        state.audioMeterAnalyser.disconnect();
      } catch (_) {}
    }
    if (state.audioMeterContext) {
      state.audioMeterContext.close().catch(() => {});
    }
    state.audioMeterContext = null;
    state.audioMeterAnalyser = null;
    state.audioMeterSource = null;
    state.audioMeterData = null;
    state.audioMeterLevel = 0;
    setAudioMeterLevel(0);
  };

  const startAudioMeter = () => {
    stopAudioMeter();
    updateAudioMeterState();
    if (!state.stream || !streamHasAudio(state.stream)) return;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    let audioContext;
    try {
      audioContext = new AudioCtx();
    } catch (_) {
      return;
    }
    let source;
    let analyser;
    try {
      source = audioContext.createMediaStreamSource(state.stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.85;
      source.connect(analyser);
    } catch (_) {
      audioContext.close().catch(() => {});
      return;
    }

    state.audioMeterContext = audioContext;
    state.audioMeterSource = source;
    state.audioMeterAnalyser = analyser;
    state.audioMeterData = new Uint8Array(analyser.fftSize);
    state.audioMeterLevel = 0;

    const tick = () => {
      if (!state.captureActive || !state.stream || !streamHasAudio(state.stream)) {
        stopAudioMeter();
        updateAudioMeterState();
        return;
      }
      analyser.getByteTimeDomainData(state.audioMeterData);
      let sum = 0;
      for (let i = 0; i < state.audioMeterData.length; i++) {
        const value = (state.audioMeterData[i] - 128) / 128;
        sum += value * value;
      }
      const rms = Math.sqrt(sum / state.audioMeterData.length);
      const normalized = Math.min(1, rms * 2.6);
      state.audioMeterLevel = state.audioMeterLevel * 0.75 + normalized * 0.25;
      setAudioMeterLevel(state.audioMeterLevel);
      state.audioMeterId = requestAnimationFrame(tick);
    };
    state.audioMeterId = requestAnimationFrame(tick);
  };

  const clampPointToFrame = (point, frame) => ({
    x: Math.min(Math.max(point.x, frame.offsetX), frame.offsetX + frame.displayWidth),
    y: Math.min(Math.max(point.y, frame.offsetY), frame.offsetY + frame.displayHeight)
  });

  const isPointInFrame = (point, frame) => (
    point.x >= frame.offsetX &&
    point.x <= frame.offsetX + frame.displayWidth &&
    point.y >= frame.offsetY &&
    point.y <= frame.offsetY + frame.displayHeight
  );

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

  const updateButtons = () => {
    const hasRecording = Boolean(state.recordedUrl);
    if (el.startCapture) {
      el.startCapture.disabled = !supportsCapture || !supportsRecorder || state.captureActive || state.recording;
    }
    if (el.stopCapture) {
      el.stopCapture.disabled = !state.captureActive || state.recording;
    }
    if (el.startRecord) {
      el.startRecord.disabled = !state.captureActive || state.recording;
    }
    if (el.pauseRecord) {
      el.pauseRecord.disabled = !state.recording;
      el.pauseRecord.textContent = state.paused ? 'Resume' : 'Pause';
    }
    if (el.stopRecord) {
      el.stopRecord.disabled = !state.recording;
    }
    if (el.audioToggle) {
      el.audioToggle.disabled = !supportsCapture || state.captureActive || state.recording;
    }
    if (el.fpsSelect) {
      el.fpsSelect.disabled = !supportsCapture || state.captureActive || state.recording;
    }
    if (el.audioLevel) {
      const audioLocked = !supportsCapture || state.captureActive || state.recording || !el.audioToggle?.checked;
      el.audioLevel.disabled = audioLocked;
    }
    if (el.testCapture) {
      el.testCapture.disabled = !supportsCapture || !supportsRecorder || state.captureActive || state.recording;
    }
    if (el.cropToggle) {
      el.cropToggle.disabled = !state.captureActive || state.recording;
      const label = state.cropSelecting || state.cropRegion ? 'Cancel crop' : 'Crop';
      el.cropToggle.textContent = label;
    }
    if (el.downloadLinks.length) {
      const hasDownload = Boolean(state.downloadUrl);
      el.downloadLinks.forEach((link) => {
        link.hidden = !hasRecording || !hasDownload;
      });
    }
    updateAudioMeterState();
  };

  const startTimer = () => {
    if (state.timerId) return;
    state.timerStart = performance.now();
    state.totalPausedMs = 0;
    state.timerPausedAt = 0;
    state.timerId = setInterval(() => {
      const now = performance.now();
      const elapsed = now - state.timerStart - state.totalPausedMs;
      if (el.timer) {
        el.timer.textContent = formatDuration(elapsed);
      }
    }, 250);
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

  const clearRecordedClip = () => {
    if (state.recordedUrl) {
      URL.revokeObjectURL(state.recordedUrl);
    }
    state.recordedUrl = null;
    state.recordedBlob = null;
    state.recordedDuration = 0;
    state.recordedHasAudio = false;
    state.recordedMimeType = '';
    clearDownload();
    resetTimer();
    updatePlaceholder();
  };

  const setLivePreview = () => {
    if (!state.stream) return;
    el.video.src = '';
    el.video.srcObject = state.stream;
    el.video.controls = false;
    el.video.loop = false;
    el.video.muted = true;
    el.video.play().catch(() => {});
    updatePlaceholder();
  };

  const setRecordedClip = (blob, mimeType) => {
    if (!blob) return;
    if (state.recordedUrl) {
      URL.revokeObjectURL(state.recordedUrl);
    }
    state.recordedBlob = blob;
    state.recordedMimeType = mimeType || blob.type || '';
    state.recordedUrl = URL.createObjectURL(blob);
    el.video.srcObject = null;
    el.video.src = state.recordedUrl;
    el.video.controls = true;
    el.video.loop = false;
    el.video.muted = false;
    el.video.play().catch(() => {});
    updatePlaceholder();

    const handleMetadata = () => {
      state.recordedDuration = Number.isFinite(el.video.duration) ? el.video.duration : 0;
      resetTimer(state.recordedDuration * 1000);
      updateButtons();
    };
    if (el.video.readyState >= 1) {
      handleMetadata();
    } else {
      el.video.addEventListener('loadedmetadata', handleMetadata, { once: true });
    }

    setDownload(blob, state.recordedMimeType);
    setStatus('Clip ready to download. Stop capture when you are done.', 'ready');
  };

  const startCapture = async () => {
    if (!supportsCapture || !supportsRecorder || state.captureActive || state.recording) return false;
    setStatus('Requesting capture permission...', 'pending');

    const includeAudio = Boolean(el.audioToggle?.checked);
    const fps = getSelectedFps();
    const videoConstraints = fps ? { frameRate: { ideal: fps } } : true;
    const constraints = { video: videoConstraints, audio: includeAudio };
    let stream = null;
    let audioFallback = false;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia(constraints);
    } catch (_) {
      if (includeAudio) {
        try {
          stream = await navigator.mediaDevices.getDisplayMedia({ video: videoConstraints, audio: false });
          audioFallback = true;
        } catch (err) {
          setStatus('Capture request failed. Check permissions.', 'error');
          return false;
        }
      } else {
        setStatus('Capture request failed. Check permissions.', 'error');
        return false;
      }
    }

    if (!stream) {
      setStatus('Capture did not start.', 'error');
      return false;
    }

    if (!includeAudio) {
      stream.getAudioTracks().forEach((track) => {
        stream.removeTrack(track);
        track.stop();
      });
    }

    clearRecordedClip();
    state.stream = stream;
    state.captureActive = true;
    state.stopReason = '';
    setLivePreview();
    setView();
    updateCaptureMeta();
    updateButtons();
    resetTimer();
    startAudioMeter();

    const track = stream.getVideoTracks()[0];
    track.addEventListener('ended', () => {
      if (state.recording) {
        state.stopReason = 'Capture ended by the browser.';
        stopRecording();
      } else {
        stopCapture('Capture ended by the browser.');
      }
    }, { once: true });

    if (audioFallback) {
      setStatus('Audio capture is not available. Capturing without audio.', 'warn');
    } else {
      setStatus('Capture ready. Start recording when you are ready.', 'ready');
    }
    updateCropOverlay();
    return true;
  };

  const stopCapture = (reason) => {
    if (!state.captureActive) return;
    stopTracks(state.stream);
    state.stream = null;
    state.captureActive = false;
    if (state.testTimerId) {
      clearTimeout(state.testTimerId);
      state.testTimerId = null;
    }
    state.testMode = false;
    cleanupRecordingPipeline();
    updateCaptureMeta();
    updateButtons();
    setView();
    if (!state.recordedUrl) {
      el.video.srcObject = null;
    }
    resetTimer(state.recordedDuration ? state.recordedDuration * 1000 : 0);
    const message = reason || state.stopReason || (state.recordedUrl ? 'Capture stopped. Clip ready to download.' : 'Capture stopped.');
    setStatus(message, state.recordedUrl ? 'ready' : 'idle');
    state.stopReason = '';
    clearCrop();
    updatePlaceholder();
    stopAudioMeter();
  };

  const startTestCapture = async () => {
    if (state.captureActive || state.recording) return;
    state.testMode = true;
    const started = await startCapture();
    if (!started) {
      state.testMode = false;
      return;
    }
    startRecording();
    if (state.recording) {
      state.testTimerId = setTimeout(() => {
        stopRecording();
      }, 5000);
    } else {
      state.testMode = false;
    }
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

  const getCropPixels = (videoWidth, videoHeight) => {
    if (!state.cropRegion) {
      return { sx: 0, sy: 0, sw: videoWidth, sh: videoHeight };
    }
    const sx = Math.max(0, Math.round(state.cropRegion.x * videoWidth));
    const sy = Math.max(0, Math.round(state.cropRegion.y * videoHeight));
    const sw = Math.max(1, Math.round(state.cropRegion.width * videoWidth));
    const sh = Math.max(1, Math.round(state.cropRegion.height * videoHeight));
    return {
      sx,
      sy,
      sw: Math.min(sw, videoWidth - sx),
      sh: Math.min(sh, videoHeight - sy)
    };
  };

  const createCroppedStream = (fps) => {
    if (!state.cropRegion || !el.video) return null;
    if (!('captureStream' in HTMLCanvasElement.prototype)) return null;
    const track = state.stream ? state.stream.getVideoTracks()[0] : null;
    const settings = track && track.getSettings ? track.getSettings() : {};
    const videoWidth = el.video.videoWidth || settings.width;
    const videoHeight = el.video.videoHeight || settings.height;
    if (!videoWidth || !videoHeight) return null;
    const crop = getCropPixels(videoWidth, videoHeight);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    canvas.width = crop.sw;
    canvas.height = crop.sh;

    const drawFrame = () => {
      if (!state.captureActive || !state.stream) {
        state.cropAnimationId = null;
        return;
      }
      if (el.video.readyState < 2) {
        state.cropAnimationId = requestAnimationFrame(drawFrame);
        return;
      }
      ctx.drawImage(
        el.video,
        crop.sx,
        crop.sy,
        crop.sw,
        crop.sh,
        0,
        0,
        canvas.width,
        canvas.height
      );
      state.cropAnimationId = requestAnimationFrame(drawFrame);
    };

    drawFrame();
    const stream = fps ? canvas.captureStream(fps) : canvas.captureStream();
    const cleanup = () => {
      if (state.cropAnimationId) {
        cancelAnimationFrame(state.cropAnimationId);
        state.cropAnimationId = null;
      }
      stopTracks(stream);
    };
    return { stream, cleanup };
  };

  const createAdjustedAudioTrack = (stream, gainValue) => {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    try {
      const audioContext = new AudioCtx();
      const source = audioContext.createMediaStreamSource(stream);
      const gainNode = audioContext.createGain();
      gainNode.gain.value = gainValue;
      const destination = audioContext.createMediaStreamDestination();
      source.connect(gainNode);
      gainNode.connect(destination);
      state.audioContext = audioContext;
      const track = destination.stream.getAudioTracks()[0];
      const cleanup = () => {
        try {
          track.stop();
        } catch (_) {}
        source.disconnect();
        gainNode.disconnect();
        destination.disconnect();
        audioContext.close().catch(() => {});
        state.audioContext = null;
      };
      return { track, cleanup };
    } catch (_) {
      return null;
    }
  };

  const buildRecordingStream = () => {
    if (!state.stream) return null;
    const outputStream = new MediaStream();
    const cleanupTasks = [];
    const fps = getSelectedFps();
    const sourceVideoTrack = state.stream.getVideoTracks()[0];
    let videoTrack = sourceVideoTrack;

    if (state.cropRegion) {
      const cropped = createCroppedStream(fps);
      if (cropped && cropped.stream) {
        videoTrack = cropped.stream.getVideoTracks()[0];
        cleanupTasks.push(cropped.cleanup);
      } else {
        setStatus('Crop preview unavailable. Recording full frame.', 'warn');
      }
    }

    if (videoTrack) {
      outputStream.addTrack(videoTrack);
    }

    if (streamHasAudio(state.stream)) {
      const gainValue = getAudioGain();
      if (gainValue !== 1) {
        const adjusted = createAdjustedAudioTrack(state.stream, gainValue);
        if (adjusted && adjusted.track) {
          outputStream.addTrack(adjusted.track);
          cleanupTasks.push(adjusted.cleanup);
        } else {
          outputStream.addTrack(state.stream.getAudioTracks()[0]);
        }
      } else {
        outputStream.addTrack(state.stream.getAudioTracks()[0]);
      }
    }

    const cleanup = () => {
      cleanupTasks.forEach((fn) => fn());
    };
    return { stream: outputStream, cleanup };
  };

  const cleanupRecordingPipeline = () => {
    if (state.recordingCleanup) {
      state.recordingCleanup();
    }
    state.recordingCleanup = null;
    state.recordingStream = null;
  };

  const startRecording = () => {
    if (!state.captureActive || state.recording || !state.stream) return;
    clearDownload();
    clearRecordedClip();

    setLivePreview();

    const recordingStreamInfo = buildRecordingStream();
    if (!recordingStreamInfo || !recordingStreamInfo.stream) {
      setStatus('Recording failed to initialize in this browser.', 'error');
      if (state.testMode) {
        state.testMode = false;
      }
      return;
    }
    state.recordingStream = recordingStreamInfo.stream;
    state.recordingCleanup = recordingStreamInfo.cleanup;

    const mimeType = getSelectedMimeType();
    let recorderInfo;
    try {
      recorderInfo = createRecorder(recordingStreamInfo.stream, mimeType);
    } catch (_) {
      setStatus('Recording failed to initialize in this browser.', 'error');
      cleanupRecordingPipeline();
      if (state.testMode) {
        state.testMode = false;
      }
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
        state.recordedHasAudio = streamHasAudio(recordingStreamInfo.stream);
        setRecordedClip(blob, mime);
      } else {
        setStatus('Recording stopped. No data captured.', 'warn');
      }
      cleanupRecordingPipeline();
      if (state.testTimerId) {
        clearTimeout(state.testTimerId);
        state.testTimerId = null;
      }
      if (state.testMode) {
        state.testMode = false;
        stopCapture('Test capture complete. Clip ready to download.');
      }
      updateButtons();
    }, { once: true });

    try {
      state.recorder.start(200);
      if (state.testMode) {
        setStatus('Recording 5-second test...', 'recording');
      } else {
        setStatus('Recording...', 'recording');
      }
      updateButtons();
    } catch (err) {
      state.recording = false;
      stopTimer();
      setStatus('Recording failed to start.', 'error');
      cleanupRecordingPipeline();
      if (state.testMode) {
        state.testMode = false;
      }
      updateButtons();
    }
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

  const toggleCropSelection = () => {
    if (!state.captureActive || state.recording) return;
    if (state.cropSelecting || state.cropRegion) {
      clearCrop();
      return;
    }
    state.cropSelecting = true;
    state.cropStart = null;
    state.cropCurrent = null;
    updateCropOverlay();
    updateButtons();
  };

  const handleCropPointerDown = (event) => {
    if (!el.cropOverlay) return;
    if (event.button && event.button !== 0) return;
    const frame = getVideoFrameRect();
    if (!frame) {
      setStatus('Preview not ready for cropping.', 'warn');
      return;
    }
    const point = {
      x: event.clientX - frame.stageRect.left,
      y: event.clientY - frame.stageRect.top
    };
    if (state.cropSelecting) {
      if (!isPointInFrame(point, frame)) return;
      const clamped = clampPointToFrame(point, frame);
      state.cropStart = clamped;
      state.cropCurrent = clamped;
      updateCropSelectionBox({ x: clamped.x, y: clamped.y, width: 0, height: 0 });
      el.cropOverlay.setPointerCapture(event.pointerId);
      return;
    }

    if (!state.cropRegion) return;
    const box = getCropBoxFromRegion(frame);
    if (!box) return;
    const inside = point.x >= box.x && point.x <= box.x + box.width &&
      point.y >= box.y && point.y <= box.y + box.height;
    if (!inside) return;
    state.cropDragging = true;
    state.cropDragOffset = {
      x: point.x - box.x,
      y: point.y - box.y
    };
    el.cropOverlay.setPointerCapture(event.pointerId);
  };

  const handleCropPointerMove = (event) => {
    if (!state.cropStart && !state.cropDragging) return;
    const frame = getVideoFrameRect();
    if (!frame) return;
    const point = {
      x: event.clientX - frame.stageRect.left,
      y: event.clientY - frame.stageRect.top
    };
    if (state.cropStart) {
      const clamped = clampPointToFrame(point, frame);
      state.cropCurrent = clamped;
      const box = {
        x: Math.min(state.cropStart.x, clamped.x),
        y: Math.min(state.cropStart.y, clamped.y),
        width: Math.abs(clamped.x - state.cropStart.x),
        height: Math.abs(clamped.y - state.cropStart.y)
      };
      updateCropSelectionBox(box);
      return;
    }

    if (!state.cropRegion || !state.cropDragOffset) return;
    const box = getCropBoxFromRegion(frame);
    if (!box) return;
    const desiredX = point.x - state.cropDragOffset.x;
    const desiredY = point.y - state.cropDragOffset.y;
    const maxX = Math.max(frame.offsetX, frame.offsetX + frame.displayWidth - box.width);
    const maxY = Math.max(frame.offsetY, frame.offsetY + frame.displayHeight - box.height);
    const nextX = Math.min(Math.max(desiredX, frame.offsetX), maxX);
    const nextY = Math.min(Math.max(desiredY, frame.offsetY), maxY);
    state.cropRegion = {
      x: (nextX - frame.offsetX) / frame.displayWidth,
      y: (nextY - frame.offsetY) / frame.displayHeight,
      width: box.width / frame.displayWidth,
      height: box.height / frame.displayHeight
    };
    updateCropSelectionBox({ x: nextX, y: nextY, width: box.width, height: box.height });
  };

  const handleCropPointerUp = (event) => {
    if (!state.cropStart && !state.cropDragging) return;
    const frame = getVideoFrameRect();
    if (!frame) {
      state.cropStart = null;
      state.cropCurrent = null;
      state.cropDragging = false;
      state.cropDragOffset = null;
      return;
    }
    if (state.cropStart) {
      const point = {
        x: event.clientX - frame.stageRect.left,
        y: event.clientY - frame.stageRect.top
      };
      const clamped = clampPointToFrame(point, frame);
      const box = {
        x: Math.min(state.cropStart.x, clamped.x),
        y: Math.min(state.cropStart.y, clamped.y),
        width: Math.abs(clamped.x - state.cropStart.x),
        height: Math.abs(clamped.y - state.cropStart.y)
      };
      const minSize = 24;
      if (box.width >= minSize && box.height >= minSize) {
        state.cropRegion = {
          x: (box.x - frame.offsetX) / frame.displayWidth,
          y: (box.y - frame.offsetY) / frame.displayHeight,
          width: box.width / frame.displayWidth,
          height: box.height / frame.displayHeight
        };
      }
      state.cropSelecting = false;
      state.cropStart = null;
      state.cropCurrent = null;
    } else if (state.cropDragging) {
      state.cropDragging = false;
      state.cropDragOffset = null;
    }
    updateCropOverlay();
    updateButtons();
    if (el.cropOverlay && el.cropOverlay.hasPointerCapture(event.pointerId)) {
      el.cropOverlay.releasePointerCapture(event.pointerId);
    }
  };

  const init = () => {
    setFormatOptions();
    updateAudioLevelValue();
    updateCaptureMeta();
    updateButtons();
    setView();

    el.startCapture?.addEventListener('click', startCapture);
    el.stopCapture?.addEventListener('click', () => stopCapture());
    el.startRecord?.addEventListener('click', startRecording);
    el.pauseRecord?.addEventListener('click', togglePause);
    el.stopRecord?.addEventListener('click', stopRecording);
    el.testCapture?.addEventListener('click', startTestCapture);
    el.cropToggle?.addEventListener('click', toggleCropSelection);
    el.audioToggle?.addEventListener('change', updateButtons);
    el.audioLevel?.addEventListener('input', updateAudioLevelValue);
    el.fpsSelect?.addEventListener('change', updateButtons);
    el.cropOverlay?.addEventListener('pointerdown', handleCropPointerDown);
    el.cropOverlay?.addEventListener('pointermove', handleCropPointerMove);
    el.cropOverlay?.addEventListener('pointerup', handleCropPointerUp);
    el.cropOverlay?.addEventListener('pointercancel', handleCropPointerUp);
    window.addEventListener('resize', syncCropSelection);
    el.downloadLinks.forEach((link) => {
      link.addEventListener('click', (event) => {
        if (!state.downloadUrl) {
          event.preventDefault();
          setStatus('Record a clip before downloading.', 'warn');
        }
      });
    });

    el.video?.addEventListener('loadedmetadata', () => {
      updateCaptureMeta();
      syncCropSelection();
      handleVideoReady();
    });
    el.video?.addEventListener('loadeddata', handleVideoReady);
    el.video?.addEventListener('canplay', handleVideoReady);
    el.video?.addEventListener('playing', handleVideoReady);
    el.video?.addEventListener('emptied', updatePlaceholder);
  };

  init();
})();
