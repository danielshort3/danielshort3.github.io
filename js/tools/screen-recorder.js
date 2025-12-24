(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);

  const el = {
    formatOptions: $('[data-screenrec="format-options"]'),
    formatHelp: $('[data-screenrec="format-help"]'),
    audioToggle: $('[data-screenrec="audio-toggle"]'),
    audioHelp: $('[data-screenrec="audio-help"]'),
    startCapture: $('[data-screenrec="start-capture"]'),
    stopCapture: $('[data-screenrec="stop-capture"]'),
    video: $('[data-screenrec="video"]'),
    placeholder: $('[data-screenrec="placeholder"]'),
    status: $('[data-screenrec="status"]'),
    captureMeta: $('[data-screenrec="capture-meta"]'),
    timer: $('[data-screenrec="timer"]'),
    startRecord: $('[data-screenrec="start-record"]'),
    pauseRecord: $('[data-screenrec="pause-record"]'),
    stopRecord: $('[data-screenrec="stop-record"]'),
    download: $('[data-screenrec="download"]'),
    downloadNote: $('[data-screenrec="download-note"]')
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
    if (el.download) {
      el.download.hidden = !hasRecording;
    }
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
    if (el.placeholder) el.placeholder.hidden = true;

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
    setStatus('Clip ready to download.', 'ready');
  };

  const startCapture = async () => {
    if (!supportsCapture || !supportsRecorder || state.captureActive || state.recording) return;
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

    clearRecordedClip();
    state.stream = stream;
    state.captureActive = true;
    state.stopReason = '';
    setLivePreview();
    updateCaptureMeta();
    updateButtons();
    resetTimer();

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
    const message = reason || state.stopReason || (state.recordedUrl ? 'Capture stopped. Clip ready to download.' : 'Capture stopped.');
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
    if (!state.captureActive || state.recording || !state.stream) return;
    clearDownload();
    clearRecordedClip();

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
        setRecordedClip(blob, mime);
      } else {
        setStatus('Recording stopped. No data captured.', 'warn');
      }
      stopCapture();
      updateButtons();
    }, { once: true });

    try {
      state.recorder.start(200);
      setStatus('Recording...', 'recording');
      updateButtons();
    } catch (err) {
      state.recording = false;
      stopTimer();
      setStatus('Recording failed to start.', 'error');
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

  const init = () => {
    setFormatOptions();
    updateCaptureMeta();
    updateButtons();

    el.startCapture?.addEventListener('click', startCapture);
    el.stopCapture?.addEventListener('click', () => stopCapture());
    el.startRecord?.addEventListener('click', startRecording);
    el.pauseRecord?.addEventListener('click', togglePause);
    el.stopRecord?.addEventListener('click', stopRecording);
    el.download?.addEventListener('click', (event) => {
      if (!state.downloadUrl) {
        event.preventDefault();
        setStatus('Record a clip before downloading.', 'warn');
      }
    });

    el.video?.addEventListener('loadedmetadata', updateCaptureMeta);
  };

  init();
})();
