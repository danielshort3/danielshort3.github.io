(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const el = {
    formatOptions: $('[data-screenrec="format-options"]'),
    formatHelp: $('[data-screenrec="format-help"]'),
    fpsSelect: $('[data-screenrec="fps-select"]'),
    qualitySelect: $('[data-screenrec="quality-select"]'),
    audioToggle: $('[data-screenrec="audio-toggle"]'),
    audioHelp: $('[data-screenrec="audio-help"]'),
    audioLevel: $('[data-screenrec="audio-level"]'),
    audioLevelValue: $('[data-screenrec="audio-level-value"]'),
    audioLevelHelp: $('[data-screenrec="audio-level-help"]'),
    audioMeter: $('[data-screenrec="audio-meter"]'),
    audioMeterFill: $('[data-screenrec="audio-meter-fill"]'),
    micLevel: $('[data-screenrec="mic-level"]'),
    micLevelValue: $('[data-screenrec="mic-level-value"]'),
    micToggle: $('[data-screenrec="mic-toggle"]'),
    micSelect: $('[data-screenrec="mic-select"]'),
    micHelp: $('[data-screenrec="mic-help"]'),
    audioStatus: $('[data-screenrec="audio-status"]'),
    startCapture: $('[data-screenrec="start-capture"]'),
    stopCapture: $('[data-screenrec="stop-capture"]'),
    captureActions: $('[data-screenrec="capture-actions"]'),
    testCapture: $('[data-screenrec="test-capture"]'),
    cropToggle: $('[data-screenrec="crop-toggle"]'),
    cropLabel: $('[data-screenrec="crop-label"]'),
    cropPresetsToggle: $('[data-screenrec="crop-presets-toggle"]'),
    cropPresets: $('[data-screenrec="crop-presets"]'),
    cropPresetsLabel: $('[data-screenrec="crop-presets-label"]'),
    cropPresetButtons: $$('[data-screenrec="crop-preset"]'),
    controlsToggle: $('[data-screenrec="controls-toggle"]'),
    controlsBody: $('[data-screenrec="controls-body"]'),
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
    downloadAll: $('[data-screenrec="download-all"]'),
    downloadNote: $('[data-screenrec="download-note"]'),
    downloadPanel: $('[data-screenrec="download-panel"]')
  };

  if (!el.startCapture || !el.video || !el.startRecord) return;

  const supportsCapture = Boolean(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia);
  const supportsRecorder = typeof window.MediaRecorder !== 'undefined';
  const supportsMicrophone = Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  const state = {
    stream: null,
    recorders: [],
    primaryRecorder: null,
    recording: false,
    paused: false,
    captureActive: false,
    timerId: null,
    timerStart: 0,
    timerPausedAt: 0,
    totalPausedMs: 0,
    downloadUrls: [],
    downloadFiles: [],
    downloadZipUrl: null,
    downloadZipName: '',
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
    cropResizing: false,
    cropResizeHandle: null,
    cropResizeStart: null,
    cropAnimationId: null,
    cropAspectValue: 'free',
    cropAspectLabel: 'Free',
    cropPresetsOpen: false,
    audioContext: null,
    audioMeterContext: null,
    audioMeterAnalyser: null,
    audioMeterSources: [],
    audioMeterData: null,
    audioMeterId: null,
    audioMeterLevel: 0,
    micStream: null,
    micDeviceId: ''
  };

  const MP4_AAC_MIME = 'video/mp4;codecs=avc1.42E01E,mp4a.40.2';
  const MP4_H264_MIME = 'video/mp4;codecs=avc1.42E01E';
  const MP4_BASE_MIME = 'video/mp4';

  const MIME_TYPE_OPTIONS = [
    { label: 'MP4 (H.264 + AAC)', mimeType: MP4_AAC_MIME },
    { label: 'MP4 (H.264)', mimeType: MP4_H264_MIME },
    { label: 'MP4 (default)', mimeType: MP4_BASE_MIME },
    { label: 'WebM (VP9)', mimeType: 'video/webm;codecs=vp9' },
    { label: 'WebM (VP8)', mimeType: 'video/webm;codecs=vp8' },
    { label: 'WebM (default)', mimeType: 'video/webm' }
  ];

  const PREFERRED_MIME_TYPES = [
    MP4_AAC_MIME,
    MP4_H264_MIME,
    MP4_BASE_MIME,
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm'
  ];

  const QUALITY_PRESETS = {
    auto: { label: 'Auto', video: 0, audio: 0 },
    tiny: { label: 'Tiny preview', video: 800000, audio: 64000 },
    low: { label: 'Small', video: 1500000, audio: 80000 },
    medium: { label: 'Balanced', video: 4000000, audio: 128000 },
    high: { label: 'High', video: 8000000, audio: 192000 }
  };

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

  const buildFilename = (ext, suffix, stampOverride) => {
    const stamp = stampOverride || new Date().toISOString().replace(/[:.]/g, '-');
    const safeSuffix = suffix ? `-${suffix}` : '';
    return `screen-recording-${stamp}${safeSuffix}.${ext}`;
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

    options.forEach((opt) => {
      const label = document.createElement('label');
      label.className = 'screenrec-radio';

      const input = document.createElement('input');
      input.type = 'checkbox';
      input.name = 'screenrec-format';
      input.value = opt.mimeType;
      if (opt.mimeType === 'auto') {
        input.dataset.auto = 'true';
      }

      const text = document.createElement('span');
      text.textContent = opt.label || opt.mimeType;

      label.appendChild(input);
      label.appendChild(text);
      el.formatOptions.appendChild(label);
    });

    if (el.formatHelp) {
      if (!supported.length) {
        el.formatHelp.textContent = 'No explicit formats detected. Auto will use the browser default.';
      } else {
        const labels = supported.map((opt) => opt.label).join(', ');
        const mp4Supported = supported.some((opt) => opt.mimeType.includes('mp4'));
        const mp4AacSupported = supported.some((opt) => opt.mimeType === MP4_AAC_MIME);
        let mp4Note = 'MP4 recording is not supported here.';
        if (mp4Supported) {
          mp4Note = mp4AacSupported
            ? 'MP4 with AAC audio is supported in this browser.'
            : 'MP4 audio may be limited; MP4 exports may be video-only for compatibility.';
        }
        el.formatHelp.textContent = `Select one or more formats. Supported: ${labels}. ${mp4Note}`;
      }
    }

    const formatInputs = Array.from(el.formatOptions.querySelectorAll('input[name="screenrec-format"]'));
    const autoInput = formatInputs.find((input) => input.dataset.auto === 'true');
    const mp4DefaultInput = formatInputs.find((input) => input.value === MP4_BASE_MIME);
    const webmDefaultInput = formatInputs.find((input) => input.value === 'video/webm');
    const mp4FallbackInput = formatInputs.find((input) => input.dataset.auto !== 'true' && input.value.includes('mp4'));
    const defaultInputs = [
      mp4DefaultInput || mp4FallbackInput,
      webmDefaultInput
    ].filter(Boolean);

    if (defaultInputs.length) {
      defaultInputs.forEach((input) => {
        input.checked = true;
      });
      if (autoInput) autoInput.checked = false;
    } else if (autoInput) {
      autoInput.checked = true;
    }
    formatInputs.forEach((input) => {
      input.addEventListener('change', () => {
        if (input.dataset.auto === 'true' && input.checked) {
          formatInputs.forEach((other) => {
            if (other !== input) other.checked = false;
          });
          return;
        }
        if (input.dataset.auto !== 'true' && input.checked && autoInput) {
          autoInput.checked = false;
        }
        const anyExplicitChecked = formatInputs.some((item) => item.dataset.auto !== 'true' && item.checked);
        if (!anyExplicitChecked && autoInput) {
          autoInput.checked = true;
        }
      });
    });
  };

  const getSelectedMimeTypes = () => {
    if (!el.formatOptions) return [];
    const inputs = Array.from(el.formatOptions.querySelectorAll('input[name="screenrec-format"]'));
    const explicitInputs = inputs.filter((input) => input.dataset.auto !== 'true');
    const selectedExplicit = explicitInputs.filter((input) => input.checked).map((input) => input.value);
    if (selectedExplicit.length) {
      return Array.from(new Set(selectedExplicit));
    }
    const autoInput = inputs.find((input) => input.dataset.auto === 'true');
    if (autoInput?.checked || !selectedExplicit.length) {
      const preferred = getPreferredMimeType();
      return preferred ? [preferred] : [''];
    }
    return [''];
  };

  const slugify = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');

  const formatLabelFromMime = (mimeType) => {
    if (!mimeType) return 'Default';
    const match = MIME_TYPE_OPTIONS.find((opt) => opt.mimeType === mimeType);
    if (match) return match.label;
    if (mimeType.includes('mp4')) return 'MP4';
    if (mimeType.includes('webm')) return 'WebM';
    return 'Video';
  };

  const isMp4Mime = (mimeType) => Boolean(mimeType && mimeType.includes('mp4'));

  const supportsMp4Aac = () => (
    supportsRecorder &&
    typeof MediaRecorder !== 'undefined' &&
    typeof MediaRecorder.isTypeSupported === 'function' &&
    MediaRecorder.isTypeSupported(MP4_AAC_MIME)
  );

  const resolveMp4Recording = (mimeType, hasAudio) => {
    if (!isMp4Mime(mimeType)) {
      return { mimeType, stripAudio: false };
    }
    if (!hasAudio) {
      return { mimeType, stripAudio: false };
    }
    if (supportsMp4Aac()) {
      return { mimeType: MP4_AAC_MIME, stripAudio: false };
    }
    return { mimeType, stripAudio: true };
  };

  const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let j = 0; j < 8; j++) {
        c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
      }
      table[i] = c;
    }
    return table;
  })();

  const crc32 = (data) => {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < data.length; i++) {
      crc = CRC_TABLE[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  };

  const getDosTimestamp = () => {
    const now = new Date();
    const year = Math.max(1980, now.getFullYear());
    const month = now.getMonth() + 1;
    const day = now.getDate();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const seconds = Math.floor(now.getSeconds() / 2);
    return {
      dosDate: ((year - 1980) << 9) | (month << 5) | day,
      dosTime: (hours << 11) | (minutes << 5) | seconds
    };
  };

  const buildZipBlob = async (files) => {
    const parts = [];
    const centralParts = [];
    const encoder = new TextEncoder();
    const { dosDate, dosTime } = getDosTimestamp();
    const fileCount = Math.min(files.length, 65535);
    let offset = 0;
    let centralSize = 0;
    const zipFlags = 0x0800;
    const zipVersion = 20;

    for (let i = 0; i < fileCount; i++) {
      const file = files[i];
      const nameBytes = encoder.encode(file.name);
      const data = new Uint8Array(await file.blob.arrayBuffer());
      const crc = crc32(data);
      const localHeader = new Uint8Array(30 + nameBytes.length);
      const view = new DataView(localHeader.buffer);
      view.setUint32(0, 0x04034b50, true);
      view.setUint16(4, zipVersion, true);
      view.setUint16(6, zipFlags, true);
      view.setUint16(8, 0, true);
      view.setUint16(10, dosTime, true);
      view.setUint16(12, dosDate, true);
      view.setUint32(14, crc, true);
      view.setUint32(18, data.length, true);
      view.setUint32(22, data.length, true);
      view.setUint16(26, nameBytes.length, true);
      view.setUint16(28, 0, true);
      localHeader.set(nameBytes, 30);
      const localOffset = offset;
      parts.push(localHeader, data);
      offset += localHeader.length + data.length;

      const centralHeader = new Uint8Array(46 + nameBytes.length);
      const cview = new DataView(centralHeader.buffer);
      cview.setUint32(0, 0x02014b50, true);
      cview.setUint16(4, zipVersion, true);
      cview.setUint16(6, zipVersion, true);
      cview.setUint16(8, zipFlags, true);
      cview.setUint16(10, 0, true);
      cview.setUint16(12, dosTime, true);
      cview.setUint16(14, dosDate, true);
      cview.setUint32(16, crc, true);
      cview.setUint32(20, data.length, true);
      cview.setUint32(24, data.length, true);
      cview.setUint16(28, nameBytes.length, true);
      cview.setUint16(30, 0, true);
      cview.setUint16(32, 0, true);
      cview.setUint16(34, 0, true);
      cview.setUint16(36, 0, true);
      cview.setUint32(38, 0, true);
      cview.setUint32(42, localOffset, true);
      centralHeader.set(nameBytes, 46);
      centralParts.push(centralHeader);
      centralSize += centralHeader.length;
    }

    const centralOffset = offset;
    parts.push(...centralParts);

    const endRecord = new Uint8Array(22);
    const endView = new DataView(endRecord.buffer);
    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(4, 0, true);
    endView.setUint16(6, 0, true);
    endView.setUint16(8, fileCount, true);
    endView.setUint16(10, fileCount, true);
    endView.setUint32(12, centralSize, true);
    endView.setUint32(16, centralOffset, true);
    endView.setUint16(20, 0, true);
    parts.push(endRecord);

    return new Blob(parts, { type: 'application/zip' });
  };

  const setDownloads = (files) => {
    if (!el.downloadAll || !el.downloadNote) return;
    clearDownload();
    if (!files.length) return;
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const extCounts = files.reduce((acc, file) => {
      const ext = extensionFromMime(file.mimeType);
      acc[ext] = (acc[ext] || 0) + 1;
      return acc;
    }, {});
    const noteParts = [];
    state.downloadZipName = buildFilename('zip', '', stamp);
    state.downloadFiles = files.map((file) => {
      const ext = extensionFromMime(file.mimeType);
      const label = file.label || formatLabelFromMime(file.mimeType);
      const suffix = extCounts[ext] > 1 ? slugify(label) : '';
      const name = buildFilename(ext, suffix, stamp);
      const url = URL.createObjectURL(file.blob);
      noteParts.push(`${label} ${formatBytes(file.blob.size)}`);
      return { url, name, mimeType: file.mimeType, label, blob: file.blob };
    });
    state.downloadUrls = state.downloadFiles.map((file) => file.url);
    const fileCount = state.downloadFiles.length;
    el.downloadAll.disabled = fileCount === 0;
    el.downloadAll.textContent = fileCount > 1 ? `Download ${fileCount} formats` : 'Download format';
    if (fileCount > 1) {
      el.downloadNote.textContent = `Ready: ${noteParts.join(' · ')}. Download as zip.`;
    } else {
      el.downloadNote.textContent = `Ready: ${noteParts.join(' · ')}.`;
    }
  };

  const clearDownload = () => {
    if (state.downloadUrls.length) {
      state.downloadUrls.forEach((url) => URL.revokeObjectURL(url));
    }
    state.downloadUrls = [];
    state.downloadFiles = [];
    if (state.downloadZipUrl) {
      URL.revokeObjectURL(state.downloadZipUrl);
    }
    state.downloadZipUrl = null;
    state.downloadZipName = '';
    if (el.downloadAll) {
      el.downloadAll.disabled = true;
      el.downloadAll.textContent = 'Download formats';
    }
    if (el.downloadNote) {
      el.downloadNote.textContent = 'Record a clip to enable download.';
    }
  };

  const triggerDownloads = async () => {
    if (!state.downloadFiles.length) return;
    if (state.downloadFiles.length === 1) {
      const file = state.downloadFiles[0];
      const link = document.createElement('a');
      link.href = file.url;
      link.download = file.name;
      link.rel = 'noopener';
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setStatus('Download started.', 'ready');
      return;
    }

    const button = el.downloadAll;
    const originalText = button?.textContent || '';
    if (button) {
      button.disabled = true;
      button.textContent = 'Preparing zip...';
    }
    setStatus('Preparing zip...', 'pending');

    try {
      if (!state.downloadZipUrl) {
        const zipBlob = await buildZipBlob(state.downloadFiles);
        state.downloadZipUrl = URL.createObjectURL(zipBlob);
      }
      const zipName = state.downloadZipName || buildFilename('zip');
      const link = document.createElement('a');
      link.href = state.downloadZipUrl;
      link.download = zipName;
      link.rel = 'noopener';
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setStatus('Zip download started.', 'ready');
    } catch (_) {
      setStatus('Zip creation failed.', 'error');
    } finally {
      if (button) {
        button.textContent = originalText || 'Download formats';
        button.disabled = state.downloadFiles.length === 0;
      }
    }
  };

  const getSelectedFps = () => {
    if (!el.fpsSelect) return 0;
    const value = el.fpsSelect.value;
    if (value === 'auto') return 0;
    const fps = Number(value);
    return Number.isFinite(fps) && fps > 0 ? fps : 0;
  };

  const getQualitySettings = () => {
    if (!el.qualitySelect) return null;
    const value = el.qualitySelect.value || 'auto';
    const preset = QUALITY_PRESETS[value] || QUALITY_PRESETS.auto;
    if (!preset || value === 'auto') return null;
    const options = {};
    if (Number.isFinite(preset.video) && preset.video > 0) {
      options.videoBitsPerSecond = preset.video;
    }
    if (Number.isFinite(preset.audio) && preset.audio > 0) {
      options.audioBitsPerSecond = preset.audio;
    }
    return Object.keys(options).length ? options : null;
  };

  const getSystemGain = () => {
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

  const getMicGain = () => {
    if (!el.micLevel) return 1;
    const value = Number(el.micLevel.value);
    if (!Number.isFinite(value)) return 1;
    return Math.max(0, value) / 100;
  };

  const updateMicLevelValue = () => {
    if (!el.micLevel || !el.micLevelValue) return;
    const value = Number(el.micLevel.value);
    const safeValue = Number.isFinite(value) ? value : 100;
    el.micLevelValue.textContent = `${Math.round(safeValue)}%`;
  };

  const getActiveAudioStreams = () => {
    const streams = [];
    if (state.captureActive && streamHasAudio(state.stream) && el.audioToggle?.checked) {
      streams.push(state.stream);
    }
    if (streamHasAudio(state.micStream) && el.micToggle?.checked) {
      streams.push(state.micStream);
    }
    return streams;
  };

  const getRecordingAudioSources = () => {
    const sources = [];
    if (streamHasAudio(state.stream) && el.audioToggle?.checked) {
      sources.push({ stream: state.stream, gain: getSystemGain() });
    }
    if (streamHasAudio(state.micStream) && el.micToggle?.checked) {
      sources.push({ stream: state.micStream, gain: getMicGain() });
    }
    return sources;
  };

  const getAudioSummary = () => {
    const systemAvailable = streamHasAudio(state.stream);
    const micAvailable = streamHasAudio(state.micStream);
    const systemRequested = Boolean(el.audioToggle?.checked);
    const micRequested = Boolean(el.micToggle?.checked);
    const activeSources = [];
    if (systemAvailable) activeSources.push('System');
    if (micAvailable) activeSources.push('Mic');
    if (activeSources.length) {
      return `Audio on (${activeSources.join(' + ')})`;
    }
    const requestedSources = [];
    if (systemRequested) requestedSources.push('System');
    if (micRequested) requestedSources.push('Mic');
    if (requestedSources.length) {
      return `Audio requested (${requestedSources.join(' + ')})`;
    }
    return 'Audio off';
  };

  const updateAudioStatus = () => {
    if (!el.audioStatus) return;
    const systemAvailable = streamHasAudio(state.stream);
    const micAvailable = streamHasAudio(state.micStream);
    const systemRequested = Boolean(el.audioToggle?.checked);
    const micRequested = Boolean(el.micToggle?.checked);
    if (systemAvailable || micAvailable) {
      const parts = [];
      if (systemAvailable) parts.push('System audio detected.');
      if (micAvailable) parts.push('Microphone detected.');
      if (systemRequested && !systemAvailable) {
        if (state.captureActive) {
          parts.push('System audio not detected. Enable "Share audio" in the browser prompt.');
        } else {
          parts.push('System audio appears after you start capture and enable "Share audio".');
        }
      }
      if (micRequested && !micAvailable) {
        parts.push('Microphone not detected.');
      }
      el.audioStatus.textContent = parts.join(' ');
      return;
    }
    if (systemRequested || micRequested) {
      const parts = [];
      if (systemRequested) {
        if (state.captureActive) {
          parts.push('System audio not detected. Enable "Share audio" in the browser prompt.');
        } else {
          parts.push('System audio appears after you start capture and enable "Share audio".');
        }
      }
      if (micRequested) {
        parts.push('Microphone not detected.');
      }
      el.audioStatus.textContent = parts.join(' ');
      return;
    }
    el.audioStatus.textContent = 'No audio sources selected.';
  };

  const setAudioMeterLevel = (value) => {
    if (!el.audioMeter || !el.audioMeterFill) return;
    const clamped = Math.max(0, Math.min(1, value));
    el.audioMeterFill.style.transform = `scaleX(${clamped})`;
    el.audioMeter.setAttribute('aria-valuenow', `${Math.round(clamped * 100)}`);
  };

  const updateAudioMeterState = () => {
    if (!el.audioMeter) return;
    const active = getActiveAudioStreams().length > 0;
    if (active) {
      el.audioMeter.dataset.active = 'true';
    } else {
      delete el.audioMeter.dataset.active;
      setAudioMeterLevel(0);
    }
  };

  const setControlsCollapsed = (collapsed) => {
    if (!el.controlsPanel || !el.controlsToggle || !el.controlsBody) return;
    el.controlsPanel.dataset.collapsed = collapsed ? 'true' : 'false';
    el.controlsToggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    el.controlsBody.hidden = collapsed;
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

  const MIN_CROP_SIZE = 24;

  const parseAspectRatioValue = (value) => {
    if (!value || value === 'free') return null;
    const parts = value.split(':').map((part) => Number(part));
    if (parts.length !== 2) return null;
    const [width, height] = parts;
    if (!Number.isFinite(width) || !Number.isFinite(height) || height === 0) return null;
    return width / height;
  };

  const getCropAspectRatio = () => {
    const ratio = parseAspectRatioValue(state.cropAspectValue);
    return Number.isFinite(ratio) && ratio > 0 ? ratio : null;
  };

  const getMinCropSizeForRatio = (ratio) => {
    if (!ratio || !Number.isFinite(ratio) || ratio <= 0) {
      return { minWidth: MIN_CROP_SIZE, minHeight: MIN_CROP_SIZE };
    }
    if (ratio >= 1) {
      return { minWidth: MIN_CROP_SIZE * ratio, minHeight: MIN_CROP_SIZE };
    }
    return { minWidth: MIN_CROP_SIZE, minHeight: MIN_CROP_SIZE / ratio };
  };

  const getHandleDirection = (handle) => ({
    x: handle && handle.includes('w') ? -1 : 1,
    y: handle && handle.includes('n') ? -1 : 1
  });

  const getResizeAnchor = (box, handle) => ({
    x: handle && handle.includes('w') ? box.x + box.width : box.x,
    y: handle && handle.includes('n') ? box.y + box.height : box.y
  });

  const getFreeformCropBox = (anchor, point, frame, direction, enforceMin) => {
    let width = Math.abs(point.x - anchor.x);
    let height = Math.abs(point.y - anchor.y);
    const maxWidth = direction.x > 0
      ? frame.offsetX + frame.displayWidth - anchor.x
      : anchor.x - frame.offsetX;
    const maxHeight = direction.y > 0
      ? frame.offsetY + frame.displayHeight - anchor.y
      : anchor.y - frame.offsetY;
    width = Math.min(width, maxWidth);
    height = Math.min(height, maxHeight);
    if (enforceMin) {
      width = Math.max(width, MIN_CROP_SIZE);
      height = Math.max(height, MIN_CROP_SIZE);
      width = Math.min(width, maxWidth);
      height = Math.min(height, maxHeight);
    }
    const x = direction.x > 0 ? anchor.x : anchor.x - width;
    const y = direction.y > 0 ? anchor.y : anchor.y - height;
    return { x, y, width, height };
  };

  const getRatioCropBox = (anchor, point, frame, direction, ratio, enforceMin) => {
    if (!ratio || !Number.isFinite(ratio) || ratio <= 0) {
      return getFreeformCropBox(anchor, point, frame, direction, enforceMin);
    }
    let width = Math.abs(point.x - anchor.x);
    let height = Math.abs(point.y - anchor.y);
    if (width === 0 && height === 0) {
      if (!enforceMin) {
        return { x: anchor.x, y: anchor.y, width: 0, height: 0 };
      }
      const min = getMinCropSizeForRatio(ratio);
      width = min.minWidth;
      height = width / ratio;
    } else if (height === 0) {
      height = width / ratio;
    } else if (width === 0) {
      width = height * ratio;
    } else if (width / height > ratio) {
      width = height * ratio;
    } else {
      height = width / ratio;
    }

    const maxWidth = direction.x > 0
      ? frame.offsetX + frame.displayWidth - anchor.x
      : anchor.x - frame.offsetX;
    const maxHeight = direction.y > 0
      ? frame.offsetY + frame.displayHeight - anchor.y
      : anchor.y - frame.offsetY;
    if (width > 0 && height > 0) {
      const scale = Math.min(1, maxWidth / width, maxHeight / height);
      width *= scale;
      height *= scale;
    }
    if (enforceMin) {
      const min = getMinCropSizeForRatio(ratio);
      if (width < min.minWidth || height < min.minHeight) {
        width = min.minWidth;
        height = width / ratio;
        const scale = Math.min(1, maxWidth / width, maxHeight / height);
        width *= scale;
        height *= scale;
      }
    }
    const x = direction.x > 0 ? anchor.x : anchor.x - width;
    const y = direction.y > 0 ? anchor.y : anchor.y - height;
    return { x, y, width, height };
  };

  const normalizeCropBox = (box, frame) => ({
    x: Math.min(Math.max((box.x - frame.offsetX) / frame.displayWidth, 0), 1),
    y: Math.min(Math.max((box.y - frame.offsetY) / frame.displayHeight, 0), 1),
    width: Math.min(Math.max(box.width / frame.displayWidth, 0), 1),
    height: Math.min(Math.max(box.height / frame.displayHeight, 0), 1)
  });

  const clampCropBox = (box, frame, handle) => {
    let { x, y, width, height } = box;
    const minX = frame.offsetX;
    const minY = frame.offsetY;
    const maxX = frame.offsetX + frame.displayWidth;
    const maxY = frame.offsetY + frame.displayHeight;

    if (width < MIN_CROP_SIZE) {
      if (handle && handle.includes('w')) {
        x -= MIN_CROP_SIZE - width;
      }
      width = MIN_CROP_SIZE;
    }
    if (height < MIN_CROP_SIZE) {
      if (handle && handle.includes('n')) {
        y -= MIN_CROP_SIZE - height;
      }
      height = MIN_CROP_SIZE;
    }

    if (x < minX) {
      if (handle && handle.includes('w')) {
        width -= minX - x;
      }
      x = minX;
    }
    if (y < minY) {
      if (handle && handle.includes('n')) {
        height -= minY - y;
      }
      y = minY;
    }

    if (x + width > maxX) {
      if (handle && handle.includes('e')) {
        width = maxX - x;
      } else {
        x = maxX - width;
      }
    }
    if (y + height > maxY) {
      if (handle && handle.includes('s')) {
        height = maxY - y;
      } else {
        y = maxY - height;
      }
    }

    width = Math.max(MIN_CROP_SIZE, Math.min(width, maxX - minX));
    height = Math.max(MIN_CROP_SIZE, Math.min(height, maxY - minY));
    x = Math.min(Math.max(x, minX), maxX - width);
    y = Math.min(Math.max(y, minY), maxY - height);

    return { x, y, width, height };
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
    state.cropResizing = false;
    state.cropResizeHandle = null;
    state.cropResizeStart = null;
    updateCropOverlay();
    updateButtons();
  };

  const updateCropPresetUI = () => {
    if (el.cropPresetsLabel) {
      el.cropPresetsLabel.textContent = state.cropAspectLabel || 'Free';
    }
    if (el.cropPresetsToggle) {
      el.cropPresetsToggle.title = `Crop preset: ${state.cropAspectLabel || 'Free'}`;
    }
    if (el.cropPresetButtons && el.cropPresetButtons.length) {
      el.cropPresetButtons.forEach((button) => {
        const value = button.dataset.ratio || 'free';
        const isActive = value === state.cropAspectValue;
        button.classList.toggle('is-active', isActive);
        button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      });
    }
  };

  const setCropPresetsOpen = (open) => {
    state.cropPresetsOpen = Boolean(open);
    if (el.cropPresets) {
      el.cropPresets.hidden = !state.cropPresetsOpen;
    }
    if (el.cropPresetsToggle) {
      el.cropPresetsToggle.setAttribute('aria-expanded', state.cropPresetsOpen ? 'true' : 'false');
    }
  };

  const applyCropAspectRatio = () => {
    const ratio = getCropAspectRatio();
    if (!ratio || !state.cropRegion) return;
    const frame = getVideoFrameRect();
    if (!frame) return;
    const box = getCropBoxFromRegion(frame);
    if (!box) return;
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    let width = box.width;
    let height = box.height;
    if (width / height > ratio) {
      width = height * ratio;
    } else {
      height = width / ratio;
    }
    const min = getMinCropSizeForRatio(ratio);
    if (width < min.minWidth || height < min.minHeight) {
      width = min.minWidth;
      height = width / ratio;
    }
    const maxWidth = frame.displayWidth;
    const maxHeight = frame.displayHeight;
    if (width > maxWidth || height > maxHeight) {
      const scale = Math.min(1, maxWidth / width, maxHeight / height);
      width *= scale;
      height *= scale;
    }
    let x = centerX - width / 2;
    let y = centerY - height / 2;
    x = Math.min(Math.max(x, frame.offsetX), frame.offsetX + frame.displayWidth - width);
    y = Math.min(Math.max(y, frame.offsetY), frame.offsetY + frame.displayHeight - height);
    state.cropRegion = normalizeCropBox({ x, y, width, height }, frame);
    updateCropSelectionBox({ x, y, width, height });
    updateCropOverlay();
  };

  const setCropPreset = (value, label) => {
    const nextValue = value || 'free';
    const ratio = parseAspectRatioValue(nextValue);
    if (!ratio && nextValue !== 'free') {
      state.cropAspectValue = 'free';
      state.cropAspectLabel = 'Free';
    } else {
      state.cropAspectValue = nextValue;
      state.cropAspectLabel = label || (nextValue === 'free' ? 'Free' : nextValue);
    }
    updateCropPresetUI();
    if (state.cropRegion) {
      applyCropAspectRatio();
    }
  };

  const stopAudioMeter = () => {
    if (state.audioMeterId) {
      cancelAnimationFrame(state.audioMeterId);
      state.audioMeterId = null;
    }
    if (state.audioMeterSources.length) {
      state.audioMeterSources.forEach((source) => {
        try {
          source.disconnect();
        } catch (_) {}
      });
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
    state.audioMeterSources = [];
    state.audioMeterData = null;
    state.audioMeterLevel = 0;
    setAudioMeterLevel(0);
  };

  const startAudioMeter = () => {
    stopAudioMeter();
    updateAudioMeterState();
    const meterStreams = getActiveAudioStreams();
    if (!meterStreams.length) return;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    let audioContext;
    try {
      audioContext = new AudioCtx();
    } catch (_) {
      return;
    }
    const sources = [];
    let analyser;
    try {
      meterStreams.forEach((stream) => {
        sources.push(audioContext.createMediaStreamSource(stream));
      });
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.85;
      sources.forEach((source) => source.connect(analyser));
    } catch (_) {
      audioContext.close().catch(() => {});
      return;
    }

    state.audioMeterContext = audioContext;
    state.audioMeterSources = sources;
    state.audioMeterAnalyser = analyser;
    state.audioMeterData = new Uint8Array(analyser.fftSize);
    state.audioMeterLevel = 0;

    const tick = () => {
      if (!getActiveAudioStreams().length) {
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

  const streamHasAudio = (stream) => Boolean(
    stream &&
    stream.getAudioTracks &&
    stream.getAudioTracks().some((track) => track.readyState !== 'ended')
  );

  const isCaptureLive = () => {
    if (!state.captureActive || !state.stream || !state.stream.active) return false;
    const tracks = state.stream.getVideoTracks();
    return tracks.length > 0 && tracks.some((track) => track.readyState === 'live');
  };

  const setMicHelp = (text) => {
    if (!el.micHelp) return;
    el.micHelp.textContent = text;
  };

  const stopMicStream = () => {
    if (!state.micStream) return;
    stopTracks(state.micStream);
    state.micStream = null;
  };

  const updateMicDevices = async () => {
    if (!el.micSelect || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    let devices = [];
    try {
      devices = await navigator.mediaDevices.enumerateDevices();
    } catch (_) {
      return;
    }
    const audioInputs = devices.filter((device) => device.kind === 'audioinput');
    const currentValue = state.micDeviceId || el.micSelect.value;
    el.micSelect.innerHTML = '';

    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Default microphone';
    el.micSelect.appendChild(defaultOption);

    audioInputs.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Microphone ${index + 1}`;
      el.micSelect.appendChild(option);
    });

    if (currentValue && audioInputs.some((device) => device.deviceId === currentValue)) {
      el.micSelect.value = currentValue;
    } else {
      el.micSelect.value = '';
    }
  };

  const ensureMicStream = async () => {
    if (!el.micToggle?.checked) {
      stopMicStream();
      updateAudioStatus();
      startAudioMeter();
      return false;
    }
    if (!supportsMicrophone) {
      setMicHelp('Microphone capture is not supported in this browser.');
      if (el.micToggle) {
        el.micToggle.checked = false;
      }
      updateButtons();
      return false;
    }

    const requestedId = el.micSelect?.value || '';
    if (state.micStream && requestedId === state.micDeviceId) {
      setMicHelp('Microphone ready.');
      updateAudioStatus();
      startAudioMeter();
      return true;
    }

    const constraints = requestedId ? { deviceId: { exact: requestedId } } : true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
      stopMicStream();
      state.micStream = stream;
      state.micDeviceId = requestedId;
      const micTrack = stream.getAudioTracks()[0];
      micTrack?.addEventListener('ended', () => {
        stopMicStream();
        updateAudioStatus();
        startAudioMeter();
        updateCaptureMeta();
      }, { once: true });
      setMicHelp('Microphone ready.');
      await updateMicDevices();
      updateAudioStatus();
      startAudioMeter();
      updateCaptureMeta();
      return true;
    } catch (_) {
      stopMicStream();
      if (el.micToggle) {
        el.micToggle.checked = false;
      }
      setMicHelp('Microphone access denied or unavailable.');
      setStatus('Microphone access denied or unavailable.', 'warn');
      updateAudioStatus();
      startAudioMeter();
      updateButtons();
      return false;
    }
  };

  const updateCaptureMeta = () => {
    if (!el.captureMeta) {
      updateAudioStatus();
      return;
    }
    if (!state.stream) {
      el.captureMeta.textContent = state.recordedUrl ? 'Recorded clip' : 'Not started';
      updateAudioStatus();
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
    parts.push(getAudioSummary());
    el.captureMeta.textContent = parts.length ? parts.join(' | ') : 'Capture active';
    updateAudioStatus();
  };

  const updateButtons = () => {
    const captureLive = isCaptureLive();
    if (el.startCapture) {
      el.startCapture.disabled = !supportsCapture || !supportsRecorder || captureLive || state.recording;
    }
    if (el.stopCapture) {
      el.stopCapture.disabled = !captureLive || state.recording;
    }
    if (el.startRecord) {
      el.startRecord.disabled = !captureLive || state.recording;
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
    if (el.micToggle) {
      el.micToggle.disabled = !supportsMicrophone || state.captureActive || state.recording;
    }
    if (el.micSelect) {
      const micLocked = !supportsMicrophone || state.captureActive || state.recording || !el.micToggle?.checked;
      el.micSelect.disabled = micLocked;
    }
    if (el.fpsSelect) {
      el.fpsSelect.disabled = !supportsCapture || state.captureActive || state.recording;
    }
    if (el.qualitySelect) {
      el.qualitySelect.disabled = !supportsRecorder || state.captureActive || state.recording;
    }
    if (el.audioLevel) {
      const audioLocked = !supportsCapture || state.captureActive || state.recording || !el.audioToggle?.checked;
      el.audioLevel.disabled = audioLocked;
    }
    if (el.micLevel) {
      const micLevelLocked = !supportsMicrophone || state.captureActive || state.recording || !el.micToggle?.checked;
      el.micLevel.disabled = micLevelLocked;
    }
    if (el.formatOptions) {
      el.formatOptions.querySelectorAll('input').forEach((input) => {
        input.disabled = state.recording;
      });
    }
    if (el.testCapture) {
      el.testCapture.disabled = !supportsCapture || !supportsRecorder || state.captureActive || state.recording;
    }
    if (el.cropToggle) {
      el.cropToggle.disabled = !captureLive || state.recording;
      const label = state.cropSelecting || state.cropRegion ? 'Cancel crop' : 'Crop';
      el.cropToggle.setAttribute('aria-label', label);
      el.cropToggle.title = label;
      if (el.cropLabel) {
        el.cropLabel.textContent = label;
      }
    }
    if (el.cropPresetsToggle) {
      el.cropPresetsToggle.disabled = !captureLive || state.recording;
      if (el.cropPresetsToggle.disabled && state.cropPresetsOpen) {
        setCropPresetsOpen(false);
      }
    }
    if (el.downloadAll) {
      const hasDownload = state.downloadUrls.length > 0;
      el.downloadAll.disabled = !hasDownload;
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

  const setRecordedClip = (blob, mimeType, files) => {
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

    const downloadFiles = Array.isArray(files) && files.length
      ? files
      : [{ blob, mimeType: state.recordedMimeType, label: formatLabelFromMime(state.recordedMimeType) }];
    setDownloads(downloadFiles);
    setStatus('Clip ready to download. Stop capture when you are done.', 'ready');
  };

  const startCapture = async () => {
    if (state.captureActive && (!state.stream || !state.stream.active)) {
      stopCapture('Capture ended by the browser.');
    }
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
    if (el.micToggle?.checked && !streamHasAudio(state.micStream)) {
      await ensureMicStream();
    }
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
    } else if (includeAudio && !streamHasAudio(stream)) {
      setStatus('System audio not detected. Enable "Share audio" in the browser prompt.', 'warn');
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
    startAudioMeter();
  };

  const startTestCapture = async () => {
    if (state.captureActive || state.recording) return;
    state.testMode = true;
    const started = await startCapture();
    if (!started) {
      state.testMode = false;
      return;
    }
    await startRecording();
    if (state.recording) {
      state.testTimerId = setTimeout(() => {
        stopRecording();
      }, 5000);
    } else {
      state.testMode = false;
    }
  };

  const createRecorder = (stream, mimeType, allowFallback = true, fallbackStream = stream, baseOptions = null) => {
    const options = { ...(baseOptions || {}) };
    if (mimeType) options.mimeType = mimeType;
    let recorder;
    let finalMime = mimeType;
    try {
      recorder = new MediaRecorder(stream, options);
    } catch (err) {
      if (mimeType && allowFallback) {
        const fallbackOptions = { ...(baseOptions || {}) };
        recorder = new MediaRecorder(fallbackStream, fallbackOptions);
        finalMime = '';
        setStatus('Selected format not supported. Using browser default.', 'warn');
      } else {
        throw err;
      }
    }
    return { recorder, mimeType: finalMime };
  };

  const buildRecorderSet = (stream) => {
    const selections = getSelectedMimeTypes();
    const uniqueSelections = selections.length ? Array.from(new Set(selections)) : [''];
    const hasAudio = streamHasAudio(stream);
    const videoOnlyStream = hasAudio ? new MediaStream(stream.getVideoTracks()) : stream;
    const qualityOptions = getQualitySettings();

    const recorders = [];
    const failed = [];
    let mp4AudioStripped = false;
    uniqueSelections.forEach((mimeType, index) => {
      const resolved = resolveMp4Recording(mimeType, hasAudio);
      if (resolved.stripAudio) {
        mp4AudioStripped = true;
      }
      const selection = {
        mimeType: resolved.mimeType,
        label: formatLabelFromMime(resolved.mimeType || mimeType),
        isPrimary: index === 0
      };
      const recorderStream = resolved.stripAudio ? videoOnlyStream : stream;
      const recorderOptions = qualityOptions ? { ...qualityOptions } : null;
      if (recorderOptions && recorderOptions.audioBitsPerSecond && !streamHasAudio(recorderStream)) {
        delete recorderOptions.audioBitsPerSecond;
      }
      let recorderInfo;
      try {
        recorderInfo = createRecorder(recorderStream, selection.mimeType, selection.isPrimary, stream, recorderOptions);
      } catch (_) {
        failed.push(selection);
        return;
      }
      recorders.push({
        recorder: recorderInfo.recorder,
        mimeType: recorderInfo.mimeType || selection.mimeType,
        label: selection.label,
        isPrimary: selection.isPrimary,
        chunks: [],
        file: null
      });
    });
    return { recorders, failed, mp4AudioStripped };
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

  const createMixedAudioTrack = (sources) => {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    try {
      const audioContext = new AudioCtx();
      const destination = audioContext.createMediaStreamDestination();
      const sourceNodes = [];
      const gainNodes = [];
      sources.forEach((entry) => {
        const source = audioContext.createMediaStreamSource(entry.stream);
        const gainNode = audioContext.createGain();
        gainNode.gain.value = Number.isFinite(entry.gain) ? entry.gain : 1;
        source.connect(gainNode);
        gainNode.connect(destination);
        sourceNodes.push(source);
        gainNodes.push(gainNode);
      });
      state.audioContext = audioContext;
      const track = destination.stream.getAudioTracks()[0];
      const cleanup = () => {
        try {
          track.stop();
        } catch (_) {}
        sourceNodes.forEach((source) => source.disconnect());
        gainNodes.forEach((gainNode) => gainNode.disconnect());
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

    const audioSources = getRecordingAudioSources();
    if (audioSources.length) {
      const shouldMix = audioSources.length > 1 || audioSources.some((entry) => entry.gain !== 1);
      if (shouldMix) {
        const mixed = createMixedAudioTrack(audioSources);
        if (mixed && mixed.track) {
          outputStream.addTrack(mixed.track);
          cleanupTasks.push(mixed.cleanup);
        } else {
          outputStream.addTrack(audioSources[0].stream.getAudioTracks()[0]);
        }
      } else {
        outputStream.addTrack(audioSources[0].stream.getAudioTracks()[0]);
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

  const startRecording = async () => {
    if (!isCaptureLive() || state.recording || !state.stream) return;
    clearRecordedClip();

    setLivePreview();
    if (el.micToggle?.checked && !streamHasAudio(state.micStream)) {
      await ensureMicStream();
    }

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

    const recorderSet = buildRecorderSet(recordingStreamInfo.stream);
    if (!recorderSet.recorders.length) {
      setStatus('Recording failed to initialize in this browser.', 'error');
      cleanupRecordingPipeline();
      if (state.testMode) {
        state.testMode = false;
      }
      return;
    }

    const mp4AudioStripped = recorderSet.mp4AudioStripped;
    state.recorders = recorderSet.recorders;
    state.primaryRecorder = state.recorders.find((entry) => entry.isPrimary)?.recorder || state.recorders[0]?.recorder || null;
    state.recording = true;
    state.paused = false;
    startTimer();

    let stoppedCount = 0;
    const finalizeRecording = () => {
      state.recording = false;
      state.paused = false;
      stopTimer();
      const files = state.recorders
        .map((entry) => entry.file)
        .filter(Boolean);
      const primaryEntry = state.recorders.find((entry) => entry.isPrimary && entry.file);
      const primaryFile = primaryEntry?.file || files[0];
      if (files.length && primaryFile) {
        state.recordedHasAudio = streamHasAudio(recordingStreamInfo.stream);
        setRecordedClip(primaryFile.blob, primaryFile.mimeType, files);
      } else {
        setStatus('Recording stopped. No data captured.', 'warn');
      }
      cleanupRecordingPipeline();
      state.recorders = [];
      state.primaryRecorder = null;
      if (state.testTimerId) {
        clearTimeout(state.testTimerId);
        state.testTimerId = null;
      }
      if (state.testMode) {
        state.testMode = false;
        stopCapture('Test capture complete. Clip ready to download.');
        return;
      }
      if (state.captureActive && state.stream && !state.stream.active) {
        stopCapture(state.stopReason || 'Capture ended by the browser.');
        return;
      }
      updateButtons();
    };

    const handleRecorderStop = (entry) => {
      if (!state.recorders.includes(entry)) return;
      const mime = entry.recorder.mimeType || entry.mimeType || 'video/webm';
      if (entry.chunks.length) {
        const blob = new Blob(entry.chunks, { type: mime });
        entry.chunks = [];
        entry.file = {
          blob,
          mimeType: mime,
          label: formatLabelFromMime(mime),
          isPrimary: entry.isPrimary
        };
      }
      stoppedCount += 1;
      if (stoppedCount === state.recorders.length) {
        finalizeRecording();
      }
    };

    state.recorders.forEach((entry) => {
      entry.recorder.addEventListener('dataavailable', (event) => {
        if (event.data && event.data.size > 0) {
          entry.chunks.push(event.data);
        }
      });
      entry.recorder.addEventListener('stop', () => {
        handleRecorderStop(entry);
      }, { once: true });
    });

    try {
      const started = [];
      const failedStarts = [];
      state.recorders.forEach((entry) => {
        try {
          entry.recorder.start(200);
          started.push(entry);
        } catch (_) {
          failedStarts.push(entry);
        }
      });

      state.recorders = started;
      state.primaryRecorder = state.recorders.find((entry) => entry.isPrimary)?.recorder || state.recorders[0]?.recorder || null;

      if (!state.recorders.length) {
        throw new Error('Recorder start failed.');
      }

      const primaryMissing = !state.recorders.some((entry) => entry.isPrimary);
      const formatsSkipped = recorderSet.failed.length > 0 || failedStarts.length > 0;
      let statusMessage = state.testMode ? 'Recording 5-second test...' : 'Recording...';
      if (primaryMissing && formatsSkipped) {
        statusMessage = `${statusMessage} Primary format unavailable; some formats skipped.`;
      } else if (primaryMissing) {
        statusMessage = `${statusMessage} Primary format unavailable.`;
      } else if (formatsSkipped) {
        statusMessage = `${statusMessage} Some formats skipped.`;
      }
      if (mp4AudioStripped) {
        statusMessage = `${statusMessage} MP4 audio isn't supported here; MP4 will be video-only.`;
      }
      setStatus(statusMessage, 'recording');
      updateButtons();
    } catch (err) {
      state.recording = false;
      stopTimer();
      setStatus('Recording failed to start.', 'error');
      cleanupRecordingPipeline();
      state.recorders = [];
      state.primaryRecorder = null;
      if (state.testMode) {
        state.testMode = false;
      }
      updateButtons();
    }
  };

  const togglePause = () => {
    if (!state.recorders.length || !state.recording) return;
    if (state.primaryRecorder && state.primaryRecorder.state === 'recording') {
      state.recorders.forEach((entry) => {
        if (entry.recorder.state === 'recording') {
          try {
            entry.recorder.pause();
          } catch (_) {}
        }
      });
      state.paused = true;
      pauseTimer();
      setStatus('Recording paused.', 'warn');
    } else if (state.primaryRecorder && state.primaryRecorder.state === 'paused') {
      state.recorders.forEach((entry) => {
        if (entry.recorder.state === 'paused') {
          try {
            entry.recorder.resume();
          } catch (_) {}
        }
      });
      state.paused = false;
      resumeTimer();
      setStatus('Recording...', 'recording');
    }
    updateButtons();
  };

  const stopRecording = () => {
    if (!state.recorders.length || !state.recording) return;
    state.recorders.forEach((entry) => {
      if (entry.recorder.state !== 'inactive') {
        entry.recorder.stop();
      }
    });
  };

  const toggleCropSelection = () => {
    if (!isCaptureLive() || state.recording) return;
    if (state.cropPresetsOpen) {
      setCropPresetsOpen(false);
    }
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
    const resizeHandle = event.target.closest('[data-screenrec-resize]');
    if (resizeHandle) {
      const box = getCropBoxFromRegion(frame);
      if (!box) return;
      state.cropResizing = true;
      state.cropResizeHandle = resizeHandle.dataset.screenrecResize || '';
      state.cropResizeStart = { box, point };
      el.cropOverlay.setPointerCapture(event.pointerId);
      return;
    }

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
    if (!state.cropStart && !state.cropDragging && !state.cropResizing) return;
    const frame = getVideoFrameRect();
    if (!frame) return;
    const point = {
      x: event.clientX - frame.stageRect.left,
      y: event.clientY - frame.stageRect.top
    };
    const ratio = getCropAspectRatio();

    if (state.cropResizing && state.cropResizeStart) {
      const { box: startBox, point: startPoint } = state.cropResizeStart;
      const dx = point.x - startPoint.x;
      const dy = point.y - startPoint.y;
      const handle = state.cropResizeHandle || '';
      if (ratio) {
        const clamped = clampPointToFrame(point, frame);
        const anchor = getResizeAnchor(startBox, handle);
        const direction = getHandleDirection(handle);
        const nextBox = getRatioCropBox(anchor, clamped, frame, direction, ratio, true);
        state.cropRegion = normalizeCropBox(nextBox, frame);
        updateCropSelectionBox(nextBox);
      } else {
        let nextBox = { ...startBox };
        if (handle.includes('n')) {
          nextBox.y = startBox.y + dy;
          nextBox.height = startBox.height - dy;
        }
        if (handle.includes('s')) {
          nextBox.height = startBox.height + dy;
        }
        if (handle.includes('w')) {
          nextBox.x = startBox.x + dx;
          nextBox.width = startBox.width - dx;
        }
        if (handle.includes('e')) {
          nextBox.width = startBox.width + dx;
        }
        nextBox = clampCropBox(nextBox, frame, handle);
        state.cropRegion = normalizeCropBox(nextBox, frame);
        updateCropSelectionBox(nextBox);
      }
      return;
    }

    if (state.cropStart) {
      const clamped = clampPointToFrame(point, frame);
      state.cropCurrent = clamped;
      let box;
      if (ratio) {
        const direction = {
          x: clamped.x >= state.cropStart.x ? 1 : -1,
          y: clamped.y >= state.cropStart.y ? 1 : -1
        };
        box = getRatioCropBox(state.cropStart, clamped, frame, direction, ratio, false);
      } else {
        box = {
          x: Math.min(state.cropStart.x, clamped.x),
          y: Math.min(state.cropStart.y, clamped.y),
          width: Math.abs(clamped.x - state.cropStart.x),
          height: Math.abs(clamped.y - state.cropStart.y)
        };
      }
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
    if (!state.cropStart && !state.cropDragging && !state.cropResizing) return;
    const frame = getVideoFrameRect();
    if (!frame) {
      state.cropStart = null;
      state.cropCurrent = null;
      state.cropDragging = false;
      state.cropDragOffset = null;
      state.cropResizing = false;
      state.cropResizeHandle = null;
      state.cropResizeStart = null;
      return;
    }
    if (state.cropResizing) {
      state.cropResizing = false;
      state.cropResizeHandle = null;
      state.cropResizeStart = null;
    } else if (state.cropStart) {
      const point = {
        x: event.clientX - frame.stageRect.left,
        y: event.clientY - frame.stageRect.top
      };
      const clamped = clampPointToFrame(point, frame);
      const ratio = getCropAspectRatio();
      let box;
      if (ratio) {
        const direction = {
          x: clamped.x >= state.cropStart.x ? 1 : -1,
          y: clamped.y >= state.cropStart.y ? 1 : -1
        };
        box = getRatioCropBox(state.cropStart, clamped, frame, direction, ratio, false);
        const min = getMinCropSizeForRatio(ratio);
        if (box.width >= min.minWidth && box.height >= min.minHeight) {
          state.cropRegion = normalizeCropBox(box, frame);
        }
      } else {
        box = {
          x: Math.min(state.cropStart.x, clamped.x),
          y: Math.min(state.cropStart.y, clamped.y),
          width: Math.abs(clamped.x - state.cropStart.x),
          height: Math.abs(clamped.y - state.cropStart.y)
        };
        if (box.width >= MIN_CROP_SIZE && box.height >= MIN_CROP_SIZE) {
          state.cropRegion = {
            x: (box.x - frame.offsetX) / frame.displayWidth,
            y: (box.y - frame.offsetY) / frame.displayHeight,
            width: box.width / frame.displayWidth,
            height: box.height / frame.displayHeight
          };
        }
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
    updateMicLevelValue();
    updateCropPresetUI();
    updateCaptureMeta();
    updateButtons();
    setView();
    if (el.controlsPanel && el.controlsBody && el.controlsToggle) {
      const shouldCollapse = el.controlsPanel.dataset.collapsed === 'true' || el.controlsBody.hidden;
      setControlsCollapsed(shouldCollapse);
    }
    updateMicDevices();
    if (!supportsMicrophone) {
      setMicHelp('Microphone capture is not supported in this browser.');
    }

    el.startCapture?.addEventListener('click', startCapture);
    el.stopCapture?.addEventListener('click', () => stopCapture());
    el.startRecord?.addEventListener('click', startRecording);
    el.pauseRecord?.addEventListener('click', togglePause);
    el.stopRecord?.addEventListener('click', stopRecording);
    el.testCapture?.addEventListener('click', startTestCapture);
    el.cropToggle?.addEventListener('click', toggleCropSelection);
    el.cropPresetsToggle?.addEventListener('click', (event) => {
      event.stopPropagation();
      if (el.cropPresetsToggle?.disabled) return;
      setCropPresetsOpen(!state.cropPresetsOpen);
    });
    el.cropPresets?.addEventListener('click', (event) => {
      const target = event.target.closest('[data-screenrec="crop-preset"]');
      if (!target) return;
      const value = target.dataset.ratio || 'free';
      const label = target.dataset.label || target.textContent.trim();
      setCropPreset(value, label);
      setCropPresetsOpen(false);
    });
    document.addEventListener('click', (event) => {
      if (!state.cropPresetsOpen) return;
      if (el.cropPresets?.contains(event.target) || el.cropPresetsToggle?.contains(event.target)) return;
      setCropPresetsOpen(false);
    });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || !state.cropPresetsOpen) return;
      setCropPresetsOpen(false);
    });
    el.controlsToggle?.addEventListener('click', () => {
      if (!el.controlsPanel) return;
      const collapsed = el.controlsPanel.dataset.collapsed === 'true';
      setControlsCollapsed(!collapsed);
    });
    el.audioToggle?.addEventListener('change', () => {
      updateButtons();
      updateAudioStatus();
    });
    el.micToggle?.addEventListener('change', () => {
      if (el.micToggle.checked) {
        ensureMicStream();
      } else {
        stopMicStream();
        updateAudioStatus();
        startAudioMeter();
        updateCaptureMeta();
      }
      updateButtons();
    });
    el.micSelect?.addEventListener('change', () => {
      state.micDeviceId = el.micSelect.value;
      if (el.micToggle?.checked) {
        ensureMicStream();
      }
    });
    el.audioLevel?.addEventListener('input', updateAudioLevelValue);
    el.micLevel?.addEventListener('input', updateMicLevelValue);
    el.fpsSelect?.addEventListener('change', updateButtons);
    el.cropOverlay?.addEventListener('pointerdown', handleCropPointerDown);
    el.cropOverlay?.addEventListener('pointermove', handleCropPointerMove);
    el.cropOverlay?.addEventListener('pointerup', handleCropPointerUp);
    el.cropOverlay?.addEventListener('pointercancel', handleCropPointerUp);
    window.addEventListener('resize', syncCropSelection);
    el.downloadAll?.addEventListener('click', () => {
      if (!state.downloadUrls.length) {
        setStatus('Record a clip before downloading.', 'warn');
        return;
      }
      triggerDownloads();
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
    navigator.mediaDevices?.addEventListener('devicechange', updateMicDevices);
  };

  init();
})();
