(() => {
  'use strict';

  const RECORDINGS = {
    ocean: '/audio/ocean/gentle-water.json',
    cove: '/audio/ocean/quiet-beach.json'
  };
  const CROSSFADE = 6;
  const clamp = (value, fallback = 0) => Number.isFinite(Number(value))
    ? Math.max(0, Math.min(1, Number(value))) : fallback;

  const create = ({ onStatus = () => {}, onError = () => {} } = {}) => {
    let context = null;
    let master = null;
    let enabled = false;
    let visible = true;
    let disposed = false;
    let volume = 0.35;
    let fade = 1;
    let scene = 'ocean';
    let epoch = 0;
    let scheduler = 0;
    let suspendTimer = 0;
    let currentLayer = null;
    const layers = new Set();
    const buffers = new Map();
    const requests = new Map();
    const controllers = new Set();

    const notify = (state) => {
      if (!disposed) onStatus({ state, enabled });
    };
    const shouldPlay = () => enabled && visible && fade > 0 && !disposed;
    const disconnect = (node) => { try { node.disconnect(); } catch (_) {} };
    const hold = (parameter, time) => {
      if (typeof parameter.cancelAndHoldAtTime === 'function') parameter.cancelAndHoldAtTime(time);
      else {
        const value = parameter.value;
        parameter.cancelScheduledValues(time);
        parameter.setValueAtTime(value, time);
      }
    };
    const updateGain = (seconds = 0.65) => {
      if (!master || !context || disposed) return;
      const now = context.currentTime;
      hold(master.gain, now);
      master.gain.linearRampToValueAtTime(shouldPlay() ? Math.pow(volume, 1.35) * fade : 0, now + seconds);
    };
    const stopLayer = (layer) => {
      if (!layers.has(layer)) return;
      for (const clip of layer.clips) {
        clip.source.onended = null;
        try { clip.source.stop(); } catch (_) {}
        disconnect(clip.source);
        disconnect(clip.gain);
      }
      layer.clips.clear();
      disconnect(layer.gain);
      layers.delete(layer);
      if (currentLayer === layer) currentLayer = null;
    };
    const stopLayers = () => {
      window.clearInterval(scheduler);
      scheduler = 0;
      [...layers].forEach(stopLayer);
    };
    const retireLayer = (layer) => {
      if (!layer || layer.retireAt) return;
      const now = context.currentTime;
      hold(layer.gain.gain, now);
      layer.gain.gain.linearRampToValueAtTime(0, now + CROSSFADE);
      layer.retireAt = now + CROSSFADE + 0.1;
    };

    const load = (name) => {
      if (buffers.has(name)) return Promise.resolve(buffers.get(name));
      if (requests.has(name)) return requests.get(name);
      const controller = new AbortController();
      controllers.add(controller);
      const timeout = window.setTimeout(() => controller.abort(), 20000);
      const request = (async () => {
        try {
          const response = await fetch(RECORDINGS[name], { signal: controller.signal, credentials: 'same-origin' });
          if (!response.ok || response.status === 204) throw new Error('The ocean recording could not be loaded.');
          // A JSON envelope keeps download managers from taking over background
          // media requests. The enclosed recording remains an ordinary MP3.
          const payload = await response.json();
          if (disposed || payload.encoding !== 'base64' || payload.mediaType !== 'audio/mpeg'
            || typeof payload.data !== 'string' || !payload.data.length || payload.data.length > 11 * 1024 * 1024) {
            throw new Error('The ocean recording is unavailable.');
          }
          const binary = window.atob(payload.data);
          const bytes = new Uint8Array(binary.length);
          for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
          const buffer = await context.decodeAudioData(bytes.buffer);
          if (disposed) throw new Error('Audio has been closed.');
          if (buffer.duration < 15 || buffer.duration > 240 || buffer.numberOfChannels > 2) {
            throw new Error('The ocean recording has an unsupported format.');
          }
          buffers.set(name, buffer);
          return buffer;
        } finally {
          window.clearTimeout(timeout);
          controllers.delete(controller);
          requests.delete(name);
        }
      })();
      requests.set(name, request);
      return request;
    };

    const scheduleClip = (layer, when) => {
      // Long, unaltered field-recording passages preserve natural wave timing.
      // Changing the passage and its length avoids an obvious short repeating loop.
      const duration = Math.min(layer.buffer.duration - 0.5, 48 + Math.random() * 24);
      let offset = Math.random() * Math.max(0, layer.buffer.duration - duration - 0.25);
      if (Math.abs(offset - layer.lastOffset) < 8) {
        offset = (offset + layer.buffer.duration * 0.31) % Math.max(0.01, layer.buffer.duration - duration - 0.25);
      }
      layer.lastOffset = offset;
      const source = context.createBufferSource();
      const gain = context.createGain();
      source.buffer = layer.buffer;
      source.connect(gain);
      gain.connect(layer.gain);
      const fadeIn = new Float32Array(65);
      const fadeOut = new Float32Array(65);
      for (let index = 0; index < fadeIn.length; index += 1) {
        const angle = index / (fadeIn.length - 1) * Math.PI * 0.5;
        fadeIn[index] = Math.sin(angle);
        fadeOut[index] = Math.cos(angle);
      }
      gain.gain.setValueAtTime(0, when);
      gain.gain.setValueCurveAtTime(fadeIn, when, CROSSFADE);
      gain.gain.setValueAtTime(1, when + duration - CROSSFADE);
      gain.gain.setValueCurveAtTime(fadeOut, when + duration - CROSSFADE, CROSSFADE);
      const clip = { source, gain };
      layer.clips.add(clip);
      source.onended = () => {
        layer.clips.delete(clip);
        disconnect(source);
        disconnect(gain);
      };
      source.start(when, offset, duration);
      layer.nextAt = when + duration - CROSSFADE;
    };

    const pump = () => {
      if (!shouldPlay() || !context || context.state !== 'running') return;
      const now = context.currentTime;
      for (const layer of layers) {
        if (layer.retireAt && now >= layer.retireAt) stopLayer(layer);
      }
      if (currentLayer && currentLayer.nextAt <= now + 2) {
        // Timers can be throttled without notice. Do not schedule in the past.
        scheduleClip(currentLayer, Math.max(now + 0.025, currentLayer.nextAt));
      }
    };

    const beginLayer = (name, buffer) => {
      if (currentLayer && currentLayer.name === name) return;
      retireLayer(currentLayer);
      const gain = context.createGain();
      gain.connect(master);
      gain.gain.value = 1;
      const layer = { name, buffer, gain, clips: new Set(), nextAt: 0, retireAt: 0, lastOffset: -100 };
      layers.add(layer);
      currentLayer = layer;
      scheduleClip(layer, context.currentTime + 0.04);
      if (!scheduler) scheduler = window.setInterval(pump, 1000);
    };

    const activate = async () => {
      const token = ++epoch;
      window.clearTimeout(suspendTimer);
      if (!shouldPlay()) return false;
      try {
        if (!context) {
          const AudioContext = window.AudioContext || window.webkitAudioContext;
          if (!AudioContext) throw new Error('Ocean sound is not supported in this browser.');
          context = new AudioContext({ latencyHint: 'playback', sampleRate: 32000 });
          master = context.createGain();
          master.gain.value = 0;
          master.connect(context.destination);
        }
        const requestedScene = scene;
        // Resume immediately inside the sound-button gesture, before fetching.
        const resumed = context.resume();
        if (!buffers.has(requestedScene)) notify('loading');
        const [buffer] = await Promise.all([load(requestedScene), resumed]);
        if (token !== epoch || !shouldPlay()) return false;
        beginLayer(requestedScene, buffer);
        updateGain(2);
        notify('playing');
        return true;
      } catch (error) {
        if (disposed || token !== epoch) return false;
        enabled = false;
        stopLayers();
        if (context && context.state !== 'closed') context.suspend().catch(() => {});
        notify('error');
        onError(error);
        return false;
      }
    };

    const suspend = () => {
      epoch += 1;
      window.clearTimeout(suspendTimer);
      if (!context) return;
      updateGain(0.18);
      suspendTimer = window.setTimeout(() => {
        if (shouldPlay() || disposed) return;
        stopLayers();
        context.suspend().catch(() => {});
      }, 240);
    };

    return {
      get enabled() { return enabled; },
      async setEnabled(value) {
        if (disposed) return false;
        enabled = Boolean(value);
        if (enabled && shouldPlay()) return activate();
        suspend();
        notify(enabled ? 'suspended' : 'off');
        return enabled;
      },
      setVolume(value) {
        volume = clamp(value, volume);
        updateGain();
      },
      setScene(value) {
        const next = Object.prototype.hasOwnProperty.call(RECORDINGS, value) ? value : 'ocean';
        if (disposed || next === scene) return;
        scene = next;
        if (shouldPlay()) void activate();
      },
      setFade(value) {
        const wasPlaying = shouldPlay();
        fade = clamp(value, fade);
        if (disposed) return;
        if (wasPlaying && !shouldPlay()) suspend();
        else if (!wasPlaying && shouldPlay()) void activate();
        else updateGain(0.3);
      },
      setVisible(value) {
        const next = Boolean(value);
        if (disposed || next === visible) return;
        visible = next;
        if (shouldPlay()) void activate();
        else {
          suspend();
          if (enabled) notify('suspended');
        }
      },
      dispose() {
        if (disposed) return;
        disposed = true;
        enabled = false;
        epoch += 1;
        window.clearTimeout(suspendTimer);
        controllers.forEach((controller) => controller.abort());
        controllers.clear();
        stopLayers();
        buffers.clear();
        requests.clear();
        if (master) disconnect(master);
        if (context && context.state !== 'closed') context.close().catch(() => {});
        master = null;
      }
    };
  };

  window.OceanWaveAudio = { create };
})();
