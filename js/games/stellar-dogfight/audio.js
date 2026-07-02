(() => {
  "use strict";

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function createAudioController(options = {}) {
    let enabled = options.enabled !== false;
    let context = null;
    let master = null;
    let musicGain = null;
    let musicOscA = null;
    let musicOscB = null;
    let musicLfo = null;
    let musicLfoGain = null;
    let shotCooldown = 0;

    function ensureContext() {
      if (context) return true;
      const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextCtor) return false;
      context = new AudioContextCtor();
      master = context.createGain();
      master.gain.value = enabled ? 0.16 : 0;
      master.connect(context.destination);
      return true;
    }

    function now() {
      if (!context) return 0;
      return context.currentTime;
    }

    function resume() {
      if (!ensureContext()) return;
      if (context.state === "suspended") {
        context.resume().catch(() => {});
      }
    }

    function setEnabled(next) {
      enabled = !!next;
      if (!ensureContext()) return;
      const target = enabled ? 0.16 : 0;
      master.gain.setTargetAtTime(target, now(), 0.05);
      if (!enabled) {
        stopMusic();
      }
    }

    function setMode(mode) {
      if (!enabled) {
        stopMusic();
        return;
      }
      if (mode === "flight" || mode === "training") {
        startMusic();
      } else {
        stopMusic();
      }
    }

    function playTone(freq, duration, type, gainValue, whenOffset = 0, attack = 0.004, release = 0.03) {
      if (!enabled || !ensureContext()) return;
      resume();
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const startAt = now() + whenOffset;
      const peak = clamp(gainValue, 0, 0.35);
      oscillator.type = type || "triangle";
      oscillator.frequency.setValueAtTime(freq, startAt);
      gain.gain.setValueAtTime(0.0001, startAt);
      gain.gain.exponentialRampToValueAtTime(peak, startAt + attack);
      gain.gain.exponentialRampToValueAtTime(0.0001, startAt + duration + release);
      oscillator.connect(gain);
      gain.connect(master);
      oscillator.start(startAt);
      oscillator.stop(startAt + duration + release + 0.01);
    }

    function playNoiseBurst(duration = 0.06, gainValue = 0.03, highpassHz = 500) {
      if (!enabled || !ensureContext()) return;
      resume();
      const length = Math.max(1, Math.floor(context.sampleRate * duration));
      const buffer = context.createBuffer(1, length, context.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < length; i += 1) {
        data[i] = (Math.random() * 2 - 1) * (1 - i / length);
      }
      const src = context.createBufferSource();
      src.buffer = buffer;
      const filter = context.createBiquadFilter();
      filter.type = "highpass";
      filter.frequency.value = highpassHz;
      const gain = context.createGain();
      const startAt = now();
      gain.gain.setValueAtTime(0.0001, startAt);
      gain.gain.exponentialRampToValueAtTime(clamp(gainValue, 0, 0.25), startAt + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, startAt + duration);
      src.connect(filter);
      filter.connect(gain);
      gain.connect(master);
      src.start(startAt);
      src.stop(startAt + duration + 0.01);
    }

    function startMusic() {
      if (!enabled || !ensureContext()) return;
      resume();
      if (musicOscA || musicOscB) return;

      musicGain = context.createGain();
      musicGain.gain.value = 0.0001;
      musicGain.connect(master);

      musicOscA = context.createOscillator();
      musicOscA.type = "sine";
      musicOscA.frequency.value = 98;
      musicOscA.connect(musicGain);

      musicOscB = context.createOscillator();
      musicOscB.type = "triangle";
      musicOscB.frequency.value = 147;
      musicOscB.connect(musicGain);

      musicLfo = context.createOscillator();
      musicLfo.type = "sine";
      musicLfo.frequency.value = 0.14;
      musicLfoGain = context.createGain();
      musicLfoGain.gain.value = 0.06;
      musicLfo.connect(musicLfoGain);
      musicLfoGain.connect(musicGain.gain);

      const startAt = now();
      musicGain.gain.exponentialRampToValueAtTime(0.022, startAt + 0.5);
      musicOscA.start(startAt);
      musicOscB.start(startAt);
      musicLfo.start(startAt);
    }

    function stopMusic() {
      if (!context) return;
      if (!musicOscA && !musicOscB) return;
      const stopAt = now() + 0.22;
      if (musicGain) {
        musicGain.gain.exponentialRampToValueAtTime(0.0001, stopAt);
      }
      if (musicOscA) musicOscA.stop(stopAt + 0.03);
      if (musicOscB) musicOscB.stop(stopAt + 0.03);
      if (musicLfo) musicLfo.stop(stopAt + 0.03);
      setTimeout(() => {
        musicOscA = null;
        musicOscB = null;
        musicLfo = null;
        musicLfoGain = null;
        musicGain = null;
      }, 320);
    }

    function play(eventName, payload = {}) {
      if (!enabled) return;
      if (eventName === "shot") {
        const t = performance.now();
        if (t < shotCooldown) return;
        shotCooldown = t + 55;
        playTone(640, 0.028, "square", 0.05, 0, 0.001, 0.02);
        return;
      }
      if (eventName === "launch") {
        playTone(220, 0.08, "triangle", 0.09);
        playTone(330, 0.08, "triangle", 0.08, 0.05);
        playTone(440, 0.12, "triangle", 0.07, 0.1);
        return;
      }
      if (eventName === "ability") {
        playTone(520, 0.09, "sine", 0.07);
        playTone(760, 0.12, "sine", 0.06, 0.05);
        return;
      }
      if (eventName === "secondary") {
        playTone(300, 0.08, "triangle", 0.08);
        playTone(230, 0.1, "triangle", 0.06, 0.07);
        return;
      }
      if (eventName === "upgrade") {
        playTone(420, 0.06, "triangle", 0.06);
        playTone(560, 0.08, "triangle", 0.06, 0.05);
        return;
      }
      if (eventName === "player-hit") {
        playNoiseBurst(0.06, 0.04, 720);
        return;
      }
      if (eventName === "enemy-down") {
        const elite = !!payload.elite;
        playTone(elite ? 280 : 220, elite ? 0.12 : 0.08, "triangle", elite ? 0.08 : 0.05);
        if (elite) {
          playTone(360, 0.13, "triangle", 0.07, 0.05);
        }
        return;
      }
      if (eventName === "gameover") {
        playTone(190, 0.16, "sawtooth", 0.07);
        playTone(140, 0.22, "sawtooth", 0.06, 0.09);
        return;
      }
      if (eventName === "victory") {
        playTone(392, 0.1, "triangle", 0.07);
        playTone(523, 0.11, "triangle", 0.07, 0.06);
        playTone(659, 0.14, "triangle", 0.07, 0.12);
      }
    }

    return {
      resume,
      setEnabled,
      setMode,
      play
    };
  }

  window.STELLAR_DOGFIGHT_AUDIO = {
    createAudioController
  };
})();
