(function initProjectStarfallEngineAudio(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  const AUDIO_CUE_THROTTLE_SECONDS = 0.035;
  const MUSIC_LOOP_INITIAL_DELAY_MS = 20;
  const MUSIC_LOOP_STEP_DELAY_MS = 620;
  const MUSIC_LOOP_NOTE_SECONDS = 0.5;
  const MUSIC_LOOP_ATTACK_SECONDS = 0.04;
  const MUSIC_LOOP_RELEASE_SECONDS = 0.48;
  const MUSIC_LOOP_GAIN = 0.035;
  const MUSIC_LOOP_NOTES = Object.freeze([196, 246.94, 293.66, 329.63, 392, 329.63, 293.66, 246.94]);

  function getMusicLoopFrequency(step) {
    return MUSIC_LOOP_NOTES[Number(step) % MUSIC_LOOP_NOTES.length];
  }

  function canPlayAudioCue(cue, audio, options) {
    const state = audio && typeof audio === 'object' ? audio : {};
    return !!(cue && (state.sfxEnabled || state.enabled || (options && options.bypassEnabled)));
  }

  function createAudioCuePlayback(cue, intensity) {
    const source = cue && typeof cue === 'object' ? cue : {};
    const duration = clamp(Number(source.duration || 0.1), 0.03, 0.6);
    return {
      type: source.type || '',
      oscillatorType: source.type === 'chime' ? 'triangle' : 'sine',
      gainAmount: clamp(Number(source.gain || 0.04) * clamp(Number(intensity || 1), 0.4, 1.8), 0.005, 0.16),
      duration,
      frequency: Math.max(60, Number(source.frequency || 440)),
      filterFrequency: Math.max(80, Number(source.frequency || 220)),
      endFrequency: source.endFrequency ? Math.max(60, Number(source.endFrequency)) : 0
    };
  }

  function writeAudioNoiseEnvelope(data, random) {
    if (!data || typeof data.length !== 'number') return data;
    const nextRandom = typeof random === 'function' ? random : Math.random;
    const length = Math.max(1, data.length);
    for (let index = 0; index < length; index += 1) {
      data[index] = (nextRandom() * 2 - 1) * Math.pow(1 - index / length, 1.8);
    }
    return data;
  }

  const api = {
    AUDIO_CUE_THROTTLE_SECONDS,
    MUSIC_LOOP_INITIAL_DELAY_MS,
    MUSIC_LOOP_STEP_DELAY_MS,
    MUSIC_LOOP_NOTE_SECONDS,
    MUSIC_LOOP_ATTACK_SECONDS,
    MUSIC_LOOP_RELEASE_SECONDS,
    MUSIC_LOOP_GAIN,
    MUSIC_LOOP_NOTES,
    getMusicLoopFrequency,
    canPlayAudioCue,
    createAudioCuePlayback,
    writeAudioNoiseEnvelope
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.audio = Object.assign({}, modules.audio || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
