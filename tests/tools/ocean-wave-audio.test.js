'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const root = path.join(__dirname, '../..');
const source = fs.readFileSync(path.join(root, 'js/tools/ocean-wave-audio.js'), 'utf8');
const flush = async () => { for (let index = 0; index < 12; index += 1) await Promise.resolve(); };

function harness() {
  const contexts = [];
  const pending = [];
  const timeouts = new Map();
  const intervals = new Map();
  let nextTimer = 0;
  const parameter = () => ({
    value: 0, events: [],
    cancelAndHoldAtTime(time) { this.events.push(['hold', time]); },
    setValueAtTime(value, time) { this.events.push(['value', value, time]); },
    linearRampToValueAtTime(value, time) { this.events.push(['ramp', value, time]); },
    setTargetAtTime(value, time, constant) { this.events.push(['target', value, time, constant]); },
    setValueCurveAtTime(curve, time, duration) { this.events.push(['curve', curve, time, duration]); }
  });
  class AudioContext {
    constructor() {
      this.currentTime = 0;
      this.sampleRate = 32000;
      this.state = 'suspended';
      this.destination = {};
      this.sources = [];
      this.gains = [];
      this.resumeCount = 0;
      contexts.push(this);
    }
    resume() { this.resumeCount += 1; this.state = 'running'; return Promise.resolve(); }
    suspend() { this.state = 'suspended'; return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    async decodeAudioData(bytes) {
      const name = new Uint8Array(bytes)[0] === 1 ? 'cove' : 'ocean';
      return { duration: name === 'cove' ? 98 : 175, numberOfChannels: 2, name };
    }
    createGain() {
      const node = { gain: parameter(), connect(target) { this.output = target; }, disconnect() { this.disconnected = true; } };
      this.gains.push(node);
      return node;
    }
    createBiquadFilter() {
      return { frequency: parameter(), Q: parameter(), connect(target) { this.output = target; }, disconnect() { this.disconnected = true; } };
    }
    createBuffer(channels, length, sampleRate) {
      const data = new Float32Array(length);
      return { name: 'wind', numberOfChannels: channels, duration: length / sampleRate, getChannelData: () => data };
    }
    createBufferSource() {
      const node = {
        connect(target) { this.output = target; }, disconnect() { this.disconnected = true; },
        start(...args) { this.started = args; }, stop() { this.stopped = true; }
      };
      this.sources.push(node);
      return node;
    }
  }
  const window = {
    AudioContext,
    atob,
    setTimeout(callback, delay) { const id = ++nextTimer; timeouts.set(id, { callback, delay }); return id; },
    clearTimeout(id) { timeouts.delete(id); },
    setInterval(callback) { const id = ++nextTimer; intervals.set(id, callback); return id; },
    clearInterval(id) { intervals.delete(id); }
  };
  vm.runInNewContext(source, {
    window, AbortController, Float32Array,
    fetch(url, options) {
      return new Promise((resolve, reject) => {
        const name = url.includes('quiet-beach') ? 'cove' : 'ocean';
        const data = Buffer.from(new Uint8Array(2048).fill(name === 'cove' ? 1 : 2)).toString('base64');
        pending.push({ name, resolve: (overrides = {}) => resolve({ ok: true, status: 200, json: async () => ({ encoding: 'base64', mediaType: 'audio/mpeg', data }), ...overrides }), reject, signal: options.signal });
        options.signal.addEventListener('abort', () => reject(new Error('Aborted')));
      });
    }
  });
  const statuses = [];
  const errors = [];
  const audio = window.OceanWaveAudio.create({ onStatus: value => statuses.push(value), onError: error => errors.push(error) });
  return {
    audio, contexts, pending, statuses, errors, intervals, timeouts,
    tick(seconds) {
      contexts.forEach(context => { context.currentTime += seconds; });
      intervals.forEach(callback => callback());
    },
    runSuspend() {
      for (const [id, timer] of [...timeouts]) {
        if (timer.delay === 240) { timeouts.delete(id); timer.callback(); }
      }
    }
  };
}

(async () => {
  const lazy = harness();
  lazy.audio.setVolume(0.4);
  lazy.audio.setScene('cove');
  lazy.audio.setConditions({ wind: 10, waveHeight: 2.5, shore: 0.8 });
  assert.equal(lazy.contexts.length, 0, 'Loading a page and selecting a scene never creates an audio context.');
  assert.equal(lazy.pending.length, 0, 'Recordings are not downloaded before Sound is enabled.');
  const enabling = lazy.audio.setEnabled(true);
  assert.equal(lazy.contexts.length, 1);
  assert.equal(lazy.contexts[0].resumeCount, 1, 'Resume occurs synchronously within the user gesture.');
  await lazy.audio.setEnabled(false);
  lazy.pending.forEach(request => request.resolve());
  assert.equal(await enabling, false);
  assert.equal(lazy.contexts[0].sources.length, 0, 'Disabling during loading cannot produce late playback.');
  lazy.runSuspend();
  assert.equal(lazy.contexts[0].state, 'suspended');
  assert.equal(await lazy.audio.setEnabled(true), true);
  assert.equal(lazy.pending.length, 2, 'Both successfully decoded recordings are cached.');
  assert.equal(lazy.contexts.length, 1, 'The unlocked context is reused.');
  const context = lazy.contexts[0];
  const first = context.sources[0];
  assert.equal(context.sources.length, 3, 'Two recording beds and a single quiet wind loop are active.');
  assert.ok(first.started[2] >= 48 && first.started[2] <= 72);
  assert.ok(first.started[1] + first.started[2] <= first.buffer.duration);
  lazy.tick(first.started[2] - 7);
  const second = context.sources.find(node => node !== first && node.buffer.name === first.buffer.name);
  assert.equal(second.started[0], first.started[0] + first.started[2] - 6, 'Recorded passages overlap by six seconds.');
  const envelopes = context.gains[2].gain.events.filter(event => event[0] === 'curve');
  const middlePower = envelopes[0][1][32] ** 2 + envelopes[1][1][32] ** 2;
  assert.ok(Math.abs(middlePower - 1) < 1e-6, 'Crossfades preserve the energy of uncorrelated recordings.');
  lazy.audio.setVisible(false);
  lazy.runSuspend();
  assert.equal(context.state, 'suspended');
  assert.ok(context.sources.every(node => node.stopped && node.disconnected));
  assert.ok(context.gains.every(node => node === context.gains[0] || node.disconnected), 'All recording and wind gains are disconnected on suspension.');
  assert.equal(lazy.intervals.size, 0, 'Offscreen playback does not leave the scheduler running.');
  lazy.audio.setVisible(true);
  await flush();
  assert.equal(context.state, 'running');
  assert.equal(lazy.audio.enabled, true);
  lazy.audio.setFade(0);
  lazy.runSuspend();
  assert.equal(context.state, 'suspended', 'A completed timer releases active playback work.');
  lazy.audio.setFade(1);
  await flush();
  assert.equal(context.state, 'running');
  lazy.audio.dispose();
  assert.equal(context.state, 'closed');
  assert.equal(lazy.intervals.size, 0);
  assert.equal(lazy.timeouts.size, 0);
  assert.equal(await lazy.audio.setEnabled(true), false);

  const race = harness();
  const initial = race.audio.setEnabled(true);
  race.audio.setScene('cove');
  race.audio.setConditions({ wind: 1, waveHeight: 0.2, shore: 1 });
  race.audio.setScene('ocean');
  race.pending.forEach(request => request.resolve());
  assert.equal(await initial, true);
  const mixedContext = race.contexts[0];
  const ocean = mixedContext.sources.find(node => node.buffer.name === 'ocean');
  const surf = mixedContext.sources.find(node => node.buffer.name === 'cove');
  const wind = mixedContext.sources.find(node => node.buffer.name === 'wind');
  const target = parameter => parameter.events.filter(event => event[0] === 'target').at(-1)[1];
  const oceanGain = ocean.output.output.gain;
  const surfGain = surf.output.output.gain;
  const windGain = wind.output.output.output.gain;
  const calmSurf = target(surfGain);
  const calmWind = target(windGain);
  const calmFrequency = target(wind.output.frequency);
  assert.ok(target(oceanGain) > calmSurf * 5, 'Scene changes during loading apply the latest open-water mix.');
  race.audio.setConditions({ wind: 20, waveHeight: 5, shore: 0 });
  const roughSurf = target(surfGain);
  assert.ok(roughSurf > calmSurf * 3, 'Increasing swell brings the surf recording forward.');
  assert.ok(target(windGain) > calmWind, 'Stronger breeze gradually raises the quiet wind layer.');
  assert.ok(target(wind.output.frequency) > calmFrequency, 'Wind gets a little brighter with stronger breeze.');
  race.audio.setScene('cove');
  race.audio.setConditions({ shore: 0 });
  const distantOcean = target(oceanGain);
  race.audio.setConditions({ shore: 1 });
  assert.ok(target(surfGain) > roughSurf * 2, 'Moving toward shore brings the surf recording closer.');
  assert.ok(target(oceanGain) < distantOcean, 'Open water recedes when shore wash comes forward.');
  assert.ok(target(oceanGain) + target(surfGain) <= 0.75, 'Recording gains retain conservative headroom at maximum conditions.');
  assert.ok(target(windGain) <= 0.12, 'Procedural wind remains below the recording beds.');
  assert.equal(mixedContext.sources.length, 3, 'Condition and scene updates do not restart passages or create extra audio nodes.');
  assert.equal(race.pending.length, 2, 'Condition changes do not download more audio.');
  const beforeInvalid = [target(oceanGain), target(surfGain), target(windGain)];
  race.audio.setConditions({ wind: NaN, waveHeight: Infinity, shore: null });
  race.audio.setConditions(null);
  race.audio.setConditions({ wind: '', waveHeight: true, shore: undefined });
  assert.deepEqual([target(oceanGain), target(surfGain), target(windGain)], beforeInvalid, 'Malformed conditions preserve the current mix.');
  race.audio.setConditions({ wind: -1, waveHeight: -3, shore: 12 });
  assert.equal(target(windGain), 0, 'Zero breeze silences the procedural layer.');
  for (const gain of [oceanGain, surfGain, windGain]) {
    assert.ok(gain.events.filter(event => event[0] === 'target').every(event => event[1] >= 0 && Number.isFinite(event[1]) && event[3] >= 0.8), 'Mix changes use finite, gradual targets.');
  }
  race.audio.setVolume(0);
  assert.equal(mixedContext.gains[0].gain.events.at(-1)[1], 0, 'Master volume silences recordings and procedural wind together.');
  const samples = wind.buffer.getChannelData(0);
  assert.ok(samples.every(Number.isFinite));
  assert.ok(Math.abs(samples[0] - samples.at(-1)) <= 0.2, 'The generated wind loop avoids a large seam discontinuity.');
  assert.equal(wind.loop, true);
  race.audio.dispose();
  assert.ok([wind, wind.output, wind.output.output, wind.output.output.output].every(node => node.disconnected), 'Wind source and both filters are disconnected on disposal.');

  const closing = harness();
  const late = closing.audio.setEnabled(true);
  closing.audio.dispose();
  await late;
  assert.ok(closing.pending.every(request => request.signal.aborted));
  assert.equal(closing.contexts[0].sources.length, 0);
  assert.equal(closing.errors.length, 0, 'Closing during a fetch is quiet.');
  assert.equal(closing.timeouts.size, 0);

  const failed = harness();
  const failedStart = failed.audio.setEnabled(true);
  failed.pending[0].reject(new Error('Offline'));
  assert.equal(await failedStart, false);
  assert.equal(failed.audio.enabled, false);
  assert.equal(failed.errors.length, 1);
  assert.equal(failed.contexts[0].state, 'suspended');
  assert.equal(failed.pending[1].signal.aborted, true, 'A failed recording also cancels the other pending recording download.');
  failed.audio.dispose();

  const intercepted = harness();
  const interceptedStart = intercepted.audio.setEnabled(true);
  intercepted.pending[0].resolve({ status: 204 });
  assert.equal(await interceptedStart, false);
  assert.equal(intercepted.errors.length, 1);
  assert.match(intercepted.errors[0].message, /could not be loaded/);
  assert.equal(intercepted.contexts[0].sources.length, 0, 'Empty intercepted responses do not reach the audio decoder.');
  intercepted.audio.dispose();

  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'audio/ocean/sources.json'), 'utf8'));
  assert.equal(manifest.license, 'CC0 1.0 Universal');
  for (const recording of manifest.recordings) {
    const bytes = fs.readFileSync(path.join(root, 'audio/ocean', recording.file));
    assert.equal(bytes.length, recording.bytes);
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), recording.sha256, 'Served recordings match their provenance manifest.');
    const payload = JSON.parse(bytes);
    const recordingBytes = Buffer.from(payload.data, 'base64');
    assert.equal(payload.mediaType, 'audio/mpeg');
    assert.equal(crypto.createHash('sha256').update(recordingBytes).digest('hex'), recording.audioSha256);
  }
  console.log('Ocean condition-responsive audio tests passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
