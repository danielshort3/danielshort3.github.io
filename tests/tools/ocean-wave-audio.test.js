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
    setValueCurveAtTime(curve, time, duration) { this.events.push(['curve', curve, time, duration]); }
  });
  class AudioContext {
    constructor() {
      this.currentTime = 0;
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
      const node = { gain: parameter(), connect() {}, disconnect() { this.disconnected = true; } };
      this.gains.push(node);
      return node;
    }
    createBufferSource() {
      const node = {
        connect() {}, disconnect() { this.disconnected = true; },
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
  assert.equal(lazy.contexts.length, 0, 'Loading a page and selecting a scene never creates an audio context.');
  assert.equal(lazy.pending.length, 0, 'Recordings are not downloaded before Sound is enabled.');
  const enabling = lazy.audio.setEnabled(true);
  assert.equal(lazy.contexts.length, 1);
  assert.equal(lazy.contexts[0].resumeCount, 1, 'Resume occurs synchronously within the user gesture.');
  await lazy.audio.setEnabled(false);
  lazy.pending[0].resolve();
  assert.equal(await enabling, false);
  assert.equal(lazy.contexts[0].sources.length, 0, 'Disabling during loading cannot produce late playback.');
  lazy.runSuspend();
  assert.equal(lazy.contexts[0].state, 'suspended');
  assert.equal(await lazy.audio.setEnabled(true), true);
  assert.equal(lazy.pending.length, 1, 'Successfully decoded recordings are cached.');
  assert.equal(lazy.contexts.length, 1, 'The unlocked context is reused.');
  const context = lazy.contexts[0];
  const first = context.sources[0];
  assert.ok(first.started[2] >= 48 && first.started[2] <= 72);
  assert.ok(first.started[1] + first.started[2] <= first.buffer.duration);
  lazy.tick(first.started[2] - 7);
  const second = context.sources[1];
  assert.equal(second.started[0], first.started[0] + first.started[2] - 6, 'Recorded passages overlap by six seconds.');
  const envelopes = context.gains[2].gain.events.filter(event => event[0] === 'curve');
  const middlePower = envelopes[0][1][32] ** 2 + envelopes[1][1][32] ** 2;
  assert.ok(Math.abs(middlePower - 1) < 1e-6, 'Crossfades preserve the energy of uncorrelated recordings.');
  lazy.audio.setVisible(false);
  lazy.runSuspend();
  assert.equal(context.state, 'suspended');
  assert.ok(context.sources.every(node => node.stopped && node.disconnected));
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
  race.pending[0].resolve();
  await initial;
  race.audio.setScene('cove');
  race.audio.setScene('ocean');
  await flush();
  race.pending[1].resolve();
  await flush();
  assert.ok(race.contexts[0].sources.every(node => node.buffer.name === 'ocean'), 'A stale scene response cannot replace the latest scene.');
  race.audio.dispose();

  const closing = harness();
  const late = closing.audio.setEnabled(true);
  closing.audio.dispose();
  await late;
  assert.equal(closing.pending[0].signal.aborted, true);
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
  console.log('Ocean recorded audio tests passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
