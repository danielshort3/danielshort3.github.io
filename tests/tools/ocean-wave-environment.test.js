'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const root = path.join(__dirname, '../..');
const source = fs.readFileSync(path.join(root, 'js/tools/ocean-wave-environment.js'), 'utf8');
const flush = async () => { for (let index = 0; index < 16; index++) await Promise.resolve(); };
const arrayBuffer = bytes => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const fixture = Buffer.concat([
  Buffer.from('#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n'),
  Buffer.from(Array.from({ length: 8 }, (_, index) => [64 + index * 4, 80 + index * 3, 96 + index * 2, 128]).flat()),
]);

function harness({ floating = true, half = true, maximumSize = 4096, respectAbort = true } = {}) {
  const requests = [];
  const changes = [];
  const errors = [];
  const textures = [];
  const uploads = [];
  const filters = [];
  let currentTexture = null;
  let lost = false;
  const gl = {
    MAX_TEXTURE_SIZE: 1, TEXTURE_BINDING_2D: 2, UNPACK_FLIP_Y_WEBGL: 3, UNPACK_PREMULTIPLY_ALPHA_WEBGL: 4,
    TEXTURE_2D: 5, TEXTURE_MIN_FILTER: 6, TEXTURE_MAG_FILTER: 7, TEXTURE_WRAP_S: 8, TEXTURE_WRAP_T: 9,
    LINEAR: 10, LINEAR_MIPMAP_LINEAR: 11, REPEAT: 12, CLAMP_TO_EDGE: 13, RGBA: 14, FLOAT: 15, UNSIGNED_BYTE: 16, HALF_FLOAT: 17,
    NO_ERROR: 0,
    getExtension: name => floating && (half || !name.includes('half_float')) ? { HALF_FLOAT_OES: 17 } : null,
    getParameter: name => name === gl.MAX_TEXTURE_SIZE ? maximumSize : name === gl.TEXTURE_BINDING_2D ? currentTexture : false,
    getError: () => 0,
    isContextLost: () => lost,
    createTexture() { const texture = { id: textures.length, deleted: 0 }; textures.push(texture); return texture; },
    deleteTexture(texture) { texture.deleted++; },
    bindTexture(type, texture) { currentTexture = texture; },
    pixelStorei() {},
    texParameteri(type, name, value) { filters.push({ texture: currentTexture, name, value }); },
    texImage2D(type, level, format, width, height, border, layout, storage, pixels) {
      uploads.push({ texture: currentTexture, width, height, storage, pixels });
    },
    generateMipmap() { currentTexture.mipmaps = true; },
  };
  const window = {};
  vm.runInNewContext(source, {
    window, AbortController, Float32Array, Uint8Array,
    fetch(url, options) {
      return new Promise((resolve, reject) => {
        const request = { url, signal: options.signal, reject,
          resolve: () => resolve({ ok: true, headers: { get: () => String(fixture.length) }, arrayBuffer: async () => arrayBuffer(fixture) }),
        };
        requests.push(request);
        if (respectAbort) options.signal.addEventListener('abort', () => reject(new Error('Aborted')));
      });
    },
  });
  const environment = window.OceanWaveEnvironment.create(gl, { onChange: sky => changes.push(sky), onError: error => errors.push(error) });
  return { environment, requests, changes, errors, textures, uploads, filters, gl, lose: () => { lost = true; } };
}

(async () => {
  const app = harness();
  const selected = app.environment.select('golden', 'ultra');
  assert.match(app.requests[0].url, /_1k\.hdr$/);
  app.requests[0].resolve();
  await flush();
  assert.equal(app.changes.length, 1, 'The lightweight sky is visible while Ultra detail loads.');
  assert.match(app.requests[1].url, /_4k\.hdr$/);
  for (let frame = 0; frame < 60; frame++) await app.environment.select('golden', 'ultra');
  assert.equal(app.requests.length, 2, 'Repeated renderer selections never duplicate pending requests.');
  app.requests[1].resolve();
  assert.equal(await selected, true);
  assert.equal(app.environment.current.resolution, '4k');
  assert.equal(app.changes.length, 2);
  assert.equal(app.environment.current.textureEncoding, 0);
  assert.equal(app.environment.current.halfTexture, true);
  assert.equal(app.uploads[0].storage, app.gl.HALF_FLOAT);
  assert.equal(app.uploads[0].pixels.BYTES_PER_ELEMENT, 2, 'Half-float panorama uploads halve GPU memory without clipping HDR radiance.');
  assert.equal(app.uploads[0].pixels[3], 0x3c00, 'Half conversion preserves opaque alpha exactly.');
  assert.ok(app.textures.every(texture => texture.mipmaps), 'Both sky resolutions have filtered mip levels.');
  assert.ok(app.filters.some(filter => filter.name === app.gl.TEXTURE_MIN_FILTER && filter.value === app.gl.LINEAR_MIPMAP_LINEAR));
  const sharpSky = app.environment.current;
  app.environment.releaseUnused();
  assert.equal(sharpSky.texture.deleted, 0, 'The selected high-resolution sky remains alive without renderer references.');
  for (const quality of ['low', 'auto', 'high']) {
    await app.environment.select('dawn', quality);
    assert.equal(app.environment.current.resolution, '1k', `${quality} restores the lightweight sky even when Ultra is cached.`);
    await app.environment.select('golden', 'ultra');
    assert.equal(app.environment.current, sharpSky, 'The exact requested resolution remains reusable while retained.');
  }
  await app.environment.select('dawn', 'low');
  assert.equal(app.requests.length, 2, 'Dawn and golden hour share the same cached panorama.');
  app.environment.releaseUnused([sharpSky, app.environment.current, null]);
  assert.equal(sharpSky.texture.deleted, 0, 'A renderer reference preserves the old sky during its crossfade.');
  app.environment.releaseUnused();
  assert.equal(sharpSky.texture.deleted, 1, 'Completing the crossfade releases the unreferenced Ultra texture.');
  assert.equal(app.environment.current.texture.deleted, 0);
  app.environment.releaseUnused();
  assert.equal(sharpSky.texture.deleted, 1, 'Repeated cache sweeps do not delete a texture twice.');
  const reselected = app.environment.select('golden', 'ultra');
  await flush();
  assert.equal(app.requests.length, 3, 'Returning to Ultra reloads its released detail without reloading the 1k sky.');
  assert.match(app.requests[2].url, /_4k\.hdr$/);
  app.requests[2].resolve();
  await reselected;
  assert.notEqual(app.environment.current, sharpSky);
  app.environment.dispose();
  assert.ok(app.textures.every(texture => texture.deleted === 1), 'Disposal releases every cached low/high resolution texture exactly once.');

  const race = harness();
  const stale = race.environment.select('golden', 'ultra');
  const latest = race.environment.select('daylight', 'high');
  race.requests[0].resolve();
  await flush();
  assert.equal(await stale, false);
  assert.equal(race.requests.length, 2, 'A superseded Ultra mood does not start its optional large download.');
  race.requests[1].resolve();
  await latest;
  assert.equal(race.changes.length, 1);
  assert.match(race.environment.current.assetId, /kloofendal/);
  race.environment.dispose();

  const cancelled = harness();
  const cancelledSelection = cancelled.environment.select('golden', 'ultra');
  cancelled.requests[0].resolve();
  await flush();
  await cancelled.environment.select('golden', 'low');
  assert.equal(cancelled.requests[1].signal.aborted, true, 'A quality downgrade stops its obsolete large download.');
  assert.equal(await cancelledSelection, false);
  assert.equal(cancelled.environment.current.resolution, '1k');
  assert.equal(cancelled.uploads.length, 1);
  assert.equal(cancelled.errors.length, 0, 'Cancellation is not reported as a sky failure.');
  cancelled.environment.dispose();

  const late = harness({ respectAbort: false });
  const obsolete = late.environment.select('golden', 'ultra');
  late.requests[0].resolve();
  await flush();
  await late.environment.select('golden', 'low');
  const replacement = late.environment.select('golden', 'ultra');
  await flush();
  assert.equal(late.requests.length, 3, 'A new Ultra request can start before its aborted predecessor settles.');
  assert.equal(late.requests[2].signal.aborted, false);
  late.requests[1].resolve();
  assert.equal(await obsolete, false);
  assert.equal(late.uploads.length, 1, 'A late aborted response cannot decode and upload obsolete Ultra detail.');
  const differentMood = late.environment.select('daylight', 'high');
  assert.equal(late.requests[2].signal.aborted, true, 'The stale request finally handler cannot remove its pending replacement.');
  late.requests[2].resolve();
  assert.equal(await replacement, false);
  late.requests[3].resolve();
  assert.equal(await differentMood, true);
  assert.equal(late.uploads.length, 2, 'Only the two requested 1k skies were uploaded.');
  assert.equal(late.errors.length, 0);
  late.environment.dispose();

  const shared = harness();
  const sharedSelection = shared.environment.select('golden', 'ultra');
  shared.requests[0].resolve();
  await flush();
  await shared.environment.select('dawn', 'ultra');
  assert.equal(shared.requests.length, 2, 'Moods using the same resolution and asset share their pending upgrade.');
  assert.equal(shared.requests[1].signal.aborted, false);
  shared.requests[1].resolve();
  assert.equal(await sharedSelection, true);
  shared.environment.dispose();

  const repeated = harness();
  const firstMood = repeated.environment.select('golden', 'high');
  const secondMood = repeated.environment.select('daylight', 'high');
  const returnedMood = repeated.environment.select('dawn', 'low');
  assert.equal(repeated.requests.length, 2, 'Returning to the same pending 1k asset does not duplicate its download.');
  repeated.requests[0].resolve();
  assert.equal(await firstMood, false);
  assert.equal(await returnedMood, true);
  repeated.requests[1].resolve();
  assert.equal(await secondMood, false);
  assert.equal(repeated.changes.length, 1, 'A late response cannot replace the latest selected mood.');
  await repeated.environment.select('daylight', 'auto');
  assert.equal(repeated.requests.length, 2, 'Small sky downloads remain cached after a superseding selection.');
  repeated.environment.dispose();

  const blending = harness();
  const retainedSkies = [];
  for (const mood of ['golden', 'daylight', 'dusk']) {
    const requestIndex = blending.requests.length;
    const selection = blending.environment.select(mood, 'ultra');
    blending.requests[requestIndex].resolve();
    await flush();
    blending.requests[requestIndex + 1].resolve();
    await selection;
    retainedSkies.push(blending.environment.current);
  }
  blending.environment.releaseUnused(retainedSkies);
  assert.ok(retainedSkies.every(sky => sky.texture.deleted === 0), 'Renderer previous/current/pending references can coexist during queued crossfades.');
  blending.environment.releaseUnused([retainedSkies[1]]);
  assert.equal(retainedSkies[0].texture.deleted, 1, 'Superseded detail is released even when other skies remain in use.');
  assert.equal(retainedSkies[1].texture.deleted, 0, 'The active renderer fade reference stays valid.');
  assert.equal(retainedSkies[2].texture.deleted, 0, 'The latest selected sky stays valid while queued for rendering.');
  blending.environment.releaseUnused();
  assert.equal(retainedSkies[1].texture.deleted, 1);
  blending.environment.dispose();
  assert.ok(blending.textures.every(texture => texture.deleted === 1), 'Eviction and disposal release all sky textures exactly once.');

  const upgradeFailure = harness();
  const upgrade = upgradeFailure.environment.select('golden', 'ultra');
  upgradeFailure.requests[0].resolve();
  await flush();
  upgradeFailure.requests[1].reject(new Error('Offline'));
  assert.equal(await upgrade, true);
  assert.equal(upgradeFailure.environment.current.resolution, '1k', 'Optional detail failures preserve the working sky.');
  assert.equal(upgradeFailure.errors.length, 0);
  upgradeFailure.environment.dispose();

  const compatibility = harness({ floating: false, maximumSize: 1024 });
  const compatible = compatibility.environment.select('dusk', 'ultra');
  compatibility.requests[0].resolve();
  await compatible;
  assert.equal(compatibility.requests.length, 1, 'GPU texture limits prevent unsupported 2K upgrades.');
  assert.equal(compatibility.environment.current.textureEncoding, 1, 'Byte textures advertise gamma decoding for smooth dark gradients.');
  assert.equal(compatibility.environment.current.textureScale, 8);
  assert.equal(compatibility.uploads[0].storage, compatibility.gl.UNSIGNED_BYTE);
  assert.ok(compatibility.uploads[0].pixels[0] > 64, 'The byte path retains dark sky precision rather than dividing linear radiance into coarse steps.');
  compatibility.environment.dispose();

  const closing = harness();
  const pending = closing.environment.select('daylight', 'ultra');
  closing.environment.dispose();
  assert.equal(await pending, false);
  assert.equal(closing.requests[0].signal.aborted, true);
  assert.equal(closing.uploads.length, 0);
  const limited = harness({ half: false, maximumSize: 2048 });
  const limitedSelection = limited.environment.select('golden', 'ultra');
  limited.requests[0].resolve();
  await flush();
  assert.match(limited.requests[1].url, /_2k\.hdr$/, 'A GPU limited to 2048 uses the supported sunset asset.');
  limited.requests[1].resolve();
  await limitedSelection;
  assert.equal(limited.environment.current.halfTexture, false);
  assert.equal(limited.uploads[0].storage, limited.gl.FLOAT, 'Float32 storage remains available without half-float texture support.');
  await limited.environment.select('golden', 'high');
  assert.equal(limited.environment.current.resolution, '1k', 'A cached 2k fallback also downgrades when leaving Ultra.');
  limited.environment.dispose();
  const loss = harness();
  const lost = loss.environment.select('golden', 'ultra');
  loss.lose();
  loss.requests[0].resolve();
  assert.equal(await lost, false);
  assert.equal(loss.uploads.length, 0, 'A lost context cannot receive late HDR uploads.');
  loss.environment.dispose();

  const scope = { window: {} };
  vm.runInNewContext(source.replace('window.OceanWaveEnvironment = { create };', 'window.OceanWaveEnvironment = { create, decode, prepare, ASSETS };'), scope);
  const api = scope.window.OceanWaveEnvironment;
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'img/games/ocean/sources.json'), 'utf8'));
  const prepared = new Map();
  for (const asset of manifest.assets) {
    const bytes = fs.readFileSync(path.join(root, 'img/games/ocean', asset.file));
    assert.equal(bytes.length, asset.bytes);
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), asset.sha256);
    const config = asset.file.startsWith('kloppenheim') ? api.ASSETS.golden : asset.file.startsWith('kloofendal') ? api.ASSETS.daylight : api.ASSETS.dusk;
    const sky = api.prepare(api.decode(arrayBuffer(bytes)), config);
    assert.equal(sky.width, asset.file.endsWith('_4k.hdr') ? 4096 : asset.file.endsWith('_2k.hdr') ? 2048 : 1024);
    assert.ok(sky.data.every(Number.isFinite));
    assert.ok(sky.sunRadiance.every(value => Number.isFinite(value) && value >= 0));
    prepared.set(asset.file, { sunDirection: sky.sunDirection, sunRadiance: sky.sunRadiance });
  }
  const low = prepared.get('kloppenheim_06_puresky_1k.hdr');
  const high = prepared.get('kloppenheim_06_puresky_2k.hdr');
  assert.deepEqual(low.sunDirection, high.sunDirection, 'A resolution upgrade cannot relocate the sunset to a different bright cloud.');
  assert.ok(high.sunRadiance[0] / low.sunRadiance[0] > .85 && high.sunRadiance[0] / low.sunRadiance[0] < 1.15, 'Extracted solar energy remains consistent between panorama resolutions.');
  assert.ok(high.sunRadiance[0] > high.sunRadiance[2] * 5, 'The sunset retains its photographed warm solar spectrum.');
  assert.ok(prepared.get('kloofendal_48d_partly_cloudy_puresky_2k.hdr').sunRadiance[0] > 1000, 'Daylight sunlight retains HDR energy instead of clipping to 50.');
  assert.deepEqual(Array.from(prepared.get('qwantani_dusk_1_puresky_2k.hdr').sunRadiance), [0, 0, 0], 'The sun stays below the horizon in the dusk photograph.');
  console.log('Ocean HDR sky tests passed: progressive quality, downgrades, texture eviction, cancellation races, mipmaps, solar extraction, and disposal.');
})().catch(error => { console.error(error); process.exitCode = 1; });
