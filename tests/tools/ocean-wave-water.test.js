'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = name => fs.readFileSync(path.join(__dirname, '../../js/tools', name), 'utf8');
const spectrumSource = source('ocean-wave-spectrum.js');
const shaderSource = source('ocean-wave-shaders.js');
const window = {};
const sandbox = vm.createContext({ window, Float32Array, Math, Number, Object });
vm.runInContext(spectrumSource, sandbox);
vm.runInContext(shaderSource, sandbox);

const mockGpu = ({ linear = true, framebufferComplete = true, compile = true } = {}) => {
  const created = new Set();
  const deleted = new Set();
  const uniforms = {};
  const bindings = new Map();
  const uploads = [];
  const draws = [];
  let active = 100;
  let target;
  let program;
  let lost = false;
  let sequence = 0;
  const allocate = kind => { const resource = { id: ++sequence, kind }; created.add(resource); return resource; };
  const gl = {
    TEXTURE0: 100, TEXTURE1: 101, TEXTURE2: 102, TEXTURE_2D: 1,
    FRAMEBUFFER: 2, FRAMEBUFFER_COMPLETE: 3, COLOR_ATTACHMENT0: 4,
    NO_ERROR: 0, FLOAT: 5, RGBA: 6, LINEAR: 7, NEAREST: 8,
    TEXTURE_MIN_FILTER: 9, TEXTURE_MAG_FILTER: 10, TEXTURE_WRAP_S: 11, TEXTURE_WRAP_T: 12,
    REPEAT: 13, CLAMP_TO_EDGE: 14, LINEAR_MIPMAP_LINEAR: 15,
    FRAGMENT_SHADER: 16, VERTEX_SHADER: 17, HIGH_FLOAT: 18, COMPILE_STATUS: 19,
    LINK_STATUS: 20, ARRAY_BUFFER: 21, STATIC_DRAW: 22, TRIANGLES: 23,
    isContextLost: () => lost,
    getShaderPrecisionFormat: () => ({ precision: 23 }),
    getExtension: name => name === 'OES_texture_float_linear' && !linear ? null : {},
    getParameter: () => null,
    getError: () => 0,
    createTexture: () => allocate('texture'),
    createProgram: () => allocate('program'),
    createShader: () => allocate('shader'),
    createFramebuffer: () => allocate('framebuffer'),
    createBuffer: () => allocate('buffer'),
    deleteTexture: resource => deleted.add(resource),
    deleteProgram: resource => deleted.add(resource),
    deleteShader: resource => deleted.add(resource),
    deleteFramebuffer: resource => deleted.add(resource),
    deleteBuffer: resource => deleted.add(resource),
    getShaderParameter: () => compile,
    getProgramParameter: () => true,
    getShaderInfoLog: () => 'Deliberate shader failure',
    checkFramebufferStatus: () => framebufferComplete ? 3 : -1,
    shaderSource: (shader, text) => { shader.source = text; },
    attachShader: (resource, shader) => { if (shader.source.includes('uPrevious')) resource.foam = true; },
    getUniformLocation: (_program, name) => name,
    uniform1i: (name, a) => { uniforms[name] = a; },
    uniform1f: (name, a) => { uniforms[name] = a; },
    uniform2f: (name, a, b) => { uniforms[name] = [a, b]; },
    activeTexture: unit => { active = unit; },
    bindTexture: (_kind, texture) => { bindings.set(active, texture); },
    useProgram: resource => { program = resource; },
    framebufferTexture2D: (_fb, _attachment, _kind, texture) => { target = texture; },
    texSubImage2D: (...args) => uploads.push({ texture: bindings.get(active), values: new Float32Array(args[8]) }),
    drawArrays: () => {
      if (program?.foam) draws.push({ target, previous: bindings.get(100), uniforms: structuredClone(uniforms) });
    },
  };
  for (const name of ['bindFramebuffer', 'bindBuffer', 'compileShader', 'bindAttribLocation', 'linkProgram',
    'texParameteri', 'texImage2D', 'generateMipmap', 'bufferData', 'disable', 'colorMask', 'viewport',
    'enableVertexAttribArray', 'vertexAttribPointer']) gl[name] = () => {};
  return { gl, created, deleted, uploads, draws, lose: () => { lost = true; } };
};

for (const linear of [true, false]) {
  const gpu = mockGpu({ linear });
  const spectrum = window.OceanWaveSpectrum.create(gpu.gl);
  assert.ok(spectrum, 'The spectrum and foam must initialize with or without float-linear filtering.');
  const descriptor = spectrum.foam;
  const resources = gpu.created.size;
  spectrum.setView('ocean', 0, 0);
  assert.equal(spectrum.update(0, 12, 2), true);
  const first = gpu.draws.at(-1);
  assert.equal(first.uniforms.uReset, 1);
  assert.notEqual(first.target, first.previous, 'A foam draw must never sample its own framebuffer attachment.');
  assert.equal(spectrum.foam.texture, first.target);
  spectrum.update(1 / 30, 12, 2);
  const second = gpu.draws.at(-1);
  assert.equal(second.previous, first.target, 'The next frame must consume actual prior-frame foam.');
  assert.equal(second.target, first.previous, 'Foam targets should alternate without new allocations.');
  assert.equal(second.uniforms.uReset, 0);
  assert.ok(Math.abs(second.uniforms.uDelta - 1 / 30) < 1e-10);
  spectrum.update(1 / 30 + .25, 12, 2);
  assert.ok(Math.abs(gpu.draws.at(-1).uniforms.uDelta - .25) < 1e-10, 'Slow frames must age foam by elapsed seconds.');
  spectrum.update(1 / 30, 12, 2);
  const count = gpu.draws.length;
  spectrum.update(1 / 30, 12, 2);
  assert.equal(gpu.draws.length, count, 'A duplicate paused frame must not evolve foam.');
  spectrum.setView('ocean', 3, -3);
  spectrum.update(1 / 30, 12, 2);
  const reproject = gpu.draws.at(-1);
  assert.equal(reproject.uniforms.uDelta, 0);
  assert.equal(reproject.uniforms.uReset, 0, 'Moving the camera should reproject history instead of erase it.');
  assert.deepEqual(reproject.uniforms.uPreviousOrigin, [-90, -90]);
  assert.notDeepEqual(reproject.uniforms.uOrigin, reproject.uniforms.uPreviousOrigin);
  assert.equal(spectrum.foam, descriptor, 'The public mapping descriptor should remain stable.');
  spectrum.setView('cove', 3, -3);
  spectrum.update(1 / 30, 12, 2);
  assert.equal(gpu.draws.at(-1).uniforms.uReset, 1, 'Changing scenes must clear incompatible foam sources.');
  spectrum.update(0, 12, 2);
  assert.equal(gpu.draws.at(-1).uniforms.uReset, 1, 'Rewinding simulation time must not revive stale foam.');
  assert.equal(gpu.created.size, resources, 'Rendering, advection and view changes must reuse GPU allocations.');
  gpu.lose();
  assert.equal(spectrum.update(1, 12, 2), false);
  spectrum.dispose();
  spectrum.dispose();
  assert.equal(gpu.deleted.size, gpu.created.size, 'Every allocation, including both foam buffers, must be released.');
}

// Measure uploaded spectral energy rather than compare profile constants.
// At the same wind and height, long swell must move energy toward longer waves
// while short chop must move energy toward shorter waves.
const meanWaveNumber = values => {
  let energy = 0;
  let moment = 0;
  const size = Math.sqrt(values.length / 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const offset = (y * size + x) * 4;
      const power = values[offset] ** 2 + values[offset + 1] ** 2;
      const k = Math.hypot(x < size / 2 ? x : x - size, y < size / 2 ? y : y - size) * Math.PI * 2 / 180;
      energy += power;
      moment += power * k;
    }
  }
  return moment / energy;
};
{
  const gpu = mockGpu();
  const spectrum = window.OceanWaveSpectrum.create(gpu.gl);
  const moments = {};
  for (const swell of ['balanced', 'long', 'chop']) {
    spectrum.update(0, 4.8, .95, swell);
    moments[swell] = meanWaveNumber(gpu.uploads.at(-2).values);
  }
  assert.ok(moments.long < moments.balanced * .85, 'Long swell must contain materially longer dominant wavelengths.');
  assert.ok(moments.chop > moments.balanced * 1.2, 'Short chop must contain materially shorter dominant wavelengths.');
  const uploads = gpu.uploads.length;
  spectrum.update(0, 4.8, .95, 'invalid');
  assert.equal(gpu.uploads.length, uploads + 2, 'Invalid character values must restore the balanced spectrum.');
  assert.ok(Math.abs(meanWaveNumber(gpu.uploads.at(-2).values) - moments.balanced) < 1e-12);
  spectrum.dispose();
}

for (const options of [{ framebufferComplete: false }, { compile: false }]) {
  const gpu = mockGpu(options);
  assert.equal(window.OceanWaveSpectrum.create(gpu.gl), null);
  assert.equal(gpu.deleted.size, gpu.created.size, 'Failed setup must release partially created resources.');
}
{
  const { getWaterDepth, getShoreProximity, constrainCoveCamera } = window.OceanWaveShaders;
  for (const z of [-180, 0, 245, 600, 1250]) {
    const left = { x: -10000, z };
    const right = { x: 10000, z };
    constrainCoveCamera(left);
    constrainCoveCamera(right);
    const middle = (left.x + right.x) / 2;
    assert.ok(getWaterDepth(left.x, z) > 0 && getWaterDepth(right.x, z) > 0);
    assert.ok(getWaterDepth(middle, z) > getWaterDepth(left.x, z));
    assert.ok(getShoreProximity(left.x, z) > getShoreProximity(middle, z));
    assert.ok(getShoreProximity(middle, z) >= 0 && getShoreProximity(left.x, z) <= 1);
    assert.ok(Math.abs(getWaterDepth(left.x - 2.5, z)) < 1e-10, 'The CPU depth and camera shoreline must agree.');
  }
}

const verifyGpu = async () => {
  const { chromium } = require('playwright');
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.setContent('<canvas id="water" width="128" height="128"></canvas>');
    await page.addScriptTag({ content: spectrumSource });
    await page.addScriptTag({ content: shaderSource });
    const result = await page.evaluate(() => {
      const gl = document.querySelector('canvas').getContext('webgl', { antialias: false });
      if (!gl) throw Error('WebGL is unavailable for the explicit GPU verification.');
      const shader = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(shader, window.OceanWaveShaders.fragment);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(shader));
      const spectrum = window.OceanWaveSpectrum.create(gl);
      if (!spectrum) throw Error('The actual spectrum/foam GPU programs failed to initialize.');
      const target = gl.createFramebuffer();
      const pixels = new Float32Array(spectrum.foam.size ** 2 * 4);
      const readFoam = () => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, target);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, spectrum.foam.texture, 0);
        gl.readPixels(0, 0, spectrum.foam.size, spectrum.foam.size, gl.RGBA, gl.FLOAT, pixels);
        if (gl.getError() !== gl.NO_ERROR) throw Error('Foam readback failed.');
        let sum = 0;
        for (let i = 0; i < pixels.length; i += 4) {
          if (!Number.isFinite(pixels[i]) || pixels[i] < 0 || pixels[i] > 1) throw Error('Invalid foam density.');
          sum += pixels[i];
        }
        return sum;
      };
      spectrum.setView('ocean', 0, 0);
      for (let frame = 0; frame < 60; frame++) spectrum.update(frame / 30, 16, 3, 'chop');
      const energetic = readFoam();
      spectrum.update(2, 0, 0);
      const retained = readFoam();
      for (let frame = 61; frame <= 120; frame++) spectrum.update(frame / 30, 0, 0);
      const decayed = readFoam();
      spectrum.setView('cove', 0, 0);
      for (let frame = 121; frame <= 180; frame++) spectrum.update(frame / 30, 2.4, .65);
      const shore = readFoam();
      spectrum.setView('ocean', 0, 0);
      for (let frame = 181; frame <= 210; frame++) spectrum.update(frame / 30, 2.4, .65);
      const calm = readFoam();
      spectrum.dispose();
      gl.deleteShader(shader);
      gl.deleteFramebuffer(target);
      return { energetic, retained, decayed, shore, calm };
    });
    assert.ok(result.energetic > 1, 'Energetic crests should generate nonzero foam.');
    assert.ok(result.retained > result.energetic * .85, 'Foam should survive after its generating crests stop.');
    assert.ok(result.decayed < result.retained * .8 && result.decayed > 0, 'Foam must decay over seconds, not disappear in one frame.');
    assert.ok(result.shore > result.calm + .1, 'The cove should generate shore foam even in calm wind.');
    assert.ok(result.calm < result.energetic * .01, 'Default calm water should remain almost foam-free.');
    console.log('Ocean GPU shader compilation, persistent foam generation/decay, shoreline sources and calm-water sparsity passed.', result);
  } finally {
    await browser.close();
  }
};

console.log('Ocean spectral character, foam history/reprojection, resource cleanup and shoreline geometry checks passed.');
if (process.argv.includes('--gpu')) verifyGpu().catch(error => { console.error(error); process.exitCode = 1; });
