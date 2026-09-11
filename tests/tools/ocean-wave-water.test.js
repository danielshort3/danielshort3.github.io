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

const mockGpu = ({ linear = true, framebufferComplete = true, compile = true, maxTexture = 4096,
  maxRenderbuffer = 4096, allocationLimit = Infinity, mipmapFailureAt = 0 } = {}) => {
  const created = new Set();
  const deleted = new Set();
  const uniforms = {};
  const bindings = new Map();
  const uploads = [];
  const draws = [];
  const viewports = [];
  let active = 100;
  let target;
  let program;
  let lost = false;
  let sequence = 0;
  let error = 0;
  let mipmaps = 0;
  let viewport;
  const allocate = kind => { const resource = { id: ++sequence, kind }; created.add(resource); return resource; };
  const gl = {
    TEXTURE0: 100, TEXTURE1: 101, TEXTURE2: 102, TEXTURE_2D: 1,
    FRAMEBUFFER: 2, FRAMEBUFFER_COMPLETE: 3, COLOR_ATTACHMENT0: 4,
    NO_ERROR: 0, FLOAT: 5, RGBA: 6, LINEAR: 7, NEAREST: 8,
    TEXTURE_MIN_FILTER: 9, TEXTURE_MAG_FILTER: 10, TEXTURE_WRAP_S: 11, TEXTURE_WRAP_T: 12,
    REPEAT: 13, CLAMP_TO_EDGE: 14, LINEAR_MIPMAP_LINEAR: 15,
    FRAGMENT_SHADER: 16, VERTEX_SHADER: 17, HIGH_FLOAT: 18, COMPILE_STATUS: 19,
    LINK_STATUS: 20, ARRAY_BUFFER: 21, STATIC_DRAW: 22, TRIANGLES: 23,
    MAX_TEXTURE_SIZE: 24, MAX_RENDERBUFFER_SIZE: 25,
    isContextLost: () => lost,
    getShaderPrecisionFormat: () => ({ precision: 23 }),
    getExtension: name => name === 'OES_texture_float_linear' && !linear ? null : {},
    getParameter: name => name === 24 ? maxTexture : name === 25 ? maxRenderbuffer : null,
    getError: () => { const value = error; error = 0; return value; },
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
    texImage2D: (...args) => {
      const texture = bindings.get(active);
      texture.width = args[3];
      texture.height = args[4];
      if (args[3] > allocationLimit || args[4] > allocationLimit) error = 1285;
    },
    texParameteri: (_target, parameter, value) => {
      const texture = bindings.get(active);
      texture.parameters ||= new Map();
      texture.parameters.set(parameter, value);
    },
    generateMipmap: () => { mipmaps++; if (mipmaps === mipmapFailureAt) error = 1282; },
    viewport: (...values) => { viewport = values; viewports.push(values); },
    drawArrays: () => {
      if (program?.foam) draws.push({ target, previous: bindings.get(100), viewport, uniforms: structuredClone(uniforms) });
    },
  };
  for (const name of ['bindFramebuffer', 'bindBuffer', 'compileShader', 'bindAttribLocation', 'linkProgram',
    'bufferData', 'disable', 'colorMask',
    'enableVertexAttribArray', 'vertexAttribPointer']) gl[name] = () => {};
  return { gl, created, deleted, uploads, draws, viewports, lose: () => { lost = true; } };
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
  const gpu = mockGpu();
  const spectrum = window.OceanWaveSpectrum.create(gpu.gl, { quality: 'low' });
  const field = spectrum.fields[0];
  const foam = spectrum.foam;
  spectrum.setView('cove', 8, -3);
  spectrum.update(7.5, 4.8, .95, 'long');
  const profiles = [['low', 64, 128], ['medium', 128, 256], ['high', 256, 512], ['ultra', 512, 1024], ['low', 64, 128]];
  for (let index = 0; index < profiles.length; index++) {
    const [quality, size, foamSize] = profiles[index];
    const previousAllocations = [...gpu.created].filter(resource => !gpu.deleted.has(resource));
    const previousFoam = spectrum.foam.texture;
    const changed = spectrum.setQuality(quality);
    assert.equal(changed, index > 0, 'Only a different effective quality allocation should rebuild the spectrum.');
    assert.equal(spectrum.fields[0], field, 'Water descriptors must remain valid across quality changes.');
    assert.equal(spectrum.foam, foam, 'Foam descriptors must remain valid across quality changes.');
    assert.equal(field.size, size);
    assert.equal(spectrum.fields[1].size, size);
    assert.equal(field.texture.width, size, 'Reported field resolution must match its actual texture allocation.');
    assert.equal(foam.size, foamSize);
    assert.equal(foam.texture.width, foamSize);
    assert.deepEqual(gpu.draws.at(-1).viewport, [0, 0, foamSize, foamSize], 'Foam must render across its independently sized target.');
    assert.deepEqual(gpu.draws.at(-1).uniforms.uFieldSizes, [size, size]);
    if (index > 0) {
      assert.equal(gpu.draws.at(-1).previous, previousFoam, 'A quality switch must reproject existing foam into the new target.');
      assert.equal(gpu.draws.at(-1).uniforms.uDelta, 0, 'Changing detail must not age or regenerate the current foam.');
      assert.equal(gpu.draws.at(-1).uniforms.uReset, 0);
      assert.equal(gpu.draws.at(-1).uniforms.uTime, 7.5, 'Quality changes must preserve the displayed simulation time.');
      assert.equal(gpu.draws.at(-1).uniforms.uWind, 4.8);
      assert.ok(previousAllocations.every(resource => gpu.deleted.has(resource)), 'Replacing quality must free every former GPU resource.');
    }
    const allocated = gpu.created.size;
    assert.equal(spectrum.setQuality(quality), false);
    spectrum.update(7.5, 4.8, .95, 'long');
    assert.equal(gpu.created.size, allocated, 'Repeated quality settings and paused renders must reuse resources.');
  }
  assert.equal(spectrum.setQuality('invalid'), false);
  spectrum.dispose();
  assert.equal(gpu.created.size, gpu.deleted.size);
  assert.equal(spectrum.setQuality('ultra'), false, 'Disposed spectra must never allocate again.');
}

for (const cap of [{ maxTexture: 128 }, { maxRenderbuffer: 128 }]) {
  const gpu = mockGpu(cap);
  const spectrum = window.OceanWaveSpectrum.create(gpu.gl, { quality: 'ultra' });
  assert.equal(spectrum.fields[0].size, 128, 'Spectrum dimensions must respect both GPU allocation limits.');
  assert.equal(spectrum.foam.size, 128);
  assert.ok([...gpu.created].filter(resource => resource.kind === 'texture').every(texture => texture.width <= 128));
  const allocations = gpu.created.size;
  assert.equal(spectrum.setQuality('high'), false, 'Quality labels constrained to identical allocations must not churn resources.');
  assert.equal(gpu.created.size, allocations);
  spectrum.dispose();
  assert.equal(gpu.created.size, gpu.deleted.size);
}

{
  const gpu = mockGpu({ allocationLimit: 256 });
  const spectrum = window.OceanWaveSpectrum.create(gpu.gl, { quality: 'ultra' });
  assert.ok(spectrum, 'An unsupported large allocation must fall back to a smaller working field set.');
  assert.equal(spectrum.fields[0].size, 128);
  assert.equal(spectrum.foam.size, 256);
  assert.ok(gpu.deleted.size > 0, 'Failed large candidates must release their partial GPU resources.');
  const allocated = gpu.created.size;
  assert.equal(spectrum.setQuality('ultra'), false, 'A known allocation fallback must not be retried every frame.');
  assert.equal(gpu.created.size, allocated);
  assert.equal(spectrum.update(3, 5, 1), true);
  spectrum.dispose();
  assert.equal(gpu.created.size, gpu.deleted.size);
}

for (const mipmapFailureAt of [1, 2, 3, 4]) {
  const gpu = mockGpu({ mipmapFailureAt });
  const spectrum = window.OceanWaveSpectrum.create(gpu.gl, { quality: 'medium' });
  spectrum.update(0, 4.8, .95);
  assert.equal(spectrum.mipmapped, false, 'Mipmap failure in either allocated or first rendered field must select the fallback.');
  for (const field of spectrum.fields) {
    assert.equal(field.texture.parameters.get(gpu.gl.TEXTURE_MIN_FILTER), gpu.gl.LINEAR,
      'Every field must stop sampling mip levels when one mipmap generation fails.');
  }
  assert.equal(spectrum.update(1, 4.8, .95), true, 'A non-mipmapped float-linear field must remain usable.');
  spectrum.dispose();
  assert.equal(gpu.created.size, gpu.deleted.size);
}

for (const linear of [true, false]) {
  const gpu = mockGpu({ linear });
  const queries = { getError: 0, getParameter: 0 };
  for (const name of Object.keys(queries)) {
    const query = gpu.gl[name];
    gpu.gl[name] = (...args) => { queries[name]++; return query(...args); };
  }
  const spectrum = window.OceanWaveSpectrum.create(gpu.gl, { quality: 'medium' });
  assert.ok(queries.getError > 0 && queries.getParameter > 0, 'Setup must retain allocation and capability validation.');
  spectrum.update(0, 4.8, .95);
  for (const quality of ['medium', 'high']) {
    spectrum.setQuality(quality);
    const validatedQueries = { ...queries };
    const drawCount = gpu.draws.length;
    for (let frame = 1; frame <= 20; frame++) {
      spectrum.setView('cove', frame * .1, 0);
      assert.equal(spectrum.update(frame / 30, frame < 10 ? 4.8 : 4.9, .95), true);
    }
    assert.equal(gpu.draws.length, drawCount + 20, 'The query check must exercise real simulation updates.');
    assert.deepEqual(queries, validatedQueries,
      'Ordinary animation and camera/wind changes must not synchronously query GPU state after initial validation.');
  }
  spectrum.dispose();
  assert.equal(gpu.created.size, gpu.deleted.size);
}

{
  const gpu = mockGpu({ linear: false });
  const spectrum = window.OceanWaveSpectrum.create(gpu.gl, { quality: 'ultra' });
  spectrum.update(2, 8, 1.2, 'chop');
  assert.equal(spectrum.linear, false);
  assert.equal(spectrum.mipmapped, false);
  assert.equal(gpu.draws.at(-1).uniforms.uLinear, 0, 'Foam source reconstruction must select its manual interpolation fallback.');
  assert.deepEqual(gpu.draws.at(-1).uniforms.uFieldSizes, [512, 512]);
  assert.equal(gpu.draws.at(-1).uniforms.uPreviousSize, 1024);
  spectrum.dispose();
}

{
  const uploaded = [];
  for (const quality of ['medium', 'ultra']) {
    const gpu = mockGpu();
    const spectrum = window.OceanWaveSpectrum.create(gpu.gl, { quality });
    spectrum.update(3, 4.8, .95);
    uploaded.push({ size: spectrum.fields[0].size, swell: gpu.uploads.at(-2).values, ripple: gpu.uploads.at(-1).values });
    spectrum.dispose();
  }
  const at = (field, x, y) => (((y + field.size) % field.size) * field.size + (x + field.size) % field.size) * 4;
  for (const [x, y] of [[2, 3], [5, -4], [-5, 6], [-8, -3]]) {
    const phases = uploaded.map(field => {
      const offset = at(field, x, y);
      return Math.atan2(field.swell[offset + 1], field.swell[offset]);
    });
    assert.ok(Math.abs(phases[0] - phases[1]) < .000001, 'Adding finer detail must preserve the phase of existing large waves.');
  }
  const fine = uploaded[1];
  let slopeEnergy = 0;
  let addedSlopeEnergy = 0;
  for (let y = 0; y < fine.size; y++) {
    for (let x = 0; x < fine.size; x++) {
      const kx = x < fine.size / 2 ? x : x - fine.size;
      const ky = y < fine.size / 2 ? y : y - fine.size;
      const offset = (y * fine.size + x) * 4;
      const energy = (kx * kx + ky * ky) * (fine.ripple[offset] ** 2 + fine.ripple[offset + 1] ** 2);
      slopeEnergy += energy;
      if (Math.abs(kx) >= 64 || Math.abs(ky) >= 64) addedSlopeEnergy += energy;
    }
  }
  assert.ok(addedSlopeEnergy > slopeEnergy * .03, 'Ultra must add meaningful small-wave slope energy beyond the former 128-bin limit.');
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
  const inspectGpu = mockGpu({ linear: false });
  const inspectSpectrum = window.OceanWaveSpectrum.create(inspectGpu.gl);
  const foamShader = [...inspectGpu.created].find(resource => resource.source?.includes('float previousDensity'));
  const reconstruction = foamShader.source.slice(foamShader.source.indexOf('vec3 sampleWave'), foamShader.source.indexOf('float previousDensity'));
  inspectSpectrum.dispose();
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.setContent('<canvas id="water" width="128" height="128"></canvas>');
    await page.addScriptTag({ content: spectrumSource });
    await page.addScriptTag({ content: shaderSource });
    const result = await page.evaluate((sampleWaveSource) => {
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

      const fine = window.OceanWaveSpectrum.create(gl, { quality: 'low' });
      fine.setView('cove', 0, 0);
      for (let frame = 0; frame < 20; frame++) fine.update(frame / 30, 16, 3, 'chop');
      const meanDensity = () => {
        const values = new Float32Array(fine.foam.size ** 2 * 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fine.foam.texture, 0);
        gl.readPixels(0, 0, fine.foam.size, fine.foam.size, gl.RGBA, gl.FLOAT, values);
        if (gl.getError() !== gl.NO_ERROR) throw Error('Resized foam readback failed.');
        let density = 0;
        for (let index = 0; index < values.length; index += 4) density += values[index];
        return density / (fine.foam.size ** 2);
      };
      const beforeResize = meanDensity();
      fine.setQuality('ultra');
      const afterResize = meanDensity();
      const fieldSize = fine.fields[1].size;
      const foamSize = fine.foam.size;
      const fieldValues = new Float32Array(fieldSize ** 2 * 4);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fine.fields[1].texture, 0);
      gl.readPixels(0, 0, fieldSize, fieldSize, gl.RGBA, gl.FLOAT, fieldValues);
      if (gl.getError() !== gl.NO_ERROR) throw Error('Ultra spectrum readback failed.');
      let slopeEnergy = 0;
      for (let index = 0; index < fieldValues.length; index += 4) {
        if (!Number.isFinite(fieldValues[index]) || !Number.isFinite(fieldValues[index + 1])
          || !Number.isFinite(fieldValues[index + 2])) throw Error('Non-finite Ultra FFT output.');
        slopeEnergy += fieldValues[index + 1] ** 2 + fieldValues[index + 2] ** 2;
      }
      fine.dispose();

      // Exercise the production reconstruction helper on a tiny known field.
      // A nearest-filter-only device must recover sub-texel height and both
      // derivative channels, rather than turn every pair of pixels into a block.
      const vertex = gl.createShader(gl.VERTEX_SHADER);
      gl.shaderSource(vertex, 'attribute vec2 aPosition; varying vec2 vUv; void main(){vUv=aPosition*.5+.5;gl_Position=vec4(aPosition,0.,1.);}');
      gl.compileShader(vertex);
      const reconstructionShader = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(reconstructionShader, `precision highp float; uniform sampler2D field; uniform float uLinear; varying vec2 vUv;
        ${sampleWaveSource} void main(){gl_FragColor=vec4(sampleWave(field,vUv,2.0),1.0);}`);
      gl.compileShader(reconstructionShader);
      if (!gl.getShaderParameter(reconstructionShader, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(reconstructionShader));
      const reconstructionProgram = gl.createProgram();
      gl.attachShader(reconstructionProgram, vertex);
      gl.attachShader(reconstructionProgram, reconstructionShader);
      gl.bindAttribLocation(reconstructionProgram, 0, 'aPosition');
      gl.linkProgram(reconstructionProgram);
      if (!gl.getProgramParameter(reconstructionProgram, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(reconstructionProgram));
      const texture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 2, 2, 0, gl.RGBA, gl.FLOAT,
        new Float32Array([0,1,2,1, 2,1,2,1, 4,1,2,1, 6,1,2,1]));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      const reconstructedTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, reconstructedTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 8, 8, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, reconstructedTexture, 0);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.useProgram(reconstructionProgram);
      gl.uniform1i(gl.getUniformLocation(reconstructionProgram, 'field'), 0);
      gl.uniform1f(gl.getUniformLocation(reconstructionProgram, 'uLinear'), 0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.viewport(0, 0, 8, 8);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const reconstructed = new Float32Array(8 * 8 * 4);
      gl.readPixels(0, 0, 8, 8, gl.RGBA, gl.FLOAT, reconstructed);
      if (gl.getError() !== gl.NO_ERROR) throw Error('Sub-texel reconstruction readback failed.');
      const interpolation = [2,3,4,5].map(x => Array.from(reconstructed.slice((2 * 8 + x) * 4, (2 * 8 + x) * 4 + 3)));
      gl.deleteBuffer(buffer);
      gl.deleteTexture(texture);
      gl.deleteTexture(reconstructedTexture);
      gl.deleteProgram(reconstructionProgram);
      gl.deleteShader(vertex);
      gl.deleteShader(reconstructionShader);
      gl.deleteShader(shader);
      gl.deleteFramebuffer(target);
      return { energetic, retained, decayed, shore, calm, beforeResize, afterResize, fieldSize, foamSize, slopeEnergy, interpolation };
    }, reconstruction);
    assert.ok(result.energetic > 1, 'Energetic crests should generate nonzero foam.');
    assert.ok(result.retained > result.energetic * .85, 'Foam should survive after its generating crests stop.');
    assert.ok(result.decayed < result.retained * .8 && result.decayed > 0, 'Foam must decay over seconds, not disappear in one frame.');
    assert.ok(result.shore > result.calm + .1, 'The cove should generate shore foam even in calm wind.');
    assert.ok(result.calm < result.energetic * .01, 'Default calm water should remain almost foam-free.');
    assert.equal(result.fieldSize, 512, 'Ultra must execute its larger FFT on a real GPU.');
    assert.equal(result.foamSize, 1024);
    assert.ok(result.slopeEnergy > 0, 'The larger FFT must produce finite, nonzero surface derivatives.');
    assert.ok(result.beforeResize > 0 && Math.abs(result.afterResize - result.beforeResize) < result.beforeResize * .01,
      'Changing resolution must preserve the visible foam density instead of clearing history.');
    for (let index = 0; index < result.interpolation.length; index++) {
      const [height, slopeX, slopeZ] = result.interpolation[index];
      assert.ok(Math.abs(height - (.75 + index * .5)) < .00001, 'Manual field reconstruction must produce continuous sub-texel heights.');
      assert.ok(Math.abs(slopeX - 1) < .00001 && Math.abs(slopeZ - 2) < .00001,
        'Manual field reconstruction must preserve both slope channels.');
    }
    console.log('Ocean GPU shader compilation, persistent foam generation/decay, shoreline sources and calm-water sparsity passed.', result);
  } finally {
    await browser.close();
  }
};

console.log('Ocean spectral character, foam history/reprojection, resource cleanup and shoreline geometry checks passed.');
if (process.argv.includes('--gpu')) verifyGpu().catch(error => { console.error(error); process.exitCode = 1; });
