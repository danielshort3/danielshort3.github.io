(() => {
  'use strict';

  const MAX_BYTES = 24 * 1024 * 1024;
  const MAX_PIXELS = 4096 * 2048;
  const TAU = Math.PI * 2;
  const ASSETS = {
    dawn: { id: 'kloppenheim_06_puresky', hasSun: true, ultraResolution: '4k', sunDirection: [.6468873, .0807796, .7582951] },
    golden: { id: 'kloppenheim_06_puresky', hasSun: true, ultraResolution: '4k', sunDirection: [.6468873, .0807796, .7582951] },
    daylight: { id: 'kloofendal_48d_partly_cloudy_puresky', hasSun: true, sunDirection: [.3774921, .7417488, .5543541] },
    dusk: { id: 'qwantani_dusk_1_puresky', hasSun: false },
  };

  // Radiance scanlines become linear RGB, with row zero at the zenith.
  const decode = (buffer) => {
    const bytes = new Uint8Array(buffer);
    if (bytes.length < 32 || bytes.length > MAX_BYTES) throw new Error('Invalid ocean HDR size.');
    let offset = 0;
    const line = () => {
      const start = offset;
      while (offset < bytes.length && bytes[offset] !== 10) {
        offset++;
        if (offset - start > 4096 || offset > 65536) throw new Error('Invalid ocean HDR header.');
      }
      if (offset >= bytes.length) throw new Error('Truncated ocean HDR header.');
      const value = String.fromCharCode(...bytes.subarray(start, offset)).replace(/\r$/, '');
      offset++;
      return value;
    };
    if (!/^#\?(RADIANCE|RGBE)$/.test(line())) throw new Error('Invalid ocean HDR signature.');
    let format = false;
    let header = '';
    do {
      header = line();
      if (header === 'FORMAT=32-bit_rle_rgbe') format = true;
    } while (header);
    if (!format) throw new Error('Unsupported ocean HDR format.');
    const dimensions = /^([+-])Y (\d+) ([+-])X (\d+)$/.exec(line());
    if (!dimensions) throw new Error('Unsupported ocean HDR orientation.');
    const width = Number(dimensions[4]);
    const height = Number(dimensions[2]);
    if (width < 1 || height < 1 || width > 4096 || height > 4096 || width * height > MAX_PIXELS) {
      throw new Error('Invalid ocean HDR dimensions.');
    }
    const rgbe = new Uint8Array(width * height * 4);
    const requireBytes = (count) => {
      if (offset + count > bytes.length) throw new Error('Truncated ocean HDR pixels.');
    };
    const modern = width >= 8 && width <= 32767 && bytes[offset] === 2
      && bytes[offset + 1] === 2 && !(bytes[offset + 2] & 128);
    if (modern) {
      for (let y = 0; y < height; y++) {
        requireBytes(4);
        if (bytes[offset] !== 2 || bytes[offset + 1] !== 2
          || ((bytes[offset + 2] << 8) | bytes[offset + 3]) !== width) {
          throw new Error('Invalid ocean HDR scanline.');
        }
        offset += 4;
        for (let channel = 0; channel < 4; channel++) {
          let x = 0;
          while (x < width) {
            requireBytes(1);
            const code = bytes[offset++];
            const count = code > 128 ? code - 128 : code;
            if (!count || x + count > width) throw new Error('Invalid ocean HDR run length.');
            requireBytes(code > 128 ? 1 : count);
            if (code > 128) {
              const value = bytes[offset++];
              for (let n = 0; n < count; n++) rgbe[((y * width + x++) * 4) + channel] = value;
            } else {
              for (let n = 0; n < count; n++) rgbe[((y * width + x++) * 4) + channel] = bytes[offset++];
            }
          }
        }
      }
    } else {
      let pixel = 0;
      let shift = 0;
      while (pixel < width * height) {
        requireBytes(4);
        if (bytes[offset] === 1 && bytes[offset + 1] === 1 && bytes[offset + 2] === 1) {
          const count = bytes[offset + 3] * Math.pow(2, shift);
          if (!pixel || shift > 24 || pixel + count > width * height) {
            throw new Error('Invalid legacy ocean HDR run.');
          }
          const previous = rgbe.subarray((pixel - 1) * 4, pixel * 4);
          for (let n = 0; n < count; n++) rgbe.set(previous, pixel++ * 4);
          shift += 8;
        } else {
          rgbe.set(bytes.subarray(offset, offset + 4), pixel++ * 4);
          shift = 0;
        }
        offset += 4;
      }
    }

    const data = new Float32Array(width * height * 4);
    const exponentScale = Array.from({ length: 256 }, (_, e) => e ? Math.pow(2, e - 136) : 0);
    for (let y = 0; y < height; y++) {
      const targetY = dimensions[1] === '-' ? y : height - y - 1;
      for (let x = 0; x < width; x++) {
        const targetX = dimensions[3] === '+' ? x : width - x - 1;
        const source = (y * width + x) * 4;
        const target = (targetY * width + targetX) * 4;
        const scale = exponentScale[rgbe[source + 3]];
        data[target] = rgbe[source] * scale;
        data[target + 1] = rgbe[source + 1] * scale;
        data[target + 2] = rgbe[source + 2] * scale;
        data[target + 3] = 1;
      }
    }
    return { width, height, data };
  };

  const luminance = (data, index) => data[index] * 0.2126 + data[index + 1] * 0.7152 + data[index + 2] * 0.0722;
  const directionAt = (x, y, width, height) => {
    const longitude = ((x + 0.5) / width - 0.5) * TAU;
    const latitude = (y + 0.5) / height * Math.PI;
    return [Math.sin(longitude) * Math.sin(latitude), Math.cos(latitude), Math.cos(longitude) * Math.sin(latitude)];
  };

  const prepare = (decoded, asset) => {
    const { width, height, data } = decoded;
    const upperRows = Math.floor(height / 2);
    const samples = [];
    let brightest = 0;
    let brightestX = 0;
    let brightestY = 0;
    const sampleStep = Math.max(2, Math.floor(width / 512));
    for (let y = 0; y < upperRows; y++) {
      for (let x = 0; x < width; x++) {
        const value = luminance(data, (y * width + x) * 4);
        if (!Number.isFinite(value)) throw new Error('Non-finite ocean HDR radiance.');
        if (!(x % sampleStep) && !(y % sampleStep)) samples.push(value);
        if (value > brightest) {
          brightest = value;
          brightestX = x;
          brightestY = y;
        }
      }
    }
    samples.sort((a, b) => a - b);
    const ceiling = samples[Math.floor(samples.length * 0.98)];
    let weighted = 0;
    let totalWeight = 0;
    for (let y = 0; y < upperRows; y += sampleStep) {
      const weight = Math.sin((y + 0.5) / height * Math.PI);
      for (let x = 0; x < width; x += sampleStep) {
        weighted += Math.min(ceiling, luminance(data, (y * width + x) * 4)) * weight;
        totalWeight += weight;
      }
    }
    const mean = weighted / totalWeight;
    if (!(mean > 0) || !Number.isFinite(mean)) throw new Error('Empty ocean HDR sky.');
    const normalization = 0.75 / mean;
    const peakDirection = directionAt(brightestX, brightestY, width, height);
    const sunDirection = [0, 0, 0];
    let sunWeight = 0;
    for (let y = Math.max(0, brightestY - 10); y < Math.min(upperRows, brightestY + 11); y++) {
      for (let dx = -10; dx <= 10; dx++) {
        const x = (brightestX + dx + width) % width;
        const value = luminance(data, (y * width + x) * 4);
        if (value < brightest * 0.5) continue;
        const direction = directionAt(x, y, width, height);
        const alignment = direction[0] * peakDirection[0] + direction[1] * peakDirection[1] + direction[2] * peakDirection[2];
        if (alignment < 0.998) continue;
        for (let n = 0; n < 3; n++) sunDirection[n] += direction[n] * value;
        sunWeight += value;
      }
    }
    const length = Math.hypot(...sunDirection);
    if (!sunWeight || !length) sunDirection.splice(0, 3, ...peakDirection);
    else for (let n = 0; n < 3; n++) sunDirection[n] /= length;
    if (asset.sunDirection) {
      // Coordinates measured from the 2K source remain fixed during upgrades.
      // In the 1K sunset, the brightest cloud can outshine the actual sun.
      sunDirection.splice(0, 3, ...asset.sunDirection);
      brightestX = Math.floor((Math.atan2(sunDirection[0], sunDirection[2]) / TAU + .5) * width);
      brightestY = Math.floor(Math.acos(sunDirection[1]) / Math.PI * height);
    }
    // Separate only the small solar core from its photographed atmosphere.
    // Reconstructing that disk at display resolution avoids enlarged HDR
    // texels and lets crossfades move one sun instead of overlaying two suns.
    const sunRadiance = [0, 0, 0];
    if (asset.hasSun) {
      const radius = Math.ceil(height * .026 / Math.PI) + 2;
      const region = [];
      const background = [0, 0, 0];
      let backgroundWeight = 0;
      for (let y = Math.max(0, brightestY - radius); y < Math.min(upperRows, brightestY + radius + 1); y++) {
        const solidAngle = Math.sin((y + .5) / height * Math.PI) * TAU / width * Math.PI / height;
        for (let dx = -radius; dx <= radius; dx++) {
          const x = (brightestX + dx + width) % width;
          const direction = directionAt(x, y, width, height);
          const alignment = direction[0] * sunDirection[0] + direction[1] * sunDirection[1] + direction[2] * sunDirection[2];
          const angle = Math.acos(Math.max(-1, Math.min(1, alignment)));
          const index = (y * width + x) * 4;
          if (angle >= .013 && angle <= .024) {
            for (let n = 0; n < 3; n++) background[n] += data[index + n] * solidAngle;
            backgroundWeight += solidAngle;
          }
          if (angle < .011) region.push({ index, angle, solidAngle });
        }
      }
      if (backgroundWeight > 0) {
        for (let n = 0; n < 3; n++) background[n] /= backgroundWeight;
        const solarSolidAngle = TAU * (1 - Math.cos(.00465));
        for (const { index, angle, solidAngle } of region) {
          const edge = Math.max(0, Math.min(1, (angle - .0065) / .0045));
          const retained = edge * edge * (3 - 2 * edge);
          for (let n = 0; n < 3; n++) {
            const excess = Math.max(0, data[index + n] - background[n]);
            sunRadiance[n] += excess * (1 - retained) * solidAngle * normalization / solarSolidAngle;
            data[index + n] -= excess * (1 - retained);
          }
        }
      }
    }
    for (let pixel = 0; pixel < width * height; pixel++) {
      const index = pixel * 4;
      for (let n = 0; n < 3; n++) data[index + n] = Math.min(256, Math.max(0, data[index + n] * normalization));
    }
    return { width, height, data, sunDirection, sunRadiance, hasSun: asset.hasSun, normalization,
      peakLuminance: brightest * normalization };
  };

  const readBounded = async (response) => {
    if (!response.ok) throw new Error('Ocean sky unavailable.');
    const length = Number(response.headers.get('content-length'));
    if (length > MAX_BYTES) throw new Error('Ocean sky is too large.');
    if (!response.body || !response.body.getReader) {
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength > MAX_BYTES) throw new Error('Ocean sky is too large.');
      return buffer;
    }
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > MAX_BYTES) {
          await reader.cancel();
          throw new Error('Ocean sky is too large.');
        }
        chunks.push(value);
      }
    } finally {
      reader.releaseLock();
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return bytes.buffer;
  };

  const create = (gl, options = {}) => {
    if (!gl) return null;
    const floatStorage = !!gl.getExtension('OES_texture_float') || typeof gl.texStorage2D === 'function';
    const floatLinear = floatStorage && !!gl.getExtension('OES_texture_float_linear');
    const webgl2 = typeof gl.texStorage2D === 'function';
    const halfExtension = webgl2 ? null : gl.getExtension('OES_texture_half_float');
    const halfLinear = webgl2 || !!(halfExtension && gl.getExtension('OES_texture_half_float_linear'));
    const linearStorage = halfLinear || floatLinear;
    const maximumSize = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE)) || 2048;
    const cache = new Map();
    const pending = new Map();
    let current = null;
    let disposed = false;
    let selection = 0;
    let selectionKey = '';

    const upload = (sky) => {
      const texture = gl.createTexture();
      if (!texture) throw new Error('Ocean sky texture unavailable.');
      const previous = gl.getParameter(gl.TEXTURE_BINDING_2D);
      const previousFlip = gl.getParameter(gl.UNPACK_FLIP_Y_WEBGL);
      const previousPremultiply = gl.getParameter(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL);
      // Gamma encoding gives the byte-texture fallback many more levels in
      // darker sky gradients; all shading still decodes back to linear RGB.
      const textureScale = linearStorage ? 1 : 8;
      const textureEncoding = linearStorage ? 0 : 1;
      let pixels = sky.data;
      if (halfLinear) {
        // Half float retains much more radiance precision than the RGBE
        // source needs, while halving Ultra panorama memory on the GPU.
        pixels = new Uint16Array(sky.data.length);
        const bits = new Uint32Array(sky.data.buffer, sky.data.byteOffset, sky.data.length);
        for (let index = 0; index < bits.length; index++) {
          const exponent = (bits[index] >>> 23) & 255;
          const mantissa = bits[index] & 0x7fffff;
          pixels[index] = exponent < 103 ? 0 : exponent < 113
            ? ((mantissa | 0x800000) + (1 << (125 - exponent))) >>> (126 - exponent)
            : Math.min(0x7bff, ((exponent - 112) << 10) + ((mantissa + 0x1000) >>> 13));
        }
      } else if (!linearStorage) {
        pixels = new Uint8Array(sky.data.length);
        for (let i = 0; i < pixels.length; i++) {
          pixels[i] = i % 4 === 3 ? 255 : Math.round(Math.pow(Math.min(1, sky.data[i] / textureScale), 1 / 2.2) * 255);
        }
      }
      try {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        const format = webgl2 && linearStorage ? halfLinear ? gl.RGBA16F : gl.RGBA32F : gl.RGBA;
        const storage = halfLinear ? webgl2 ? gl.HALF_FLOAT : halfExtension.HALF_FLOAT_OES : floatLinear ? gl.FLOAT : gl.UNSIGNED_BYTE;
        gl.texImage2D(gl.TEXTURE_2D, 0, format, sky.width, sky.height, 0, gl.RGBA, storage, pixels);
        // Without mipmaps GLSL's blur argument has no effect. These filtered
        // radiance levels make distant reflections and diffuse light stable.
        gl.generateMipmap(gl.TEXTURE_2D);
        if (gl.getError() === gl.NO_ERROR) {
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        }
      } catch (error) {
        gl.deleteTexture(texture);
        throw error;
      } finally {
        gl.bindTexture(gl.TEXTURE_2D, previous);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, previousFlip);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, previousPremultiply);
      }
      return { texture, textureScale, textureEncoding, width: sky.width, height: sky.height, sunDirection: sky.sunDirection,
        sunRadiance: sky.sunRadiance,
        hasSun: sky.hasSun, exposure: 1, peakLuminance: sky.peakLuminance, floatTexture: linearStorage, halfTexture: halfLinear };
    };

    const load = (asset, resolution = '1k') => {
      const key = `${asset.id}:${resolution}`;
      if (cache.has(key)) return Promise.resolve(cache.get(key));
      if (pending.has(key)) return pending.get(key).promise;
      const request = { controller: new AbortController(), resolution, promise: null };
      request.promise = fetch(`/img/games/ocean/${asset.id}_${resolution}.hdr`, { signal: request.controller.signal, cache: 'force-cache' })
        .then(readBounded)
        .then((buffer) => {
          if (disposed || request.controller.signal.aborted || gl.isContextLost()) return null;
          const sky = upload(prepare(decode(buffer), asset));
          sky.assetId = asset.id;
          sky.resolution = resolution;
          cache.set(key, sky);
          return sky;
        })
        .finally(() => {
          // A cancelled request may settle after a new request for this sky.
          if (pending.get(key) === request) pending.delete(key);
        });
      pending.set(key, request);
      return request.promise;
    };

    return {
      get current() { return current; },
      get ready() { return !!current && !disposed; },
      async select(mood, qualityMode = 'auto') {
        if (disposed) return false;
        const asset = ASSETS[mood] || ASSETS.dawn;
        const resolution = qualityMode === 'ultra' && maximumSize >= 2048
          ? asset.ultraResolution === '4k' && maximumSize >= 4096 ? '4k' : '2k' : '1k';
        const key = `${asset.id}:${resolution}`;
        if (key === selectionKey) return !!current;
        selectionKey = key;
        const selected = ++selection;
        for (const [pendingKey, request] of pending) {
          if (request.resolution === '1k' || pendingKey === key) continue;
          pending.delete(pendingKey);
          request.controller.abort();
        }
        const publish = (sky) => {
          if (!sky || disposed || selected !== selection || gl.isContextLost()) return false;
          if (current !== sky) {
            current = sky;
            if (typeof options.onChange === 'function') options.onChange(current);
          }
          return true;
        };
        try {
          const cached = cache.get(key);
          if (cached) return publish(cached);
          if (!publish(await load(asset))) return false;
          if (resolution !== '1k') {
            // Ultra upgrades photographic detail after the smaller sky is
            // visible. A failed optional upgrade keeps that working sky.
            try { publish(await load(asset, resolution)); } catch {}
          }
          return !disposed && selected === selection && !gl.isContextLost();
        } catch (error) {
          if (!disposed && selected === selection && typeof options.onError === 'function') options.onError(error);
          return false;
        }
      },
      releaseUnused(skies = []) {
        if (disposed) return;
        // The renderer owns crossfade references; keep those until its fade
        // bookkeeping releases them, along with the latest selected sky.
        const retained = new Set([current, ...skies]);
        for (const [key, sky] of cache) {
          if (sky.resolution === '1k' || retained.has(sky)) continue;
          cache.delete(key);
          gl.deleteTexture(sky.texture);
        }
      },
      dispose() {
        if (disposed) return;
        disposed = true;
        selection++;
        for (const request of pending.values()) request.controller.abort();
        for (const sky of cache.values()) gl.deleteTexture(sky.texture);
        cache.clear();
        pending.clear();
        current = null;
      },
    };
  };

  window.OceanWaveEnvironment = { create };
})();
