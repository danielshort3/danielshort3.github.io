(() => {
  'use strict';

  const MAX_BYTES = 12 * 1024 * 1024;
  const MAX_PIXELS = 2048 * 1024;
  const TAU = Math.PI * 2;
  const ASSETS = {
    dawn: { id: 'kloppenheim_06_puresky', hasSun: true },
    golden: { id: 'kloppenheim_06_puresky', hasSun: true },
    daylight: { id: 'kloofendal_48d_partly_cloudy_puresky', hasSun: true },
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
    for (let y = 0; y < upperRows; y++) {
      for (let x = 0; x < width; x++) {
        const value = luminance(data, (y * width + x) * 4);
        if (!Number.isFinite(value)) throw new Error('Non-finite ocean HDR radiance.');
        if (!(x % 2) && !(y % 2)) samples.push(value);
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
    for (let y = 0; y < upperRows; y += 2) {
      const weight = Math.sin((y + 0.5) / height * Math.PI);
      for (let x = 0; x < width; x += 2) {
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
    for (let pixel = 0; pixel < width * height; pixel++) {
      const index = pixel * 4;
      for (let n = 0; n < 3; n++) data[index + n] = Math.min(50, Math.max(0, data[index + n] * normalization));
    }
    return { width, height, data, sunDirection, hasSun: asset.hasSun, normalization,
      peakLuminance: Math.min(50, brightest * normalization) };
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
    const cache = new Map();
    const pending = new Map();
    const abort = new AbortController();
    let current = null;
    let disposed = false;
    let selection = 0;

    const upload = (sky) => {
      const texture = gl.createTexture();
      if (!texture) throw new Error('Ocean sky texture unavailable.');
      const previous = gl.getParameter(gl.TEXTURE_BINDING_2D);
      const previousFlip = gl.getParameter(gl.UNPACK_FLIP_Y_WEBGL);
      const previousPremultiply = gl.getParameter(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL);
      // Float textures retain highlights; the compatibility path trades their
      // dynamic range for enough linear precision in the main sky gradient.
      const textureScale = floatLinear ? 1 : 6;
      let pixels = sky.data;
      if (!floatLinear) {
        pixels = new Uint8Array(sky.data.length);
        for (let i = 0; i < pixels.length; i++) {
          pixels[i] = i % 4 === 3 ? 255 : Math.round(Math.min(1, sky.data[i] / textureScale) * 255);
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
        const format = floatLinear && typeof gl.texStorage2D === 'function' ? gl.RGBA32F : gl.RGBA;
        gl.texImage2D(gl.TEXTURE_2D, 0, format, sky.width, sky.height, 0, gl.RGBA, floatLinear ? gl.FLOAT : gl.UNSIGNED_BYTE, pixels);
      } catch (error) {
        gl.deleteTexture(texture);
        throw error;
      } finally {
        gl.bindTexture(gl.TEXTURE_2D, previous);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, previousFlip);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, previousPremultiply);
      }
      return { texture, textureScale, width: sky.width, height: sky.height, sunDirection: sky.sunDirection,
        hasSun: sky.hasSun, exposure: 1, peakLuminance: sky.peakLuminance, floatTexture: floatLinear };
    };

    const load = (asset) => {
      if (cache.has(asset.id)) return Promise.resolve(cache.get(asset.id));
      if (pending.has(asset.id)) return pending.get(asset.id);
      const promise = fetch(`/img/games/ocean/${asset.id}_1k.hdr`, { signal: abort.signal, cache: 'force-cache' })
        .then(readBounded)
        .then((buffer) => {
          if (disposed || gl.isContextLost()) return null;
          const sky = upload(prepare(decode(buffer), asset));
          cache.set(asset.id, sky);
          return sky;
        })
        .finally(() => pending.delete(asset.id));
      pending.set(asset.id, promise);
      return promise;
    };

    return {
      get current() { return current; },
      get ready() { return !!current && !disposed; },
      async select(mood) {
        if (disposed) return false;
        const selected = ++selection;
        try {
          const sky = await load(ASSETS[mood] || ASSETS.dawn);
          if (!sky || disposed || selected !== selection || gl.isContextLost()) return false;
          current = sky;
          if (typeof options.onChange === 'function') options.onChange(current);
          return true;
        } catch (error) {
          if (!disposed && selected === selection && typeof options.onError === 'function') options.onError(error);
          return false;
        }
      },
      dispose() {
        if (disposed) return;
        disposed = true;
        selection++;
        abort.abort();
        for (const sky of cache.values()) gl.deleteTexture(sky.texture);
        cache.clear();
        pending.clear();
        current = null;
      },
    };
  };

  window.OceanWaveEnvironment = { create };
})();
