(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersNumbers = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // Nonnegative scientific numbers: wallet size is not limited by Number.MAX_VALUE.
  // Mantissas retain ordinary floating-point precision; comparisons never expand exponents.
  function valid(x) {
    return x !== null && typeof x === 'object' && !Array.isArray(x) &&
      Number.isFinite(x.m) && Number.isSafeInteger(x.e) && Math.abs(x.e) <= 1e12 &&
      ((x.m === 0 && x.e === 0) || (x.m >= 1 && x.m < 10));
  }
  function normal(m, e) {
    if (!Number.isFinite(m) || !Number.isSafeInteger(e) || m < 0) throw new RangeError('Invalid scientific number.');
    if (m === 0) return { m: 0, e: 0 };
    const shift = Math.floor(Math.log10(m));
    let mantissa = shift < -308 ? Number(m.toExponential().split('e')[0]) : m / Math.pow(10, shift);
    let exponent = e + shift;
    if (mantissa >= 10) { mantissa /= 10; exponent += 1; }
    if (mantissa < 1) { mantissa *= 10; exponent -= 1; }
    if (!Number.isSafeInteger(exponent)) throw new RangeError('Scientific exponent exceeds supported precision.');
    return { m: mantissa, e: exponent };
  }
  function from(x) {
    if (valid(x)) return { m: x.m, e: x.e };
    if (typeof x === 'number' && Number.isFinite(x) && x >= 0) return normal(x, 0);
    if (typeof x === 'string' && x.length < 80 && /^\d+(?:\.\d+)?(?:e[+-]?\d+)?$/i.test(x)) {
      const parts = x.toLowerCase().split('e');
      return normal(Number(parts[0]), Number(parts[1] || 0));
    }
    throw new TypeError('Expected a nonnegative number.');
  }
  const zero = () => ({ m: 0, e: 0 });
  function cmp(a, b) {
    a = from(a); b = from(b);
    if (!a.m || !b.m) return Math.sign(a.m - b.m);
    return a.e === b.e ? Math.sign(a.m - b.m) : Math.sign(a.e - b.e);
  }
  function add(a, b) {
    a = from(a); b = from(b);
    if (!a.m) return b;
    if (!b.m) return a;
    if (a.e < b.e) { const t = a; a = b; b = t; }
    return a.e - b.e > 18 ? a : normal(a.m + b.m * Math.pow(10, b.e - a.e), a.e);
  }
  function sub(a, b) {
    a = from(a); b = from(b);
    if (cmp(a, b) <= 0) return zero();
    return a.e - b.e > 18 ? a : normal(a.m - b.m * Math.pow(10, b.e - a.e), a.e);
  }
  function mul(a, b) { a = from(a); b = from(b); return !a.m || !b.m ? zero() : normal(a.m * b.m, a.e + b.e); }
  function div(a, b) { a = from(a); b = from(b); if (!b.m) throw new RangeError('Division by zero.'); return a.m ? normal(a.m / b.m, a.e - b.e) : zero(); }
  function pow(a, power) {
    a = from(a);
    if (!Number.isFinite(power)) throw new RangeError('Invalid power.');
    if (!a.m) return power === 0 ? from(1) : zero();
    const log = (a.e + Math.log10(a.m)) * power;
    if (!Number.isFinite(log) || Math.abs(log) > Number.MAX_SAFE_INTEGER - 2) throw new RangeError('Power exceeds supported exponent precision.');
    const e = Math.floor(log);
    return normal(Math.pow(10, log - e), e);
  }
  function toNumber(a) { a = from(a); return a.e > 308 ? Infinity : a.m * Math.pow(10, a.e); }
  function log10(a) { a = from(a); return a.m ? a.e + Math.log10(a.m) : -Infinity; }
  function floor(a) { a = from(a); return a.e > 14 ? a : from(Math.floor(toNumber(a) + 1e-10)); }
  function min(a, b) { return cmp(a, b) <= 0 ? from(a) : from(b); }
  function max(a, b) { return cmp(a, b) >= 0 ? from(a) : from(b); }
  function format(a, digits) {
    a = from(a); digits = digits === undefined ? 2 : digits;
    if (!a.m) return '0';
    if (a.e >= 9 || a.e < -3) return a.m.toFixed(digits).replace(/\.?0+$/, '') + 'e' + a.e;
    const value = toNumber(a);
    if (value >= 1000) return Math.floor(value).toLocaleString('en-US');
    return value.toFixed(value >= 100 ? 0 : digits).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  }
  return { from, zero, valid, normal, cmp, add, sub, mul, div, pow, toNumber, log10, floor, min, max, format };
});
