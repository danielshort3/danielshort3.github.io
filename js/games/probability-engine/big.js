(() => {
  "use strict";

  const PE = window.ProbabilityEngine = window.ProbabilityEngine || {};

class Big {
  constructor(mantissa = 0, exponent = 0) {
    this.m = Number(mantissa) || 0;
    this.e = Number(exponent) || 0;
    this.normalize();
  }

  static zero() {
    return new Big(0, 0);
  }

  static one() {
    return new Big(1, 0);
  }

  static from(value) {
    if (value instanceof Big) {
      return value.clone();
    }
    if (typeof value === "number") {
      if (!Number.isFinite(value) || value === 0) {
        return Big.zero();
      }
      const exponent = Math.floor(Math.log10(Math.abs(value)));
      const mantissa = value / Math.pow(10, exponent);
      return new Big(mantissa, exponent);
    }
    if (typeof value === "string") {
      if (!value || value === "0") {
        return Big.zero();
      }
      if (value.includes("e")) {
        const parts = value.split("e");
        return new Big(Number(parts[0]), Number(parts[1]));
      }
      return Big.from(Number(value));
    }
    if (Array.isArray(value) && value.length === 2) {
      return new Big(value[0], value[1]);
    }
    if (value && typeof value === "object" && "m" in value && "e" in value) {
      return new Big(value.m, value.e);
    }
    return Big.zero();
  }

  clone() {
    return new Big(this.m, this.e);
  }

  normalize() {
    if (!Number.isFinite(this.m) || this.m === 0) {
      this.m = 0;
      this.e = 0;
      return this;
    }
    while (Math.abs(this.m) >= 10) {
      this.m /= 10;
      this.e += 1;
    }
    while (Math.abs(this.m) < 1) {
      this.m *= 10;
      this.e -= 1;
    }
    return this;
  }

  isZero() {
    return this.m === 0;
  }

  abs() {
    return new Big(Math.abs(this.m), this.e);
  }

  neg() {
    return new Big(-this.m, this.e);
  }

  cmp(otherValue) {
    const other = Big.from(otherValue);
    if (this.m === 0 && other.m === 0) {
      return 0;
    }
    if (this.m >= 0 && other.m < 0) {
      return 1;
    }
    if (this.m < 0 && other.m >= 0) {
      return -1;
    }
    const sign = this.m < 0 ? -1 : 1;
    if (this.e !== other.e) {
      return this.e > other.e ? sign : -sign;
    }
    if (this.m === other.m) {
      return 0;
    }
    return this.m > other.m ? sign : -sign;
  }

  gte(other) {
    return this.cmp(other) >= 0;
  }

  gt(other) {
    return this.cmp(other) > 0;
  }

  lte(other) {
    return this.cmp(other) <= 0;
  }

  lt(other) {
    return this.cmp(other) < 0;
  }

  add(otherValue) {
    const other = Big.from(otherValue);
    if (this.m === 0) {
      return other;
    }
    if (other.m === 0) {
      return this.clone();
    }
    let a = this.clone();
    let b = other.clone();
    if (a.e < b.e) {
      const temp = a;
      a = b;
      b = temp;
    }
    const diff = a.e - b.e;
    if (diff > 16) {
      return a;
    }
    return new Big(a.m + b.m * Math.pow(10, -diff), a.e);
  }

  sub(otherValue) {
    return this.add(Big.from(otherValue).neg());
  }

  mul(otherValue) {
    const other = Big.from(otherValue);
    if (this.m === 0 || other.m === 0) {
      return Big.zero();
    }
    return new Big(this.m * other.m, this.e + other.e);
  }

  div(otherValue) {
    const other = Big.from(otherValue);
    if (other.m === 0) {
      return Big.zero();
    }
    if (this.m === 0) {
      return Big.zero();
    }
    return new Big(this.m / other.m, this.e - other.e);
  }

  pow(power) {
    if (this.m === 0) {
      return Big.zero();
    }
    if (power === 0) {
      return Big.one();
    }
    const sign = this.m < 0 && Math.abs(power % 2) === 1 ? -1 : 1;
    const log10 = Math.log10(Math.abs(this.m)) + this.e;
    const resultLog = log10 * power;
    if (!Number.isFinite(resultLog)) {
      return new Big(sign, 999999);
    }
    const exponent = Math.floor(resultLog);
    const mantissa = sign * Math.pow(10, resultLog - exponent);
    return new Big(mantissa, exponent);
  }

  log10() {
    if (this.m === 0) {
      return -Infinity;
    }
    return Math.log10(Math.abs(this.m)) + this.e;
  }

  toNumber() {
    if (this.m === 0) {
      return 0;
    }
    if (this.e > 308) {
      return this.m > 0 ? Infinity : -Infinity;
    }
    if (this.e < -324) {
      return 0;
    }
    return this.m * Math.pow(10, this.e);
  }

  toArray() {
    return [this.m, this.e];
  }
}

  PE.Big = Big;
})();
