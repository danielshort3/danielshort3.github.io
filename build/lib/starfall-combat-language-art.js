'use strict';

// Authoritative editable artwork for the existing code-native combat FX family.
// Category cores remain fixed; class/element color is a small secondary accent.
const COLORS = Object.freeze({ heal: '#62D995', damage: '#F06A60', buff: '#F2C45E', shield: '#63D7E8', debuff: '#B88AF3', resource: '#668FFF', movement: '#EDE9DD' });
const PEARL = '#FFF7E7';
function category(entry, row) {
  const id = entry.id || '', meta = entry.meta || {};
  if (entry.kind === 'enemy') return row === 'buff' ? (meta.behavior === 'healer' || /oracle/i.test(id) ? 'heal' : 'buff') : 'damage';
  if (entry.kind === 'basic') return 'damage';
  if (meta.category === 'mobility' && row !== 'impact') return 'movement';
  if (/^(fighter_guard|mage_mana_shield|guardian_impact_guard|guardian_oath_barrier|guardian_shield_wall)$/.test(id)) return 'shield';
  if (meta.category === 'buff') return 'buff';
  if (row === 'area' && /power_break|spell_mark|burning_mark|rune_mark|mana_seal|weak_point_mark|pierce_armor|snare|tripwire/.test(id)) return 'debuff';
  return 'damage';
}
function attrs(properties) { return Object.entries(properties).map(([key, value]) => `${key}="${value}"`).join(' '); }
function el(tag, properties) { return `<${tag} ${attrs(properties)}/>`; }
function path(d, properties) { return el('path', { d, ...properties }); }
function circle(x, y, radius, properties) { return el('circle', { cx: x, cy: y, r: radius, ...properties }); }
function plus(x, y, size, color, opacity) { return path(`M${x - size} ${y}H${x + size}M${x} ${y - size}V${y + size}`, { fill: 'none', stroke: color, 'stroke-width': size * 0.72, 'stroke-linecap': 'square', opacity }); }
function chevron(x, y, size, color, down = false) { return path(`M${x - size} ${y + (down ? -size / 2 : size / 2)}L${x} ${y + (down ? size / 2 : -size / 2)}L${x + size} ${y + (down ? -size / 2 : size / 2)}`, { fill: 'none', stroke: color, 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }); }
function spark(x, y, radius, color, angle = 0, opacity = 1) {
  return path(`M${x} ${y - radius}Q${x + radius * 0.16} ${y - radius * 0.16} ${x + radius} ${y}Q${x + radius * 0.16} ${y + radius * 0.16} ${x} ${y + radius}Q${x - radius * 0.16} ${y + radius * 0.16} ${x - radius} ${y}Q${x - radius * 0.16} ${y - radius * 0.16} ${x} ${y - radius}Z`, { fill: color, opacity, transform: `rotate(${angle} ${x} ${y})` });
}
function draw(entry, row, frame, count) {
  const p = frame / Math.max(1, count - 1), loop = frame / Math.max(1, count);
  const seed = Number(entry.identity && entry.identity.seed || Array.from(entry.id || '').reduce((a, c) => a * 31 + c.charCodeAt(0), 7)) >>> 0;
  const kind = category(entry, row), color = COLORS[kind], enemy = entry.kind === 'enemy';
  const accent = entry.palette && entry.palette.accent || PEARL;
  const pulse = Math.sin(loop * Math.PI * 2), phase = (seed % 23) * 0.07;
  const parts = [];
  const line = { fill: 'none', stroke: color, 'stroke-width': 4, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' };
  const boundary = enemy ? { 'stroke-dasharray': '10 5' } : {};
  if (kind === 'heal') {
    const radius = 23 + p * 32;
    parts.push(el('ellipse', { cx: 80, cy: 117, rx: radius, ry: radius * 0.3, ...line, ...boundary, opacity: 0.95 - p * 0.5 }));
    for (let i = 0; i < 3; i += 1) parts.push(plus(54 + i * 26, 104 - ((p + i * 0.17) % 1) * 57, 5 + (i === 1 ? 2 : 0), color, 0.95));
    parts.push(spark(80, 89, 8 + 6 * Math.sin(p * Math.PI), PEARL));
  } else if (kind === 'shield') {
    const width = 34 + 3 * pulse;
    parts.push(path(`M80 28Q${80 + width} 49 124 48L116 99Q105 122 80 132Q55 122 44 99L36 48Q${80 - width} 49 80 28Z`, { fill: color, 'fill-opacity': 0.09, ...line, 'stroke-width': 5 }));
    parts.push(path(`M80 42L108 56L102 99L80 116L58 99L52 56Z`, { fill: 'none', stroke: PEARL, 'stroke-width': 2, opacity: 0.6 }));
    parts.push(spark(62 + loop * 36, 44 + Math.sin(loop * Math.PI) * 4, 5, PEARL));
  } else if (kind === 'buff') {
    for (let i = 0; i < 3; i += 1) parts.push(chevron(80, 112 - i * 24 - p * 12, 12 + i * 2, color));
    for (let i = 0; i < 4; i += 1) parts.push(spark(47 + i * 22, 113 - ((loop + i * 0.22) % 1) * 80, 3 + (seed + i) % 3, i % 2 ? PEARL : color));
    if (enemy) parts.push(el('ellipse', { cx: 80, cy: 128, rx: 36, ry: 8, ...line, ...boundary, 'stroke-width': 2 }));
  } else if (kind === 'debuff') {
    parts.push(circle(80, 82, 34, { ...line, 'stroke-dasharray': '32 12', transform: `rotate(${p * 26} 80 82)` }));
    parts.push(chevron(80, 77 + p * 14, 13, color, true));
    const id = entry.id || '';
    if (/snare|seal|tripwire/.test(id)) parts.push(path('M61 62L99 100M99 62L61 100', { ...line, 'stroke-width': 3 }));
    else parts.push(circle(80, 68 + p * 12, 5, { fill: color }));
  } else if (kind === 'movement') {
    for (let i = 0; i < 3; i += 1) {
      const x = 45 + i * 27 + p * 7;
      parts.push(path(`M${x - 13} 51Q${x + 21} 80 ${x - 13} 109L${x - 3} 81Z`, { fill: color, opacity: 0.28 + i * 0.25 }));
    }
    parts.push(spark(63 + p * 48, 80, 5, accent, 0, 0.7));
  } else if (row === 'telegraph') {
    // Footprint geometry is runtime-owned; this compact gathering cue identifies its source.
    parts.push(el('ellipse', { cx: 80, cy: 116, rx: 38, ry: 11, ...line, ...boundary, opacity: 0.65 + 0.3 * pulse }));
    parts.push(path('M80 53L96 82H87V99H73V82H64Z', { fill: color, opacity: 0.65 + loop * 0.3 }));
    parts.push(circle(80, 109, 3, { fill: PEARL }));
  } else if (row === 'projectile') {
    const y = 80 + Math.sin(loop * Math.PI * 2) * 2;
    if (/arrow|archer|sniper|bow/i.test(entry.id || '')) {
      parts.push(path(`M31 ${y}H121M113 ${y - 10}L132 ${y}L113 ${y + 10}M36 ${y - 8}L47 ${y}L36 ${y + 8}`, { ...line, 'stroke-width': 4 }));
    } else {
      parts.push(path(`M25 ${y - 8}Q71 ${y - 14} 101 ${y - 21}Q133 ${y} 101 ${y + 21}Q74 ${y + 9} 25 ${y + 8}L66 ${y}Z`, { fill: color }));
      parts.push(path(`M66 ${y}Q97 ${y - 12} 120 ${y}Q100 ${y + 8} 66 ${y}Z`, { fill: PEARL }));
    }
    parts.push(spark(51 + loop * 40, 80, 3 + (seed % 4), accent, 30, 0.65));
  } else if (row === 'melee' || row === 'trail' || (row === 'impact' && entry.profile === 'melee')) {
    const tilt = (seed % 5 - 2) * 7;
    parts.push(`<g transform="rotate(${tilt} 80 80)" opacity="${0.95 - p * 0.38}">`);
    parts.push(path(`M30 121Q${47 + p * 18} 22 131 40Q82 45 56 106Z`, { fill: color }));
    parts.push(path('M31 121Q61 32 131 40Q77 42 43 112Z', { fill: PEARL }));
    parts.push('</g>');
    if (row === 'impact') parts.push(spark(110, 65, 12 + p * 12, color, 20, 1 - p * 0.6));
  } else if (row === 'cast') {
    const radius = 36 - p * 20;
    for (let i = 0; i < 5; i += 1) {
      const angle = phase + i * Math.PI * 2 / 5 + p * 0.8;
      parts.push(spark(80 + Math.cos(angle) * radius, 80 + Math.sin(angle) * radius, 6 + p * 4, color, angle * 180 / Math.PI));
    }
    parts.push(circle(80, 80, 4 + p * 13, { fill: color, opacity: 0.4 + p * 0.45 }));
    parts.push(spark(80, 80, 4 + p * 9, PEARL));
  } else if (row === 'area') {
    parts.push(el('ellipse', { cx: 80, cy: 107, rx: 47, ry: 17, ...line, ...boundary, 'stroke-width': 4 }));
    for (let i = 0; i < 5; i += 1) {
      const x = 43 + i * 18, height = 12 + ((i * 9 + seed) % 17) + 6 * pulse;
      parts.push(path(`M${x - 5} 105L${x} ${100 - height}L${x + 6} 105Z`, { fill: color, opacity: 0.5 + 0.4 * Math.sin((loop + i * 0.13) * Math.PI) }));
    }
  } else {
    const envelope = Math.sin((p * 0.82 + 0.08) * Math.PI), radius = 13 + 25 * envelope;
    parts.push(spark(80, 80, radius, color, phase * 17, 1 - p * 0.55));
    parts.push(spark(80, 80, radius * 0.58, PEARL, phase * 17));
    for (let i = 0; i < 5; i += 1) {
      const angle = phase + i * Math.PI * 2 / 5, r = 24 + p * 30;
      parts.push(spark(80 + Math.cos(angle) * r, 80 + Math.sin(angle) * r, 6 - p * 3, color, angle * 180 / Math.PI, 1 - p * 0.7));
    }
  }
  // A tiny authored accent differentiates abilities without changing their meaning.
  parts.push(spark(48 + seed % 61, 40 + (seed >>> 6) % 25 + pulse * 3, 2 + (seed >>> 12) % 3, accent, seed % 90, 0.45));
  const stageScale = row === 'cast' ? 0.72 + p * 0.2 : row === 'projectile' ? 0.78 + pulse * 0.025 : row === 'impact' ? 1 - p * 0.14 : row === 'area' ? 0.95 + pulse * 0.02 : 1;
  return `<g transform="translate(80 80) scale(${stageScale}) translate(-80 -80)">${parts.join('')}</g>`;
}
module.exports = { COLORS, category, draw };
