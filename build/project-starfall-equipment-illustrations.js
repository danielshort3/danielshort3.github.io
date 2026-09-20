'use strict';

// Editable vector masters for attachment art. Geometry stays inside radius 58;
// grip centers, atlas angles, row meanings and exported pivots remain unchanged.
const INK = '#242735';
const pick = (item, keys, fallback) => keys.map(key => item[key]).find(Boolean) || fallback;
const shade = (color, amount) => {
  const n = parseInt(color.replace('#', ''), 16);
  return '#' + [n >> 16, n >> 8 & 255, n & 255].map(v => Math.max(0, Math.min(255, Math.round(v + amount))).toString(16).padStart(2, '0')).join('');
};
const path = (d, fill, stroke = INK, width = 1.6) => `<path d="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round"/>`;
const line = (d, color, width = 1.5) => path(d, 'none', color, width);
const oval = (x, y, rx, ry, color, stroke = INK, width = 1.5) => `<ellipse cx="${x}" cy="${y}" rx="${rx}" ry="${ry}" fill="${color}" stroke="${stroke}" stroke-width="${width}"/>`;
const star = (x, y, size, color) => path(`M${x} ${y - size} L${x + size * .28} ${y - size * .28} L${x + size} ${y} L${x + size * .28} ${y + size * .28} L${x} ${y + size} L${x - size * .28} ${y + size * .28} L${x - size} ${y} L${x - size * .28} ${y - size * .28} Z`, color, INK, .9);

function family(id) {
  if (/eclipse|umbral|corona|sovereign|penumbra|sunfall/.test(id)) return 'eclipse';
  if (/thorn|briar|root|leaf|pathfinder|ranger|windrunner/.test(id)) return 'forest';
  if (/ember|molten|furnace|lava|scorch|cinder/.test(id)) return 'ember';
  if (/gear|chrono|ratchet|titan|clock|gyro|springstep/.test(id)) return 'clockwork';
  if (/colossus|geode|oreline|deepcore|bedrock|quarry|stonewake/.test(id)) return 'stone';
  if (/storm|cloud|sky|roc|tempest|lightning|gale/.test(id)) return 'storm';
  if (/star|rune|index|comet|archivist|astral|scribe|orbit|aether|channeler/.test(id)) return 'astral';
  return 'frontier';
}

function ornament(type, x, y, color, scale = 1) {
  let body;
  if (type === 'forest') body = path('M-7 5 Q-8 -5 6 -8 Q10 2 -7 5Z', color) + line('M-7 5 L4 -5', '#f5f0cc', 1);
  else if (type === 'ember') body = path('M0 -9 Q7 -2 4 5 Q0 10 -5 4 Q-7 0 -2 -3 Q0 0 0 -9Z', color) + path('M0 0 L3 5 L-2 5Z', '#fff1ad', 'none');
  else if (type === 'clockwork') body = path('M-3 -8 L3 -8 L4 -5 L8 -3 L8 3 L5 4 L3 8 L-3 8 L-4 5 L-8 3 L-8 -3 L-5 -4Z', color) + oval(0, 0, 3, 3, INK);
  else if (type === 'stone') body = path('M-8 -4 L-2 -9 L7 -5 L8 4 L0 9 L-7 4Z', color) + line('M-2 -9 L0 0 L8 4 M0 0 L-7 4', '#f2f4f0', 1.2);
  else if (type === 'storm') body = path('M3 -10 L-6 1 L0 1 L-3 10 L7 -3 L1 -3Z', color);
  else if (type === 'eclipse') body = oval(0, 0, 8, 8, color) + oval(2, -2, 6.5, 6.5, '#313349', 'none');
  else body = star(0, 0, 8, color);
  return `<g transform="translate(${x} ${y}) scale(${scale})">${body}</g>`;
}

function drawItem(item, state, suffix = '') {
  const type = family(item.id), prefix = item.fileId + '-' + suffix;
  const base = pick(item, ['blade', 'metal', 'cloth', 'leather', 'wood', 'rod', 'core'], '#8d9caf');
  const trim = pick(item, ['trim', 'buckle', 'bright', 'edge', 'shine'], '#d9b567');
  const dark = pick(item, ['dark', 'sole', 'stitch'], '#3e3640');
  const grip = pick(item, ['grip', 'haft', 'rod', 'leather', 'wood'], '#815b3f');
  const gem = pick(item, ['gem', 'core', 'glow', 'accent', 'lens'], '#83d5ed');
  const gradient = (id, color) => `<linearGradient id="${prefix}-${id}" x1="0" y1="0" x2=".72" y2="1"><stop stop-color="${shade(color, 48)}"/><stop offset=".38" stop-color="${color}"/><stop offset="1" stop-color="${shade(color, -42)}"/></linearGradient>`;
  const defs = '<defs>' + gradient('base', base) + gradient('trim', trim) + gradient('grip', grip) + gradient('gem', gem) + '</defs>';
  const paint = id => `url(#${prefix}-${id})`;
  const motif = (x, y, scale) => ornament(type, x, y, trim, scale);
  const handle = (start, end, y = 0) => path(`M${start} ${y - 4} Q${(start + end) / 2} ${y - 6} ${end} ${y - 3} L${end} ${y + 4} Q${(start + end) / 2} ${y + 6} ${start} ${y + 4}Z`, paint('grip')) + line(`M${start + 4} ${y - 3} L${start + 8} ${y + 4} M${start + 10} ${y - 3} L${start + 14} ${y + 4}`, shade(grip, -30), 1.3);
  let art = '';
  if (item.kind === 'sword') {
    const curved = /cutlass|saber|eclipse/.test(item.id), broad = /greatsword|vanguard/.test(item.id);
    const blade = curved ? 'M-17 -5 Q14 -7 47 -20 Q41 4 7 10 L-17 5Z' : `M-17 -6 L${broad ? 27 : 32} ${broad ? -11 : -7} L48 0 L${broad ? 27 : 32} ${broad ? 11 : 7} L-17 6Z`;
    art = handle(-39, -17) + oval(-39, 0, 5, 6, paint('trim')) + path(blade, paint('base')) + path(curved ? 'M-15 -4 Q14 -6 44 -17 Q25 1 -15 1Z' : 'M-14 -4 L32 -5 L45 0 L-14 0Z', '#f4f1e7', 'none') + path('M-23 -11 Q-18 -15 -15 -8 L-14 8 Q-18 15 -23 11 L-20 2 L-20 -2Z', paint('trim'));
    if (type !== 'frontier') art += motif(5, 0, .45);
    if (type === 'forest') art += path('M4 -7 L10 -15 L13 -7 M20 8 L25 15 L27 8', base);
  } else if (item.kind === 'axe') {
    art = handle(-41, 33) + oval(-41, 0, 4, 5, paint('trim'));
    if (/maul/.test(item.id)) art += path('M20 -23 L44 -20 L49 -12 L47 20 L23 23 L17 14 L18 -15Z', paint('base')) + path('M21 -19 L43 -16 L43 -8 L21 -10Z', shade(base, 45)) + motif(32, 1, .85);
    else art += path('M23 -4 Q18 -19 29 -24 Q34 -12 49 -18 Q52 0 43 11 Q32 7 24 4Z', paint('base')) + path('M29 -22 Q38 -11 48 -16 Q47 -5 42 0 Q36 -7 26 -6Z', shade(base, 54), 'none') + path('M24 4 Q20 14 30 23 Q39 19 42 11 L34 8Z', paint('base')) + motif(29, 0, .6);
  } else if (item.kind === 'staff' || item.kind === 'wand') {
    const staff = item.kind === 'staff', start = staff ? -43 : -33, end = staff ? 34 : 25;
    art = handle(start, end) + oval(start, 0, 3, 5, paint('trim')) + path(`M${end - 8} -7 Q${end - 2} -20 ${end + 8} -15 Q${end + 18} -1 ${end + 8} 15 Q${end - 2} 20 ${end - 8} 7 L${end - 2} 6 Q${end + 13} 1 ${end + 2} -9 L${end - 2} -4Z`, paint('trim'));
    if (/codex/.test(item.id)) art += path('M24 -17 L43 -14 L49 12 L29 17 L22 10Z', paint('base')) + path('M28 -12 L42 -10 L45 8 L30 12Z', '#eadfbf') + star(36, 0, 7, gem);
    else if (type === 'clockwork') art += ornament(type, end + 4, 0, trim, 1.25) + oval(end + 4, 0, 5, 5, paint('gem'));
    else if (type === 'eclipse') art += ornament(type, end + 5, 0, trim, 1.6) + star(end + 4, 0, 4, gem);
    else art += path(`M${end + 4} -15 L${end + 14} -2 L${end + 4} 13 L${end - 5} -1Z`, paint('gem')) + path(`M${end + 4} -12 L${end + 4} 8 L${end - 2} -1Z`, '#e5f8ff', 'none');
    if (type === 'forest') art += ornament('forest', 15, -8, '#91b97b', .8);
  } else if (item.kind === 'bow') {
    const half = item.long ? 43 : 38, pull = state === 'draw' ? -22 : state === 'release' ? -7 : 0;
    const xStart = state === 'draw' ? -30 : state === 'release' ? -14 : -10, end = state === 'release' ? 44 : 35;
    art = path(`M-3 ${-half} Q15 ${-half + 10} 14 -15 L10 0 L14 15 Q15 ${half - 10} -3 ${half} L1 ${half - 8} Q10 23 5 9 L4 0 L5 -9 Q10 -23 1 ${-half + 8}Z`, paint('base')) + line(`M0 ${-half + 1} L${pull} 0 L0 ${half - 1}`, '#ede3ca', 1.2) + line(`M2 ${-half + 6} Q12 -22 8 -10 M8 10 Q12 22 2 ${half - 6}`, shade(base, 50), 1.8) + handle(3, 14) + line(`M${xStart} 0 L${end} 0`, '#d5b878', 2) + path(`M${end} -4 L${end + 8} 0 L${end} 4 L${end + 2} 0Z`, paint('trim')) + path(`M${xStart} 0 L${xStart + 7} -5 L${xStart + 5} 0 L${xStart + 7} 5Z`, '#e9e4d7');
    if (type !== 'frontier') art += motif(9, -22, .55) + motif(9, 22, .55);
  } else if (item.kind === 'chest') {
    const robes = /robe|mantle/.test(item.id), plate = /plate|bulwark|harness/.test(item.id);
    const hem = robes ? 41 : 32;
    art = path(`M-11 -34 Q0 -25 11 -34 L28 -24 L23 4 L${robes ? 31 : 23} ${hem} Q0 ${hem + 5} ${robes ? -31 : -23} ${hem} L-23 4 L-28 -24Z`, paint('base')) + path('M-11 -34 L-4 -23 L0 -17 L4 -23 L11 -34 L18 -29 L10 -13 L0 -9 L-10 -13 L-18 -29Z', paint('trim')) + path(`M-23 3 Q-10 10 -8 ${hem - 2} L0 ${hem} L0 -9 L-10 -13Z`, shade(base, -24), 'none') + line(`M4 -11 L4 ${hem - 5} M-22 ${hem - 4} Q0 ${hem + 1} 22 ${hem - 4}`, trim, 2.5) + path('M-23 16 Q0 20 23 16 L23 22 Q0 26 -23 22Z', paint('grip')) + path('M-5 16 L5 16 L5 24 L-5 24Z', paint('trim')) + motif(0, 3, .7);
    if (plate) art += path('M-28 -24 L-15 -30 L-11 -17 L-30 -12 L-34 -17Z', paint('trim')) + path('M28 -24 L15 -30 L11 -17 L30 -12 L34 -17Z', paint('trim')) + line('M-18 0 Q0 7 18 0 M-17 8 Q0 14 17 8', shade(base, 35), 1.5);
    if (type === 'forest') art += ornament('forest', -23, 28, '#8bab64', .9) + ornament('forest', 23, 28, '#8bab64', .9);
  } else if (item.kind === 'boots') {
    art = path('M-12 -29 Q0 -33 12 -29 L11 8 Q15 14 25 16 Q30 23 23 27 L-14 27 Q-18 22 -15 15 L-12 8Z', paint('base')) + path('M-14 -29 Q0 -24 14 -29 L13 -20 Q0 -15 -14 -20Z', paint('trim')) + path('M-15 18 Q0 23 26 20 L25 27 Q3 31 -15 27Z', dark) + path('M-10 -18 L-4 -16 L-5 11 Q0 17 11 18 L0 20 Q-10 16 -12 11Z', shade(base, 28), 'none') + line('M-12 -5 L11 -5', grip, 5) + path('M0 -9 L8 -9 L8 -1 L0 -1Z', paint('trim'));
    if (type !== 'frontier') art += motif(0, -15, .48);
  } else if (item.kind === 'head') {
    const crown = /crown/.test(item.id), mask = /mask/.test(item.id);
    if (crown) art = path('M-32 4 L-35 -23 L-19 -13 L-9 -35 L0 -16 L15 -36 L23 -11 L35 -24 L31 9 Q0 22 -32 4Z', paint('trim')) + path('M-29 1 Q0 12 29 1 L29 11 Q0 24 -29 11Z', paint('base')) + motif(0, 0, 1.1);
    else art = path('M-31 18 L-29 -12 Q-23 -36 0 -37 Q23 -36 29 -12 L31 18 L20 29 L13 13 L-13 13 L-20 29Z', paint('base')) + path('M-23 -10 Q-18 -30 0 -31 L0 -12 L-23 1Z', shade(base, 35), 'none') + path('M-28 2 Q0 -7 28 2 L24 15 L10 11 L0 19 L-10 11 L-24 15Z', dark) + line('M0 -34 L0 -13 M-26 0 Q0 -9 26 0', trim, 3) + motif(0, -18, .7);
    if (mask || type === 'storm') art += path('M-29 -8 L-43 -24 L-34 3 L-26 9 M29 -8 L43 -24 L34 3 L26 9', paint('trim'));
  } else if (item.kind === 'gloves' || item.kind === 'grip') {
    art = path('M-13 13 L-17 -1 Q-18 -7 -13 -10 L-9 -4 L-9 -14 Q-6 -20 -1 -16 Q3 -21 6 -16 Q11 -18 13 -12 L17 2 Q16 12 6 19Z', paint('base')) + path('M-16 9 L10 13 L7 23 L-16 18Z', paint('grip')) + line('M-7 -12 L-4 0 M0 -14 L3 -1 M7 -12 L9 0', shade(base, -40), 1.3) + path('M-11 -3 L8 -6 L11 6 L-6 9Z', paint('trim')) + motif(0, 1, .4);
  } else if (item.kind === 'ring' || item.kind === 'amulet') {
    if (item.kind === 'ring') art = oval(0, 7, 20, 16, paint('trim')) + oval(0, 7, 12, 9, 'none', INK, 3) + path('M-8 -11 L0 -22 L10 -11 L0 0Z', paint('gem')) + path('M0 -19 L0 -4 L-5 -11Z', '#f4fffa', 'none');
    else art = line('M-27 -30 Q-5 8 0 0 Q12 1 27 -30', INK, 5) + line('M-27 -30 Q-5 8 0 0 Q12 1 27 -30', trim, 2.7) + oval(0, 0, 4, 5, 'none', trim, 2) + path('M0 2 L16 17 L0 35 L-16 17Z', paint('trim')) + path('M0 8 L10 18 L0 28 L-10 18Z', paint('gem')) + star(0, 17, 6, '#f0f7e5');
  } else if (item.kind === 'shield') {
    art = path('M0 -39 Q22 -31 36 -32 L32 18 Q19 35 0 44 Q-19 35 -32 18 L-36 -32 Q-22 -31 0 -39Z', paint('trim')) + path('M0 -31 Q19 -25 28 -25 L25 13 Q16 28 0 35 Q-16 28 -25 13 L-28 -25Q-19 -25 0 -31Z', paint('base')) + path('M0 -30 L0 34 Q-15 26 -24 12 L-27 -23Z', shade(base, 25), 'none') + star(0, -1, 20, trim) + oval(0, -1, 6, 6, paint('gem'));
  } else if (item.kind === 'core' || item.kind === 'focus') {
    art = path('M0 -35 L25 -19 L31 13 L12 34 L-19 28 L-30 4 L-22 -24Z', paint('trim')) + path('M0 -29 L20 -16 L24 10 L10 27 L-15 22 L-24 2 L-17 -19Z', paint('gem')) + path('M0 -29 L1 0 L-24 2 L-17 -19Z', '#effcf0', 'none') + path('M1 0 L20 -16 L24 10 L10 27Z', shade(gem, -42), 'none') + ornament(item.kind === 'core' ? 'ember' : 'astral', 1, 1, '#fff4c5', 1.1);
  } else if (item.kind === 'scope') {
    art = path('M-36 -9 L27 -13 L35 -8 L35 11 L26 15 L-36 9Z', paint('base')) + path('M-26 -12 L-17 -12 L-17 13 L-26 13Z', paint('trim')) + path('M22 -17 L33 -17 L37 -9 L37 11 L32 18 L22 18Z', paint('trim')) + oval(34, 0, 7, 13, paint('gem')) + oval(33, -4, 2, 4, '#edffff', 'none') + path('M-13 10 L1 12 L0 26 L-14 24Z', paint('base'));
  } else if (item.kind === 'kit') {
    art = path('M-34 -24 Q-6 -34 29 -24 L34 19 Q23 36 -27 30 L-37 15Z', paint('grip')) + path('M-34 -24 Q-2 -35 29 -24 L27 -5 Q0 4 -32 -8Z', paint('base')) + path('M-7 -24 L5 -24 L6 23 L-7 23Z', dark) + path('M-8 -8 L7 -8 L7 5 L-8 5Z', paint('trim')) + line('M-27 16 Q0 26 27 15', shade(grip, 46), 1.4) + path('M-27 -24 L-22 -39 L-16 -38 L-16 -25 M18 -25 L22 -39 L28 -36 L25 -23', paint('trim')) + line('M30 -9 Q46 1 34 24', '#cec4a8', 3);
  } else throw new Error('No illustrated equipment renderer for ' + item.kind);
  return defs + art;
}

module.exports = { drawItem, family };
