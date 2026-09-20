(function initProjectStarfallEquipmentAttachments(global) {
  'use strict';

const ATLAS_CELL_SIZE = 128;
const ATLAS_ANGLE_SETS = Object.freeze({
  weapon: Object.freeze([-62, -34, -7, 8, 25, 42, 62, 78]),
  body: Object.freeze([-42, -18, -7, 0, 16, 42, 72, 90]),
  limb: Object.freeze([-15, -7, 0, 5, 10, 15, 20, 30])
});
const WEAPON_KINDS = Object.freeze(['sword', 'axe', 'wand', 'staff', 'bow']);
const LIMB_KINDS = Object.freeze(['boots', 'gloves', 'grip', 'ring', 'focus']);

const ROWS = Object.freeze(['idle', 'run', 'jump', 'fall', 'climb', 'basic', 'skill', 'party', 'hit', 'defeat']);

function attachment(torso, head, mainHand, offHand, feet, options) {
  const settings = options || {};
  return Object.freeze({
    torso: Object.freeze({ x: torso[0], y: torso[1], width: torso[2], height: torso[3], angle: torso[4] || 0 }),
    head: Object.freeze({ x: head[0], y: head[1], angle: head[2] || 0 }),
    mainHand: Object.freeze({ x: mainHand[0], y: mainHand[1], angle: mainHand[2] || 0 }),
    offHand: Object.freeze({ x: offHand[0], y: offHand[1], angle: offHand[2] || 0 }),
    feet: Object.freeze(feet.map((foot) => Object.freeze({ x: foot[0], y: foot[1], angle: foot[2] || 0 }))),
    weapon: Object.freeze({
      x: settings.weapon && settings.weapon[0] != null ? settings.weapon[0] : mainHand[0],
      y: settings.weapon && settings.weapon[1] != null ? settings.weapon[1] : mainHand[1],
      angle: settings.weapon && settings.weapon[2] != null ? settings.weapon[2] : 55,
      mode: settings.weaponMode || 'held'
    }),
    shield: Object.freeze({
      x: settings.shield && settings.shield[0] != null ? settings.shield[0] : offHand[0],
      y: settings.shield && settings.shield[1] != null ? settings.shield[1] : offHand[1],
      angle: settings.shield && settings.shield[2] != null ? settings.shield[2] : torso[4] || 0
    }),
    bowAngle: Number(settings.bowAngle || 0),
    scale: Number(settings.scale || 1)
  });
}

const A = attachment;

const ATTACHMENTS = Object.freeze({
  idle: Object.freeze([
    A([81, 94, 30, 36, 0], [82, 42, 0], [105, 96], [55, 97], [[98, 148], [64, 148]], { weapon: [103, 95, 62] }),
    A([80, 94, 30, 36, 0], [81, 42, 0], [104, 96], [54, 97], [[98, 148], [63, 148]], { weapon: [102, 95, 62] }),
    A([76, 93, 30, 36, 0], [77, 41, 0], [101, 95], [50, 96], [[93, 146], [59, 146]], { weapon: [99, 94, 62] }),
    A([78, 94, 30, 36, 0], [79, 42, 0], [102, 96], [52, 97], [[95, 148], [61, 148]], { weapon: [100, 95, 62] }),
    A([79, 94, 30, 36, 0], [80, 42, 0], [103, 96], [53, 97], [[96, 147], [62, 147]], { weapon: [101, 95, 62] }),
    A([77, 94, 30, 36, 0], [78, 42, 0], [101, 96], [51, 97], [[94, 148], [60, 148]], { weapon: [99, 95, 62] })
  ]),
  run: Object.freeze([
    A([78, 96, 32, 35, -12], [77, 48, -8], [108, 80, -12], [39, 86, -12], [[113, 143, 18], [42, 145, -8]], { weapon: [105, 79, 32] }),
    A([77, 97, 32, 35, -13], [76, 49, -8], [109, 80, -12], [38, 88, -12], [[109, 144, 10], [47, 146, -12]], { weapon: [106, 79, 30] }),
    A([75, 91, 32, 35, -12], [75, 43, -8], [106, 75, -12], [37, 82, -12], [[102, 135, 18], [45, 134, -8]], { weapon: [103, 74, 30] }),
    A([78, 92, 32, 35, -12], [78, 44, -8], [108, 76, -12], [40, 83, -12], [[108, 139, 16], [46, 137, -8]], { weapon: [105, 75, 30] }),
    A([76, 91, 32, 35, -12], [76, 43, -8], [105, 75, -12], [39, 83, -12], [[101, 137, 18], [46, 137, -8]], { weapon: [102, 74, 30] }),
    A([77, 98, 32, 35, -10], [77, 50, -7], [106, 82, -10], [41, 89, -10], [[102, 147, 12], [49, 145, -6]], { weapon: [103, 81, 30] })
  ]),
  jump: Object.freeze([
    A([77, 115, 32, 33, 2], [77, 75, 0], [103, 124], [48, 125], [[100, 147, 2], [58, 148, -4]], { weapon: [100, 122, 42] }),
    A([75, 91, 32, 34, -14], [74, 48, -10], [111, 82], [39, 101], [[91, 136, 15], [50, 132, -8]], { weapon: [107, 81, 30] }),
    A([78, 78, 34, 33, -18], [80, 37, -12], [112, 73], [39, 91], [[101, 115, 20], [53, 123, -8]], { weapon: [108, 72, 23] }),
    A([78, 74, 34, 33, -18], [81, 34, -12], [119, 71], [38, 84], [[99, 104, 18], [57, 109, -7]], { weapon: [114, 70, 15] }),
    A([72, 99, 32, 34, -9], [74, 58, -7], [104, 96], [39, 106], [[91, 145, 10], [54, 143, -6]], { weapon: [101, 95, 38] }),
    A([75, 116, 32, 33, 1], [75, 76, 0], [102, 124], [48, 129], [[98, 148, 2], [59, 148, -3]], { weapon: [99, 122, 48] })
  ]),
  fall: Object.freeze([
    A([77, 112, 34, 34, -42], [74, 82, -35], [113, 114], [39, 126], [[123, 143, 14], [96, 143, 4]], { weapon: [109, 112, 16] }),
    A([78, 94, 34, 34, -38], [76, 62, -30], [119, 76], [34, 87], [[121, 122, 14], [94, 124, 4]], { weapon: [115, 75, 8] }),
    A([79, 82, 34, 34, -34], [79, 51, -28], [124, 65], [29, 76], [[116, 112, 15], [95, 128, 3]], { weapon: [120, 64, 4] }),
    A([79, 78, 34, 34, -33], [79, 47, -27], [126, 61], [30, 72], [[116, 105, 14], [96, 124, 3]], { weapon: [122, 60, 2] }),
    A([78, 96, 34, 34, -38], [78, 64, -30], [123, 82], [32, 94], [[124, 131, 15], [98, 138, 3]], { weapon: [119, 81, 9] }),
    A([79, 113, 34, 34, -42], [76, 88, -35], [128, 104], [38, 122], [[131, 139, 14], [101, 142, 3]], { weapon: [124, 103, 16] })
  ]),
  climb: Object.freeze([
    A([72, 91, 30, 35, 0], [73, 45, 0], [97, 27], [53, 29], [[84, 130, 3], [60, 143, -4]], { weapon: [52, 102, 78], weaponMode: 'stowed' }),
    A([70, 88, 30, 35, 0], [71, 41, 0], [97, 21], [55, 24], [[76, 121, 3], [53, 139, -4]], { weapon: [50, 99, 78], weaponMode: 'stowed' }),
    A([77, 86, 30, 35, 0], [79, 40, 0], [104, 18], [65, 24], [[97, 118, 3], [68, 133, -4]], { weapon: [57, 97, 78], weaponMode: 'stowed' }),
    A([73, 93, 30, 35, 0], [74, 46, 0], [100, 25], [55, 27], [[88, 133, 3], [60, 148, -4]], { weapon: [53, 104, 78], weaponMode: 'stowed' }),
    A([71, 92, 30, 35, 0], [72, 46, 0], [95, 27], [50, 27], [[74, 138, 3], [50, 131, -4]], { weapon: [51, 103, 78], weaponMode: 'stowed' }),
    A([77, 91, 30, 35, 0], [78, 44, 0], [101, 23], [58, 25], [[91, 132, 3], [64, 143, -4]], { weapon: [57, 102, 78], weaponMode: 'stowed' })
  ]),
  basic: Object.freeze([
    A([76, 94, 32, 36, 0], [77, 43, 0], [108, 89], [46, 98], [[96, 146], [57, 146]], { weapon: [105, 88, -62] }),
    A([77, 92, 33, 35, -5], [78, 42, -3], [116, 89], [47, 94], [[107, 144, 7], [49, 145, -5]], { weapon: [108, 88, -34] }),
    A([75, 92, 34, 35, -7], [76, 42, -4], [124, 88], [44, 94], [[104, 143, 8], [48, 144, -5]], { weapon: [109, 87, -6] }),
    A([77, 92, 34, 35, -7], [78, 42, -4], [129, 88], [46, 94], [[106, 144, 8], [49, 145, -5]], { weapon: [111, 87, 8] }),
    A([72, 97, 34, 34, -11], [75, 48, -8], [122, 94], [42, 101], [[103, 144, 9], [46, 144, -5]], { weapon: [108, 92, 25] }),
    A([77, 94, 32, 36, 0], [78, 43, 0], [111, 94], [52, 98], [[98, 146], [60, 146]], { weapon: [107, 93, 56] })
  ]),
  skill: Object.freeze([
    A([73, 95, 31, 36, 0], [74, 44, 0], [103, 94], [49, 99], [[93, 146], [57, 146]], { weapon: [101, 93, -58] }),
    A([75, 95, 32, 36, -2], [76, 45, -2], [109, 91], [47, 99], [[96, 146], [55, 146]], { weapon: [106, 90, -35] }),
    A([78, 93, 34, 35, -6], [79, 43, -4], [116, 89], [46, 97], [[101, 145], [52, 145]], { weapon: [111, 88, -8] }),
    A([78, 93, 34, 35, -6], [79, 43, -4], [117, 88], [46, 97], [[102, 146], [52, 146]], { weapon: [112, 87, 5] }),
    A([75, 98, 33, 34, -8], [77, 48, -6], [116, 94], [45, 101], [[99, 145], [53, 145]], { weapon: [110, 92, 25] }),
    A([71, 95, 31, 36, 0], [72, 43, 0], [101, 94], [47, 99], [[91, 146], [55, 146]], { weapon: [99, 93, 58] })
  ]),
  party: Object.freeze([
    A([74, 96, 31, 36, 0], [75, 45, 0], [96, 27], [51, 99], [[93, 151], [58, 151]], { weapon: [48, 105, 78], weaponMode: 'stowed' }),
    A([73, 96, 31, 36, 0], [74, 46, 0], [95, 28], [50, 99], [[92, 151], [57, 151]], { weapon: [47, 105, 78], weaponMode: 'stowed' }),
    A([71, 95, 31, 36, 0], [72, 44, 0], [94, 27], [48, 98], [[90, 150], [55, 150]], { weapon: [45, 104, 78], weaponMode: 'stowed' }),
    A([73, 96, 31, 36, 0], [74, 46, 0], [95, 28], [50, 99], [[92, 151], [57, 151]], { weapon: [47, 105, 78], weaponMode: 'stowed' }),
    A([71, 95, 31, 36, 0], [72, 44, 0], [94, 31], [48, 98], [[90, 150], [55, 150]], { weapon: [45, 104, 78], weaponMode: 'stowed' }),
    A([72, 96, 31, 36, 0], [73, 45, 0], [101, 94], [49, 99], [[91, 151], [56, 151]], { weapon: [99, 93, 58] })
  ]),
  hit: Object.freeze([
    A([74, 95, 31, 36, -4], [75, 42, -4], [102, 91], [48, 96], [[93, 147], [58, 147]], { weapon: [100, 90, 62] }),
    A([69, 96, 31, 35, 8], [70, 47, 8], [99, 94], [44, 98], [[87, 147], [54, 148]], { weapon: [97, 93, 70] }),
    A([71, 97, 31, 35, 6], [72, 48, 6], [101, 95], [44, 99], [[90, 146], [55, 147]], { weapon: [99, 94, 68] }),
    A([74, 97, 31, 35, 3], [75, 47, 3], [103, 95], [47, 100], [[93, 148], [58, 148]], { weapon: [101, 94, 65] }),
    A([73, 95, 31, 36, 1], [74, 44, 1], [102, 93], [47, 98], [[92, 148], [57, 148]], { weapon: [100, 92, 63] }),
    A([74, 94, 31, 36, 0], [75, 42, 0], [103, 92], [48, 97], [[93, 149], [58, 149]], { weapon: [101, 91, 62] })
  ]),
  defeat: Object.freeze([
    A([75, 105, 34, 34, 12], [78, 67, 10], [101, 119], [49, 123], [[99, 133, 8], [57, 134, -5]], { weapon: [91, 136, 5], weaponMode: 'dropped', shield: [56, 126, 12] }),
    A([75, 105, 35, 34, 16], [78, 65, 14], [103, 120], [51, 122], [[103, 131, 8], [59, 132, -5]], { weapon: [89, 137, 4], weaponMode: 'dropped', shield: [58, 126, 16] }),
    A([65, 111, 35, 36, 72], [91, 93, 72], [112, 121], [38, 122], [[31, 124, 0], [49, 127, 0]], { weapon: [84, 137, 2], weaponMode: 'dropped', shield: [44, 120, 72] }),
    A([61, 112, 36, 37, 84], [98, 101, 84], [121, 122], [31, 122], [[22, 126, 0], [42, 127, 0]], { weapon: [80, 137, 1], weaponMode: 'dropped', shield: [39, 121, 84] }),
    A([59, 112, 36, 37, 88], [101, 102, 88], [125, 123], [27, 122], [[17, 126, 0], [38, 128, 0]], { weapon: [77, 137, 0], weaponMode: 'dropped', shield: [35, 122, 88] }),
    A([56, 113, 36, 37, 90], [104, 103, 90], [128, 124], [24, 123], [[13, 127, 0], [35, 129, 0]], { weapon: [74, 137, 0], weaponMode: 'dropped', shield: [32, 123, 90] })
  ])
});

const PLAYER_FRAME_REGISTRATIONS = Object.freeze({
  idle: Object.freeze([
    Object.freeze({ originX: 81, groundY: 153, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 80, groundY: 153, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 76, groundY: 151, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 78, groundY: 153, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 79, groundY: 152, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 77, groundY: 153, authoredBodyHeight: 143 })
  ]),
  run: Object.freeze([
    Object.freeze({ originX: 78, groundY: 149, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 77, groundY: 149, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 75, groundY: 138, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 78, groundY: 144, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 76, groundY: 141, authoredBodyHeight: 143 }),
    Object.freeze({ originX: 77, groundY: 150, authoredBodyHeight: 143 })
  ])
});

// BEGIN ILLUSTRATED PLAYER SOCKETS
const OVERHAUL_ATTACHMENTS = Object.freeze({
  idle: Object.freeze([
    A([82,88,30,34,0], [96,46,0], [117,108,0], [65,102,0], [[105,143,0],[71,143,0]], {"weapon":[117,108,62],"weaponMode":"held"}),
    A([82,89,30,34,0], [93,46,0], [114,108,0], [62,102,0], [[102,143,0],[68,143,0]], {"weapon":[114,108,62],"weaponMode":"held"}),
    A([82,88,30,34,0], [94,46,0], [115,108,0], [63,102,0], [[103,143,0],[69,143,0]], {"weapon":[115,108,62],"weaponMode":"held"}),
    A([82,89,30,34,0], [92,46,0], [113,108,0], [61,103,0], [[101,143,0],[67,143,0]], {"weapon":[113,108,62],"weaponMode":"held"}),
    A([82,90,30,34,0], [94,49,0], [116,110,0], [63,104,0], [[103,143,0],[69,143,0]], {"weapon":[116,110,62],"weaponMode":"held"}),
    A([82,92,30,34,0], [93,50,0], [114,110,0], [62,105,0], [[102,144,0],[68,144,0]], {"weapon":[114,110,62],"weaponMode":"held"}),
    A([82,91,30,34,0], [95,49,0], [117,110,0], [64,104,0], [[104,143,0],[70,143,0]], {"weapon":[117,110,62],"weaponMode":"held"}),
    A([82,90,30,34,0], [93,49,0], [115,110,0], [62,104,0], [[102,143,0],[68,143,0]], {"weapon":[115,110,62],"weaponMode":"held"})
  ]),
  run: Object.freeze([
    A([82,86,30,34,0], [92,49,0], [117,95,0], [55,104,0], [[114,142,15],[49,129,-10]], {"weapon":[117,95,30],"weaponMode":"held"}),
    A([82,86,30,34,0], [100,49,0], [121,93,0], [69,104,0], [[91,142,15],[65,128,-10]], {"weapon":[121,93,24],"weaponMode":"held"}),
    A([82,84,30,34,0], [97,45,0], [121,92,0], [60,102,0], [[119,140,15],[54,126,-10]], {"weapon":[121,92,35],"weaponMode":"held"}),
    A([82,82,30,34,0], [93,48,0], [118,87,0], [54,101,0], [[117,137,15],[48,125,-10]], {"weapon":[118,87,40],"weaponMode":"held"}),
    A([82,86,30,34,0], [86,50,0], [120,97,0], [49,105,0], [[109,142,15],[43,129,-10]], {"weapon":[120,97,30],"weaponMode":"held"}),
    A([82,85,30,34,0], [93,49,0], [120,97,0], [61,104,0], [[83,142,15],[56,128,-10]], {"weapon":[120,97,24],"weaponMode":"held"}),
    A([82,81,30,34,0], [86,47,0], [122,94,0], [48,102,0], [[109,140,15],[42,126,-10]], {"weapon":[122,94,35],"weaponMode":"held"}),
    A([82,82,30,34,0], [85,47,0], [122,88,0], [45,101,0], [[110,138,15],[39,124,-10]], {"weapon":[122,88,40],"weaponMode":"held"})
  ]),
  jump: Object.freeze([
    A([82,99,30,34,0], [104,69,0], [116,132,0], [66,113,0], [[115,145,0],[73,145,0]], {"weapon":[116,132,62],"weaponMode":"held"}),
    A([82,91,30,34,0], [94,55,0], [120,107,0], [58,107,0], [[104,144,0],[64,144,0]], {"weapon":[120,107,62],"weaponMode":"held"}),
    A([82,78,30,34,0], [102,46,0], [131,50,0], [69,103,0], [[112,143,0],[75,143,0]], {"weapon":[131,50,62],"weaponMode":"held"}),
    A([82,85,30,34,0], [87,58,0], [110,88,0], [53,108,0], [[96,144,0],[59,144,0]], {"weapon":[110,88,62],"weaponMode":"held"}),
    A([82,85,30,34,0], [99,54,0], [122,81,0], [62,106,0], [[109,143,0],[69,143,0]], {"weapon":[122,81,62],"weaponMode":"held"}),
    A([82,88,30,34,0], [94,58,0], [123,89,0], [60,108,0], [[104,144,0],[66,144,0]], {"weapon":[123,89,62],"weaponMode":"held"}),
    A([82,90,30,34,0], [97,58,0], [131,91,0], [59,108,0], [[108,144,0],[66,144,0]], {"weapon":[131,91,62],"weaponMode":"held"}),
    A([82,84,30,34,0], [93,55,0], [128,89,0], [56,107,0], [[104,144,0],[62,144,0]], {"weapon":[128,89,62],"weaponMode":"held"})
  ]),
  fall: Object.freeze([
    A([82,86,30,34,0], [97,55,0], [130,91,0], [60,107,0], [[107,143,0],[67,143,0]], {"weapon":[130,91,62],"weaponMode":"held"}),
    A([82,85,30,34,0], [90,55,0], [115,93,0], [56,107,0], [[100,144,0],[63,144,0]], {"weapon":[115,93,62],"weaponMode":"held"}),
    A([82,87,30,34,0], [95,57,0], [125,97,0], [57,108,0], [[106,144,0],[64,144,0]], {"weapon":[125,97,62],"weaponMode":"held"}),
    A([82,87,30,34,0], [104,64,0], [128,107,0], [67,111,0], [[115,144,0],[74,144,0]], {"weapon":[128,107,62],"weaponMode":"held"}),
    A([82,89,30,34,0], [101,62,0], [124,108,0], [65,110,0], [[112,144,0],[72,144,0]], {"weapon":[124,108,62],"weaponMode":"held"}),
    A([82,95,30,34,0], [90,66,0], [118,111,0], [58,112,0], [[99,145,0],[64,145,0]], {"weapon":[118,111,62],"weaponMode":"held"}),
    A([82,94,30,34,0], [92,70,0], [119,114,0], [60,114,0], [[101,145,0],[66,145,0]], {"weapon":[119,114,62],"weaponMode":"held"}),
    A([82,93,30,34,0], [90,70,0], [117,113,0], [56,113,0], [[99,144,0],[62,144,0]], {"weapon":[117,113,62],"weaponMode":"held"})
  ]),
  climb: Object.freeze([
    A([79,104,30,34,0], [80,54,0], [115,32,0], [56,106,0], [[98,144,0],[62,144,0]], {"weapon":[57,117,78],"weaponMode":"stowed"}),
    A([79,104,30,34,0], [80,54,0], [116,62,0], [57,107,0], [[98,144,0],[63,144,0]], {"weapon":[57,117,78],"weaponMode":"stowed"}),
    A([79,104,30,34,0], [80,54,0], [114,35,0], [56,107,0], [[98,144,0],[62,144,0]], {"weapon":[57,117,78],"weaponMode":"stowed"}),
    A([79,104,30,34,0], [80,54,0], [114,30,0], [56,106,0], [[97,144,0],[62,144,0]], {"weapon":[57,117,78],"weaponMode":"stowed"}),
    A([80,102,30,34,0], [81,51,0], [47,34,0], [56,105,0], [[99,143,0],[62,143,0]], {"weapon":[58,115,78],"weaponMode":"stowed"}),
    A([79,102,30,34,0], [80,51,0], [50,62,0], [57,105,0], [[97,143,0],[63,143,0]], {"weapon":[57,115,78],"weaponMode":"stowed"}),
    A([95,79,30,34,0], [80,51,0], [50,35,0], [57,105,0], [[97,143,0],[63,143,0]], {"weapon":[73,92,78],"weaponMode":"stowed"}),
    A([95,82,30,34,0], [80,52,0], [51,31,0], [56,105,0], [[97,143,0],[62,143,0]], {"weapon":[73,95,78],"weaponMode":"stowed"})
  ]),
  basic: Object.freeze([
    A([82,86,30,34,0], [98,46,0], [119,92,0], [66,103,0], [[107,143,0],[72,143,0]], {"weapon":[119,92,-62],"weaponMode":"held"}),
    A([82,84,30,34,0], [98,46,0], [125,82,0], [61,103,0], [[109,143,0],[68,143,0]], {"weapon":[125,82,-34],"weaponMode":"held"}),
    A([82,85,30,34,0], [97,49,0], [127,76,0], [58,104,0], [[108,143,0],[65,143,0]], {"weapon":[127,76,-7],"weaponMode":"held"}),
    A([82,84,30,34,0], [87,49,0], [130,79,0], [45,104,0], [[100,144,0],[53,144,0]], {"weapon":[130,79,8],"weaponMode":"held"}),
    A([82,86,30,34,0], [94,50,0], [135,79,0], [51,105,0], [[106,144,0],[59,144,0]], {"weapon":[135,79,25],"weaponMode":"held"}),
    A([82,86,30,34,0], [96,50,0], [125,86,0], [58,104,0], [[107,143,0],[65,143,0]], {"weapon":[125,86,42],"weaponMode":"held"}),
    A([82,87,30,34,0], [97,48,0], [116,93,0], [65,104,0], [[107,143,0],[71,143,0]], {"weapon":[116,93,62],"weaponMode":"held"}),
    A([82,88,30,34,0], [96,48,0], [117,93,0], [64,104,0], [[105,143,0],[70,143,0]], {"weapon":[117,93,62],"weaponMode":"held"})
  ]),
  skill: Object.freeze([
    A([82,88,30,34,0], [96,47,0], [123,101,0], [59,104,0], [[107,143,0],[66,143,0]], {"weapon":[123,101,62],"weaponMode":"held"}),
    A([82,84,30,34,0], [93,46,0], [123,99,0], [51,103,0], [[105,143,0],[59,143,0]], {"weapon":[123,99,62],"weaponMode":"held"}),
    A([82,82,30,34,0], [92,46,0], [110,89,0], [55,103,0], [[103,143,0],[62,143,0]], {"weapon":[110,89,62],"weaponMode":"held"}),
    A([82,84,30,34,0], [87,46,0], [107,88,0], [46,103,0], [[99,143,0],[53,143,0]], {"weapon":[107,88,62],"weaponMode":"held"}),
    A([82,81,30,34,0], [101,44,0], [143,72,0], [56,102,0], [[115,143,0],[64,143,0]], {"weapon":[143,72,62],"weaponMode":"held"}),
    A([82,81,30,34,0], [80,46,0], [125,78,0], [34,103,0], [[93,143,0],[42,143,0]], {"weapon":[125,78,62],"weaponMode":"held"}),
    A([82,83,30,34,0], [91,45,0], [114,92,0], [52,102,0], [[102,143,0],[59,143,0]], {"weapon":[114,92,62],"weaponMode":"held"}),
    A([82,87,30,34,0], [96,45,0], [123,99,0], [59,102,0], [[107,143,0],[65,143,0]], {"weapon":[123,99,62],"weaponMode":"held"})
  ]),
  party: Object.freeze([
    A([82,89,30,34,0], [96,47,0], [123,101,0], [60,101,0], [[106,143,0],[67,143,0]], {"weapon":[60,102,78],"weaponMode":"stowed"}),
    A([82,89,30,34,0], [88,46,0], [100,102,0], [52,102,0], [[98,143,0],[59,143,0]], {"weapon":[60,102,78],"weaponMode":"stowed"}),
    A([82,83,30,34,0], [90,47,0], [107,88,0], [55,88,0], [[100,143,0],[62,143,0]], {"weapon":[60,96,78],"weaponMode":"stowed"}),
    A([82,86,30,34,0], [94,47,0], [108,88,0], [60,88,0], [[104,143,0],[66,143,0]], {"weapon":[60,99,78],"weaponMode":"stowed"}),
    A([82,80,30,34,0], [91,48,0], [133,86,0], [46,86,0], [[104,143,0],[54,143,0]], {"weapon":[60,93,78],"weaponMode":"stowed"}),
    A([82,80,30,34,0], [88,47,0], [129,86,0], [47,86,0], [[100,143,0],[55,143,0]], {"weapon":[60,93,78],"weaponMode":"stowed"}),
    A([82,88,30,34,0], [87,47,0], [114,112,0], [52,112,0], [[98,143,0],[58,143,0]], {"weapon":[60,101,78],"weaponMode":"stowed"}),
    A([82,89,30,34,0], [96,47,0], [121,101,0], [61,101,0], [[106,143,0],[67,143,0]], {"weapon":[60,102,78],"weaponMode":"stowed"})
  ]),
  hit: Object.freeze([
    A([82,84,30,34,0], [90,46,0], [115,95,0], [57,103,0], [[100,143,0],[63,143,0]], {"weapon":[115,95,62],"weaponMode":"held"}),
    A([82,84,30,34,0], [82,49,0], [110,89,0], [46,104,0], [[92,143,0],[53,143,0]], {"weapon":[110,89,62],"weaponMode":"held"}),
    A([82,89,30,34,0], [92,59,0], [119,104,0], [58,109,0], [[102,144,0],[64,144,0]], {"weapon":[119,104,62],"weaponMode":"held"}),
    A([82,88,30,34,0], [98,50,0], [127,101,0], [59,105,0], [[109,144,0],[66,144,0]], {"weapon":[127,101,62],"weaponMode":"held"}),
    A([82,82,30,34,0], [99,50,0], [123,97,0], [64,105,0], [[109,144,0],[71,144,0]], {"weapon":[123,97,62],"weaponMode":"held"}),
    A([82,87,30,34,0], [97,53,0], [116,105,0], [59,106,0], [[109,144,0],[66,144,0]], {"weapon":[116,105,62],"weaponMode":"held"}),
    A([82,85,30,34,0], [96,47,0], [118,96,0], [64,103,0], [[105,143,0],[70,143,0]], {"weapon":[118,96,62],"weaponMode":"held"}),
    A([82,86,30,34,0], [93,48,0], [116,97,0], [60,104,0], [[102,143,0],[66,143,0]], {"weapon":[116,97,62],"weaponMode":"held"})
  ]),
  defeat: Object.freeze([
    A([82,88,30,34,0], [101,50,0], [126,113,0], [65,105,0], [[111,144,0],[71,144,0]], {"weapon":[126,113,62],"weaponMode":"held"}),
    A([82,92,30,34,0], [100,53,0], [119,121,0], [63,106,0], [[111,144,0],[69,144,0]], {"weapon":[119,121,62],"weaponMode":"held"}),
    A([82,94,30,34,0], [101,60,0], [116,129,0], [60,109,0], [[113,144,0],[68,144,0]], {"weapon":[116,129,62],"weaponMode":"held"}),
    A([82,103,30,34,0], [99,71,0], [126,140,0], [58,115,0], [[111,145,0],[66,145,0]], {"weapon":[126,140,62],"weaponMode":"held"}),
    A([82,109,30,34,0], [101,86,0], [118,143,0], [57,121,0], [[114,146,0],[66,146,0]], {"weapon":[118,143,62],"weaponMode":"held"}),
    A([82,119,30,34,80], [120,111,80], [97,141,0], [76,137,0], [[26,138,0],[47,140,0]], {"weapon":[97,141,62],"weaponMode":"dropped"}),
    A([74,122,30,34,80], [118,121,80], [97,142,0], [76,140,0], [[29,141,0],[49,142,0]], {"weapon":[97,142,62],"weaponMode":"dropped"}),
    A([84,124,30,34,80], [120,124,80], [100,144,0], [76,141,0], [[26,142,0],[48,143,0]], {"weapon":[100,144,62],"weaponMode":"dropped"})
  ])
});
// END ILLUSTRATED PLAYER SOCKETS

function getEquipmentAttachment(row, frame) {
  const rowId = OVERHAUL_ATTACHMENTS[row] ? row : 'idle';
  const frames = OVERHAUL_ATTACHMENTS[rowId];
  const frameIndex = Math.max(0, Math.min(frames.length - 1, Math.floor(Number(frame) || 0)));
  return frames[frameIndex];
}

function getPlayerSpriteRegistration(row, frame, fallback) {
  // New production sheets are registered during import, including flight bob.
  // Legacy per-frame corrections must not be applied a second time.
  return Object.freeze({ originX: 80, groundY: 150, authoredBodyHeight: 140 });
}

function getEquipmentAngleSet(kind) {
  const itemKind = String(kind || '').trim().toLowerCase();
  if (WEAPON_KINDS.includes(itemKind)) return ATLAS_ANGLE_SETS.weapon;
  if (LIMB_KINDS.includes(itemKind)) return ATLAS_ANGLE_SETS.limb;
  return ATLAS_ANGLE_SETS.body;
}

function getNearestAtlasAngle(angle, angles) {
  const choices = Array.isArray(angles) && angles.length ? angles : ATLAS_ANGLE_SETS.weapon;
  const target = Number(angle || 0);
  let bestIndex = 0;
  let bestError = Number.POSITIVE_INFINITY;
  choices.forEach((choice, index) => {
    const error = Math.abs(target - Number(choice || 0));
    if (error < bestError) {
      bestIndex = index;
      bestError = error;
    }
  });
  return Object.freeze({
    index: bestIndex,
    angle: Number(choices[bestIndex] || 0),
    error: bestError
  });
}

function getEquipmentAtlasVariantRow(kind, state, frame) {
  if (String(kind || '').toLowerCase() !== 'bow') return 0;
  const stateId = String(state || '').toLowerCase();
  const frameIndex = Math.max(0, Math.floor(Number(frame) || 0));
  if (!['basic', 'skill', 'party'].includes(stateId)) return 0;
  if (frameIndex <= 0 || frameIndex >= 5) return 0;
  if (frameIndex >= 4) return 2;
  return 1;
}

function getAtlasPivot(atlas, angleIndex) {
  const cellSize = Math.max(1, Number(atlas && atlas.frameWidth || ATLAS_CELL_SIZE));
  const pivots = atlas && Array.isArray(atlas.pivots) ? atlas.pivots : [];
  const source = pivots[angleIndex] || {};
  return Object.freeze({
    x: Number.isFinite(Number(source.x)) ? Number(source.x) : Number(atlas && atlas.pivotX || cellSize / 2),
    y: Number.isFinite(Number(source.y)) ? Number(source.y) : Number(atlas && atlas.pivotY || cellSize / 2)
  });
}

function createAtlasPart(visual, rig, state, frame, partId, anchor, options) {
  const atlas = visual && visual.atlas;
  if (!atlas || !atlas.sheet || !anchor) return null;
  const settings = options || {};
  const kind = String(visual.kind || atlas.kind || '').toLowerCase();
  const angles = Array.isArray(atlas.angles) && atlas.angles.length ? atlas.angles : getEquipmentAngleSet(kind);
  const nearest = getNearestAtlasAngle(anchor.angle, angles);
  const pivot = getAtlasPivot(atlas, nearest.index);
  const frameWidth = Math.max(1, Number(atlas.frameWidth || ATLAS_CELL_SIZE));
  const frameHeight = Math.max(1, Number(atlas.frameHeight || frameWidth));
  const order = Number.isFinite(Number(settings.order)) ? Number(settings.order) : Number(visual.order || 0);
  return Object.freeze({
    visualId: visual.id || '',
    partId,
    mode: settings.mode || 'attached',
    order,
    angle: Number(anchor.angle || 0),
    orientationAngle: nearest.angle,
    angleError: nearest.error,
    socket: Object.freeze({ x: Number(anchor.x || 0), y: Number(anchor.y || 0) }),
    pivot,
    scaleX: Math.max(0.05, Number(settings.scaleX || 1)),
    scaleY: Math.max(0.05, Number(settings.scaleY || 1)),
    frame: Object.freeze({
      sheet: atlas.sheet,
      row: getEquipmentAtlasVariantRow(kind, state, frame),
      frameIndex: nearest.index,
      frameWidth,
      frameHeight
    })
  });
}

function resolveEquipmentAtlasParts(visual, state, frame) {
  if (!visual || !visual.atlas || !visual.atlas.sheet) return [];
  const rig = getEquipmentAttachment(state, frame);
  const kind = String(visual.kind || visual.atlas.kind || '').toLowerCase();
  const parts = [];
  const add = (partId, anchor, options) => {
    const part = createAtlasPart(visual, rig, state, frame, partId, anchor, options);
    if (part) parts.push(part);
  };
  if (WEAPON_KINDS.includes(kind)) {
    const anchor = kind === 'bow'
      ? Object.assign({}, rig.weapon, { angle: Number(rig.bowAngle || rig.weapon.angle || 0) })
      : rig.weapon;
    add('weapon', anchor, {
      mode: rig.weapon.mode,
      order: rig.weapon.mode === 'stowed' ? -10 : Number(visual.order || 40)
    });
  } else if (kind === 'shield') {
    add('shield', rig.shield);
  } else if (kind === 'chest') {
    add('chest', rig.torso, {
      scaleX: Number(rig.torso.width || 32) / 32,
      scaleY: Number(rig.torso.height || 36) / 36
    });
  } else if (kind === 'head') {
    add('head', rig.head);
  } else if (kind === 'boots') {
    rig.feet.forEach((foot, index) => add(`boot-${index}`, foot));
  } else if (kind === 'gloves' || kind === 'grip') {
    add(`${kind}-off`, rig.offHand);
    add(`${kind}-main`, rig.mainHand);
  } else if (kind === 'ring') {
    add('ring', rig.mainHand);
  } else if (kind === 'amulet') {
    add('amulet', rig.torso);
  } else if (kind === 'core') {
    add('core', { x: rig.torso.x - 19, y: rig.torso.y - 7, angle: rig.torso.angle });
  } else if (kind === 'focus') {
    add('focus', { x: rig.mainHand.x + 8, y: rig.mainHand.y - 13, angle: 0 });
  } else if (kind === 'scope') {
    add('scope', { x: rig.head.x + 12, y: rig.head.y + 2, angle: rig.head.angle });
  } else if (kind === 'kit') {
    add('kit', { x: rig.torso.x - 18, y: rig.torso.y + 13, angle: rig.torso.angle });
  }
  return parts.sort((a, b) => a.order - b.order || a.partId.localeCompare(b.partId));
}

const api = Object.freeze({
  ATLAS_CELL_SIZE,
  ATLAS_ANGLE_SETS,
  ROWS,
  ATTACHMENTS,
  PLAYER_FRAME_REGISTRATIONS,
  getEquipmentAttachment,
  getPlayerSpriteRegistration,
  getEquipmentAngleSet,
  getNearestAtlasAngle,
  getEquipmentAtlasVariantRow,
  resolveEquipmentAtlasParts
});

const modules = global.ProjectStarfallEngineModules || {};
modules.equipmentAttachments = Object.assign({}, modules.equipmentAttachments || {}, api);
global.ProjectStarfallEngineModules = modules;

if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
