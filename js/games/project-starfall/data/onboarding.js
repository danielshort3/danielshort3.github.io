(function initProjectStarfallDataOnboarding(global) {
  'use strict';

  const ONBOARDING_STEPS = Object.freeze([
    Object.freeze({ id: 'learn_move', event: 'move', title: 'Get your footing', summary: 'Use the arrow keys or WASD to move through Starfall Crossing.' }),
    Object.freeze({ id: 'learn_attack', event: 'attack', title: 'Ready your weapon', summary: 'Press J or Left Shift to practice a basic attack.' }),
    Object.freeze({ id: 'travel_greenroot', event: 'travel', mapId: 'greenrootMeadow', title: 'Reach Starfall Verge', summary: 'Take the frontier gate into the first expedition field.' }),
    Object.freeze({ id: 'defeat_enemy', event: 'defeat', enemyId: 'glassback', mapId: 'greenrootMeadow', count: 3, title: 'Break the first threat', summary: 'Defeat three Glassbacks in Starfall Verge.' }),
    Object.freeze({ id: 'loot_drop', event: 'loot', mapId: 'greenrootMeadow', title: 'Recover a field drop', summary: 'Collect one dropped item before leaving the frontier.' }),
    Object.freeze({ id: 'claim_first_steps', event: 'questClaim', questId: 'first_steps', title: 'Report to the Quartermaster', summary: 'Finish First Expedition, then claim its reward from the Verge Quartermaster.' })
  ]);

  const AUDIO_CUES = Object.freeze({
    uiConfirm: Object.freeze({ label: 'UI Confirm', type: 'tone', frequency: 660, duration: 0.08, gain: 0.045 }),
    attack: Object.freeze({ label: 'Basic Attack', type: 'noise', frequency: 180, duration: 0.07, gain: 0.05 }),
    skill: Object.freeze({ label: 'Skill Cast', type: 'sweep', frequency: 360, endFrequency: 780, duration: 0.12, gain: 0.06 }),
    buff: Object.freeze({ label: 'Buff Cast', type: 'chime', frequency: 520, endFrequency: 1040, duration: 0.18, gain: 0.055 }),
    loot: Object.freeze({ label: 'Loot Pickup', type: 'chime', frequency: 740, endFrequency: 1180, duration: 0.11, gain: 0.045 }),
    level: Object.freeze({ label: 'Level Up', type: 'chime', frequency: 660, endFrequency: 1320, duration: 0.26, gain: 0.07 }),
    upgradeSuccess: Object.freeze({ label: 'Upgrade Success', type: 'chime', frequency: 580, endFrequency: 980, duration: 0.22, gain: 0.065 }),
    upgradeFail: Object.freeze({ label: 'Upgrade Fail', type: 'sweep', frequency: 260, endFrequency: 120, duration: 0.18, gain: 0.045 }),
    damage: Object.freeze({ label: 'Player Hit', type: 'noise', frequency: 120, duration: 0.09, gain: 0.05 }),
    defeat: Object.freeze({ label: 'Enemy Defeat', type: 'sweep', frequency: 420, endFrequency: 190, duration: 0.13, gain: 0.05 }),
    travel: Object.freeze({ label: 'Map Travel', type: 'chime', frequency: 440, endFrequency: 760, duration: 0.2, gain: 0.05 }),
    partyAssist: Object.freeze({ label: 'Party Assist', type: 'chime', frequency: 510, endFrequency: 820, duration: 0.14, gain: 0.04 })
  });

  const STARTER_ITEMS = Object.freeze({
    fighter: ['training_sword'],
    mage: ['training_wand'],
    archer: ['training_bow']
  });

  const STARTER_CONSUMABLES = Object.freeze({
    minor_health_potion: 3,
    minor_resource_tonic: 2,
    pet_whistle: 1,
    stat_reset_scroll: 1,
    admin_worldwright_console: 1
  });

  const api = {
    ONBOARDING_STEPS,
    AUDIO_CUES,
    STARTER_ITEMS,
    STARTER_CONSUMABLES
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.onboarding = Object.assign({}, modules.onboarding || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
