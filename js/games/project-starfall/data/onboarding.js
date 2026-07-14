(function initProjectStarfallDataOnboarding(global) {
  'use strict';

  const ONBOARDING_STEPS = Object.freeze([
    Object.freeze({ id: 'choose_class', event: 'classSelected', title: 'Choose a class', summary: 'Pick Fighter, Mage, or Archer to enter Starfall Crossing.' }),
    Object.freeze({ id: 'learn_move', event: 'move', title: 'Move through town', summary: 'Use the arrow keys or WASD to move. You can change either movement binding in Keybinds.' }),
    Object.freeze({ id: 'learn_jump', event: 'jump', title: 'Practice jumping', summary: 'Press Space to jump onto platforms and clear low obstacles.' }),
    Object.freeze({ id: 'learn_attack', event: 'attack', title: 'Practice a basic attack', summary: 'Press J or Left Shift to use your equipped weapon.' }),
    Object.freeze({ id: 'learn_interact', event: 'interact', title: 'Use an interaction', summary: 'Press F at a station, Y near an NPC, or Up/W at a portal.' }),
    Object.freeze({ id: 'open_worldmap', event: 'openPanel', panelId: 'worldmap', title: 'Check the world map', summary: 'Open the world map to see the current route and nearby areas.' }),
    Object.freeze({ id: 'travel_greenroot', event: 'travel', mapId: 'greenrootMeadow', title: 'Travel to Greenroot', summary: 'Use the Greenroot Gate or world map to enter the first field.' }),
    Object.freeze({ id: 'defeat_enemy', event: 'defeat', title: 'Defeat an enemy', summary: 'Use basic attacks or a bound skill to defeat a field enemy.' }),
    Object.freeze({ id: 'loot_drop', event: 'loot', title: 'Pick up loot', summary: 'Hold the loot key near a dropped item to collect coins, gear, or materials.' }),
    Object.freeze({ id: 'open_inventory', event: 'openPanel', panelId: 'inventory', title: 'Review the item grid', summary: 'Open Inventory to compare level requirements, base strength, and item tier glow.' }),
    Object.freeze({ id: 'equip_item', event: 'equip', title: 'Equip an item', summary: 'Open inventory and equip a stronger item into the matching slot.' }),
    Object.freeze({ id: 'open_skills', event: 'openPanel', panelId: 'skills', title: 'Open your skill batch', summary: 'Open Skills to see trainable nodes, role tags, and next-rank changes.' }),
    Object.freeze({ id: 'rank_skill', event: 'rankSkill', title: 'Rank a skill', summary: 'Spend a skill point in the skill window to improve your class kit.' }),
    Object.freeze({ id: 'open_upgrade', event: 'openPanel', panelId: 'upgrade', title: 'Inspect upgrade risk', summary: 'Open the Upgrade Station to preview before/after power and failure outcomes.' }),
    Object.freeze({ id: 'upgrade_item', event: 'upgrade', title: 'Attempt an upgrade', summary: 'Use Upgrade Dust at the artisan to see the risk table and try an item upgrade.' }),
    Object.freeze({ id: 'open_party', event: 'openPanel', panelId: 'partyPanel', title: 'Open the party panel', summary: 'Open Party to see simulated ally roles, assist timing, and current self-buff behavior.' }),
    Object.freeze({ id: 'find_party', event: 'partyFind', title: 'Find prototype allies', summary: 'Open Party and fill the local simulated party slots for passive assists.' }),
    Object.freeze({ id: 'open_trials', event: 'openPanel', panelId: 'quests', title: 'Check trials', summary: 'Open Quests and switch to Trials when you are ready to preview advanced branches.' }),
    Object.freeze({ id: 'start_trial', event: 'startTrial', title: 'Start a class trial', summary: 'At Level 20, start a branch trial from the Quest window.' }),
    Object.freeze({ id: 'choose_advanced', event: 'advancedClass', title: 'Choose an advanced class', summary: 'Complete a trial and choose your permanent Level 25 advanced branch.' }),
    Object.freeze({ id: 'clear_dungeon', event: 'dungeonComplete', title: 'Clear a dungeon', summary: 'Enter a boss dungeon and defeat its boss to complete the vertical-slice loop.' })
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
