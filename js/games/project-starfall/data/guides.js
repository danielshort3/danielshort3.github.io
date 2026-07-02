(function initProjectStarfallDataGuides(global) {
  'use strict';

  const TARGET_FARM_TABLES = Object.freeze([
    Object.freeze({ enemyId: 'crackedMimic', name: 'Mimic Cache', summary: 'Higher chance for coupons, Echo Prisms, and rare gear from Mimics.' }),
    Object.freeze({ enemyId: 'emberjawGolem', name: 'Emberjaw Set Hunt', summary: 'Higher boss-set pressure while farming Emberjaw.' }),
    Object.freeze({ enemyId: 'riftAberration', name: 'Rift Splinter Hunt', summary: 'Improves rift material and elite reward streaks.' }),
    Object.freeze({ enemyId: 'rimewarden', name: 'Rimewarden Sigil Hunt', summary: 'Improves frost boss material and Relic chase odds.' }),
    Object.freeze({ enemyId: 'stormbreakRoc', name: 'Stormbreak Plume Hunt', summary: 'Improves airborne boss material and currency drops.' })
  ]);

  const ADVANCED_FEATURE_GUIDE = Object.freeze([
    Object.freeze({ id: 'status_icons', title: 'Status Icons', summary: 'Buff icons show effects active right now with warm duration bars; cooldown icons show unavailable skills with dark overlays and ready timers.', detail: 'Use this convention for future combat procedures: buffs answer what is affecting the player, cooldowns answer what can be pressed next. Rune Mage aura buffs and glyph cooldowns are intentionally separate.', panelId: 'guide' }),
    Object.freeze({ id: 'map_modifiers', title: 'Map Modifiers', summary: 'Fields, dungeons, and rifts can roll local rules that alter enemy pressure and rewards.', panelId: 'worldmap' }),
    Object.freeze({ id: 'boss_break', title: 'Boss Break Windows', summary: 'Bosses build a break gauge from control, armor-break, and focused attacks. Broken bosses take extra damage.', panelId: 'monsters' }),
    Object.freeze({ id: 'skill_modifiers', title: 'Skill Modifiers', summary: 'Skills unlock automatic modifiers as you level and invest in the class kit.', panelId: 'skills' }),
    Object.freeze({ id: 'class_mastery', title: 'Class Mastery', summary: 'Defeating enemies and using skills earns class mastery XP for small permanent class bonuses.', panelId: 'character' }),
    Object.freeze({ id: 'gear_traits', title: 'Gear Traits', summary: 'Rare and better drops can roll traits that change stat priorities and reward loops.', panelId: 'equipment' }),
    Object.freeze({ id: 'card_decks', title: 'Card Decks', summary: 'Cards stack by same card and tier, live in the Cards inventory tab, upgrade with Star Cards, and slot into a six-card deck from Equipment.', detail: 'Equipment > Cards shows the active deck, each equipped card, Star Card wallet, and full Active Deck Bonuses total. Equip one stack per card definition at a time; duplicate tiers of the same card do not stack in the deck. Right-click a stack in Inventory > Cards and choose Upgrade to spend Star Cards on the next tier. Upgrade All Cards upgrades every eligible unlocked stack as far as your Star Cards allow. Unequipped, unlocked cards can be sold for coins.', panelId: 'equipment' }),
    Object.freeze({ id: 'monster_research', title: 'Monster Research', summary: 'Monster Guide kills reveal stats, drops, and mastery bonuses while feeding target farming.', panelId: 'monsters' }),
    Object.freeze({ id: 'target_farming', title: 'Target Farming', summary: 'Repeatedly defeating the selected guide monster improves its drop pressure for a short streak.', panelId: 'monsters' }),
    Object.freeze({ id: 'roster_synergy', title: 'Roster Synergy', summary: 'Specific unlocked roster-trait combinations add extra account-style stat bonuses.', panelId: 'beta' }),
    Object.freeze({ id: 'dungeon_objectives', title: 'Dungeon Objectives', summary: 'Dungeons track bonus objectives such as boss breaks, add clears, and party survival.', panelId: 'quests' }),
    Object.freeze({ id: 'elite_affixes', title: 'Elite Affixes', summary: 'Rare elites can spawn with combat affixes that change pressure and reward priority.', panelId: 'monsters' }),
    Object.freeze({ id: 'party_commands', title: 'Party Commands', summary: 'Party commands steer visible AI allies toward boss focus, spread clears, guarding, or burst windows.', panelId: 'partyPanel' }),
    Object.freeze({ id: 'rift_ladder', title: 'Endless Rift Ladder', summary: 'Endless Rift tracks tier, mutations, and score as a repeatable performance ladder.', panelId: 'worldmap' }),
    Object.freeze({ id: 'starfall_plinko', title: 'Starfall Plinko', summary: 'Mob-earned balls and optional coin-bought balls feed a town Plinko board for coins, materials, coupons, prisms, gear, and 100-ball jackpot pity.', detail: 'Find the Plinko Host service NPC in any regional town. Monster drops are the efficient path; buying balls spends coins for a negative-expected-value sink with clear 100-ball jackpot pity progress.', panelId: 'plinko' })
  ]);

  const MONSTER_GUIDE_FUTURE_ENEMY_IDS = Object.freeze(['bristleBoar', 'dustImp']);
  const MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS = Object.freeze([
    'banditCutterDirect',
    'banditCutterReference',
    'banditCutterHybrid',
    'banditCutterPuppet',
    ...MONSTER_GUIDE_FUTURE_ENEMY_IDS
  ]);

  const api = {
    TARGET_FARM_TABLES,
    ADVANCED_FEATURE_GUIDE,
    MONSTER_GUIDE_FUTURE_ENEMY_IDS,
    MONSTER_GUIDE_COLLECTION_EXCLUDED_ENEMY_IDS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.guides = Object.assign({}, modules.guides || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
