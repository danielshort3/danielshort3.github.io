(function initProjectStarfallDataClasses(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};

  const CLASS_ROLE_PROFILES = Object.freeze({
    fighter: Object.freeze({
      primary: 'Hybrid',
      secondary: 'Melee control',
      specialty: 'Close-range impact',
      summary: 'Reliable melee pressure with short-range control, guard windows, and finishers.'
    }),
    mage: Object.freeze({
      primary: 'Hybrid',
      secondary: 'Area setup',
      specialty: 'Spell routing',
      summary: 'Ranged spell pressure with strong setup tools and the safest vertical targeting.'
    }),
    archer: Object.freeze({
      primary: 'Hybrid',
      secondary: 'Mobile range',
      specialty: 'Marks and spacing',
      summary: 'Mobile ranged pressure that can jump attack and rewards choosing the right target.'
    }),
    guardian: Object.freeze({
      primary: 'Support / Control',
      secondary: 'Boss safety',
      specialty: 'Stored Impact',
      summary: 'Stores defensive pressure, breaks bosses open, and turns survival into counter-burst.'
    }),
    berserker: Object.freeze({
      primary: 'Bossing',
      secondary: 'Risk sustain',
      specialty: 'Low-HP burst',
      summary: 'Trades safety for strong single-target damage and sustain when fighting dangerous enemies.'
    }),
    duelist: Object.freeze({
      primary: 'Bossing',
      secondary: 'Tempo burst',
      specialty: 'Same-target combos',
      summary: 'Stacks Tempo on one target and rewards clean repeated hits over wide mob clearing.'
    }),
    fireMage: Object.freeze({
      primary: 'Mobbing',
      secondary: 'Burn ramp',
      specialty: 'Burn spread',
      summary: 'Spreads burns through clustered enemies and converts Heat into larger explosions.'
    }),
    runeMage: Object.freeze({
      primary: 'Support / Control',
      secondary: 'Setup burst',
      specialty: 'Linked runes',
      summary: 'Marks, links, slows, and detonates enemies after preparing rune setups.'
    }),
    stormMage: Object.freeze({
      primary: 'Mobbing',
      secondary: 'Chain pressure',
      specialty: 'Chain lightning',
      summary: 'Excellent clustered-enemy clearing through chained delayed lightning pulses.'
    }),
    sniper: Object.freeze({
      primary: 'Bossing',
      secondary: 'Weak-point burst',
      specialty: 'Single-target payoff',
      summary: 'Marks weak points and cashes them out with high-value shots against bosses.'
    }),
    trapper: Object.freeze({
      primary: 'Mobbing / Control',
      secondary: 'Area setup',
      specialty: 'Trap networks',
      summary: 'Prepares traps that slow, trigger, and detonate groups as enemies path into them.'
    }),
    beastArcher: Object.freeze({
      primary: 'Support / Hybrid',
      secondary: 'Sustain pressure',
      specialty: 'Companion marks',
      summary: 'Coordinates companion hits that mark targets and sustain the character over longer fights.'
    })
  });

  function createClassData(options) {
    const settings = options || {};
    const classAssets = settings.classAssets || DataAssets.CLASS_ASSETS || {};
    const playerAnimationAssets = settings.playerAnimationAssets || {};
    const baseClasses = Object.freeze({
      fighter: {
        id: 'fighter',
        name: 'Fighter',
        asset: classAssets.fighter,
        animation: playerAnimationAssets.fighter,
        resourceName: 'Momentum',
        resourceColor: '#f25f4c',
        weaponType: 'melee',
        roleProfile: CLASS_ROLE_PROFILES.fighter,
        description: 'Close-range weapon class built around impact, guarded timing, and controlled aggression.',
        stats: {
          hp: 180,
          power: 18,
          defense: 8,
          speed: 220,
          jump: 520,
          range: 76,
          mpMax: 90,
          resourceMax: 100
        }
      },
      mage: {
        id: 'mage',
        name: 'Mage',
        asset: classAssets.mage,
        animation: playerAnimationAssets.mage,
        resourceName: 'Energy',
        resourceColor: '#4f8cff',
        weaponType: 'projectile',
        roleProfile: CLASS_ROLE_PROFILES.mage,
        description: 'Ranged spellcaster with area damage, utility, and resource-aware burst windows.',
        stats: {
          hp: 135,
          power: 20,
          defense: 4,
          speed: 212,
          jump: 510,
          range: 380,
          mpMax: 130,
          resourceMax: 120
        }
      },
      archer: {
        id: 'archer',
        name: 'Archer',
        asset: classAssets.archer,
        animation: playerAnimationAssets.archer,
        resourceName: 'Focus',
        resourceColor: '#3aa76d',
        weaponType: 'projectile',
        roleProfile: CLASS_ROLE_PROFILES.archer,
        description: 'Ranged physical class that rewards spacing, marking, mobility, and target selection.',
        stats: {
          hp: 150,
          power: 19,
          defense: 5,
          speed: 235,
          jump: 525,
          range: 430,
          mpMax: 100,
          resourceMax: 110
        }
      }
    });

    const advancedClasses = Object.freeze({
      guardian: {
        id: 'guardian',
        name: 'Guardian',
        asset: classAssets.guardian,
        animation: playerAnimationAssets.guardian,
        baseClass: 'fighter',
        levelRequirement: 25,
        resourceName: 'Stored Impact',
        resourceColor: '#68a9ff',
        partySkillId: 'guardian_shield_wall',
        roleProfile: CLASS_ROLE_PROFILES.guardian,
        description: 'Defensive Fighter path that converts blocked damage into counter-pressure.'
      },
      berserker: {
        id: 'berserker',
        name: 'Berserker',
        asset: classAssets.berserker,
        animation: playerAnimationAssets.berserker,
        baseClass: 'fighter',
        levelRequirement: 25,
        resourceName: 'Rage',
        resourceColor: '#ef3d55',
        partySkillId: 'berserker_war_cry',
        roleProfile: CLASS_ROLE_PROFILES.berserker,
        description: 'Risky Fighter path with low-health burst, lifesteal, and aggressive sustain.'
      },
      duelist: {
        id: 'duelist',
        name: 'Duelist',
        asset: classAssets.duelist,
        animation: playerAnimationAssets.duelist,
        baseClass: 'fighter',
        levelRequirement: 25,
        resourceName: 'Tempo',
        resourceColor: '#f0c36a',
        partySkillId: 'duelist_rallying_flourish',
        roleProfile: CLASS_ROLE_PROFILES.duelist,
        description: 'Mobile Fighter path built around quick windows, precision counters, and party haste.'
      },
      fireMage: {
        id: 'fireMage',
        name: 'Fire Mage',
        asset: classAssets.fireMage,
        animation: playerAnimationAssets.fireMage,
        baseClass: 'mage',
        levelRequirement: 25,
        resourceName: 'Heat',
        resourceColor: '#ff8a3d',
        partySkillId: 'fire_mage_ignition_aura',
        roleProfile: CLASS_ROLE_PROFILES.fireMage,
        description: 'Explosive Mage path that builds Heat and vents it into area pressure.'
      },
      runeMage: {
        id: 'runeMage',
        name: 'Rune Mage',
        asset: classAssets.runeMage,
        animation: playerAnimationAssets.runeMage,
        baseClass: 'mage',
        levelRequirement: 25,
        resourceName: 'Runic Energy',
        resourceColor: '#28c7b7',
        partySkillId: 'rune_mage_rune_circle',
        roleProfile: CLASS_ROLE_PROFILES.runeMage,
        description: 'Setup Mage path that places, links, detonates, and empowers rune fields.'
      },
      stormMage: {
        id: 'stormMage',
        name: 'Storm Mage',
        asset: classAssets.stormMage,
        animation: playerAnimationAssets.stormMage,
        baseClass: 'mage',
        levelRequirement: 25,
        resourceName: 'Charge',
        resourceColor: '#7bdff2',
        partySkillId: 'storm_mage_stormfront',
        roleProfile: CLASS_ROLE_PROFILES.stormMage,
        description: 'Fast Mage path that chains lightning through grouped targets and builds Charge through movement.'
      },
      sniper: {
        id: 'sniper',
        name: 'Sniper',
        asset: classAssets.sniper,
        animation: playerAnimationAssets.sniper,
        baseClass: 'archer',
        levelRequirement: 25,
        resourceName: 'Aim',
        resourceColor: '#c9b35c',
        partySkillId: 'sniper_eagle_eye',
        roleProfile: CLASS_ROLE_PROFILES.sniper,
        description: 'Precision Archer path focused on weak points, precision strikes, and boss damage.'
      },
      trapper: {
        id: 'trapper',
        name: 'Trapper',
        asset: classAssets.trapper,
        animation: playerAnimationAssets.trapper,
        baseClass: 'archer',
        levelRequirement: 25,
        resourceName: 'Preparation',
        resourceColor: '#b07a47',
        partySkillId: 'trapper_tactical_field',
        roleProfile: CLASS_ROLE_PROFILES.trapper,
        description: 'Tactical Archer path that wins through traps, slows, and enemy routing.'
      },
      beastArcher: {
        id: 'beastArcher',
        name: 'Beast Archer',
        asset: classAssets.beastArcher,
        animation: playerAnimationAssets.beastArcher,
        baseClass: 'archer',
        levelRequirement: 25,
        resourceName: 'Bond',
        resourceColor: '#78b26a',
        partySkillId: 'beast_archer_pack_call',
        roleProfile: CLASS_ROLE_PROFILES.beastArcher,
        description: 'Companion Archer path focused on coordinated strikes, survivability, and party resource support.'
      }
    });

    return {
      CLASS_ROLE_PROFILES,
      BASE_CLASSES: baseClasses,
      ADVANCED_CLASSES: advancedClasses
    };
  }

  const api = {
    CLASS_ROLE_PROFILES,
    createClassData
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.classes = Object.assign({}, modules.classes || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
