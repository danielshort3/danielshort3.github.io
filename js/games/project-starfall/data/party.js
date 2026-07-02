(function initProjectStarfallDataParty(global) {
  'use strict';

  function partyAiLoadout(equipment, skills) {
    return Object.freeze({
      equipment: Object.freeze(Object.assign({}, equipment || {})),
      skills: Object.freeze((skills || []).map((entry) => Object.freeze(typeof entry === 'string' ? { skillId: entry } : Object.assign({}, entry))))
    });
  }

  const PARTY_AI_LOADOUTS = Object.freeze({
    fighter: partyAiLoadout({ weapon: 'iron_sword', chest: 'party_plate', boots: 'traveler_boots' }, [
      { skillId: 'fighter_power_break', minCooldown: 6.2, priority: 3 },
      { skillId: 'fighter_ground_slam', minCooldown: 5.4, priority: 2 },
      { skillId: 'fighter_heavy_strike', minCooldown: 1.75, priority: 1 }
    ]),
    guardian: partyAiLoadout({ weapon: 'iron_sword', chest: 'party_plate', boots: 'traveler_boots', offhand: 'guardian_tower_shield' }, [
      { skillId: 'guardian_oath_barrier', minCooldown: 12, priority: 4, support: true },
      { skillId: 'guardian_retaliation_wave', minCooldown: 8, priority: 3 },
      { skillId: 'guardian_shield_bash', minCooldown: 1.85, priority: 1 }
    ]),
    berserker: partyAiLoadout({ weapon: 'iron_axe', chest: 'party_plate', boots: 'traveler_boots', offhand: 'berserker_war_grip' }, [
      { skillId: 'berserker_war_cry', minCooldown: 18, priority: 4, support: true },
      { skillId: 'berserker_crimson_recovery', minCooldown: 10, priority: 3 },
      { skillId: 'berserker_blood_cleave', minCooldown: 1.9, priority: 1 }
    ]),
    duelist: partyAiLoadout({ weapon: 'iron_sword', chest: 'party_plate', boots: 'traveler_boots' }, [
      { skillId: 'duelist_rallying_flourish', minCooldown: 16, priority: 3, support: true },
      { skillId: 'duelist_quick_cut', minCooldown: 1.65, priority: 1 }
    ]),
    mage: partyAiLoadout({ weapon: 'apprentice_staff', chest: 'party_robes', boots: 'traveler_boots' }, [
      { skillId: 'mage_spell_mark', minCooldown: 5.4, priority: 3 },
      { skillId: 'mage_arcane_burst', minCooldown: 5.6, priority: 2 },
      { skillId: 'mage_magic_bolt', minCooldown: 1.8, priority: 1 }
    ]),
    fireMage: partyAiLoadout({ weapon: 'apprentice_staff', chest: 'party_robes', boots: 'traveler_boots', offhand: 'ember_core' }, [
      { skillId: 'fire_mage_wildfire', minCooldown: 10, priority: 4 },
      { skillId: 'fire_mage_burning_mark', minCooldown: 6, priority: 3 },
      { skillId: 'fire_mage_fireball', minCooldown: 1.95, priority: 1 }
    ]),
    runeMage: partyAiLoadout({ weapon: 'apprentice_staff', chest: 'party_robes', boots: 'traveler_boots', offhand: 'rune_etched_focus' }, [
      { skillId: 'rune_mage_ground_glyph', minCooldown: 9, priority: 4 },
      { skillId: 'rune_mage_arcane_link', minCooldown: 7, priority: 3 },
      { skillId: 'rune_mage_rune_mark', minCooldown: 1.85, priority: 1 }
    ]),
    stormMage: partyAiLoadout({ weapon: 'apprentice_staff', chest: 'party_robes', boots: 'traveler_boots', offhand: 'rune_etched_focus' }, [
      { skillId: 'storm_mage_stormfront', minCooldown: 18, priority: 4, support: true },
      { skillId: 'storm_mage_chain_bolt', minCooldown: 2.05, priority: 1 }
    ]),
    archer: partyAiLoadout({ weapon: 'oak_longbow', chest: 'party_leathers', boots: 'traveler_boots' }, [
      { skillId: 'archer_marked_shot', minCooldown: 4.5, priority: 3 },
      { skillId: 'archer_piercing_arrow', minCooldown: 5.4, priority: 2 },
      { skillId: 'archer_quick_shot', minCooldown: 1.7, priority: 1 }
    ]),
    sniper: partyAiLoadout({ weapon: 'oak_longbow', chest: 'party_leathers', boots: 'traveler_boots', offhand: 'deadeye_scope' }, [
      { skillId: 'sniper_pierce_armor', minCooldown: 7, priority: 4 },
      { skillId: 'sniper_weak_point_mark', minCooldown: 6, priority: 3 },
      { skillId: 'sniper_aimed_shot', minCooldown: 2.05, priority: 1 }
    ]),
    trapper: partyAiLoadout({ weapon: 'oak_longbow', chest: 'party_leathers', boots: 'traveler_boots', offhand: 'trap_kit' }, [
      { skillId: 'trapper_spike_trap', minCooldown: 6, priority: 4 },
      { skillId: 'trapper_lure_shot', minCooldown: 7, priority: 3 },
      { skillId: 'trapper_snare_trap', minCooldown: 1.9, priority: 1 }
    ]),
    beastArcher: partyAiLoadout({ weapon: 'oak_longbow', chest: 'party_leathers', boots: 'traveler_boots', offhand: 'trap_kit' }, [
      { skillId: 'beast_archer_pack_call', minCooldown: 18, priority: 3, support: true },
      { skillId: 'beast_archer_companion_strike', minCooldown: 1.9, priority: 1 }
    ])
  });

  const PROTOTYPE_PARTY_MEMBERS = Object.freeze([
    Object.freeze({
      id: 'aegis_mira',
      name: 'Mira',
      classId: 'guardian',
      role: 'Defense',
      summary: 'A simulated Guardian ally who adds mitigation and occasional shields.',
      statBonuses: Object.freeze({ hp: 42, defense: 3, block: 2 }),
      assist: Object.freeze({ type: 'shield', cooldown: 7.5, shieldPercent: 0.08, color: '#68a9ff' })
    }),
    Object.freeze({
      id: 'cinder_jo',
      name: 'Jo',
      classId: 'fireMage',
      role: 'Mobbing',
      summary: 'A simulated Fire Mage ally who throws small area bursts into packs.',
      statBonuses: Object.freeze({ power: 2, areaDamage: 3, burnDamage: 4 }),
      assist: Object.freeze({ type: 'damage', cooldown: 4.8, powerScale: 0.34, radius: 92, color: '#ff8a3d' })
    }),
    Object.freeze({
      id: 'deadeye_len',
      name: 'Len',
      classId: 'sniper',
      role: 'Bossing',
      summary: 'A simulated Sniper ally who helps mark priority targets.',
      statBonuses: Object.freeze({ crit: 2, critDamage: 5, range: 18 }),
      assist: Object.freeze({ type: 'mark', cooldown: 5.2, powerScale: 0.3, radius: 40, color: '#ffe16a' })
    }),
    Object.freeze({
      id: 'field_tamsin',
      name: 'Tamsin',
      classId: 'trapper',
      role: 'Control',
      summary: 'A simulated Trapper ally who slows clustered enemies during long fights.',
      statBonuses: Object.freeze({ trapDamage: 3, defense: 2, speed: 4 }),
      assist: Object.freeze({ type: 'control', cooldown: 5.8, powerScale: 0.22, radius: 118, color: '#7bdff2' })
    })
  ]);

  const api = {
    partyAiLoadout,
    PARTY_AI_LOADOUTS,
    PROTOTYPE_PARTY_MEMBERS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.party = Object.assign({}, modules.party || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
