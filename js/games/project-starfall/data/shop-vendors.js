(function initProjectStarfallDataShopVendors(global) {
  'use strict';

  const SHOP_VENDOR_TYPES = Object.freeze([
    Object.freeze({ id: 'weapon', label: 'Weapon Shop', panelLabel: 'Weapons', x: 560, platformIndex: 1 }),
    Object.freeze({ id: 'armor', label: 'Armor Shop', panelLabel: 'Armor', x: 780, platformIndex: 1 }),
    Object.freeze({ id: 'supply', label: 'Supply Shop', panelLabel: 'Supplies', x: 1000, platformIndex: 1 }),
    Object.freeze({ id: 'special', label: 'Special Shop', panelLabel: 'Specials', x: 1220, platformIndex: 1 })
  ]);

  const TOWN_SHOP_THEME_BY_TOWN = Object.freeze({
    starfallCrossing: Object.freeze({ prefix: 'Crossing', specialName: 'Guild Relics', vendorAccent: '#ffd166', vendorColor: '#5e7d9f' }),
    rustcoilOutpost: Object.freeze({ prefix: 'Rustcoil', specialName: 'Scrap Exchange', vendorAccent: '#29b3ad', vendorColor: '#7a8592' }),
    cinderRefuge: Object.freeze({ prefix: 'Cinder', specialName: 'Ember Counter', vendorAccent: '#ffcf70', vendorColor: '#9b4835' }),
    frostfenCamp: Object.freeze({ prefix: 'Frostfen', specialName: 'Frost Cache', vendorAccent: '#b7f2ff', vendorColor: '#6386a8' }),
    stormbreakHaven: Object.freeze({ prefix: 'Stormbreak', specialName: 'Storm Vault', vendorAccent: '#ffe16a', vendorColor: '#4f6073' }),
    astralObservatory: Object.freeze({ prefix: 'Astral', specialName: 'Star Lens Annex', vendorAccent: '#7bdff2', vendorColor: '#6b55a8' })
  });

  function defaultFreezeQuestReward(reward) {
    const source = reward || {};
    const frozen = Object.assign({}, source);
    ['materials', 'consumables', 'items', 'timedBuffs', 'permanentStats'].forEach((key) => {
      if (frozen[key] && typeof frozen[key] === 'object') frozen[key] = Object.freeze(frozen[key]);
    });
    return Object.freeze(frozen);
  }

  function defaultTownServiceNpcId(mapId, serviceId) {
    const prefix = String(mapId || 'town')
      .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
      .replace(/[^a-zA-Z0-9_]+/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_|_$/g, '')
      .toLowerCase();
    return `${prefix}_${serviceId}_host`;
  }

  function defaultGetTownShopVendorId(townId, typeId) {
    return defaultTownServiceNpcId(townId, `${typeId}_shop`);
  }

  function shopTypeTitle(typeId) {
    return String(typeId || '')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (letter) => letter.toUpperCase())
      .replace(/\s+/g, '');
  }

  function getTownShopInteriorMapId(townId, typeId) {
    return `${townId}${shopTypeTitle(typeId)}Shop`;
  }

  function getTownShopVendorId(townId, typeId) {
    return defaultGetTownShopVendorId(townId, typeId);
  }

  function createTownShopDoorPortals(townId) {
    return SHOP_VENDOR_TYPES.map((type) => Object.freeze({
      id: `${townId}_${type.id}_shop_door`,
      label: type.label,
      destinationMapId: getTownShopInteriorMapId(townId, type.id),
      shopDoor: true,
      shopVendorType: type.id,
      x: type.x,
      platformIndex: type.platformIndex
    }));
  }

  function shopEquipment(itemId) {
    return Object.freeze({ kind: 'equipment', itemId });
  }

  function shopConsumable(consumableId, cost, quantity) {
    return Object.freeze({ kind: 'consumable', consumableId, cost, quantity: Math.max(1, Math.floor(Number(quantity || 1) || 1)) });
  }

  function createShopBundle(freezeQuestReward, id, name, cost, reward, summary) {
    return Object.freeze({
      kind: 'bundle',
      id,
      name,
      cost,
      summary: summary || '',
      reward: freezeQuestReward(reward || {})
    });
  }

  function createShopVendorCatalog(getTownShopVendorId, townId, typeId, tier, entries) {
    const theme = TOWN_SHOP_THEME_BY_TOWN[townId] || {};
    const type = SHOP_VENDOR_TYPES.find((candidate) => candidate.id === typeId) || {};
    return Object.freeze({
      id: getTownShopVendorId(townId, typeId),
      townId,
      typeId,
      tier,
      name: typeId === 'special' ? theme.specialName || type.label || 'Special Shop' : `${theme.prefix || 'Town'} ${type.label || 'Shop'}`,
      entries: Object.freeze((entries || []).map((entry) => Object.freeze(Object.assign({}, entry))))
    });
  }

  function createShopVendorData(options) {
    const settings = options || {};
    const freezeQuestReward = typeof settings.freezeQuestReward === 'function'
      ? settings.freezeQuestReward
      : defaultFreezeQuestReward;
    const getTownShopVendorId = typeof settings.getTownShopVendorId === 'function'
      ? settings.getTownShopVendorId
      : defaultGetTownShopVendorId;
    const shopBundle = (id, name, cost, reward, summary) => createShopBundle(freezeQuestReward, id, name, cost, reward, summary);
    const shopVendorCatalog = (townId, typeId, tier, entries) => createShopVendorCatalog(getTownShopVendorId, townId, typeId, tier, entries);

    const SHOP_VENDOR_CATALOGS = Object.freeze([
      shopVendorCatalog('starfallCrossing', 'weapon', 1, [
        shopEquipment('training_sword'), shopEquipment('training_wand'), shopEquipment('training_bow'),
        shopEquipment('copper_sword'), shopEquipment('birch_wand'), shopEquipment('simple_bow')
      ]),
      shopVendorCatalog('starfallCrossing', 'armor', 1, [
        shopEquipment('stitched_vest'), shopEquipment('traveler_boots'), shopEquipment('plain_ring')
      ]),
      shopVendorCatalog('starfallCrossing', 'supply', 1, [
        shopConsumable('minor_health_potion', 35, 1), shopConsumable('minor_resource_tonic', 35, 1),
        shopConsumable('camp_ration', 45, 1), shopConsumable('town_return_scroll', 60, 1)
      ]),
      shopVendorCatalog('starfallCrossing', 'special', 1, [
        shopConsumable('pet_whistle', 500, 1),
        shopBundle('crossing_training_bundle', 'Training Bundle', 180, { consumables: { minor_health_potion: 3, minor_resource_tonic: 2 }, materials: { upgradeDust: 5 } }, 'Starter supplies and upgrade dust.')
      ]),

      shopVendorCatalog('rustcoilOutpost', 'weapon', 2, [
        shopEquipment('iron_sword'), shopEquipment('iron_axe'), shopEquipment('apprentice_staff'), shopEquipment('oak_longbow')
      ]),
      shopVendorCatalog('rustcoilOutpost', 'armor', 2, [
        shopEquipment('rustcoil_field_helm'), shopEquipment('rustcoil_work_vest'), shopEquipment('rustcoil_grip_gloves'), shopEquipment('traveler_boots')
      ]),
      shopVendorCatalog('rustcoilOutpost', 'supply', 2, [
        shopConsumable('standard_health_potion', 85, 1), shopConsumable('standard_resource_tonic', 85, 1),
        shopConsumable('field_ration', 105, 1), shopConsumable('town_return_scroll', 60, 1)
      ]),
      shopVendorCatalog('rustcoilOutpost', 'special', 2, [
        shopConsumable('guard_tonic', 140, 1), shopConsumable('swiftstep_oil', 140, 1),
        shopBundle('rustcoil_catalyst_cache', 'Rustcoil Catalyst Cache', 360, { materials: { upgradeCatalyst: 2, oreChunks: 6 } }, 'Construct-region upgrade materials.')
      ]),

      shopVendorCatalog('cinderRefuge', 'weapon', 3, [
        shopEquipment('cinder_steel_sword'), shopEquipment('cinder_steel_scepter'), shopEquipment('cinder_ashwood_bow')
      ]),
      shopVendorCatalog('cinderRefuge', 'armor', 3, [
        shopEquipment('cinder_reinforced_mail'), shopEquipment('cinder_forge_boots'), shopEquipment('cinder_ember_band')
      ]),
      shopVendorCatalog('cinderRefuge', 'supply', 3, [
        shopConsumable('greater_health_potion', 175, 1), shopConsumable('greater_resource_tonic', 175, 1),
        shopConsumable('expedition_ration', 210, 1), shopConsumable('town_return_scroll', 60, 1)
      ]),
      shopVendorCatalog('cinderRefuge', 'special', 3, [
        shopEquipment('guardian_tower_shield'), shopEquipment('berserker_war_grip'), shopEquipment('ember_core'),
        shopEquipment('rune_etched_focus'), shopEquipment('deadeye_scope'), shopEquipment('trap_kit'),
        shopBundle('cinder_prism_cache', 'Cinder Prism Cache', 920, { consumables: { potential_cube: 1 }, materials: { cubeFragment: 4 } }, 'Early attunement support.')
      ]),

      shopVendorCatalog('frostfenCamp', 'weapon', 4, [
        shopEquipment('frostfen_silver_saber'), shopEquipment('frostfen_moonlit_staff'), shopEquipment('frostfen_frostpine_bow')
      ]),
      shopVendorCatalog('frostfenCamp', 'armor', 4, [
        shopEquipment('frostfen_iceguard_coat'), shopEquipment('frostfen_snowstep_boots'), shopEquipment('frostfen_rime_ring')
      ]),
      shopVendorCatalog('frostfenCamp', 'supply', 4, [
        shopConsumable('greater_health_potion', 160, 1), shopConsumable('greater_resource_tonic', 160, 1),
        shopConsumable('expedition_ration', 195, 1), shopConsumable('magnet_charm', 220, 1)
      ]),
      shopVendorCatalog('frostfenCamp', 'special', 4, [
        shopConsumable('xp_coupon_1_2_1h', 520, 1), shopConsumable('drop_coupon_1_2_1h', 560, 1),
        shopBundle('frostfen_slot_cache', 'Frostfen Slot Cache', 1200, { consumables: { usable_slot_coupon: 1, etc_slot_coupon: 1 } }, 'Utility inventory expansion.')
      ]),

      shopVendorCatalog('stormbreakHaven', 'weapon', 5, [
        shopEquipment('stormbreak_stormforged_blade'), shopEquipment('stormbreak_thunder_rod'), shopEquipment('stormbreak_gale_longbow')
      ]),
      shopVendorCatalog('stormbreakHaven', 'armor', 5, [
        shopEquipment('stormbreak_tempest_mantle'), shopEquipment('stormbreak_cloudrunner_boots'), shopEquipment('stormbreak_lightning_charm')
      ]),
      shopVendorCatalog('stormbreakHaven', 'supply', 5, [
        shopConsumable('superior_health_potion', 320, 1), shopConsumable('superior_resource_tonic', 320, 1),
        shopConsumable('hero_ration', 380, 1), shopConsumable('town_return_scroll', 60, 1)
      ]),
      shopVendorCatalog('stormbreakHaven', 'special', 5, [
        shopConsumable('xp_coupon_1_5_1h', 1350, 1), shopConsumable('drop_coupon_1_5_1h', 1450, 1),
        shopBundle('stormbreak_echo_cache', 'Stormbreak Echo Cache', 1800, { consumables: { preservation_cube: 1 }, materials: { refinementCore: 2 } }, 'High-risk upgrade support.')
      ]),

      shopVendorCatalog('astralObservatory', 'weapon', 6, [
        shopEquipment('astral_index_blade'), shopEquipment('astral_star_lens_staff'), shopEquipment('astral_comet_bow')
      ]),
      shopVendorCatalog('astralObservatory', 'armor', 6, [
        shopEquipment('astral_starwoven_robes'), shopEquipment('astral_orbitstep_boots'), shopEquipment('astral_lens_amulet')
      ]),
      shopVendorCatalog('astralObservatory', 'supply', 6, [
        shopConsumable('superior_health_potion', 300, 1), shopConsumable('superior_resource_tonic', 300, 1),
        shopConsumable('hero_ration', 360, 1), shopConsumable('magnet_charm', 240, 1)
      ]),
      shopVendorCatalog('astralObservatory', 'special', 6, [
        shopConsumable('xp_coupon_2_0_1h', 3200, 1), shopConsumable('drop_coupon_2_0_1h', 3400, 1),
        shopBundle('astral_slot_cache', 'Astral Slot Cache', 2400, { consumables: { equipment_slot_coupon: 1, card_slot_coupon: 1 }, materials: { cubeFragment: 12 } }, 'Late-game account expansion.')
      ])
    ]);

    return Object.freeze({
      SHOP_VENDOR_CATALOGS
    });
  }

  const defaultShopVendorData = createShopVendorData();
  const api = Object.assign({
    SHOP_VENDOR_TYPES,
    TOWN_SHOP_THEME_BY_TOWN,
    createShopVendorData,
    defaultFreezeQuestReward,
    defaultGetTownShopVendorId,
    shopTypeTitle,
    getTownShopInteriorMapId,
    getTownShopVendorId,
    createTownShopDoorPortals,
    shopEquipment,
    shopConsumable
  }, defaultShopVendorData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.shopVendors = Object.assign({}, modules.shopVendors || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
