(() => {
  "use strict";

  const hasAchievement = (context, id) => !!(context.achievements && context.achievements[id]);

  const features = [
    {
      id: "upgrades",
      label: "Wave Boosts",
      desc: "Pick one boost after each cleared wave.",
      hint: "Clear Wave 1 or fly for 45 seconds.",
      panel: "upgrades",
      tip: {
        title: "Boosts unlocked!",
        message: "After each wave, pick one boost."
      },
      unlocksWhen: (context) => context.flightSeconds >= 45 || context.waveProgress >= 2
    },
    {
      id: "ability",
      label: "Ship Ability",
      desc: "Your ship ability is now part of the combat loop.",
      hint: "Fly for 90 seconds, reach Wave 3, or hit Rank 2.",
      panel: "systems",
      tip: {
        title: "Ability ready!",
        message: (context) => `Press ${context.abilityKey} to use it.`
      },
      unlocksWhen: (context) => context.unlocked("upgrades")
        && (context.flightSeconds >= 90 || context.waveProgress >= 3 || context.rank >= 2)
    },
    {
      id: "secondary",
      label: "Secondary System",
      desc: "Secondary fire opens up more combat options and cooldown management.",
      hint: "Use abilities, then reach Wave 4 or Rank 3.",
      panel: "systems",
      tip: {
        title: "Secondary ready!",
        message: (context) => `Press ${context.secondaryKey} to deploy it.`
      },
      unlocksWhen: (context) => context.unlocked("ability")
        && (context.flightSeconds >= 140 || context.waveProgress >= 4 || context.rank >= 3)
    },
    {
      id: "hangar",
      label: "Hangar Boosts",
      desc: "Permanent hangar upgrades let you shape a long-term build.",
      hint: "Reach Rank 2 or complete your first run.",
      panel: "upgrades",
      tip: {
        title: "Hangar open!",
        message: "Spend tech points in Boosts."
      },
      unlocksWhen: (context) => context.rank >= 2 || context.runHistoryCount > 0
    },
    {
      id: "scoreMode",
      label: "Score Attack",
      desc: "Timed scoring runs are now available from Options.",
      hint: "Reach Rank 2 or Campaign Wave 3.",
      panel: "settings",
      tip: {
        title: "Score Attack unlocked!",
        message: "Timed scoring runs are available from Options."
      },
      unlocksWhen: (context) => context.rank >= 2 || context.bestWave >= 3
    },
    {
      id: "armory",
      label: "Gear Locker",
      desc: "Weapons, attachments, and secondaries can now be compared and equipped.",
      hint: "Reach Rank 3 or destroy 20 enemies.",
      panel: "armory",
      tip: {
        title: "Gear locker open!",
        message: "Swap weapons and parts in Gear."
      },
      unlocksWhen: (context) => context.rank >= 3 || context.totalKills >= 20
    },
    {
      id: "shipyard",
      label: "Shipyard",
      desc: "New hulls unlock and expand your build identity.",
      hint: "Reach Rank 4 or find a blueprint.",
      panel: "shipyard",
      tip: {
        title: "Shipyard open!",
        message: "Unlock new ships in Ships."
      },
      unlocksWhen: (context) => context.rank >= 4 || context.blueprints >= 1
    },
    {
      id: "bossRush",
      label: "Boss Rush",
      desc: "Capital-wave challenge runs are now available from Options.",
      hint: "Reach Rank 5 or defeat a dreadnought.",
      panel: "settings",
      tip: {
        title: "Boss Rush unlocked!",
        message: "Capital-wave challenge runs are available from Options."
      },
      unlocksWhen: (context) => context.rank >= 5 || hasAchievement(context, "boss-down")
    },
    {
      id: "contracts",
      label: "Tasks Board",
      desc: "Optional objectives and faction reputation are now active.",
      hint: "Reach Rank 4 and Campaign Wave 5.",
      panel: "contracts",
      tip: {
        title: "Tasks unlocked!",
        message: "Grab a task for bonus loot."
      },
      unlocksWhen: (context) => context.rank >= 4 && context.bestWave >= 5
    },
    {
      id: "premium",
      label: "Astralite Forge",
      desc: "Astralite can now be spent on permanent account upgrades.",
      hint: "Reach Rank 5, Campaign Wave 7, or earn Astralite.",
      panel: "premium",
      tip: {
        title: "Astralite Forge online!",
        message: "Spend Astralite on permanent upgrades."
      },
      unlocksWhen: (context) => context.rank >= 5 || context.bestWave >= 7 || context.premiumCurrency > 0
    },
    {
      id: "dailySector",
      label: "Daily Sector",
      desc: "Daily seeded challenge runs are now available from Options.",
      hint: "Reach Rank 5 after unlocking Score Attack.",
      panel: "settings",
      tip: {
        title: "Daily Sector unlocked!",
        message: "Seeded daily runs are available from Options."
      },
      unlocksWhen: (context) => context.rank >= 5 && context.unlocked("scoreMode")
    },
    {
      id: "salvage",
      label: "Salvage Cache",
      desc: "Keys can now be spent on salvage openings and rare gear rolls.",
      hint: "Unlock Gear, then find a salvage key.",
      panel: "armory",
      tip: {
        title: "Caches unlocked!",
        message: "Open salvage in Gear."
      },
      unlocksWhen: (context) => context.unlocked("armory") && context.salvageKeys >= 1
    },
    {
      id: "frontierMode",
      label: "Frontier",
      desc: "Long-form frontier patrols are now available from Options.",
      hint: "Reach Rank 6 with Shipyard online and Campaign Wave 8 cleared.",
      panel: "settings",
      tip: {
        title: "Frontier unlocked!",
        message: "Long-form patrol runs are available from Options."
      },
      unlocksWhen: (context) => context.rank >= 6 && context.unlocked("shipyard") && context.bestWave >= 8
    }
  ];

  const modes = [
    { id: "arcade", label: "Campaign", feature: "always" },
    { id: "score", label: "Score Attack", feature: "scoreMode" },
    { id: "boss", label: "Boss Rush", feature: "bossRush" },
    { id: "daily", label: "Daily Sector", feature: "dailySector" },
    { id: "frontier", label: "Frontier", feature: "frontierMode" }
  ];

  window.STELLAR_DOGFIGHT_UNLOCKS = {
    features,
    modes
  };
})();
