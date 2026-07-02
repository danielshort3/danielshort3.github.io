(() => {
  "use strict";

  const SPRITE_PATH = "/img/games/stellar-dogfight/";
  const RASTER_PATH = "/img/games/stellar-dogfight/raster/";

  function imageAsset(fileName, options) {
    return {
      src: `${SPRITE_PATH}${fileName}.svg`,
      rasterSrc: `${RASTER_PATH}${fileName}.png`,
      ...options
    };
  }

  window.STELLAR_DOGFIGHT_ART = {
    sprites: {
      playerScout: imageAsset("ship-player-scout", { width: 96, height: 96, frames: 1, fps: 0, scale: 2.72, anchorX: 0.5, anchorY: 0.5 }),
      playerHeavy: imageAsset("ship-player-heavy", { width: 112, height: 96, frames: 1, fps: 0, scale: 2.9, anchorX: 0.5, anchorY: 0.5 }),
      enemyScreen: imageAsset("enemy-screen", { width: 80, height: 80, frames: 1, fps: 0, scale: 2.55, anchorX: 0.5, anchorY: 0.5 }),
      enemyLine: imageAsset("enemy-line", { width: 88, height: 80, frames: 1, fps: 0, scale: 2.55, anchorX: 0.5, anchorY: 0.5 }),
      enemyInterceptor: imageAsset("enemy-interceptor", { width: 88, height: 80, frames: 1, fps: 0, scale: 2.68, anchorX: 0.5, anchorY: 0.5 }),
      enemySupport: imageAsset("enemy-support", { width: 92, height: 92, frames: 1, fps: 0, scale: 2.6, anchorX: 0.5, anchorY: 0.5 }),
      enemyArtillery: imageAsset("enemy-artillery", { width: 112, height: 72, frames: 1, fps: 0, scale: 2.82, anchorX: 0.5, anchorY: 0.5 }),
      enemySiege: imageAsset("enemy-siege", { width: 104, height: 92, frames: 1, fps: 0, scale: 2.75, anchorX: 0.5, anchorY: 0.5 }),
      enemyBrawler: imageAsset("enemy-brawler", { width: 96, height: 92, frames: 1, fps: 0, scale: 2.75, anchorX: 0.5, anchorY: 0.5 }),
      enemyCommand: imageAsset("enemy-command", { width: 132, height: 120, frames: 1, fps: 0, scale: 3.05, anchorX: 0.5, anchorY: 0.5 }),
      helperDrone: imageAsset("helper-drone", { width: 56, height: 56, frames: 1, fps: 0, scale: 2.4, anchorX: 0.5, anchorY: 0.5 }),
      asteroidRock: imageAsset("asteroid-rock", { width: 104, height: 104, frames: 1, fps: 0, scale: 2.25, anchorX: 0.5, anchorY: 0.5 }),
      obstaclePlate: imageAsset("obstacle-plate", { width: 128, height: 96, frames: 1, fps: 0, scale: 1, anchorX: 0.5, anchorY: 0.5 }),
      pickupCore: imageAsset("pickup-core", { width: 72, height: 72, frames: 1, fps: 0, scale: 2.25, anchorX: 0.5, anchorY: 0.5 }),
      mineCore: imageAsset("mine-core", { width: 64, height: 64, frames: 1, fps: 0, scale: 2.1, anchorX: 0.5, anchorY: 0.5 })
    },
    effects: {
      burst: imageAsset("vfx-burst", { width: 128, height: 128, frames: 1, fps: 0, scale: 1, anchorX: 0.5, anchorY: 0.5 }),
      shieldRipple: imageAsset("vfx-shield-ripple", { width: 128, height: 128, frames: 1, fps: 0, scale: 1, anchorX: 0.5, anchorY: 0.5 })
    },
    backgrounds: {
      nebula: imageAsset("background-nebula", { width: 1600, height: 900, frames: 1, fps: 0, scale: 1, anchorX: 0.5, anchorY: 0.5 })
    }
  };
})();
