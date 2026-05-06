(() => {
  "use strict";

  const FALLBACK_COLOR = 0xffffff;

  function createRenderer(options = {}) {
    return new StellarDogfightPixiRenderer(options);
  }

  class StellarDogfightPixiRenderer {
    constructor(options) {
      this.PIXI = options.PIXI || window.PIXI;
      this.anchorCanvas = options.anchorCanvas;
      this.artConfig = options.artConfig || {};
      this.getRenderScale = typeof options.getRenderScale === "function"
        ? options.getRenderScale
        : () => window.devicePixelRatio || 1;
      this.app = null;
      this.ready = false;
      this.failed = false;
      this.width = 0;
      this.height = 0;
      this.resolution = 1;
      this.textures = new Map();
      this.spritePools = {};
      this.nebulaSprite = null;
      this.backgroundGraphics = null;
      this.worldLayer = null;
      this.starLayer = null;
      this.worldGraphics = null;
      this.vfxGraphics = null;
      this.overlayGraphics = null;
      this.entityGraphics = null;
      this.screenGraphics = null;
    }

    async init() {
      if (!this.PIXI || !this.PIXI.Application || !this.anchorCanvas) {
        this.failed = true;
        return false;
      }
      const parent = this.anchorCanvas.parentElement;
      if (!parent) {
        this.failed = true;
        return false;
      }
      const rect = this.anchorCanvas.getBoundingClientRect();
      this.width = Math.max(1, Math.round(rect.width || 1));
      this.height = Math.max(1, Math.round(rect.height || 1));
      this.resolution = this.getSafeResolution();
      this.app = new this.PIXI.Application();
      await this.app.init({
        width: this.width,
        height: this.height,
        resolution: this.resolution,
        autoDensity: true,
        autoStart: false,
        antialias: false,
        backgroundAlpha: 0,
        clearBeforeRender: true,
        powerPreference: "high-performance",
        preference: "webgl"
      });
      if (this.app.ticker && typeof this.app.ticker.stop === "function") {
        this.app.ticker.stop();
      }
      this.app.canvas.className = "pixi-stage";
      this.app.canvas.setAttribute("aria-hidden", "true");
      parent.insertBefore(this.app.canvas, this.anchorCanvas.nextSibling);
      this.setupScene();
      await this.loadTextures();
      this.ready = true;
      document.body.classList.add("is-pixi-renderer");
      return true;
    }

    setupScene() {
      const { Container, Graphics, Sprite } = this.PIXI;
      this.backgroundGraphics = new Graphics();
      this.nebulaSprite = new Sprite(this.PIXI.Texture.WHITE);
      this.nebulaSprite.visible = false;
      this.nebulaSprite.alpha = 0.28;
      this.nebulaSprite.blendMode = "screen";
      this.worldLayer = new Container();
      this.starLayer = new Container();
      this.worldGraphics = new Graphics();
      this.vfxGraphics = new Graphics();
      this.overlayGraphics = new Graphics();
      this.entityGraphics = new Graphics();
      this.screenGraphics = new Graphics();
      const worldSprites = new Container();
      const vfxSprites = new Container();
      const entitySprites = new Container();
      this.worldLayer.addChild(
        this.starLayer,
        this.worldGraphics,
        worldSprites,
        this.vfxGraphics,
        vfxSprites,
        this.overlayGraphics,
        entitySprites,
        this.entityGraphics
      );
      this.createPool("stars", this.starLayer);
      this.createPool("world", worldSprites);
      this.createPool("vfx", vfxSprites);
      this.createPool("entities", entitySprites);
      this.app.stage.addChild(this.backgroundGraphics, this.nebulaSprite, this.worldLayer, this.screenGraphics);
    }

    createPool(name, container) {
      const pool = {
        container,
        items: [],
        active: 0
      };
      this.spritePools[name] = pool;
      return pool;
    }

    beginPools() {
      Object.values(this.spritePools).forEach((pool) => {
        pool.active = 0;
      });
    }

    hideUnusedSprites() {
      Object.values(this.spritePools).forEach((pool) => {
        for (let i = pool.active; i < pool.items.length; i += 1) {
          pool.items[i].visible = false;
        }
      });
    }

    acquireSprite(poolName) {
      const pool = this.spritePools[poolName];
      if (!pool) return null;
      let sprite = pool.items[pool.active];
      if (!sprite) {
        sprite = new this.PIXI.Sprite(this.PIXI.Texture.WHITE);
        sprite.anchor.set(0.5);
        pool.items.push(sprite);
        pool.container.addChild(sprite);
      }
      pool.active += 1;
      sprite.visible = true;
      sprite.alpha = 1;
      sprite.tint = FALLBACK_COLOR;
      sprite.blendMode = "normal";
      return sprite;
    }

    async loadTextures() {
      const definitions = this.getArtDefinitions();
      const assets = Object.entries(definitions).filter(([, definition]) => definition && definition.src);
      for (const [id, definition] of assets) {
        const texture = await this.loadTexture(definition.rasterSrc || definition.src, definition.src);
        if (texture) this.textures.set(id, texture);
      }
    }

    async loadTexture(primarySrc, fallbackSrc) {
      const sources = [primarySrc, fallbackSrc].filter(Boolean);
      for (const src of sources) {
        try {
          if (this.PIXI.Assets && typeof this.PIXI.Assets.load === "function") {
            return await this.PIXI.Assets.load(src);
          }
          return this.PIXI.Texture.from(src);
        } catch (error) {
          continue;
        }
      }
      return null;
    }

    getArtDefinitions() {
      return {
        ...(this.artConfig.sprites || {}),
        ...(this.artConfig.effects || {}),
        ...(this.artConfig.backgrounds || {})
      };
    }

    getSafeResolution() {
      return clamp(this.getRenderScale() || window.devicePixelRatio || 1, 0.2, 2);
    }

    resize(width, height) {
      if (!this.app || !this.app.renderer) return;
      const safeWidth = Math.max(1, Math.round(width || this.width || 1));
      const safeHeight = Math.max(1, Math.round(height || this.height || 1));
      const resolution = this.getSafeResolution();
      if (safeWidth === this.width && safeHeight === this.height && resolution === this.resolution) return;
      this.width = safeWidth;
      this.height = safeHeight;
      this.resolution = resolution;
      this.app.renderer.resize(safeWidth, safeHeight, resolution);
    }

    renderFrame(snapshot) {
      if (!this.ready || !this.app) return null;
      const timings = {};
      let marker = nowMs();
      this.resize(snapshot.width, snapshot.height);
      this.beginPools();
      this.clearGraphics();
      this.renderBackground(snapshot);
      marker = markTiming(timings, "background", marker);
      this.updateWorldTransform(snapshot);
      this.renderStars(snapshot);
      marker = markTiming(timings, "stars", marker);
      this.renderWorld(snapshot);
      marker = markTiming(timings, "world", marker);
      this.renderVfx(snapshot);
      marker = markTiming(timings, "vfx", marker);
      this.renderOverlays(snapshot);
      marker = markTiming(timings, "overlays", marker);
      this.renderEntities(snapshot);
      marker = markTiming(timings, "entities", marker);
      this.renderScreenOverlays(snapshot);
      this.hideUnusedSprites();
      this.app.render();
      markTiming(timings, "entities", marker);
      return timings;
    }

    clearGraphics() {
      this.backgroundGraphics.clear();
      this.worldGraphics.clear();
      this.vfxGraphics.clear();
      this.overlayGraphics.clear();
      this.entityGraphics.clear();
      this.screenGraphics.clear();
    }

    updateWorldTransform(snapshot) {
      const camera = snapshot.camera || {};
      const shake = snapshot.shake || {};
      this.worldLayer.position.set(
        (shake.x || 0) - (camera.x || 0),
        (shake.y || 0) - (camera.y || 0)
      );
    }

    renderBackground(snapshot) {
      const width = snapshot.width || this.width;
      const height = snapshot.height || this.height;
      this.backgroundGraphics
        .rect(0, 0, width, height)
        .fill({ color: 0x05090f, alpha: 1 })
        .rect(0, 0, width, height * 0.42)
        .fill({ color: 0x101c2f, alpha: 0.72 })
        .rect(0, height * 0.52, width, height * 0.48)
        .fill({ color: 0x02050a, alpha: 0.45 });
      const nebulaTexture = this.textures.get("nebula");
      if (!nebulaTexture) {
        this.nebulaSprite.visible = false;
        return;
      }
      const camera = snapshot.camera || {};
      const parallaxX = -((camera.x || 0) * 0.018) % width;
      const parallaxY = -((camera.y || 0) * 0.012) % height;
      this.nebulaSprite.visible = true;
      this.nebulaSprite.texture = nebulaTexture;
      this.nebulaSprite.position.set(parallaxX - width * 0.08, parallaxY - height * 0.08);
      this.nebulaSprite.width = width * 1.18;
      this.nebulaSprite.height = height * 1.18;
      this.nebulaSprite.alpha = snapshot.qualityLevel >= 3 ? 0.16 : 0.3;
    }

    renderStars(snapshot) {
      const bounds = snapshot.bounds;
      const stars = snapshot.stars || [];
      stars.forEach((star) => {
        if (!isInBounds(star, star.radius || 1, bounds, 40)) return;
        const sprite = this.acquireSprite("stars");
        if (!sprite) return;
        const size = Math.max(1, (star.radius || 1) * 2);
        sprite.texture = this.PIXI.Texture.WHITE;
        sprite.position.set(star.x, star.y);
        sprite.width = size;
        sprite.height = size;
        sprite.alpha = star.alpha || 0.5;
      });
    }

    renderWorld(snapshot) {
      const graphics = this.worldGraphics;
      const bounds = snapshot.bounds;
      (snapshot.obstacles || []).forEach((obstacle) => {
        if (obstacle.kind === "rock") {
          const radius = obstacle.radius || 80;
          if (!isInBounds(obstacle, radius, bounds, 80)) return;
          if (this.drawSprite("world", "asteroidRock", obstacle.x, obstacle.y, obstacle.rotation || 0, radius * 2.12, radius * 2.12, { alpha: 0.96 })) return;
          graphics.circle(obstacle.x, obstacle.y, radius).fill({ color: 0x34445d, alpha: 0.9 });
          graphics.circle(obstacle.x, obstacle.y, radius).stroke({ width: 1.5, color: 0x9bb6d2, alpha: 0.16 });
          return;
        }
        const halfWidth = (obstacle.width || 80) * 0.5;
        const halfHeight = (obstacle.height || 80) * 0.5;
        if (obstacle.x + halfWidth < bounds.left - 80 || obstacle.x - halfWidth > bounds.right + 80
          || obstacle.y + halfHeight < bounds.top - 80 || obstacle.y - halfHeight > bounds.bottom + 80) {
          return;
        }
        if (this.drawSprite("world", "obstaclePlate", obstacle.x, obstacle.y, obstacle.rotation || 0, obstacle.width || 80, obstacle.height || 80, { alpha: 0.92 })) return;
        graphics.rect(obstacle.x - halfWidth, obstacle.y - halfHeight, halfWidth * 2, halfHeight * 2).fill({ color: 0x162230, alpha: 0.85 });
        graphics.rect(obstacle.x - halfWidth, obstacle.y - halfHeight, halfWidth * 2, halfHeight * 2).stroke({ width: 1.5, color: 0xa0bedc, alpha: 0.14 });
      });
      (snapshot.fieldDrops || []).forEach((drop) => {
        const def = snapshot.getFieldDropDef ? snapshot.getFieldDropDef(drop.typeId) : null;
        const radius = drop.radius || 16;
        if (!isInBounds(drop, radius, bounds, 40)) return;
        const alpha = clamp((drop.life || 0) / Math.max(1, drop.maxLife || 1), 0, 1);
        const color = colorToNumber(def && def.color ? def.color : snapshot.getTierColor?.(def?.tier), 0xffffff);
        const bob = Math.sin((snapshot.nowSec || 0) * 2 + (drop.phase || 0)) * 4;
        const y = drop.y + bob;
        graphics.circle(drop.x, y, radius + 12).fill({ color, alpha: alpha * 0.12 });
        if (this.drawSprite("world", "pickupCore", drop.x, y, (snapshot.nowSec || 0) * 0.9 + (drop.phase || 0), radius * 2.55, radius * 2.55, { alpha })) return;
        graphics.circle(drop.x, y, radius).fill({ color: 0x080e18, alpha: 0.85 });
        graphics.circle(drop.x, y, radius).stroke({ width: 2, color, alpha });
      });
      (snapshot.mines || []).forEach((mine) => {
        const radius = mine.radius || 14;
        if (!isInBounds(mine, radius, bounds, 90)) return;
        if (this.drawSprite("world", "mineCore", mine.x, mine.y, (snapshot.nowSec || 0) * 1.8, radius * 1.35, radius * 1.35, { alpha: 0.9 })) return;
        graphics.circle(mine.x, mine.y, radius * 0.5).fill({ color: 0xf6c65f, alpha: 0.8 });
      });
      if (snapshot.decoy) {
        graphics.circle(snapshot.decoy.x, snapshot.decoy.y, 12).stroke({ width: 2, color: 0xf6c65f, alpha: 0.6 });
      }
      (snapshot.hazards || []).forEach((hazard) => {
        const radius = hazard.radius || 100;
        if (!isInBounds(hazard, radius, bounds, 30)) return;
        const lifeRatio = clamp((hazard.life || 0) / 22, 0.15, 1);
        graphics.circle(hazard.x, hazard.y, radius).fill({ color: 0xf06969, alpha: 0.05 + lifeRatio * 0.08 });
        graphics.circle(hazard.x, hazard.y, radius * 0.92).stroke({ width: 2, color: 0xff9f6b, alpha: 0.45 });
      });
    }

    renderVfx(snapshot) {
      const graphics = this.vfxGraphics;
      const bounds = snapshot.bounds;
      const qualityLevel = snapshot.qualityLevel || 0;
      const now = snapshot.nowSec || 0;
      if (snapshot.player && snapshot.showDetail && snapshot.player.auraRadius > 0 && snapshot.player.auraDamage > 0) {
        const pulse = 0.5 + Math.sin(now * 4) * 0.5;
        graphics.circle(snapshot.player.x, snapshot.player.y, snapshot.player.auraRadius).fill({ color: 0x7ca8ff, alpha: 0.05 });
        graphics.circle(snapshot.player.x, snapshot.player.y, snapshot.player.auraRadius).stroke({ width: 1.5, color: 0x7ca8ff, alpha: 0.24 + pulse * 0.22 });
      }
      (snapshot.particles || []).forEach((particle, index) => {
        if (qualityLevel >= 3 && index % 2 === 1) return;
        if (!isInBounds(particle, particle.size || 3, bounds, 90)) return;
        const ratio = clamp((particle.life || 0) / Math.max(1, particle.maxLife || 1), 0, 1);
        const alpha = 1 - Math.pow(1 - ratio, 3);
        const color = colorToNumber(particle.color, 0x7cf2b4);
        if (particle.trail) {
          graphics
            .moveTo(particle.x, particle.y)
            .lineTo(particle.x - (particle.vx || 0) * 0.035, particle.y - (particle.vy || 0) * 0.035)
            .stroke({ width: Math.max(1, (particle.size || 3) * 0.65), color, alpha: alpha * 0.5 });
        }
        graphics.circle(particle.x, particle.y, particle.size || 3).fill({ color, alpha });
      });
      (snapshot.pulses || []).forEach((pulse) => {
        if (!isInBounds(pulse, pulse.radius || 20, bounds, 120)) return;
        const ratio = clamp((pulse.life || 0) / Math.max(1, pulse.maxLife || 1), 0, 1);
        const alpha = 1 - Math.pow(1 - ratio, 3);
        const color = colorToNumber(pulse.color, 0x7ca8ff);
        if (pulse.kind === "burst") {
          this.drawSprite("vfx", "burst", pulse.x, pulse.y, 0, pulse.radius * 2.2, pulse.radius * 2.2, { alpha: alpha * 0.55, blendMode: "add" });
        }
        graphics.circle(pulse.x, pulse.y, pulse.radius).stroke({ width: 1.4 + ratio * 1.6, color, alpha });
        graphics.circle(pulse.x, pulse.y, pulse.radius * 0.72).stroke({ width: 4, color, alpha: alpha * 0.28 });
      });
      (snapshot.blackHoles || []).forEach((hole) => {
        const lifeRatio = hole.maxLife ? hole.life / hole.maxLife : 1;
        const alpha = clamp(0.2 + lifeRatio * 0.4, 0, 0.7);
        graphics.circle(hole.x, hole.y, hole.radius).fill({ color: 0x587ad0, alpha: alpha * 0.24 });
        graphics.ellipse(hole.x, hole.y, hole.radius * 0.62, hole.radius * 0.22).stroke({ width: 2, color: 0xdae0ff, alpha: alpha * 0.72 });
      });
    }

    renderOverlays(snapshot) {
      const graphics = this.overlayGraphics;
      const bounds = snapshot.bounds;
      (snapshot.enemies || []).forEach((enemy) => {
        if (!snapshot.shouldDrawEnemyTelegraph || !snapshot.shouldDrawEnemyTelegraph(enemy)) return;
        const telegraph = snapshot.getEnemyAttackTelegraph ? snapshot.getEnemyAttackTelegraph(enemy) : null;
        if (!telegraph) return;
        const color = colorToNumber(telegraph.color, 0xff9f6b);
        const lineLength = Math.min(telegraph.distance || 0, 250 + (telegraph.progress || 0) * 80);
        graphics
          .moveTo(enemy.x, enemy.y)
          .lineTo(enemy.x + Math.cos(telegraph.aimAngle) * lineLength, enemy.y + Math.sin(telegraph.aimAngle) * lineLength)
          .stroke({ width: 1.4 + (telegraph.progress || 0) * 1.2, color, alpha: 0.18 + (telegraph.progress || 0) * 0.32 });
        graphics.circle(telegraph.aimX, telegraph.aimY, 10 + (telegraph.progress || 0) * 6).stroke({ width: 1.4, color, alpha: 0.3 });
      });
      (snapshot.bullets || []).forEach((bullet) => {
        if (!isInBounds(bullet, bullet.radius || 4, bounds, 80)) return;
        const color = colorToNumber(bullet.tint || (bullet.owner === "player" ? (bullet.crit ? "#f6c65f" : "#6ee7b7") : "#f06969"), 0xffffff);
        const speed = Math.hypot(bullet.vx || 0, bullet.vy || 0);
        if (speed > 1 && snapshot.premiumVfx) {
          const tail = clamp(speed * 0.035, 8, 24);
          graphics
            .moveTo(bullet.x, bullet.y)
            .lineTo(bullet.x - (bullet.vx / speed) * tail, bullet.y - (bullet.vy / speed) * tail)
            .stroke({ width: Math.max(1, (bullet.radius || 4) * 0.8), color, alpha: 0.34 });
        }
        graphics.circle(bullet.x, bullet.y, bullet.radius || 4).fill({ color, alpha: 1 });
      });
      this.renderCrosshair(snapshot);
    }

    renderEntities(snapshot) {
      const graphics = this.entityGraphics;
      const bounds = snapshot.bounds;
      (snapshot.enemies || []).forEach((enemy) => {
        const priority = snapshot.isPriorityEnemy ? snapshot.isPriorityEnemy(enemy) : false;
        if (!isInBounds(enemy, enemy.radius || 22, bounds, priority ? 260 : 120)) return;
        if (priority || snapshot.showDetail) this.drawThreatHalo(graphics, enemy, snapshot);
        this.drawShip(enemy, snapshot.getEnemyArtId ? snapshot.getEnemyArtId(enemy) : "enemyLine", enemy.color || "#f06969", false);
        if (priority || snapshot.showDetail) {
          this.drawShield(graphics, enemy, false);
          this.drawVitals(graphics, enemy);
        }
      });
      if (snapshot.player) {
        this.drawShip(snapshot.player, snapshot.playerArtId || "playerScout", "#44d2c2", true);
        this.drawShield(graphics, snapshot.player, true);
      }
      (snapshot.helpers || []).forEach((helper) => {
        if (!isInBounds(helper, 18, bounds, 80)) return;
        if (this.drawSprite("entities", "helperDrone", helper.x, helper.y, helper.angle || 0, 20, 20, { alpha: 0.96 })) return;
        graphics
          .moveTo(helper.x + 6, helper.y)
          .lineTo(helper.x - 4, helper.y + 4)
          .lineTo(helper.x - 4, helper.y - 4)
          .closePath()
          .fill({ color: 0x6ee7b7, alpha: 1 })
          .stroke({ width: 1.2, color: 0xffffff, alpha: 0.5 });
      });
    }

    renderScreenOverlays(snapshot) {
      const graphics = this.screenGraphics;
      this.drawThreatIndicators(graphics, snapshot);
    }

    renderCrosshair(snapshot) {
      const target = snapshot.aimTarget;
      if (!target) return;
      const graphics = this.overlayGraphics;
      const assist = snapshot.targetAssist;
      const color = assist ? 0xffd166 : 0xffffff;
      const alpha = assist ? 0.42 + (assist.lockStrength || 0) * 0.38 : 0.35;
      const radius = assist ? 12 : 10;
      graphics.circle(target.x, target.y, radius).stroke({ width: assist ? 1.5 : 1, color, alpha });
      graphics.moveTo(target.x - 16, target.y).lineTo(target.x - 9, target.y).stroke({ width: 1, color, alpha });
      graphics.moveTo(target.x + 9, target.y).lineTo(target.x + 16, target.y).stroke({ width: 1, color, alpha });
      graphics.moveTo(target.x, target.y - 16).lineTo(target.x, target.y - 9).stroke({ width: 1, color, alpha });
      graphics.moveTo(target.x, target.y + 9).lineTo(target.x, target.y + 16).stroke({ width: 1, color, alpha });
      if (assist && assist.enemy) {
        const enemy = assist.enemy;
        graphics.moveTo(enemy.x, enemy.y).lineTo(assist.leadX, assist.leadY).stroke({ width: 1.2, color, alpha: 0.5 });
        graphics.circle(assist.leadX, assist.leadY, 6 + (assist.lockStrength || 0) * 4).stroke({ width: 1.4, color, alpha: 0.75 });
        graphics.circle(assist.leadX, assist.leadY, 1.8).fill({ color, alpha: 0.95 });
      }
    }

    drawThreatIndicators(graphics, snapshot) {
      const player = snapshot.player;
      if (!player || !(snapshot.enemies || []).length) return;
      const margin = 34;
      const centerX = (snapshot.width || this.width) * 0.5;
      const centerY = (snapshot.height || this.height) * 0.5;
      const camera = snapshot.camera || {};
      const priorities = (snapshot.enemies || [])
        .filter((enemy) => snapshot.isPriorityEnemy ? snapshot.isPriorityEnemy(enemy) : false)
        .sort((a, b) => distanceBetween(player, a) - distanceBetween(player, b))
        .slice(0, 6);
      priorities.forEach((enemy) => {
        const screenX = enemy.x - (camera.x || 0);
        const screenY = enemy.y - (camera.y || 0);
        const onScreen = screenX >= margin && screenX <= (snapshot.width || this.width) - margin
          && screenY >= margin && screenY <= (snapshot.height || this.height) - margin;
        if (onScreen) return;
        const x = clamp(screenX, margin, (snapshot.width || this.width) - margin);
        const y = clamp(screenY, margin, (snapshot.height || this.height) - margin);
        const angle = Math.atan2(screenY - centerY, screenX - centerX);
        const color = colorToNumber(snapshot.getPriorityEnemyColor ? snapshot.getPriorityEnemyColor(enemy) : "#f6c65f", 0xf6c65f);
        const points = [
          x + Math.cos(angle) * 15,
          y + Math.sin(angle) * 15,
          x + Math.cos(angle + 2.52) * 10,
          y + Math.sin(angle + 2.52) * 10,
          x + Math.cos(angle - 2.52) * 10,
          y + Math.sin(angle - 2.52) * 10
        ];
        graphics.poly(points).fill({ color, alpha: 0.95 }).stroke({ width: 2, color: 0x080e18, alpha: 0.9 });
      });
    }

    drawShip(entity, artId, color, isPlayer) {
      const graphics = this.entityGraphics;
      const size = entity.radius || 22;
      const shipColor = colorToNumber(color, isPlayer ? 0x44d2c2 : 0xf06969);
      if (entity.thrusting) {
        const flame = 0.8 + Math.sin(nowMs() * 0.03) * 0.2;
        const backX = entity.x - Math.cos(entity.angle || 0) * size * 0.95;
        const backY = entity.y - Math.sin(entity.angle || 0) * size * 0.95;
        graphics.circle(backX, backY, size * (0.4 + flame * 0.16)).fill({ color: 0xf6c65f, alpha: 0.42 + flame * 0.18 });
      }
      const spriteWidth = size * ((artId === "enemyCommand" || artId === "playerHeavy") ? 3.4 : 3.0);
      const spriteHeight = size * (artId === "enemyArtillery" ? 1.9 : 2.72);
      if (this.drawSprite("entities", artId, entity.x, entity.y, entity.angle || 0, spriteWidth, spriteHeight, {
        alpha: entity.hitFlash > 0 ? 0.98 : 0.94,
        tint: entity.hitFlash > 0 ? 0xffffff : 0xffffff
      })) {
        if (entity.hitFlash > 0) {
          graphics.circle(entity.x, entity.y, size * 0.82).fill({ color: 0xffffff, alpha: 0.26 });
        }
        return;
      }
      const angle = entity.angle || 0;
      const noseX = entity.x + Math.cos(angle) * size;
      const noseY = entity.y + Math.sin(angle) * size;
      const leftX = entity.x + Math.cos(angle + 2.5) * size * 0.82;
      const leftY = entity.y + Math.sin(angle + 2.5) * size * 0.82;
      const tailX = entity.x - Math.cos(angle) * size * 0.4;
      const tailY = entity.y - Math.sin(angle) * size * 0.4;
      const rightX = entity.x + Math.cos(angle - 2.5) * size * 0.82;
      const rightY = entity.y + Math.sin(angle - 2.5) * size * 0.82;
      graphics.poly([noseX, noseY, leftX, leftY, tailX, tailY, rightX, rightY]).fill({ color: entity.hitFlash > 0 ? 0xffffff : shipColor, alpha: 1 });
      graphics.poly([noseX, noseY, leftX, leftY, tailX, tailY, rightX, rightY]).stroke({ width: 1.4, color: 0xffffff, alpha: 0.5 });
    }

    drawSprite(poolName, id, x, y, angle, width, height, options = {}) {
      const texture = this.textures.get(id);
      if (!texture) return false;
      const definition = this.getArtDefinitions()[id] || {};
      const sprite = this.acquireSprite(poolName);
      if (!sprite) return false;
      sprite.texture = texture;
      sprite.anchor.set(
        Number.isFinite(definition.anchorX) ? definition.anchorX : 0.5,
        Number.isFinite(definition.anchorY) ? definition.anchorY : 0.5
      );
      sprite.position.set(x, y);
      sprite.rotation = angle || 0;
      sprite.width = Math.max(1, width || definition.width || texture.width || 1);
      sprite.height = Math.max(1, height || definition.height || texture.height || 1);
      sprite.alpha = Number.isFinite(options.alpha) ? options.alpha : 1;
      sprite.tint = colorToNumber(options.tint, FALLBACK_COLOR);
      if (options.blendMode) sprite.blendMode = options.blendMode;
      return true;
    }

    drawThreatHalo(graphics, enemy, snapshot) {
      const color = colorToNumber(snapshot.getPriorityEnemyColor ? snapshot.getPriorityEnemyColor(enemy) : "#f6c65f", 0xf6c65f);
      const radius = (enemy.radius || 22) + (enemy.id === "dreadnought" ? 16 : 11);
      const pulse = 0.45 + Math.sin(nowMs() * 0.01) * 0.18;
      graphics.circle(enemy.x, enemy.y, radius).stroke({ width: enemy.id === "dreadnought" ? 3 : 2, color, alpha: 0.45 + pulse * 0.3 });
      graphics.circle(enemy.x, enemy.y, radius + 8).stroke({ width: 1.6, color, alpha: 0.22 + pulse * 0.2 });
    }

    drawShield(graphics, entity, isPlayer) {
      if (!entity || entity.maxShield <= 0 || entity.shield <= 0) return;
      const ratio = clamp(entity.shield / entity.maxShield, 0, 1);
      const radius = (entity.radius || 20) + 6;
      const color = isPlayer ? 0x57e0ff : 0xf6c65f;
      if (isPlayer) {
        this.drawSprite("vfx", "shieldRipple", entity.x, entity.y, 0, radius * 2.3, radius * 2.3, { alpha: 0.24 + ratio * 0.18, blendMode: "screen" });
      }
      graphics.circle(entity.x, entity.y, radius).stroke({ width: 2, color, alpha: 0.2 + ratio * 0.4 });
    }

    drawVitals(graphics, enemy) {
      if (!enemy) return;
      const barWidth = Math.max(26, (enemy.radius || 20) * 3.2);
      const barHeight = 4;
      const gap = 2;
      const hasShield = enemy.maxShield > 0;
      const totalHeight = barHeight + (hasShield ? barHeight + gap : 0);
      const x = enemy.x - barWidth * 0.5;
      let y = Math.max(6, enemy.y - (enemy.radius || 20) - 12 - totalHeight);
      graphics.rect(x - 1, y - 1, barWidth + 2, totalHeight + 2).fill({ color: 0x05080e, alpha: 0.7 });
      if (hasShield) {
        const shieldRatio = clamp(enemy.shield / enemy.maxShield, 0, 1);
        graphics.rect(x, y, barWidth * shieldRatio, barHeight).fill({ color: 0x57e0ff, alpha: 0.92 });
        y += barHeight + gap;
      }
      const hullRatio = clamp(enemy.health / Math.max(1, enemy.maxHealth || 1), 0, 1);
      graphics.rect(x, y, barWidth * hullRatio, barHeight).fill({ color: 0xf06969, alpha: 0.92 });
    }

    getInfo() {
      return {
        backend: this.ready ? "Pixi/WebGL" : this.failed ? "Canvas fallback" : "Canvas loading",
        ready: this.ready,
        failed: this.failed,
        width: this.app && this.app.renderer && this.app.canvas ? this.app.canvas.width : 0,
        height: this.app && this.app.renderer && this.app.canvas ? this.app.canvas.height : 0,
        resolution: this.resolution
      };
    }
  }

  function markTiming(timings, key, start) {
    const current = nowMs();
    timings[key] = (timings[key] || 0) + Math.max(0, current - start);
    return current;
  }

  function nowMs() {
    return typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
  }

  function isInBounds(item, radius, bounds, padding) {
    if (!item || !bounds) return false;
    return item.x + radius >= bounds.left - padding
      && item.x - radius <= bounds.right + padding
      && item.y + radius >= bounds.top - padding
      && item.y - radius <= bounds.bottom + padding;
  }

  function colorToNumber(value, fallback) {
    if (Number.isFinite(value)) return value;
    const text = String(value || "").trim();
    if (/^#[0-9a-f]{3}$/i.test(text)) {
      return parseInt(text.slice(1).split("").map((char) => char + char).join(""), 16);
    }
    if (/^#[0-9a-f]{6}$/i.test(text)) {
      return parseInt(text.slice(1), 16);
    }
    const rgb = text.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)/i);
    if (rgb) {
      return (Number(rgb[1]) << 16) + (Number(rgb[2]) << 8) + Number(rgb[3]);
    }
    return Number.isFinite(fallback) ? fallback : FALLBACK_COLOR;
  }

  function distanceBetween(a, b) {
    return Math.hypot((a.x || 0) - (b.x || 0), (a.y || 0) - (b.y || 0));
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  window.STELLAR_DOGFIGHT_PIXI_RENDERER = { createRenderer };
})();
