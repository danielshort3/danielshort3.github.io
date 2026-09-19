package me.danielshort.app.nativefeatures.games

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import kotlin.random.Random

class NativeGamesTest {
  @Test fun oceanUsesBoundedFiniteDeepWaterSurface() {
    assertEquals(0f, OceanModel.height(3f, 9f, 1f, 8f, 0f), 0f)
    assertEquals(0f, OceanModel.height(Float.NaN, 9f, 1f, 8f, 1f), 0f)
    val first = OceanModel.height(3f, 9f, 1f, 8f, 1f)
    val next = OceanModel.height(3f, 9f, 2f, 8f, 1f)
    assertTrue(first.isFinite() && next.isFinite())
    assertNotEquals(first, next)
  }

  @Test fun flightMotionCannotTeleportAfterLongFrame() {
    val game = StellarModel(Random(1)); game.targetX = 0f; game.targetY = 0f
    val x = game.x; val y = game.y; game.step(60f)
    assertTrue(kotlin.math.hypot(game.x - x, game.y - y) <= 17.1f)
    game.step(Float.NaN)
    assertTrue(game.x.isFinite() && game.y.isFinite())
  }

  @Test fun stellarOverdriveHasRealCooldownAndExpires() {
    val game = StellarModel(Random(2))
    assertTrue(game.activate()); assertFalse(game.activate())
    repeat(121) { game.step(.05f) }
    assertEquals(0f, game.overdrive, .01f)
    assertTrue(game.abilityCooldown > 9f)
  }

  @Test fun stellarUpgradesCannotOverspendAndSurviveNewFlight() {
    val game = StellarModel(Random(3))
    assertFalse(game.upgrade(true)); assertEquals(0, game.credits)
    game.credits = 60; assertTrue(game.upgrade(false)); assertEquals(0, game.credits)
    assertEquals(120f, game.maxHp, 0f)
    game.hp = 0f; game.restart()
    assertEquals(120f, game.hp, 0f); assertEquals(1, game.armor)
  }

  @Test fun flightCheckpointPreservesActiveEnemiesAndShots() {
    val game = StellarModel(Random(4)); repeat(20) { game.step(.04f) }; game.enemies[0].hp = 5f
    val restored = StellarModel(Random(50)).apply { restore(game.snapshot()) }
    assertEquals(game.x, restored.x, 0f); assertEquals(game.wave, restored.wave)
    assertEquals(game.enemies.size, restored.enemies.size); assertEquals(5f, restored.enemies[0].hp, 0f)
    assertEquals(game.shots.size, restored.shots.size)
  }

  @Test fun olympusUpgradeCostMatchesWebsiteAndCannotOverspend() {
    val game = StormbreakModel(Random(5))
    assertEquals(50.0, game.cost(0), 0.0); assertEquals(75.0, game.cost(1), 0.0)
    assertFalse(game.upgrade(0)); game.gold = 50.0; assertTrue(game.upgrade(0))
    assertEquals(0.0, game.gold, 0.0); assertEquals(81.0, game.cost(0), 0.0)
  }

  @Test fun olympusBossEveryFifthWaveAndZoneGatesAreEnforced() {
    val game = StormbreakModel(Random(6))
    assertFalse(game.selectZone(1)); game.waves[0] = 5; game.spawn()
    assertTrue(game.enemy.boss); assertEquals(1, game.target)
    game.level = 6; assertTrue(game.selectZone(1)); assertFalse(game.selectZone(2))
  }

  @Test fun olympusOfflineRewardsRespectFourHourCapAndAutoplay() {
    val first = StormbreakModel(Random(7)); val second = StormbreakModel(Random(7))
    assertEquals(0.0, first.applyOffline(59_999), 0.0)
    val fourHours = first.applyOffline(14_400_000)
    assertTrue(fourHours > 0)
    assertEquals(fourHours, second.applyOffline(5 * 14_400_000), 0.0)
    second.autoAttack = false; assertEquals(0.0, second.applyOffline(14_400_000), 0.0)
  }

  @Test fun olympusAbilityDoesNotRepeatDuringCooldown() {
    val game = StormbreakModel(Random(8)); game.hp = 20.0
    assertTrue(game.ability(2)); val hp = game.hp
    assertFalse(game.ability(2)); assertEquals(hp, game.hp, 0.0)
    assertEquals(6.0, game.shield, 0.0)
  }

  @Test fun reelNeighborsNeverWrapAcrossRows() {
    assertEquals(setOf(1, 3), ProbabilityModel.neighbors(0).toSet())
    assertEquals(setOf(5, 1), ProbabilityModel.neighbors(2).toSet())
    assertEquals(setOf(1, 7, 3, 5), ProbabilityModel.neighbors(4).toSet())
  }

  @Test fun reelConsumptionPaysOnlyOnceAndCountsOrthogonalSynergy() {
    val grid = listOf("wolf", "sheep", "coin", "coin", "coin", "coin", "coin", "coin", "coin")
    val result = ProbabilityModel.evaluate(grid, 1.0, Random(0))
    assertEquals(setOf(1), result.removed)
    assertEquals(listOf("Wolf + Sheep"), result.synergies)
    assertEquals((7 + 7 * 3 + 12) * (1 + .95 / 9), result.reward, .00001)
  }

  @Test fun reelNoSynergyReturnsSumOfAuthoredSymbolValues() {
    val result = ProbabilityModel.evaluate(List(9) { "coin" }, 1.0, Random(0))
    assertEquals(27.0, result.reward, 0.0)
    assertTrue(result.synergies.isEmpty())
  }

  @Test fun reelsPreventDoubleSpinAndProtectMinimumDeckSize() {
    val game = ProbabilityModel(Random(9))
    assertTrue(game.spin()); val credits = game.credits
    assertFalse(game.spin()); assertEquals(credits, game.credits, 0.0)
    assertFalse(game.removeSymbol("coal")); assertTrue(game.addSymbol("coal")); assertTrue(game.removeSymbol("coal"))
    assertEquals(10, game.deck.size)
  }

  @Test fun reelCheckpointRestoresDeckEconomyAndPausesAutoSpins() {
    val game = ProbabilityModel(Random(10)); game.addSymbol("magnet"); game.spin(); game.auto = true
    val restored = ProbabilityModel(Random(1)).apply { restore(game.snapshot()) }
    assertEquals(game.deck, restored.deck); assertEquals(game.credits, restored.credits, 0.0)
    assertEquals(game.grid, restored.grid); assertFalse(restored.auto)
  }

  @Test fun starfallUsesAuthoredClassValuesAndLandsWithoutDoubleJump() {
    val game = StarfallModel(Random(11)); game.chooseClass(1)
    assertEquals(135f, game.maxHp, 0f); assertEquals(380f, game.hero.range, 0f)
    game.jump(); val velocity = game.vy; game.jump(); assertEquals(velocity, game.vy, 0f)
    repeat(35) { game.step(.04f) }
    assertTrue(game.grounded); assertEquals(520f, game.y, .01f)
  }

  @Test fun starfallLootCannotBeEquippedWithoutOwningIt() {
    val game = StarfallModel(Random(12))
    assertFalse(game.equip(0)); assertEquals("Training weapon", game.gear.name)
    game.inventory.add(FieldLoot("Verge weapon", 3, "Common"))
    assertTrue(game.equip(0)); assertEquals(21f, game.power, 0f)
    assertEquals("Training weapon", game.inventory[0].name)
  }

  @Test fun starfallProgressPersistsWithoutHealingEnemiesOrLosingGear() {
    val game = StarfallModel(Random(13)); game.enemies[0].hp = 2f
    game.gear = FieldLoot("Star-glass weapon", 8, "Rare"); game.gold = 100; game.hp = 55f
    val restored = StarfallModel(Random(14)).apply { restore(game.snapshot()) }
    assertEquals(2f, restored.enemies[0].hp, 0f); assertEquals(game.gear, restored.gear)
    assertEquals(55f, restored.hp, 0f); assertEquals(100, restored.gold)
  }

  @Test fun malformedCheckpointsAreClampedBeforeSimulation() {
    val raw = JSONObject().put("level", -20).put("wave", 999999).put("credits", -99).put("x", "NaN")
    val flight = StellarModel().apply { restore(raw) }
    assertEquals(10, flight.wave); assertEquals(0, flight.credits); assertTrue(flight.x.isFinite())
    val starfall = StarfallModel().apply { restore(raw) }
    assertEquals(1, starfall.level); assertEquals(100, starfall.wave); assertTrue(starfall.x.isFinite())
  }

  @Test fun starfallAirborneCheckpointCannotGrantAnExtraJumpOrRetainMovement() {
    val game = StarfallModel(Random(14))
    game.jump(); game.step(.05f); game.facing = -1f; game.move = -1f; game.attackCooldown = .3f
    game.enemies[0].attack = .2f
    val restored = StarfallModel(Random(15)).apply { restore(game.snapshot()) }
    assertFalse(restored.grounded)
    assertEquals(-1f, restored.facing, 0f)
    assertEquals(0f, restored.move, 0f)
    assertEquals(.3f, restored.attackCooldown, .0001f)
    assertEquals(.2f, restored.enemies[0].attack, .0001f)
    val velocity = restored.vy
    restored.jump()
    assertEquals(velocity, restored.vy, 0f)
  }

  @Test fun olympusFinalWaveRetainsFiniteStatsAndSurvivesCheckpoint() {
    val game = StormbreakModel(Random(16))
    game.waves[0] = StormbreakModel.MAX_WAVE; game.spawn()
    game.enemy.hp = 1.0; game.strike()
    assertEquals(StormbreakModel.MAX_WAVE, game.wave)
    assertTrue(game.enemy.hp.isFinite() && game.enemy.damage.isFinite())
    val restored = StormbreakModel(Random(17)).apply { restore(game.snapshot()) }
    assertEquals(game.wave, restored.wave)
  }

  @Test fun explicitlyPausedOlympusDoesNotEarnOfflineGold() {
    val game = StormbreakModel(Random(18))
    val checkpoint = game.snapshot().put("offlineEligible", false).put("savedAt", System.currentTimeMillis() - 7_200_000)
    val restored = StormbreakModel(Random(19)).apply { restore(checkpoint) }
    assertEquals(game.gold, restored.gold, 0.0)
    assertEquals(0.0, restored.offlineReward, 0.0)
  }
}
