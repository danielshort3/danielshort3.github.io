package me.danielshort.app.nativefeatures.games

import kotlin.math.*
import kotlin.random.Random

data class OlympusZone(val name: String, val level: Int, val difficulty: Double, val gold: Double, val ambrosia: Double, val sprites: List<Int>, val boss: Int)
data class OlympusEnemy(val name: String, val sprite: Int, val boss: Boolean, val maxHp: Double, var hp: Double,
  val damage: Double, val interval: Double, var timer: Double, val gold: Double, val xp: Double)

/** Combat/progression equations ported from js/games/stormbreak/app.js. Native saves are separate. */
class StormbreakModel(private val random: Random = Random.Default) {
  var gold = 0.0
  var ambrosia = 0.0
  var level = 1
  var xp = 0.0
  var hp = 100.0
  var zone = 0
  var waves = mutableListOf(1, 1, 1)
  var upgrades = mutableListOf(1, 1, 1)
  var defeated = 0
  var totalDefeated = 0
  var autoAttack = true
  var cooldowns = mutableListOf(0.0, 0.0, 0.0)
  var shield = 0.0
  var autoTimer = .5
  var manualTimer = 0.0
  var flash = 0.0
  var message = "Command Zeus in the Temple of Ash."
  var offlineReward = 0.0
  lateinit var enemy: OlympusEnemy
  val wave get() = waves[zone]
  val maxHp get() = 100.0 + max(0, upgrades[2] - 1) * 25
  val damage get() = 13 * (1 + upgrades[0] * .12) * (1 + max(0, level - 1) * .035)
  val attackInterval get() = max(.52, .92 - min(.4, level * .006))
  val nextXp get() = min(2e9, floor(100 * 1.28.pow(max(0, level - 1))))
  val target get() = if (wave % 5 == 0) 1 else 10
  init { spawn() }

  fun spawn() {
    val z = ZONES[zone]; val boss = wave % 5 == 0
    val sprite = if (boss) z.boss else z.sprites[(wave + defeated - 1) % z.sprites.size]
    val archetype = ARCHETYPES[sprite]
    val health = floor(archetype.second * z.difficulty * 1.155.pow(wave - 1) * (1 + defeated % 4 * .045) * if (boss) 5.4 else 1.0)
    val name = if (boss) BOSSES[zone][wave / 5 % 2] else archetype.first
    enemy = OlympusEnemy(name, sprite, boss, health, health,
      ATTACK[sprite] * z.difficulty * 1.095.pow(wave - 1) * if (boss) 1.45 else 1.0,
      max(1.05, INTERVAL[sprite] - min(.45, wave * .012)), .9 + random.nextDouble() * .7,
      floor((9 + archetype.second * .065) * z.gold * (1 + upgrades[1] * .07) * 1.12.pow(wave - 1) * if (boss) 5.8 else 1.0),
      floor((11 + archetype.second * .05) * z.difficulty * 1.06.pow(wave - 1) * if (boss) 4.2 else 1.0))
  }

  fun strike(multiplier: Double = 1.18, criticalChance: Double = .16) {
    val critical = random.nextDouble() < criticalChance
    val amount = max(1.0, damage * multiplier * if (critical) 1.75 else 1.0)
    enemy.hp = max(0.0, enemy.hp - amount); flash = .22
    message = "${if (critical) "Critical · " else ""}${ceil(amount).toInt()} damage"
    if (enemy.hp <= 0) {
      gold = min(1e15, gold + enemy.gold); xp += enemy.xp; totalDefeated++
      if (enemy.boss) ambrosia = min(1e12, ambrosia + max(1.0, floor(ZONES[zone].ambrosia * (1 + wave / 20.0))))
      val oldLevel = level
      while (level < 250 && xp >= nextXp) { xp -= nextXp; level++ }
      if (level > oldLevel) hp = min(maxHp, hp + 20 + (level - oldLevel) * 5)
      defeated++
      if (defeated >= target) { waves[zone] = min(MAX_WAVE, wave + 1); defeated = 0; hp = min(maxHp, hp + maxHp * .08) }
      spawn()
    }
  }

  fun tap() { if (manualTimer <= 0) { strike(); manualTimer = .16 } }
  fun ability(index: Int): Boolean {
    if (index !in 0..2 || cooldowns[index] > 0) return false
    cooldowns[index] = listOf(5.0, 14.0, 18.0)[index]
    when (index) {
      0 -> strike(3.25, .2)
      1 -> strike(8.0, .1)
      2 -> { shield = 6.0; hp = min(maxHp, hp + maxHp * .18); message = "Aegis raised for 6 seconds." }
    }
    return true
  }

  fun cost(index: Int): Double = if (upgrades[index] >= 50) Double.POSITIVE_INFINITY else
    min(1e15, floor(listOf(50.0, 75.0, 100.0)[index] * listOf(1.62, 1.66, 1.7)[index].pow(upgrades[index] - 1)))
  fun upgrade(index: Int): Boolean {
    if (index !in 0..2 || gold < cost(index)) return false
    gold -= cost(index); upgrades[index]++; if (index == 2) hp += 25
    return true
  }
  fun selectZone(index: Int): Boolean {
    if (index !in ZONES.indices || level < ZONES[index].level || zone == index) return false
    zone = index; defeated = 0; spawn(); return true
  }

  fun step(seconds: Double) {
    if (!seconds.isFinite()) return
    val dt = seconds.coerceIn(0.0, .1)
    cooldowns.indices.forEach { cooldowns[it] = max(0.0, cooldowns[it] - dt) }
    shield = max(0.0, shield - dt); manualTimer = max(0.0, manualTimer - dt); flash = max(0.0, flash - dt)
    if (autoAttack) { autoTimer -= dt; if (autoTimer <= 0) { strike(1.0, .1); autoTimer += attackInterval } }
    enemy.timer -= dt
    if (enemy.timer <= 0) {
      hp = max(0.0, hp - max(1.0, enemy.damage * (1 - min(.45, upgrades[2] * .03)) * if (shield > 0) .28 else 1.0))
      enemy.timer += enemy.interval
      if (hp <= 0) { gold -= floor(gold * .03); hp = maxHp; enemy.hp = max(enemy.hp, enemy.maxHp * .35); shield = 2.0; message = "Zeus rallied. Retreat cost 3% gold." }
    }
    if (shield <= 0) hp = min(maxHp, hp + maxHp * .0025 * dt)
  }

  /** Same four-hour capped estimated gold model as web. No combat/XP fabricated offline. */
  fun applyOffline(elapsedMs: Long): Double {
    if (!autoAttack || elapsedMs < 60_000) return 0.0
    val z = ZONES[zone]
    val averageHp = 52 * z.difficulty * 1.155.pow(wave - 1)
    val averageReward = 12 * z.gold * (1 + upgrades[1] * .07) * 1.12.pow(wave - 1)
    val gain = floor(damage / attackInterval / max(1.0, averageHp) * averageReward * .78 *
      elapsedMs.coerceAtMost(4 * 60 * 60 * 1000L) / 1000 * .72).coerceAtLeast(1.0)
    gold = min(1e15, gold + gain); offlineReward = gain; return gain
  }

  companion object {
    const val MAX_WAVE = 200
    val ZONES = listOf(
      OlympusZone("Temple of Ash", 1, 1.0, 1.2, 1.1, listOf(0, 1, 0, 3), 2),
      OlympusZone("Cliffs of the Cyclops", 6, 2.2, 1.8, 1.3, listOf(1, 0, 2, 1), 2),
      OlympusZone("Fields of Elysium", 14, 4.4, 2.5, 1.8, listOf(3, 2, 0, 3), 3))
    private val ARCHETYPES = listOf("Ashhorn Raider" to 34.0, "Cyclopean Mauler" to 52.0, "Obsidian Minotaur" to 78.0, "Elysian Sentinel" to 66.0)
    private val ATTACK = listOf(5, 7, 9, 8)
    private val INTERVAL = listOf(2.1, 2.5, 2.35, 1.9)
    private val BOSSES = listOf(listOf("Asterion, Ash Tyrant", "Bronzehoof the Unbroken"), listOf("Brontes, Mountain Breaker", "Steropes the Relentless"), listOf("Talos, Warden of Elysium", "The Gilded Colossus"))
  }
}
