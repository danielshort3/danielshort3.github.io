package me.danielshort.app.nativefeatures.games

import kotlin.math.*
import kotlin.random.Random

data class StarfallClass(val name: String, val hp: Float, val power: Float, val defense: Float, val speed: Float, val jump: Float, val range: Float)
data class FieldPlatform(val x: Float, val y: Float, val width: Float)
data class FieldEnemy(var x: Float, var y: Float, var hp: Float, val maxHp: Float, var attack: Float = 1.5f)
data class FieldLoot(val name: String, val bonus: Int, val rarity: String)

/** Native first-expedition adaptation of Starfall's classes, gravity, platforms and gear loop. */
class StarfallModel(private val random: Random = Random.Default) {
  var classIndex = 0
  var x = 110f
  var y = 520f
  var vy = 0f
  var move = 0f
  var facing = 1f
  var grounded = true
  var hp = 180f
  var level = 1
  var xp = 0
  var gold = 0
  var kills = 0
  var wave = 1
  var attackCooldown = 0f
  var skillCooldown = 0f
  var attackFlash = 0f
  var gear = FieldLoot("Training weapon", 0, "Common")
  var inventory = mutableListOf<FieldLoot>()
  var message = "Defeat 18 enemies to secure Starfall Verge."
  var questClaimed = false
  val enemies = mutableListOf<FieldEnemy>()
  val hero get() = CLASSES[classIndex]
  val maxHp get() = hero.hp + (level - 1) * 12
  val power get() = hero.power + (level - 1) * 2 + gear.bonus
  val nextXp get() = 45 + level * 20
  val dead get() = hp <= 0
  init { spawnWave() }

  fun chooseClass(index: Int) {
    if (index !in CLASSES.indices || kills > 0) return
    classIndex = index; hp = maxHp
  }
  fun jump() { if (grounded && !dead) { vy = -hero.jump; grounded = false } }
  fun strike(skill: Boolean = false): Boolean {
    if (dead || attackCooldown > 0 || (skill && skillCooldown > 0)) return false
    attackCooldown = if (hero.range < 100) .42f else .6f
    if (skill) skillCooldown = 8f
    attackFlash = .25f
    val range = hero.range * if (skill) 1.4f else 1f
    val targets = enemies.filter { it.hp > 0 && abs(it.x - x) <= range && abs(it.y - y) < 100 && (it.x - x) * facing > -30 }
    targets.forEach {
      it.hp = (it.hp - power * if (skill) 2.4f else 1f).coerceAtLeast(0f)
      it.x = (it.x + facing * if (skill) 38 else 14).coerceIn(40f, 4160f)
    }
    claimKills()
    return true
  }
  private fun claimKills() {
    enemies.filter { it.hp <= 0 }.forEach {
      kills++; gold = (gold + 8 + wave * 2).coerceAtMost(1_000_000); xp += 12 + wave * 3
      if (kills % 3 == 0) {
        val rare = random.nextFloat() < .22f
        val drop = FieldLoot(if (rare) "Star-glass weapon" else "Verge weapon", wave + level + if (rare) 4 else 1, if (rare) "Rare" else "Common")
        inventory.add(drop); if (inventory.size > 24) inventory.removeAt(0)
        message = "Found ${drop.name} · +${drop.bonus} power"
      }
    }
    enemies.removeAll { it.hp <= 0 }
    while (level < 50 && xp >= nextXp) { xp -= nextXp; level++; hp = maxHp }
    if (kills >= 18 && !questClaimed) { questClaimed = true; gold += 180; message = "Starfall Verge secured · +180 gold" }
  }
  fun spawnWave() {
    enemies.clear()
    listOf(440f, 780f, 1140f, 1570f, 1900f, 2400f, 2830f, 3210f).forEach { position ->
      val hp = 35f + wave * 12
      // Ground encounters keep the first native expedition accessible to every class.
      enemies += FieldEnemy(position, 520f, hp, hp)
    }
  }
  fun nextWave(): Boolean {
    if (enemies.isNotEmpty()) return false
    wave = (wave + 1).coerceAtMost(100); hp = maxHp; spawnWave(); return true
  }
  fun equip(index: Int): Boolean {
    if (index !in inventory.indices) return false
    val previous = gear; gear = inventory[index]; inventory[index] = previous; return true
  }
  fun rest(): Boolean {
    if (gold < 20 || hp >= maxHp) return false
    gold -= 20; hp = maxHp; return true
  }
  fun revive() { hp = maxHp; gold -= gold / 10; x = 110f; y = 520f; vy = 0f; grounded = true; move = 0f }
  fun step(seconds: Float) {
    if (dead || !seconds.isFinite()) return
    val dt = seconds.coerceIn(0f, .05f)
    attackCooldown = max(0f, attackCooldown - dt); skillCooldown = max(0f, skillCooldown - dt); attackFlash = max(0f, attackFlash - dt)
    if (move != 0f) facing = sign(move)
    x = (x + move.coerceIn(-1f, 1f) * hero.speed * dt).coerceIn(32f, 4168f)
    val oldY = y
    vy += 1600f * dt; y += vy * dt; grounded = false
    if (vy >= 0) {
      val landing = PLATFORMS.filter { oldY <= it.y + 1 && y >= it.y && x + 17 >= it.x && x - 17 <= it.x + it.width }.minByOrNull { it.y }
      if (landing != null) { y = landing.y; vy = 0f; grounded = true }
    }
    enemies.forEach { enemy ->
      if (abs(enemy.x - x) < 550) {
        if (abs(enemy.x - x) > 42) enemy.x += sign(x - enemy.x) * (64 + wave * 2) * dt
        enemy.attack -= dt
        if (enemy.attack <= 0 && abs(enemy.x - x) < 62 && abs(enemy.y - y) < 70) {
          hp = max(0f, hp - max(1f, 9 + wave * 3 - hero.defense)); enemy.attack = 1.6f
        }
      }
    }
  }
  companion object {
    // Exact base-class stats from data/classes.js; field coordinates from map-catalog.js.
    val CLASSES = listOf(StarfallClass("Fighter", 180f, 18f, 8f, 220f, 520f, 76f),
      StarfallClass("Mage", 135f, 20f, 4f, 212f, 510f, 380f), StarfallClass("Archer", 150f, 19f, 5f, 235f, 525f, 430f))
    val PLATFORMS = listOf(FieldPlatform(0f, 520f, 4200f), FieldPlatform(260f, 452f, 340f),
      FieldPlatform(620f, 388f, 300f), FieldPlatform(960f, 318f, 280f), FieldPlatform(1360f, 452f, 410f),
      FieldPlatform(1810f, 382f, 330f), FieldPlatform(2180f, 312f, 290f), FieldPlatform(2640f, 452f, 360f),
      FieldPlatform(3040f, 388f, 320f), FieldPlatform(3420f, 322f, 300f), FieldPlatform(3840f, 452f, 420f))
  }
}
