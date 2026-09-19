package me.danielshort.app.nativefeatures.games

import kotlin.math.*
import kotlin.random.Random

data class FlightEnemy(var x: Float, var y: Float, var hp: Float, var shot: Float, val role: Int)
data class FlightShot(var x: Float, var y: Float, val vx: Float, val vy: Float, val friendly: Boolean, val damage: Float)

/** Native campaign adaptation: free movement, auto-aim, adaptive enemies, waves, and hangar upgrades. */
class StellarModel(val random: Random = Random.Default) {
  var x = 500f
  var y = 750f
  var targetX = x
  var targetY = y
  var angle = -90f
  var hp = 100f
  var wave = 1
  var credits = 0
  var weapon = 0
  var armor = 0
  var kills = 0
  var bestWave = 1
  var overdrive = 0f
  var abilityCooldown = 0f
  var completed = false
  var fireTimer = 0f
  val enemies = mutableListOf<FlightEnemy>()
  val shots = mutableListOf<FlightShot>()
  val maxHp get() = 100f + armor * 20
  val ended get() = hp <= 0f || completed
  init { spawnWave() }

  fun upgradeCost(level: Int) = (60 * 1.62.pow(level.coerceIn(0, 15))).toInt()
  fun upgrade(isWeapon: Boolean): Boolean {
    val level = if (isWeapon) weapon else armor
    if (level >= 15 || credits < upgradeCost(level)) return false
    credits -= upgradeCost(level)
    if (isWeapon) weapon++ else { armor++; hp = (hp + 20).coerceAtMost(maxHp) }
    return true
  }

  fun activate(): Boolean {
    if (ended || abilityCooldown > 0f) return false
    // Same overdrive duration/cooldown and damage/fire-rate factors as data.js.
    overdrive = 6f
    abilityCooldown = 16f
    return true
  }

  fun restart() {
    hp = maxHp; wave = 1; kills = 0; completed = false; x = 500f; y = 750f
    targetX = x; targetY = y; fireTimer = 0f; overdrive = 0f; abilityCooldown = 0f
    shots.clear(); spawnWave()
  }

  fun spawnWave() {
    enemies.clear()
    repeat((3 + wave).coerceAtMost(14)) { index ->
      enemies += FlightEnemy(80f + random.nextFloat() * 840f, 70f + random.nextFloat() * 300f,
        24f + wave * 8, .8f + random.nextFloat() * 2, index % 3)
    }
  }

  fun step(seconds: Float) {
    if (ended || !seconds.isFinite()) return
    val dt = seconds.coerceIn(0f, .05f)
    val dx = targetX - x; val dy = targetY - y
    val distance = hypot(dx, dy)
    if (distance > 1f) { val step = min(distance, 340f * dt); x += dx / distance * step; y += dy / distance * step }
    x = x.coerceIn(24f, 976f); y = y.coerceIn(24f, 976f)
    overdrive = (overdrive - dt).coerceAtLeast(0f)
    abilityCooldown = (abilityCooldown - dt).coerceAtLeast(0f)
    fireTimer -= dt
    val target = enemies.minByOrNull { hypot(it.x - x, it.y - y) }
    if (target != null) {
      angle = Math.toDegrees(atan2((target.y - y).toDouble(), (target.x - x).toDouble())).toFloat() + 90
      if (fireTimer <= 0f) {
        val d = hypot(target.x - x, target.y - y).coerceAtLeast(1f)
        shots += FlightShot(x, y, (target.x - x) / d * 780, (target.y - y) / d * 780, true,
          (14f + weapon * 4) * if (overdrive > 0) 1.2f else 1f)
        fireTimer = .32f / if (overdrive > 0) 1.3f else 1f
      }
    }
    enemies.forEach { enemy ->
      val ex = x - enemy.x; val ey = y - enemy.y; val d = hypot(ex, ey).coerceAtLeast(1f)
      val standOff = if (enemy.role == 0) 170f else 290f
      val approach = if (d > standOff) 1 else if (d < standOff - 60) -1 else 0
      val speed = 30f + wave * 4 + enemy.role * 9
      enemy.x += (ex / d * approach - ey / d * .5f) * speed * dt
      enemy.y += (ey / d * approach + ex / d * .5f) * speed * dt
      enemy.x = enemy.x.coerceIn(30f, 970f); enemy.y = enemy.y.coerceIn(30f, 970f)
      enemy.shot -= dt
      if (enemy.shot <= 0) {
        shots += FlightShot(enemy.x, enemy.y, ex / d * (140 + wave * 7), ey / d * (140 + wave * 7), false, 7f + wave)
        enemy.shot = (2.4f - wave * .08f).coerceAtLeast(.75f)
      }
    }
    val iterator = shots.iterator()
    while (iterator.hasNext()) {
      val shot = iterator.next(); shot.x += shot.vx * dt; shot.y += shot.vy * dt
      if (shot.x !in -30f..1030f || shot.y !in -30f..1030f) { iterator.remove(); continue }
      if (shot.friendly) {
        val hit = enemies.firstOrNull { it.hp > 0 && hypot(it.x - shot.x, it.y - shot.y) < 27 }
        if (hit != null) { hit.hp -= shot.damage; iterator.remove() }
      } else if (hypot(x - shot.x, y - shot.y) < 22) { hp = (hp - shot.damage).coerceAtLeast(0f); iterator.remove() }
    }
    val fallen = enemies.count { it.hp <= 0 }
    kills += fallen; credits = (credits + fallen * 12).coerceAtMost(1_000_000)
    enemies.removeAll { it.hp <= 0 }
    if (enemies.isEmpty() && hp > 0) {
      credits += wave * 15; bestWave = max(bestWave, wave)
      if (wave >= 10) completed = true else { wave++; bestWave = max(bestWave, wave); hp = min(maxHp, hp + 20); shots.clear(); spawnWave() }
    }
  }
}
