package me.danielshort.app.nativefeatures.games

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** Small bounded native game checkpoints; independent of website/localStorage saves. */
internal class GameStorage(context: Context) {
  private val preferences = context.getSharedPreferences("native_games_v1", Context.MODE_PRIVATE)
  fun read(id: String): JSONObject? = runCatching {
    val raw = preferences.getString(id, null) ?: return null
    require(raw.length <= 100_000)
    JSONObject(raw).takeIf { it.optInt("schema") == 1 }
  }.getOrNull()
  fun save(id: String, value: JSONObject) {
    value.put("schema", 1).put("savedAt", System.currentTimeMillis())
    preferences.edit().putString(id, value.toString()).apply()
  }
}

private fun JSONObject.number(key: String, fallback: Double, low: Double, high: Double): Double =
  optDouble(key, fallback).let { if (it.isFinite()) it.coerceIn(low, high) else fallback }
private fun JSONObject.int(key: String, fallback: Int, low: Int, high: Int) = number(key, fallback.toDouble(), low.toDouble(), high.toDouble()).toInt()
private fun JSONObject.float(key: String, fallback: Float, low: Float, high: Float) = number(key, fallback.toDouble(), low.toDouble(), high.toDouble()).toFloat()
private fun JSONObject.list(key: String, max: Int): List<JSONObject> = optJSONArray(key)?.let { a ->
  (0 until minOf(a.length(), max)).mapNotNull { a.optJSONObject(it) }
} ?: emptyList()

internal fun StellarModel.snapshot() = JSONObject().apply {
  put("x", x); put("y", y); put("hp", hp); put("wave", wave); put("credits", credits)
  put("weapon", weapon); put("armor", armor); put("kills", kills); put("best", bestWave); put("completed", completed)
  put("abilityCooldown", abilityCooldown); put("overdrive", overdrive)
  put("enemies", JSONArray(enemies.map { JSONObject().put("x", it.x).put("y", it.y).put("hp", it.hp).put("shot", it.shot).put("role", it.role) }))
  put("shots", JSONArray(shots.map { JSONObject().put("x", it.x).put("y", it.y).put("vx", it.vx).put("vy", it.vy).put("friendly", it.friendly).put("damage", it.damage) }))
}
internal fun StellarModel.restore(j: JSONObject?) {
  if (j == null) return
  weapon = j.int("weapon", 0, 0, 15); armor = j.int("armor", 0, 0, 15)
  x = j.float("x", 500f, 24f, 976f); y = j.float("y", 750f, 24f, 976f); targetX = x; targetY = y
  hp = j.float("hp", maxHp, 0f, maxHp); wave = j.int("wave", 1, 1, 10); credits = j.int("credits", 0, 0, 1_000_000)
  kills = j.int("kills", 0, 0, 1_000_000); bestWave = j.int("best", 1, 1, 10); completed = j.optBoolean("completed", false)
  abilityCooldown = j.float("abilityCooldown", 0f, 0f, 16f); overdrive = j.float("overdrive", 0f, 0f, 6f)
  enemies.clear(); enemies.addAll(j.list("enemies", 14).map {
    FlightEnemy(it.float("x", 500f, 0f, 1000f), it.float("y", 150f, 0f, 1000f), it.float("hp", 32f, 1f, 500f), it.float("shot", 1f, 0f, 5f), it.int("role", 0, 0, 2))
  })
  if (enemies.isEmpty() && !ended) spawnWave()
  shots.clear(); shots.addAll(j.list("shots", 200).map {
    FlightShot(it.float("x", 500f, -30f, 1030f), it.float("y", 500f, -30f, 1030f),
      it.float("vx", 0f, -1000f, 1000f), it.float("vy", 0f, -1000f, 1000f), it.optBoolean("friendly"), it.float("damage", 7f, 0f, 500f))
  })
}

internal fun StormbreakModel.snapshot() = JSONObject().apply {
  put("gold", gold); put("ambrosia", ambrosia); put("level", level); put("xp", xp); put("hp", hp); put("zone", zone)
  put("waves", JSONArray(waves)); put("upgrades", JSONArray(upgrades)); put("defeated", defeated); put("kills", totalDefeated)
  put("auto", autoAttack); put("cooldowns", JSONArray(cooldowns)); put("shield", shield); put("enemyHp", enemy.hp)
}
internal fun StormbreakModel.restore(j: JSONObject?) {
  if (j == null) return
  gold = j.number("gold", 0.0, 0.0, 1e15); ambrosia = j.number("ambrosia", 0.0, 0.0, 1e12)
  level = j.int("level", 1, 1, 250); xp = j.number("xp", 0.0, 0.0, nextXp - 1)
  upgrades = MutableList(3) { (j.optJSONArray("upgrades")?.optInt(it, 1) ?: 1).coerceIn(1, 50) }
  waves = MutableList(3) { (j.optJSONArray("waves")?.optInt(it, 1) ?: 1).coerceIn(1, StormbreakModel.MAX_WAVE) }
  hp = j.number("hp", maxHp, 0.0, maxHp); zone = j.int("zone", 0, 0, 2)
  if (level < StormbreakModel.ZONES[zone].level) zone = 0
  defeated = j.int("defeated", 0, 0, target - 1); totalDefeated = j.int("kills", 0, 0, 1_000_000_000)
  autoAttack = j.optBoolean("auto", true)
  cooldowns = MutableList(3) { (j.optJSONArray("cooldowns")?.optDouble(it, 0.0) ?: 0.0).let { d -> if (d.isFinite()) d.coerceIn(0.0, 18.0) else 0.0 } }
  shield = j.number("shield", 0.0, 0.0, 6.0); spawn(); enemy.hp = j.number("enemyHp", enemy.maxHp, 1.0, enemy.maxHp)
  offlineReward = 0.0
  if (j.optBoolean("offlineEligible", true)) applyOffline((System.currentTimeMillis() - j.optLong("savedAt", System.currentTimeMillis())).coerceAtLeast(0))
}

internal fun ProbabilityModel.snapshot() = JSONObject().apply {
  put("credits", credits); put("earned", earned); put("spins", spins); put("luck", luck); put("motor", motor)
  put("badLuck", badLuck); put("jackpot", jackpotReady); put("deck", JSONArray(deck)); put("grid", JSONArray(grid))
  put("inventory", JSONObject(inventory)); put("reward", result.reward); put("cooldown", cooldown)
}
internal fun ProbabilityModel.restore(j: JSONObject?) {
  if (j == null) return
  credits = j.number("credits", 120.0, 0.0, 1e15); earned = j.number("earned", 0.0, 0.0, 1e15)
  spins = j.int("spins", 0, 0, Int.MAX_VALUE - 1); luck = j.int("luck", 0, 0, 20); motor = j.int("motor", 0, 0, 20)
  badLuck = j.int("badLuck", 0, 0, 9); jackpotReady = j.optBoolean("jackpot")
  fun strings(key: String) = j.optJSONArray(key)?.let { a -> (0 until minOf(a.length(), 32)).map { a.optString(it) }.filter { it in ProbabilityModel.SYMBOLS } } ?: emptyList()
  strings("deck").takeIf { it.size in 10..32 }?.let { deck = it.toMutableList() }
  strings("grid").takeIf { it.size == 9 }?.let { grid = it }
  j.optJSONObject("inventory")?.let { inv -> inventory = ProbabilityModel.SYMBOLS.keys.associateWith { inv.int(it, 0, 0, 100000) }.toMutableMap() }
  result = ReelResult(j.number("reward", 0.0, 0.0, 1e9), emptyList(), emptySet())
  cooldown = j.number("cooldown", 0.0, 0.0, spinDelay)
  auto = false
}

internal fun StarfallModel.snapshot() = JSONObject().apply {
  put("class", classIndex); put("x", x); put("y", y); put("vy", vy); put("hp", hp); put("level", level); put("xp", xp)
  put("gold", gold); put("kills", kills); put("wave", wave); put("quest", questClaimed); put("skill", skillCooldown)
  put("grounded", grounded); put("facing", facing); put("attack", attackCooldown)
  fun FieldLoot.json() = JSONObject().put("name", name).put("bonus", bonus).put("rarity", rarity)
  put("gear", gear.json()); put("inventory", JSONArray(inventory.map { it.json() }))
  put("enemies", JSONArray(enemies.map { JSONObject().put("x", it.x).put("y", it.y).put("hp", it.hp).put("maxHp", it.maxHp).put("attack", it.attack) }))
}
internal fun StarfallModel.restore(j: JSONObject?) {
  if (j == null) return
  classIndex = j.int("class", 0, 0, 2); level = j.int("level", 1, 1, 50); xp = j.int("xp", 0, 0, nextXp - 1)
  x = j.float("x", 110f, 32f, 4168f); y = j.float("y", 520f, 0f, 520f); vy = j.float("vy", 0f, -600f, 1000f)
  grounded = j.optBoolean("grounded", vy == 0f && StarfallModel.PLATFORMS.any { kotlin.math.abs(y - it.y) < .01f && x >= it.x && x <= it.x + it.width })
  facing = if (j.float("facing", 1f, -1f, 1f) < 0) -1f else 1f
  move = 0f; attackFlash = 0f; attackCooldown = j.float("attack", 0f, 0f, .6f)
  hp = j.float("hp", maxHp, 0f, maxHp); gold = j.int("gold", 0, 0, 1_000_000); kills = j.int("kills", 0, 0, 1_000_000)
  wave = j.int("wave", 1, 1, 100); questClaimed = j.optBoolean("quest"); skillCooldown = j.float("skill", 0f, 0f, 8f)
  fun JSONObject.loot() = FieldLoot(optString("name", "Verge weapon").take(40), int("bonus", 0, 0, 155), if (optString("rarity") == "Rare") "Rare" else "Common")
  j.optJSONObject("gear")?.let { gear = it.loot() }; inventory = j.list("inventory", 24).map { it.loot() }.toMutableList()
  if (j.has("enemies")) {
    enemies.clear(); enemies.addAll(j.list("enemies", 20).map {
      val maxHp = it.float("maxHp", 47f, 1f, 2000f)
      FieldEnemy(it.float("x", 400f, 32f, 4168f), 520f, it.float("hp", maxHp, 1f, maxHp), maxHp, it.float("attack", 1.5f, 0f, 1.6f))
    })
  } else spawnWave()
}
