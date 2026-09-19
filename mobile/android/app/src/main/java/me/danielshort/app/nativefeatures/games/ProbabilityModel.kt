package me.danielshort.app.nativefeatures.games

import kotlin.math.*
import kotlin.random.Random

data class ReelSymbol(val id: String, val name: String, val glyph: String, val value: Double, val rarity: Int)
data class ReelResult(val reward: Double, val synergies: List<String>, val removed: Set<Int>)

/** Native 3x3 reel/deck game. Authored symbol values, orthogonal synergies and base economy. */
class ProbabilityModel(private val random: Random = Random.Default) {
  var credits = 120.0
  var earned = 0.0
  var spins = 0
  var luck = 0
  var motor = 0
  var auto = false
  var badLuck = 0
  var jackpotReady = false
  var deck = START_DECK.toMutableList()
  var grid = List(9) { "coin" }
  var result = ReelResult(0.0, emptyList(), emptySet())
  var cooldown = 0.0
  var inventory = mutableMapOf("coal" to 3, "coin" to 3, "sheep" to 2, "sprout" to 2, "gear" to 2,
    "battery" to 1, "wolf" to 1, "magnet" to 1, "farmer" to 1)
  val spinDelay get() = max(.13, 1.15 * .86.pow(motor))
  val luckCost get() = floor(60 * 2.0.pow(luck))
  val motorCost get() = floor(40 * 1.8.pow(motor))

  fun spin(): Boolean {
    if (cooldown > 0 || credits < 5 || deck.size !in 10..32) return false
    credits -= 5
    val bias = (.05 + luck * .04).coerceAtMost(.85)
    grid = List(9) {
      var symbol = deck[random.nextInt(deck.size)]
      repeat(1 + floor(bias * 3).toInt()) {
        val alternate = deck[random.nextInt(deck.size)]
        if (random.nextDouble() < bias && SYMBOLS.getValue(alternate).rarity > SYMBOLS.getValue(symbol).rarity) symbol = alternate
      }
      symbol
    }
    val jackpot = jackpotReady
    if (jackpot) {
      val available = deck.filter { SYMBOLS.getValue(it).rarity >= 2 }.ifEmpty { listOf("diamond") }
      grid = grid.toMutableList().apply { repeat(2) { set(random.nextInt(9), available.random(random)) } }
    }
    result = evaluate(grid, (.72 + luck * .02).coerceAtMost(.98), random)
    if (jackpot) result = result.copy(reward = result.reward * 2.6 + 50)
    credits = (credits + result.reward).coerceAtMost(1e15)
    earned = (earned + result.reward).coerceAtMost(1e15)
    spins = (spins + 1).coerceAtMost(Int.MAX_VALUE - 1)
    badLuck = if (result.reward >= 36) 0 else badLuck + 1
    jackpotReady = badLuck >= 10
    if (jackpotReady) badLuck = 0
    cooldown = spinDelay
    return true
  }

  fun step(dt: Double) { if (dt.isFinite()) { cooldown = max(0.0, cooldown - dt.coerceIn(0.0, .1)); if (auto && cooldown <= 0) spin() } }
  fun buyUpgrade(isLuck: Boolean): Boolean {
    val cost = if (isLuck) luckCost else motorCost
    if (credits < cost || (if (isLuck) luck else motor) >= 20) return false
    credits -= cost; if (isLuck) luck++ else motor++; return true
  }
  fun buyPack(): List<String> {
    if (credits < 40) return emptyList()
    credits -= 40
    return List(3) {
      val roll = random.nextDouble()
      val rarity = when { roll < .64 -> 0; roll < .93 -> 1; roll < .99 -> 2; else -> 3 }
      SYMBOLS.values.filter { it.rarity == rarity }.random(random).id.also { inventory[it] = (inventory[it] ?: 0) + 1 }
    }
  }
  fun addSymbol(id: String): Boolean {
    if (id !in SYMBOLS || deck.size >= 32 || (inventory[id] ?: 0) <= 0) return false
    inventory[id] = inventory.getValue(id) - 1; deck.add(id); return true
  }
  fun removeSymbol(id: String): Boolean {
    if (deck.size <= 10 || !deck.remove(id)) return false
    inventory[id] = (inventory[id] ?: 0) + 1; return true
  }

  companion object {
    val START_DECK = listOf("coal", "coal", "coin", "coin", "sheep", "sprout", "gear", "gear", "wolf", "battery")
    val SYMBOLS = listOf(
      ReelSymbol("coal", "Coal", "CL", 2.0, 0), ReelSymbol("coin", "Coin", "CN", 3.0, 0),
      ReelSymbol("sheep", "Sheep", "SP", 4.0, 0), ReelSymbol("sprout", "Sprout", "SR", 2.0, 0),
      ReelSymbol("gear", "Gear", "GR", 5.0, 0), ReelSymbol("battery", "Battery", "BT", 6.0, 1),
      ReelSymbol("wolf", "Wolf", "WF", 7.0, 1), ReelSymbol("magnet", "Magnet", "MG", 8.0, 1),
      ReelSymbol("farmer", "Farmer", "FM", 7.0, 1), ReelSymbol("bomb", "Bomb", "BM", 5.0, 1),
      ReelSymbol("diamond", "Diamond", "DM", 18.0, 2), ReelSymbol("reactor", "Reactor", "RC", 16.0, 2),
      ReelSymbol("chronos", "Chronos", "CH", 14.0, 2), ReelSymbol("oracle", "Oracle", "OR", 24.0, 3),
      ReelSymbol("singularity", "Singularity", "SG", 30.0, 3)).associateBy { it.id }

    fun neighbors(index: Int): List<Int> = buildList {
      require(index in 0..8)
      if (index >= 3) add(index - 3)
      if (index < 6) add(index + 3)
      if (index % 3 > 0) add(index - 1)
      if (index % 3 < 2) add(index + 1)
    }

    fun evaluate(grid: List<String>, chance: Double, random: Random): ReelResult {
      require(grid.size == 9 && grid.all { it in SYMBOLS })
      val removed = mutableSetOf<Int>(); val synergies = mutableListOf<String>(); var bonus = 0.0
      fun triggers(multiplier: Double) = random.nextDouble() < (chance * multiplier).coerceIn(0.0, 1.0)
      grid.indices.forEach { i ->
        if (i in removed) return@forEach
        val id = grid[i]
        if (id == "bomb") {
          val targets = neighbors(i).filter { it !in removed && SYMBOLS.getValue(grid[it]).rarity != 3 && grid[it] != "bomb" }
          if (targets.isNotEmpty() && triggers(1.06)) {
            targets.forEach { bonus += SYMBOLS.getValue(grid[it]).value * 1.5 + 3; removed.add(it); synergies += "Bomb blast" }
          }
        } else neighbors(i).forEach { n ->
          if (n !in removed) when {
            id == "wolf" && grid[n] == "sheep" && triggers(1.02) -> { removed.add(n); bonus += 12; synergies += "Wolf + Sheep" }
            id == "farmer" && grid[n] == "sprout" && triggers(1.02) -> { removed.add(n); bonus += 9; synergies += "Farmer + Sprout" }
            id == "singularity" && SYMBOLS.getValue(grid[n]).rarity == 0 && triggers(1.1) -> { removed.add(n); bonus += 20; synergies += "Singularity" }
          }
        }
      }
      var payout = 0.0
      grid.indices.filter { it !in removed }.forEach { i ->
        var multiplier = 1.0
        neighbors(i).filter { it !in removed }.forEach { n ->
          val pair = grid[i] to grid[n]
          val effect = when (pair) {
            "coal" to "gear" -> .35 to .98
            "coin" to "magnet" -> .45 to .98
            "diamond" to "magnet" -> .7 to 1.03
            "reactor" to "battery" -> .6 to 1.05
            "sheep" to "farmer" -> .32 to .96
            else -> if (grid[i] == "oracle" && SYMBOLS.getValue(grid[n]).rarity >= 2) .25 to .95
              else if (grid[i] == "chronos" && grid[n] in listOf("magnet", "reactor", "oracle")) .28 to .95 else 0.0 to 0.0
          }
          if (effect.first > 0 && triggers(effect.second)) { multiplier += effect.first; synergies += "${SYMBOLS.getValue(grid[i]).name} + ${SYMBOLS.getValue(grid[n]).name}" }
        }
        payout += SYMBOLS.getValue(grid[i]).value * multiplier
      }
      return ReelResult((payout + bonus) * (1 + synergies.size / 9.0 * .95), synergies, removed)
    }
  }
}
