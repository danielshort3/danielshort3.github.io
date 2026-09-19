package me.danielshort.app.nativefeatures

import kotlin.random.Random

enum class RouletteColor { Red, Black, Green }
enum class RouletteBet(val label: String) { Red("Red"), Black("Black"), Odd("Odd"), Even("Even") }

data class RoulettePocket(val number: Int) {
  init { require(number in 0..37) }
  val label: String get() = if (number == 37) "00" else number.toString()
  val color: RouletteColor get() = when {
    number == 0 || number == 37 -> RouletteColor.Green
    number in RED_NUMBERS -> RouletteColor.Red
    else -> RouletteColor.Black
  }
  fun wins(bet: RouletteBet): Boolean {
    if (color == RouletteColor.Green) return false
    return when (bet) {
      RouletteBet.Red -> color == RouletteColor.Red
      RouletteBet.Black -> color == RouletteColor.Black
      RouletteBet.Odd -> number % 2 == 1
      RouletteBet.Even -> number % 2 == 0
    }
  }
  companion object {
    private val RED_NUMBERS = setOf(1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36)
    fun spin(random: Random = Random.Default): RoulettePocket = RoulettePocket(random.nextInt(38))
  }
}

data class RouletteRound(val pocket: RoulettePocket, val bet: RouletteBet) {
  val won: Boolean get() = pocket.wins(bet)
}

data class RouletteGame(
  val chips: Int = 20,
  val spins: Int = 0,
  val wins: Int = 0,
  val recent: List<RouletteRound> = emptyList()
) {
  fun play(bet: RouletteBet, pocket: RoulettePocket = RoulettePocket.spin()): RouletteGame {
    if (chips == 0) return this
    val round = RouletteRound(pocket, bet)
    return copy(
      chips = chips + if (round.won) 1 else -1,
      spins = spins + 1,
      wins = wins + if (round.won) 1 else 0,
      recent = (listOf(round) + recent).take(8)
    )
  }
}
