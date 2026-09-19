package me.danielshort.app.nativefeatures

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class RouletteGameTest {
  @Test fun doubleZeroWheelHasCorrectColorAndBetDistribution() {
    val pockets = (0..37).map(::RoulettePocket)
    assertEquals(18, pockets.count { it.color == RouletteColor.Red })
    assertEquals(18, pockets.count { it.color == RouletteColor.Black })
    assertEquals(2, pockets.count { it.color == RouletteColor.Green })
    RouletteBet.entries.forEach { bet -> assertEquals(18, pockets.count { it.wins(bet) }) }
    assertEquals("00", RoulettePocket(37).label)
  }

  @Test fun greenPocketsLoseEveryOutsideBet() {
    RouletteBet.entries.forEach {
      assertFalse(RoulettePocket(0).wins(it))
      assertFalse(RoulettePocket(37).wins(it))
    }
  }

  @Test fun wagerPaysOneToOneAndRetainsOriginalBet() {
    val won = RouletteGame().play(RouletteBet.Red, RoulettePocket(1))
    assertEquals(21, won.chips)
    assertEquals(1, won.wins)
    assertTrue(won.recent.first().won)
    val lost = won.play(RouletteBet.Even, RoulettePocket(37))
    assertEquals(20, lost.chips)
    assertEquals(2, lost.spins)
    assertEquals(RouletteBet.Red, lost.recent[1].bet)
  }

  @Test fun bankruptGameCannotSpinAndHistoryIsBounded() {
    val empty = RouletteGame(chips = 0)
    assertSame(empty, empty.play(RouletteBet.Red, RoulettePocket(1)))
    var game = RouletteGame()
    repeat(20) { game = game.play(RouletteBet.Red, RoulettePocket(1)) }
    assertEquals(8, game.recent.size)
    assertEquals(20, game.spins)
  }

  @Test fun losingLastChipStopsFurtherPlayWithoutChangingRecordedResults() {
    val game = RouletteGame(chips = 1).play(RouletteBet.Even, RoulettePocket(0))
    assertEquals(0, game.chips)
    assertEquals(1, game.spins)
    assertEquals(0, game.wins)
    assertSame(game, game.play(RouletteBet.Even, RoulettePocket(2)))
  }

  @Test fun everyPocketAndBetHasTheExpectedBalanceChange() {
    for (number in 0..37) for (bet in RouletteBet.entries) {
      val pocket = RoulettePocket(number)
      val game = RouletteGame().play(bet, pocket)
      val expectedWin = when (bet) {
        RouletteBet.Red -> number in setOf(1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36)
        RouletteBet.Black -> number in setOf(2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35)
        RouletteBet.Odd -> number in 1..36 && number % 2 == 1
        RouletteBet.Even -> number in 1..36 && number % 2 == 0
      }
      assertEquals(if (expectedWin) 21 else 19, game.chips)
      assertEquals(if (expectedWin) 1 else 0, game.wins)
    }
  }
}
