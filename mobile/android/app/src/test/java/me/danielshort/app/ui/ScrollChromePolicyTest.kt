package me.danielshort.app.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ScrollChromePolicyTest {
  private fun policy() = ScrollChromePolicy(hideThreshold = 48f, showThreshold = 18f)

  @Test fun startsVisibleAndHidesAtTheAccumulatedDownwardThreshold() {
    val chrome = policy()
    assertTrue(chrome.visible)
    assertTrue(chrome.onScroll(-20f))
    assertTrue(chrome.onScroll(-27f))
    assertFalse(chrome.onScroll(-1f))
    assertFalse(chrome.visible)
  }

  @Test fun hiddenChromeRequiresTheSmallerUpwardThresholdToReturn() {
    val chrome = policy()
    assertFalse(chrome.onScroll(-48f))
    assertFalse(chrome.onScroll(10f))
    assertFalse(chrome.onScroll(7f))
    assertTrue(chrome.onScroll(1f))
    assertTrue(chrome.visible)
  }

  @Test fun reversingBeforeHidingDiscardsThePreviousDownwardRun() {
    val chrome = policy()
    assertTrue(chrome.onScroll(-40f))
    assertTrue(chrome.onScroll(4f))
    assertTrue(chrome.onScroll(-8f))
    assertFalse(chrome.onScroll(-40f))
  }

  @Test fun reversingBeforeRevealingDiscardsThePreviousUpwardRun() {
    val chrome = policy()
    assertFalse(chrome.onScroll(-48f))
    assertFalse(chrome.onScroll(15f))
    assertFalse(chrome.onScroll(-1f))
    assertFalse(chrome.onScroll(3f))
    assertTrue(chrome.onScroll(15f))
  }

  @Test fun repeatedSmallDirectionChangesDoNotFlickerEitherState() {
    val chrome = policy()
    repeat(100) {
      assertTrue(chrome.onScroll(-5f))
      assertTrue(chrome.onScroll(5f))
    }
    assertFalse(chrome.onScroll(-48f))
    repeat(100) {
      assertFalse(chrome.onScroll(5f))
      assertFalse(chrome.onScroll(-5f))
    }
  }

  @Test fun zeroMovementDoesNotToggleOrDiscardProgress() {
    val chrome = policy()
    assertTrue(chrome.onScroll(-47f))
    assertTrue(chrome.onScroll(0f))
    assertTrue(chrome.onScroll(-0f))
    assertFalse(chrome.onScroll(-1f))
    assertFalse(chrome.onScroll(17f))
    assertFalse(chrome.onScroll(0f))
    assertTrue(chrome.onScroll(1f))
  }

  @Test fun disablingAlwaysRevealsAndDoesNotAccumulateHiddenScroll() {
    val chrome = policy()
    assertFalse(chrome.onScroll(-48f))
    assertTrue(chrome.onScroll(-500f, allowed = false))
    assertTrue(chrome.visible)
    assertTrue(chrome.onScroll(-500f, allowed = false))
    assertTrue(chrome.onScroll(-47f))
    assertFalse(chrome.onScroll(-1f))
  }

  @Test fun disablingResetsPartialProgressBeforeReenabling() {
    val chrome = policy()
    assertTrue(chrome.onScroll(-40f))
    assertTrue(chrome.onScroll(0f, allowed = false))
    assertTrue(chrome.onScroll(-8f))
    assertFalse(chrome.onScroll(-40f))
  }

  @Test fun revealResetsPartialMovementEvenWhenAlreadyVisible() {
    val chrome = policy()
    assertTrue(chrome.onScroll(-40f))
    chrome.reveal()
    assertTrue(chrome.visible)
    assertTrue(chrome.onScroll(-8f))
    assertFalse(chrome.onScroll(-40f))
    chrome.reveal()
    assertTrue(chrome.visible)
    assertTrue(chrome.onScroll(-47f))
    assertFalse(chrome.onScroll(-1f))
  }

  @Test fun nonfiniteMovementIsIgnoredWithoutPoisoningAccumulation() {
    val chrome = policy()
    val invalidDeltas = listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)
    assertTrue(chrome.onScroll(-47f))
    invalidDeltas.forEach { assertTrue(chrome.onScroll(it)) }
    assertFalse(chrome.onScroll(-1f))
    assertFalse(chrome.onScroll(17f))
    invalidDeltas.forEach { assertFalse(chrome.onScroll(it)) }
    assertTrue(chrome.onScroll(1f))
  }

  @Test fun disablingTakesPriorityOverInvalidMovement() {
    listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY).forEach { delta ->
      val chrome = policy()
      assertFalse(chrome.onScroll(-48f))
      assertTrue(chrome.onScroll(delta, allowed = false))
      assertTrue(chrome.visible)
      assertTrue(chrome.onScroll(-47f))
      assertFalse(chrome.onScroll(-1f))
    }
  }

  @Test fun oneLargeMovementCanToggleButDoesNotCarryOverToTheOppositeDirection() {
    val chrome = policy()
    assertFalse(chrome.onScroll(-1000f))
    assertFalse(chrome.onScroll(17f))
    assertTrue(chrome.onScroll(1f))
    assertTrue(chrome.onScroll(1000f))
    assertTrue(chrome.onScroll(-47f))
    assertFalse(chrome.onScroll(-1f))
  }
}
