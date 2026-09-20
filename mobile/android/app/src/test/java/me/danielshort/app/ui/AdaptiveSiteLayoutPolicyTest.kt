package me.danielshort.app.ui

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.roundToInt

class AdaptiveSiteLayoutPolicyTest {
  @Test fun windowWidthDeterminesNavigationIncludingSplitWindows() {
    listOf(.75f, 1f, 1.28125f, 1.33125f, 2.625f).forEach { density ->
      listOf(360, 412, 600, 800, 839).forEach { assertFalse(usesWideSiteLayout((it * density).roundToInt(), density)) }
      listOf(840, 960, 1280, 1600).forEach { assertTrue(usesWideSiteLayout((it * density).roundToInt(), density)) }
    }
  }

  @Test fun pixelRoundingDoesNotMisclassifyAnExactly840DpWindow() {
    // Compose measures 840dp as 1076px at this valid Android density. Dividing that width
    // by density gives 839.805dp, so a floating-point dp comparison would choose compact.
    assertTrue(usesWideSiteLayout(1076, 1.28125f))
    assertFalse(usesWideSiteLayout(1075, 1.28125f))
  }

  @Test fun smallestWideWindowRetainsUsableContentAndTouchTargets() {
    val contentWidth = WIDE_SITE_MIN_WIDTH_DP - 24 - SITE_RAIL_WIDTH_DP * 5 - 6
    assertTrue(SITE_RAIL_WIDTH_DP >= 48)
    assertTrue(contentWidth >= 540)
  }
}
