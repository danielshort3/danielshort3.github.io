package me.danielshort.app.ui

import androidx.compose.ui.unit.dp
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidLayoutDensityTest {
  @Test fun fullNavigationLabelsRemainAvailableAtLargeTextSizes() {
    assertFalse(useExpandedBottomNavigation(320.dp, 1f))
    assertTrue(useExpandedBottomNavigation(320.dp, 1.6f))
    assertTrue(useExpandedBottomNavigation(390.dp, 1.6f))
    assertFalse(useExpandedBottomNavigation(600.dp, 1.6f))
  }

  @Test fun largeTextGetsFullWidthCardDescriptions() {
    assertFalse(useSpaciousCardText(280.dp, 1f))
    assertTrue(useSpaciousCardText(280.dp, 1.6f))
    assertFalse(useSpaciousCardText(620.dp, 1.6f))
  }
}
