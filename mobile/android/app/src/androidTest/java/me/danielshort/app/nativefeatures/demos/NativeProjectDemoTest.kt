package me.danielshort.app.nativefeatures.demos

import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTouchInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeProjectDemoTest {
  @get:Rule val compose = createComposeRule()

  @Test fun shapeDrawingEnablesClassificationAndClearDisablesIt() {
    compose.setContent { MaterialTheme { NativeProjectDemoScreen("shapeClassifier", {}) } }
    compose.onNodeWithText("Classify shape").assertIsNotEnabled()
    compose.onNodeWithContentDescription("Black shape drawing canvas").performTouchInput {
      down(center + Offset(-50f, 0f))
      moveTo(center + Offset(50f, 0f), 250)
      up()
    }
    compose.onNodeWithText("Classify shape").assertIsEnabled()
    compose.onNodeWithText("Clear").performScrollTo().performClick()
    compose.onNodeWithText("Classify shape").assertIsNotEnabled()
  }

  @Test fun ufoYearFilterRecalculatesActualPublishedCounts() {
    compose.setContent { MaterialTheme { NativeProjectDemoScreen("ufoDashboard", {}) } }
    compose.waitUntil(15_000) { compose.onAllNodesWithText("6,334 reported sightings").fetchSemanticsNodes().isNotEmpty() }
    compose.onNodeWithText("6,334 reported sightings").assertExists()
    compose.onNodeWithText("Year: 2013").performScrollTo().performClick()
    compose.onNode(hasText("All") and hasClickAction()).performClick()
    compose.onNodeWithText("68,859 reported sightings").assertExists()
  }

  @Test fun pizzaCityFilterRecalculatesActualPublishedDeliveryCount() {
    compose.setContent { MaterialTheme { NativeProjectDemoScreen("pizzaDashboard", {}) } }
    compose.waitUntil(15_000) { compose.onAllNodesWithText("1251 deliveries").fetchSemanticsNodes().isNotEmpty() }
    compose.onNodeWithText("City: All").performScrollTo().performClick()
    compose.onNode(hasText("Frisco") and hasClickAction()).performClick()
    compose.onNodeWithText("794 deliveries").assertExists()
  }
}
