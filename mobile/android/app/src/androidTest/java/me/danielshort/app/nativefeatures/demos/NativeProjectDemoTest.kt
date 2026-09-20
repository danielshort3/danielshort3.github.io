package me.danielshort.app.nativefeatures.demos

import androidx.compose.material3.MaterialTheme
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.ime
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.SoftwareKeyboardController
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsFocused
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTextInputSelection
import androidx.compose.ui.test.swipeUp
import androidx.compose.ui.text.TextRange
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.assertEquals
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

  @Test fun handwritingKeepsCanvasAndActionCenteredWithoutHidingTheHeaderWhileDrawing() {
    compose.setContent { MaterialTheme { NativeProjectDemoScreen("handwritingRating", {}) } }
    val canvas = compose.onNodeWithContentDescription("Black digit drawing canvas")
    val initialCanvas = canvas.fetchSemanticsNode().boundsInRoot
    val action = compose.onNodeWithText("Rate digit").fetchSemanticsNode().boundsInRoot
    assertEquals(initialCanvas.center.x, action.center.x, 1f)
    canvas.performTouchInput {
      down(center + Offset(-40f, -80f))
      moveTo(center + Offset(40f, 80f), 300)
      up()
    }
    compose.onNodeWithContentDescription("Back to project").assertExists()
    assertEquals(initialCanvas, canvas.fetchSemanticsNode().boundsInRoot)
    compose.onNodeWithText("Rate digit").assertIsEnabled()
    (0..9).forEach { compose.onNode(hasText(it.toString()) and hasClickAction()).assertExists() }
  }

  @Test fun scrollingInsideALongQueryWithTheKeyboardDismissedKeepsTheHeaderVisible() {
    var keyboard: SoftwareKeyboardController? = null
    var keyboardBottom = 0
    compose.setContent {
      keyboard = LocalSoftwareKeyboardController.current
      keyboardBottom = WindowInsets.ime.getBottom(LocalDensity.current)
      MaterialTheme { NativeProjectDemoScreen("smartSentence", {}) }
    }
    val editor = compose.onNode(hasText("Search phrase") and hasSetTextAction())
    editor.performTextInput((1..60).joinToString("\n") { "Example line $it" })
    editor.performTextInputSelection(TextRange.Zero)
    compose.runOnIdle { keyboard?.hide() }
    compose.waitUntil(5_000) { keyboardBottom == 0 }
    compose.waitForIdle()
    editor.assertIsFocused()
    val originalBounds = editor.fetchSemanticsNode().boundsInRoot
    editor.performTouchInput { swipeUp(durationMillis = 500) }
    compose.waitForIdle()
    editor.assertIsFocused()
    compose.onNodeWithContentDescription("Back to project").assertExists()
    assertEquals(originalBounds, editor.fetchSemanticsNode().boundsInRoot)
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
