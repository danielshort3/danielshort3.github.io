package me.danielshort.app.nativefeatures

import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTextReplacement
import androidx.compose.ui.text.AnnotatedString
import org.junit.Rule
import org.junit.Test

class NativeFeatureFlowTest {
  @get:Rule val compose = createComposeRule()

  private fun openCompare() {
    compose.setContent { MaterialTheme { NativeTextCompareScreen(onBack = {}) } }
  }

  private fun compareAndWait(heading: String) {
    compose.onNodeWithText("Compare", substring = false).performScrollTo().performClick()
    compose.waitUntil(5_000) {
      compose.onAllNodesWithText(heading, substring = false).fetchSemanticsNodes().isNotEmpty()
    }
  }

  @Test fun blankInputsCompareTheExampleAndClearRestoresTheEmptyEditors() {
    openCompare()
    compose.onNodeWithText("Clear").assertIsNotEnabled()
    compareAndWait("Example comparison")
    compose.onNodeWithText("Copy After", substring = true).performScrollTo().assertExists()
    compose.onNodeWithText("Clear").performScrollTo().performClick()
    compose.onNodeWithText("Example comparison").assertDoesNotExist()
    compose.onNodeWithText("Clear").assertIsNotEnabled()
    val emptyEditor = SemanticsMatcher.expectValue(SemanticsProperties.EditableText, AnnotatedString(""))
    compose.onNode(hasSetTextAction() and hasText("Before")).assert(emptyEditor)
    compose.onNode(hasSetTextAction() and hasText("After")).assert(emptyEditor)
    compareAndWait("Example comparison")
    compose.onNodeWithContentDescription("Removed: Small.", substring = true).assertExists()
    compose.onNodeWithContentDescription("Added: Thoughtful.", substring = true).assertExists()
  }

  @Test fun beforeOnlyComparesAgainstAnEmptyAfterWithoutInjectingTheExample() {
    openCompare()
    compose.onNode(hasSetTextAction() and hasText("Before")).performTextInput("Own source")
    compose.onNodeWithText(EXAMPLE_AFTER).assertDoesNotExist()
    compareAndWait("Comparison")
    compose.onNodeWithContentDescription("Removed: Own source.").assertExists()
    compose.onNodeWithText("Example comparison").assertDoesNotExist()
  }

  @Test fun afterOnlyIsAnAdditionAndEditingInvalidatesTheOldComparison() {
    openCompare()
    val afterField = compose.onNode(hasSetTextAction() and hasText("After"))
    afterField.performTextInput("My new text")
    compareAndWait("Comparison")
    compose.onNodeWithContentDescription("Added: My new text.").assertExists()
    afterField.performScrollTo().performTextReplacement("Final text")
    compose.onNodeWithText("Comparison", substring = false).assertDoesNotExist()
    compose.onNodeWithContentDescription("Added: My new text.").assertDoesNotExist()
    compareAndWait("Comparison")
    compose.onNodeWithContentDescription("Added: Final text.").assertExists()
  }

  @Test fun unchangedInputsShowNoDifferencesAndRemainComparableAfterEditing() {
    openCompare()
    compose.onNode(hasSetTextAction() and hasText("Before")).performTextInput("Same text")
    compose.onNode(hasSetTextAction() and hasText("After")).performTextInput("Same text")
    compareAndWait("No differences")
    compose.onNodeWithText("Compare", substring = false).performScrollTo().performClick()
    compose.onNodeWithText("No differences").assertExists()
    compose.onNode(hasSetTextAction() and hasText("After")).performScrollTo().performTextReplacement("New text")
    compareAndWait("Comparison")
    compose.onNodeWithText("No differences").assertDoesNotExist()
  }

  @Test fun rouletteCanPlayAndStartAFreshGame() {
    compose.setContent { MaterialTheme { NativeGameScreen(onBack = {}) } }
    compose.onNodeWithText("Black", substring = false).performClick()
    compose.onNodeWithText("Spin · 1 chip").performScrollTo().performClick()
    compose.onNodeWithText("Recent spins").assertExists()
    compose.onNodeWithText("Choose a bet, then spin.").assertDoesNotExist()
    compose.onNodeWithText("New game").performScrollTo().performClick()
    compose.onNodeWithText("Choose a bet, then spin.").assertExists()
    compose.onNodeWithText("Recent spins").assertDoesNotExist()
    compose.onNodeWithText("20", substring = false).assertExists()
  }
}
