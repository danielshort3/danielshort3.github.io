package me.danielshort.app.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class SettingsMenusTest {
  @get:Rule val compose = createComposeRule()

  @Test fun savedProjectsRequireExplicitConfirmationAndCancelDoesNothing() {
    var removals = 0
    compose.setContent { MaterialTheme { SavedProjectsMenu(3) { removals++ } } }
    compose.onNodeWithText("Remove all saved projects").assertDoesNotExist()
    compose.onNodeWithContentDescription("Saved project options").performClick()
    compose.onNodeWithText("Remove all saved projects").performClick()
    compose.onNodeWithText("This removes your 3 bookmarks. Project content, game progress, and generated files are kept.").assertIsDisplayed()
    assertEquals(0, removals)
    compose.onNodeWithText("Cancel").performClick()
    assertEquals(0, removals)
    compose.onNodeWithContentDescription("Saved project options").performClick()
    compose.onNodeWithText("Remove all saved projects").performClick()
    compose.onNodeWithTag("confirm-remove-saved").performClick()
    assertEquals(1, removals)
  }

  @Test fun noSavedProjectsCannotTriggerBulkRemoval() {
    var removals = 0
    compose.setContent { MaterialTheme { SavedProjectsMenu(0) { removals++ } } }
    compose.onNodeWithContentDescription("Saved project options").assertIsNotEnabled()
    compose.onNodeWithTag("confirm-remove-saved").assertDoesNotExist()
    assertEquals(0, removals)
  }

  @Test fun refreshLivesInOverflowAndCannotBeRepeatedWhileBusy() {
    val refreshing = mutableStateOf(false)
    var refreshes = 0
    compose.setContent { MaterialTheme { BrowseOverflowMenu(refreshing.value) { refreshes++ } } }
    compose.onNodeWithText("Refresh website content").assertDoesNotExist()
    compose.onNodeWithContentDescription("More options").assertHasClickAction().performClick()
    compose.onNodeWithText("Refresh website content").assertIsEnabled().performClick()
    assertEquals(1, refreshes)
    compose.onNodeWithText("Refresh website content").assertDoesNotExist()
    compose.runOnIdle { refreshing.value = true }
    compose.onNodeWithContentDescription("More options").performClick()
    compose.onNodeWithText("Refreshing…").assertIsNotEnabled()
    assertEquals(1, refreshes)
  }
}
