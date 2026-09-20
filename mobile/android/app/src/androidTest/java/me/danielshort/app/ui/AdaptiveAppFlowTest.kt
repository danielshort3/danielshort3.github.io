package me.danielshort.app.ui

import android.content.Context
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.ime
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.focus.FocusManager
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.test.core.app.ApplicationProvider
import me.danielshort.app.data.AppSettings
import me.danielshort.app.data.AppSettingsStore
import me.danielshort.app.data.ContentRepository
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test

@OptIn(ExperimentalTestApi::class)
class AdaptiveAppFlowTest {
  @get:Rule val compose = createComposeRule()
  private val window = mutableStateOf(DpSize(600.dp, 900.dp))
  private lateinit var settings: AppSettingsStore
  private lateinit var previousSettings: AppSettings
  private lateinit var repository: ContentRepository
  private var previousSaved = emptySet<String>()
  private var safeToInstall = false
  private lateinit var focusManager: FocusManager
  @Volatile private var keyboardVisible = false

  @Before fun prepare() {
    val context = ApplicationProvider.getApplicationContext<Context>()
    settings = AppSettingsStore(context)
    previousSettings = settings.state.value
    settings.update(previousSettings.copy(automaticUpdates = false))
    repository = ContentRepository(context, settings)
    previousSaved = repository.favorites.value
  }

  @After fun restore() {
    (repository.favorites.value union previousSaved).filter {
      (it in repository.favorites.value) != (it in previousSaved)
    }.forEach(repository::toggleSaved)
    settings.update(previousSettings)
  }

  private fun openApp() {
    compose.setContent {
      DeviceConfigurationOverride(DeviceConfigurationOverride.ForcedSize(window.value)) {
        focusManager = LocalFocusManager.current
        val imeVisible = WindowInsets.ime.getBottom(LocalDensity.current) > 0
        SideEffect { keyboardVisible = imeVisible }
        DanielShortApp(repository, onSafeToInstall = { safeToInstall = it }) { openSettings ->
          TextButton(onClick = openSettings) { Text("Review update") }
        }
      }
    }
    compose.waitUntil(10_000) { repository.state.value.content != null }
  }

  private fun finishEditing() {
    compose.runOnIdle { focusManager.clearFocus(force = true) }
    compose.waitUntil(5_000) { !keyboardVisible }
  }

  @Test fun queriesOpenProjectAndBookmarkSurviveTabletAndSplitWindowChanges() {
    openApp()
    val project = repository.state.value.content!!.projects.first()
    compose.onNodeWithText("Projects").performClick()
    compose.onNodeWithTag("project-search").performTextInput(project.title)
    // ForcedSize changes this Compose subtree's density; Android's real phone keyboard
    // keeps its physical size. Finish editing before the synthetic tablet window resize.
    finishEditing()
    compose.runOnIdle { window.value = DpSize(1280.dp, 800.dp) }
    compose.onNodeWithTag("site-rail-projects").assertIsSelected()
    compose.onNodeWithTag("project-search").assertTextContains(project.title)
    compose.onNode(hasText(project.title) and !hasSetTextAction()).performScrollTo().assertIsDisplayed().performClick()
    compose.onNodeWithContentDescription("Back to projects").assertIsDisplayed()
    compose.runOnIdle { assertFalse("Project details must prevent automatic installation", safeToInstall) }
    if (project.id !in repository.favorites.value) compose.onNodeWithContentDescription("Save project").performClick()
    compose.runOnIdle { window.value = DpSize(600.dp, 900.dp) }
    compose.onNodeWithContentDescription("Unsave project").assertIsDisplayed()
    compose.onNodeWithContentDescription("Back to projects").performClick()
    compose.onNodeWithTag("project-search").assertTextContains(project.title)
    compose.runOnIdle { assertTrue(safeToInstall); window.value = DpSize(1280.dp, 800.dp) }
    compose.onNodeWithTag("site-rail-tools").performClick()
    compose.onNodeWithTag("catalog-search").performTextInput("qr")
    finishEditing()
    compose.onNodeWithTag("site-rail-projects").performClick()
    compose.onNodeWithTag("project-search").assertTextContains(project.title)
    compose.onNodeWithTag("site-rail-tools").performClick()
    compose.onNodeWithTag("catalog-search").assertTextContains("qr")
  }

  @Test fun updateNoticeOpensConstrainedSettingsAndRailsRemainAvailable() {
    window.value = DpSize(1600.dp, 900.dp)
    openApp()
    compose.runOnIdle { assertTrue(safeToInstall) }
    compose.onNodeWithText("Review update").performClick()
    compose.runOnIdle { assertFalse(safeToInstall) }
    val panel = compose.onNodeWithTag("native-feature-panel").getUnclippedBoundsInRoot()
    assertTrue(panel.right - panel.left <= 760.dp)
    Section.entries.forEach { compose.onNodeWithTag("site-rail-${it.name.lowercase()}").assertIsDisplayed() }
    compose.onNodeWithTag("site-rail-contact").performClick().assertIsSelected()
    compose.onNodeWithText("Say hello").assertIsDisplayed()
    compose.runOnIdle { assertTrue(safeToInstall) }
  }
}
