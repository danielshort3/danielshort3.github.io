package me.danielshort.app.ui

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.StateRestorationTester
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.test.core.app.ApplicationProvider
import androidx.test.espresso.Espresso.pressBack
import kotlinx.coroutines.runBlocking
import me.danielshort.app.BuildConfig
import me.danielshort.app.data.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import java.io.File

@OptIn(ExperimentalTestApi::class)
class SettingsFlowTest {
  @get:Rule val compose = createComposeRule()
  private lateinit var context: Context
  private lateinit var settings: AppSettingsStore
  private lateinit var original: AppSettings

  @Before fun prepare() {
    context = ApplicationProvider.getApplicationContext()
    settings = AppSettingsStore(context)
    original = settings.state.value
    // The fixture is offline: no check, download, or installation is requested.
    settings.update(AppSettings(automaticUpdates = false, checkAppUpdatesOnLaunch = false))
  }

  @After fun restore() { settings.update(original) }

  @Test fun settingsPersistAcrossStoreRecreation() {
    val chosen = AppSettings(automaticUpdates = false, unmeteredOnly = true, reduceMotion = true,
      checkAppUpdatesOnLaunch = false, automaticAppUpdates = true, appUpdatesUnmeteredOnly = false)
    settings.update(chosen)
    assertEquals(chosen, AppSettingsStore(context).state.value)
  }

  @Test fun existingInstallGetsLaunchChecksWithoutEnablingAutomaticAppDownloads() {
    context.getSharedPreferences("native-settings", Context.MODE_PRIVATE).edit()
      .remove("check-app-updates-on-launch").remove("automatic-app-updates").remove("app-updates-unmetered-only").commit()
    val migrated = AppSettingsStore(context).state.value
    assertTrue(migrated.checkAppUpdatesOnLaunch)
    assertFalse(migrated.automaticAppUpdates)
    assertTrue(migrated.appUpdatesUnmeteredOnly)
    assertFalse(migrated.automaticUpdates)
  }

  @Test fun nativeModesPersistAndRememberHiddenNetworkChoiceIndependentlyOfContent() {
    val options = mutableStateOf(settings.state.value)
    compose.setContent {
      MaterialTheme {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
          AppUpdatePreferences(options.value) { options.value = it; settings.update(it) }
        }
      }
    }
    compose.onNodeWithTag("app-update-network").assertDoesNotExist()
    compose.onNodeWithTag("app-update-mode").performClick()
    compose.onNodeWithTag("choice-AUTOMATIC").performScrollTo().performClick()
    compose.onNodeWithTag("app-update-network").performScrollTo().performClick()
    compose.onNodeWithTag("choice-false").performClick()
    val restored = AppSettingsStore(context).state.value
    assertTrue(restored.automaticAppUpdates)
    assertTrue(restored.checkAppUpdatesOnLaunch)
    assertFalse(restored.appUpdatesUnmeteredOnly)
    assertFalse(restored.automaticUpdates)
    compose.onNodeWithTag("app-update-mode").performScrollTo().performClick()
    compose.onNodeWithTag("choice-MANUAL").performScrollTo().performClick()
    compose.onNodeWithTag("app-update-network").assertDoesNotExist()
    val optedOut = AppSettingsStore(context).state.value
    assertFalse(optedOut.automaticAppUpdates)
    assertFalse(optedOut.checkAppUpdatesOnLaunch)
    assertFalse(optedOut.appUpdatesUnmeteredOnly)
    compose.onNodeWithTag("app-update-mode").performClick()
    compose.onNodeWithTag("choice-AUTOMATIC").performScrollTo().performClick()
    assertFalse(AppSettingsStore(context).state.value.appUpdatesUnmeteredOnly)
  }

  @Test fun overviewHasFourRowsAndOneAccessibleImmediateSwitch() {
    val repository = ContentRepository(context, settings)
    compose.setContent { MaterialTheme { SettingsScreen(repository, onBack = {}) } }
    listOf("settings-updates", "reduce-motion", "settings-storage", "settings-information").forEach {
      compose.onNodeWithTag(it).assertExists().assertHeightIsAtLeast(48.dp)
    }
    compose.onNodeWithTag("reduce-motion").assert(SemanticsMatcher.expectValue(SemanticsProperties.Role, Role.Switch))
      .assertIsOff().performClick().assertIsOn()
    assertTrue(AppSettingsStore(context).state.value.reduceMotion)
    compose.onNodeWithText("Clear cached images").assertDoesNotExist()
    compose.onNodeWithText("Remove all saved projects").assertDoesNotExist()
    compose.onNodeWithText("Check now").assertDoesNotExist()
    compose.onNodeWithText("Version ${BuildConfig.VERSION_NAME}").assertDoesNotExist()
    capture("overview")
  }

  @Test fun contentControlsKeepManualRefreshAvailableAndDoNotChangeNativeModes() {
    val repository = ContentRepository(context, settings)
    val previousCheck = context.getSharedPreferences("native-content", Context.MODE_PRIVATE).getLong("last-checked", 0L)
    assertTrue(runBlocking { repository.refresh() })
    assertFalse(repository.state.value.refreshing)
    assertEquals(previousCheck, context.getSharedPreferences("native-content", Context.MODE_PRIVATE).getLong("last-checked", 0L))
    compose.setContent { MaterialTheme { SettingsScreen(repository, onBack = {}) } }
    compose.onNodeWithTag("settings-updates").performClick()
    compose.onNodeWithTag("refresh-content").performScrollTo().assertIsEnabled()
    compose.onNodeWithTag("content-refresh-mode").performScrollTo().performClick()
    compose.onNodeWithTag("choice-UNMETERED_ONLY").performScrollTo().performClick()
    val changed = AppSettingsStore(context).state.value
    assertTrue(changed.automaticUpdates)
    assertTrue(changed.unmeteredOnly)
    assertFalse(changed.checkAppUpdatesOnLaunch)
    assertFalse(changed.automaticAppUpdates)
    compose.onNodeWithTag("content-refresh-mode").performScrollTo().performClick()
    compose.onNodeWithTag("choice-MANUAL").performScrollTo().performClick()
    assertFalse(AppSettingsStore(context).state.value.automaticUpdates)
    assertTrue(AppSettingsStore(context).state.value.unmeteredOnly)
  }

  @Test fun pageAndOpenChoiceSurviveSavedStateRestorationWithoutWritingPreferences() {
    val repository = ContentRepository(context, settings)
    val restoration = StateRestorationTester(compose)
    restoration.setContent { MaterialTheme { SettingsScreen(repository, onBack = {}) } }
    compose.onNodeWithTag("settings-updates").performClick()
    compose.onNodeWithTag("app-update-mode").performScrollTo().performClick()
    val before = AppSettingsStore(context).state.value
    restoration.emulateSavedInstanceStateRestore()
    compose.onNodeWithTag("choice-MANUAL").assertIsSelected()
    compose.onNodeWithText("Cancel").performClick()
    compose.onNodeWithTag("app-update-mode").assertExists()
    assertEquals(before, AppSettingsStore(context).state.value)
    compose.onNodeWithTag("settings-back").performClick()
    compose.onNodeWithTag("settings-updates").assertIsDisplayed()
  }

  @Test fun largeTextNarrowScreenCanReachEveryDestinationAndKeepsBackVisible() {
    val repository = ContentRepository(context, settings)
    compose.setContent {
      DeviceConfigurationOverride(DeviceConfigurationOverride.ForcedSize(DpSize(320.dp, 640.dp))) {
        DeviceConfigurationOverride(DeviceConfigurationOverride.FontScale(2f)) {
          MaterialTheme { SettingsScreen(repository, onBack = {}) }
        }
      }
    }
    compose.onNodeWithTag("settings-information").performScrollTo().assertIsDisplayed().performClick()
    compose.onNodeWithText("Content and app updates").performScrollTo().assertIsDisplayed()
    compose.onNodeWithTag("settings-back").assertIsDisplayed().performClick()
    compose.onNodeWithTag("settings-updates").performScrollTo().performClick()
    compose.onNodeWithTag("content-refresh-mode").performScrollTo().performClick()
    compose.onNodeWithTag("choice-UNMETERED_ONLY").performScrollTo().assertIsDisplayed()
    compose.onNodeWithTag("choice-MANUAL").performScrollTo().performClick()
    compose.onNodeWithTag("refresh-content").performScrollTo().assertIsDisplayed()
    compose.onNodeWithTag("settings-back").assertIsDisplayed()
    capture("updates-large-text-320")
  }

  @Test fun wideSettingsStayBoundedAndBackReturnsToTheOriginalSection() {
    val repository = ContentRepository(context, settings)
    compose.waitUntil(10_000) { repository.state.value.content != null }
    compose.setContent {
      DeviceConfigurationOverride(DeviceConfigurationOverride.ForcedSize(DpSize(1280.dp, 800.dp))) {
        DanielShortApp(repository)
      }
    }
    compose.onNodeWithTag("site-rail-tools").performClick()
    compose.onNodeWithContentDescription("Settings").performClick()
    compose.onNodeWithTag("native-feature-panel").assertWidthIsAtMost(760.dp)
    compose.onNodeWithTag("settings-updates").assertIsDisplayed()
    capture("overview-wide")
    compose.onNodeWithTag("settings-storage").performClick()
    compose.onNodeWithTag("clear-cached-images").assertIsDisplayed()
    pressBack()
    compose.onNodeWithTag("settings-storage").assertIsDisplayed()
    pressBack()
    compose.onNodeWithTag("site-rail-tools").assertIsSelected()
    compose.onNodeWithTag("settings-body").assertDoesNotExist()
  }

  @Test fun updateNoticeOpensUpdatesDirectlyAndDoesNotMakeInstallationSafe() {
    val repository = ContentRepository(context, settings)
    var safeToInstall = false
    compose.setContent {
      DanielShortApp(repository, onSafeToInstall = { safeToInstall = it }, globalNotice = { openUpdates ->
        TextButton(onClick = openUpdates) { Text("Test update notice") }
      })
    }
    compose.onNodeWithText("Test update notice").performClick()
    compose.onNodeWithText("App updates").assertIsDisplayed()
    compose.onNodeWithTag("settings-updates").assertDoesNotExist()
    compose.runOnIdle { assertFalse(safeToInstall) }
    capture("updates")
    compose.onNodeWithTag("settings-back").performClick()
    compose.onNodeWithTag("settings-updates").assertIsDisplayed()
    compose.onNodeWithTag("settings-back").performClick()
    compose.onNodeWithText("Test update notice").assertIsDisplayed()
  }

  @Test fun settingsUseOneAccentRegardlessOfTheCallingSection() {
    val incoming = mutableStateOf(Color(0xFFC94B0A))
    var actual = Color.Unspecified
    compose.setContent {
      MaterialTheme(colorScheme = lightColorScheme(primary = incoming.value)) {
        SettingsTheme {
          val current = MaterialTheme.colorScheme.primary
          SideEffect { actual = current }
          Text("Settings color fixture")
        }
      }
    }
    compose.runOnIdle { assertEquals(Color(0xFF155DFC), actual); incoming.value = Color(0xFF087F8C) }
    compose.runOnIdle { assertEquals(Color(0xFF155DFC), actual) }
  }

  private fun capture(name: String) {
    val image = compose.onRoot().captureToImage().asAndroidBitmap()
    val directory = File(context.getExternalFilesDir(null), "settings-qa").apply { mkdirs() }
    File(directory, "$name.png").outputStream().use { assertTrue(image.compress(Bitmap.CompressFormat.PNG, 100, it)) }
  }
}
