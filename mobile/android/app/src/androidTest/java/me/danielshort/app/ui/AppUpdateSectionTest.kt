package me.danielshort.app.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import me.danielshort.app.updates.AppUpdateState
import me.danielshort.app.updates.UpdateOffer
import me.danielshort.app.updates.UpdateRetryAction
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class AppUpdateSectionTest {
  @get:Rule val compose = createComposeRule()
  private val offer = UpdateOffer(4, "0.4.0-debug", 1024L * 1024, true)

  @Test fun updateActionsAreExplicitAndDownloadCanBeCancelled() {
    val state = mutableStateOf<AppUpdateState>(AppUpdateState.Idle)
    val actions = mutableListOf<String>()
    compose.setContent {
      MaterialTheme {
        Column(Modifier.verticalScroll(rememberScrollState())) {
          AppUpdateSectionContent(state.value, onCheck = { actions += "check" }, onDownload = { actions += "download" }, onCancel = { actions += "cancel" }, onInstall = { actions += "install" })
        }
      }
    }
    compose.onNodeWithText("Check for updates").performClick()
    assertEquals(listOf("check"), actions)
    compose.runOnIdle { state.value = AppUpdateState.Available(offer) }
    compose.onNodeWithText("Download update").performClick()
    compose.runOnIdle { state.value = AppUpdateState.Downloading(offer, .5f, true) }
    compose.onNodeWithText("Cancel download").performClick()
    compose.runOnIdle { state.value = AppUpdateState.Ready(offer) }
    compose.onNodeWithText("Install update").performClick()
    assertEquals(listOf("check", "download", "cancel", "install"), actions)
  }

  @Test fun failedDownloadOffersRetryAndVerificationPreventsDoubleInstall() {
    val state = mutableStateOf<AppUpdateState>(AppUpdateState.Error("The download was interrupted.", UpdateRetryAction.DOWNLOAD))
    var downloads = 0
    compose.setContent {
      MaterialTheme {
        AppUpdateSectionContent(state.value, openingInstaller = true, onCheck = {}, onDownload = { downloads++ }, onCancel = {}, onInstall = {})
      }
    }
    compose.onNodeWithText("Retry download").performClick()
    assertEquals(1, downloads)
    compose.runOnIdle { state.value = AppUpdateState.Ready(offer) }
    compose.onNodeWithText("Verifying…").assertIsNotEnabled()
  }

  @Test fun unrecognizedBuildExplainsWhyUpdatingIsUnavailable() {
    compose.setContent {
      MaterialTheme {
        AppUpdateSectionContent(AppUpdateState.Error("This build is not a recognized published version.", UpdateRetryAction.CHECK), onCheck = {}, onDownload = {}, onCancel = {}, onInstall = {})
      }
    }
    compose.onNodeWithText("This build is not a recognized published version.").assertIsDisplayed()
    compose.onNodeWithText("Check again").assertIsDisplayed()
    compose.onNodeWithText("Install update").assertDoesNotExist()
  }

  @Test fun globalNoticeExposesUpdatesWithoutInterruptingLaunchForErrors() {
    val state = mutableStateOf<AppUpdateState>(AppUpdateState.Checking)
    var openedSettings = 0
    compose.setContent { MaterialTheme { AppUpdateNoticeContent(state.value, { openedSettings++ }) } }
    compose.onNodeWithText("View update").assertDoesNotExist()
    compose.runOnIdle { state.value = AppUpdateState.Error("Connection unavailable", UpdateRetryAction.CHECK) }
    compose.onNodeWithText("Connection unavailable").assertDoesNotExist()
    compose.runOnIdle { state.value = AppUpdateState.Available(offer) }
    compose.onNodeWithText("App update available · ${offer.versionName}").assertIsDisplayed()
    compose.onNodeWithText("View update").performClick()
    assertEquals(1, openedSettings)
    compose.runOnIdle { state.value = AppUpdateState.Ready(offer) }
    compose.onNodeWithText("App update ready · ${offer.versionName}").assertIsDisplayed()
    compose.onNodeWithText("Install update").assertDoesNotExist()
  }
}
