package me.danielshort.app.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import me.danielshort.app.updates.AppUpdateState
import me.danielshort.app.updates.UpdateOffer
import me.danielshort.app.updates.UpdateRetryAction
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class AppUpdateSectionTest {
  @get:Rule val compose = createComposeRule()
  private val offer = UpdateOffer(7, "0.5.0-debug", 1024L * 1024, true)

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
    compose.onNodeWithTag("app-update-status").assertTextEquals("Check for an app update")
    compose.onNodeWithText("Check now").performClick()
    assertEquals(listOf("check"), actions)
    compose.runOnIdle { state.value = AppUpdateState.Available(offer) }
    compose.onNodeWithTag("app-update-status").assertTextContains("Update available", substring = true)
    compose.onNodeWithText("Download update").performClick()
    compose.runOnIdle { state.value = AppUpdateState.Downloading(offer, .5f, true) }
    compose.onNodeWithTag("app-update-status").assertTextContains("Downloading update", substring = true)
    compose.onNodeWithText("Downloading patch", substring = true).assertDoesNotExist()
    compose.onNodeWithText("Cancel download").performClick()
    compose.runOnIdle { state.value = AppUpdateState.Ready(offer) }
    compose.onNodeWithTag("app-update-status").assertTextEquals("Ready to install · ${offer.versionName}")
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
    compose.onNodeWithText("The download was interrupted.").assertIsDisplayed()
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
    var openedUpdates = 0
    compose.setContent { MaterialTheme { AppUpdateNoticeContent(state.value, { openedUpdates++ }) } }
    compose.onNodeWithText("View update").assertDoesNotExist()
    compose.runOnIdle { state.value = AppUpdateState.Error("Connection unavailable", UpdateRetryAction.CHECK) }
    compose.onNodeWithText("Connection unavailable").assertDoesNotExist()
    compose.runOnIdle { state.value = AppUpdateState.Available(offer) }
    compose.onNodeWithText("App update available · ${offer.versionName}").assertIsDisplayed()
    compose.onNodeWithText("View update").performClick()
    assertEquals(1, openedUpdates)
    compose.runOnIdle { state.value = AppUpdateState.Ready(offer) }
    compose.onNodeWithText("App update ready · ${offer.versionName}").assertIsDisplayed()
    compose.onNodeWithText("Install update").assertDoesNotExist()
  }
}
