package me.danielshort.app.data

import org.junit.Assert.*
import org.junit.Test

class SettingsChoicesTest {
  @Test fun defaultsDoNotOptIntoAutomaticAppDownloads() {
    val original = AppSettings()
    assertEquals(AppUpdateMode.CHECK_AUTOMATICALLY, original.appUpdateMode)
    assertEquals(ContentRefreshMode.AUTOMATIC, original.contentRefreshMode)
    assertFalse(original.automaticAppUpdates)
    assertTrue(original.appUpdatesUnmeteredOnly)
  }

  @Test fun allExistingPreferenceCombinationsMapWithoutMutatingStorage() {
    repeat(64) { bits ->
      val original = fixture(bits)
      val snapshot = original.copy()
      assertEquals(when {
        original.automaticAppUpdates -> AppUpdateMode.AUTOMATIC
        original.checkAppUpdatesOnLaunch -> AppUpdateMode.CHECK_AUTOMATICALLY
        else -> AppUpdateMode.MANUAL
      }, original.appUpdateMode)
      assertEquals(when {
        !original.automaticUpdates -> ContentRefreshMode.MANUAL
        original.unmeteredOnly -> ContentRefreshMode.UNMETERED_ONLY
        else -> ContentRefreshMode.AUTOMATIC
      }, original.contentRefreshMode)
      assertEquals(snapshot, original)
    }
  }

  @Test fun everyAppModePreservesContentMotionAndDownloadNetworkPreferences() {
    repeat(64) { bits ->
      val original = fixture(bits)
      AppUpdateMode.entries.forEach { mode ->
        val changed = original.withAppUpdateMode(mode)
        assertEquals(mode, changed.appUpdateMode)
        assertEquals(original.automaticUpdates, changed.automaticUpdates)
        assertEquals(original.unmeteredOnly, changed.unmeteredOnly)
        assertEquals(original.reduceMotion, changed.reduceMotion)
        assertEquals(original.appUpdatesUnmeteredOnly, changed.appUpdatesUnmeteredOnly)
        assertEquals(mode == AppUpdateMode.AUTOMATIC, changed.automaticAppUpdates)
        assertEquals(mode != AppUpdateMode.MANUAL, changed.checkAppUpdatesOnLaunch)
      }
    }
  }

  @Test fun everyContentModePreservesAppUpdateAndMotionPreferences() {
    repeat(64) { bits ->
      val original = fixture(bits)
      ContentRefreshMode.entries.forEach { mode ->
        val changed = original.withContentRefreshMode(mode)
        assertEquals(mode, changed.contentRefreshMode)
        assertEquals(original.checkAppUpdatesOnLaunch, changed.checkAppUpdatesOnLaunch)
        assertEquals(original.automaticAppUpdates, changed.automaticAppUpdates)
        assertEquals(original.appUpdatesUnmeteredOnly, changed.appUpdatesUnmeteredOnly)
        assertEquals(original.reduceMotion, changed.reduceMotion)
        if (mode == ContentRefreshMode.MANUAL) assertEquals(original.unmeteredOnly, changed.unmeteredOnly)
      }
    }
  }

  @Test fun hiddenDownloadNetworkChoiceSurvivesManualMode() {
    listOf(true, false).forEach { restricted ->
      val original = AppSettings(automaticAppUpdates = true, appUpdatesUnmeteredOnly = restricted)
      val restored = original.withAppUpdateMode(AppUpdateMode.MANUAL).withAppUpdateMode(AppUpdateMode.AUTOMATIC)
      assertEquals(restricted, restored.appUpdatesUnmeteredOnly)
    }
  }

  @Test fun eitherAndroidOrTheAppCanReduceMotion() {
    assertFalse(effectiveReduceMotion(false, false))
    assertTrue(effectiveReduceMotion(true, false))
    assertTrue(effectiveReduceMotion(false, true))
    assertTrue(effectiveReduceMotion(true, true))
  }

  private fun fixture(bits: Int) = AppSettings(
    automaticUpdates = bits and 1 != 0,
    unmeteredOnly = bits and 2 != 0,
    reduceMotion = bits and 4 != 0,
    checkAppUpdatesOnLaunch = bits and 8 != 0,
    automaticAppUpdates = bits and 16 != 0,
    appUpdatesUnmeteredOnly = bits and 32 != 0
  )
}
