package me.danielshort.app.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ApplicationProvider
import android.content.Context
import kotlinx.coroutines.runBlocking
import me.danielshort.app.data.AppSettings
import me.danielshort.app.data.AppSettingsStore
import me.danielshort.app.data.ContentRepository
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test

class SettingsFlowTest {
  @get:Rule val compose = createComposeRule()
  private lateinit var context: Context
  private lateinit var settings: AppSettingsStore
  private lateinit var original: AppSettings

  @Before fun prepare() {
    context = ApplicationProvider.getApplicationContext()
    settings = AppSettingsStore(context)
    original = settings.state.value
    settings.update(AppSettings(automaticUpdates = false))
  }

  @After fun restore() { settings.update(original) }

  @Test fun settingsPersistAcrossStoreRecreation() {
    val chosen = AppSettings(automaticUpdates = false, unmeteredOnly = true, reduceMotion = true)
    settings.update(chosen)
    assertEquals(chosen, AppSettingsStore(context).state.value)
  }

  @Test fun disabledAutomaticUpdatesSkipRefreshAndKeepManualControlAvailable() {
    val repository = ContentRepository(context, settings)
    val previousCheck = context.getSharedPreferences("native-content", Context.MODE_PRIVATE).getLong("last-checked", 0L)
    assertTrue(runBlocking { repository.refresh() })
    assertFalse(repository.state.value.refreshing)
    assertEquals(previousCheck, context.getSharedPreferences("native-content", Context.MODE_PRIVATE).getLong("last-checked", 0L))
    compose.setContent { MaterialTheme { SettingsScreen(repository, onBack = {}) } }
    compose.onNodeWithText("Refresh now").performScrollTo().assertIsEnabled()
    compose.onNodeWithText("Save mobile data").assertIsNotEnabled()
    compose.onNodeWithText("Reduce motion").performScrollTo().performClick()
    assertTrue(AppSettingsStore(context).state.value.reduceMotion)
    assertFalse(AppSettingsStore(context).state.value.automaticUpdates)
    compose.onNodeWithText("Automatic content updates").performScrollTo().performClick()
    compose.onNodeWithText("Save mobile data").assertIsEnabled()
    assertTrue(AppSettingsStore(context).state.value.automaticUpdates)
  }
}
