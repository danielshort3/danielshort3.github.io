package me.danielshort.app.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

data class AppSettings(
  val automaticUpdates: Boolean = true,
  val unmeteredOnly: Boolean = false,
  val reduceMotion: Boolean = false
)

class AppSettingsStore(context: Context) {
  private val preferences = context.getSharedPreferences("native-settings", Context.MODE_PRIVATE)
  private val mutableState = MutableStateFlow(AppSettings(
    preferences.getBoolean("automatic-updates", true),
    preferences.getBoolean("unmetered-only", false),
    preferences.getBoolean("reduce-motion", false)
  ))
  val state = mutableState.asStateFlow()

  fun update(settings: AppSettings) {
    preferences.edit()
      .putBoolean("automatic-updates", settings.automaticUpdates)
      .putBoolean("unmetered-only", settings.unmeteredOnly)
      .putBoolean("reduce-motion", settings.reduceMotion)
      .apply()
    mutableState.value = settings
  }
}
