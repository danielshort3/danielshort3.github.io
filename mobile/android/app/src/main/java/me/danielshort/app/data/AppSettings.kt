package me.danielshort.app.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

data class AppSettings(
  val automaticUpdates: Boolean = true,
  val unmeteredOnly: Boolean = false,
  val reduceMotion: Boolean = false,
  val checkAppUpdatesOnLaunch: Boolean = true,
  val automaticAppUpdates: Boolean = false,
  val appUpdatesUnmeteredOnly: Boolean = true
)

class AppSettingsStore(context: Context) {
  private val preferences = context.getSharedPreferences("native-settings", Context.MODE_PRIVATE)
  private val mutableState = MutableStateFlow(AppSettings(
    preferences.getBoolean("automatic-updates", true),
    preferences.getBoolean("unmetered-only", false),
    preferences.getBoolean("reduce-motion", false),
    preferences.getBoolean("check-app-updates-on-launch", true),
    preferences.getBoolean("automatic-app-updates", false),
    preferences.getBoolean("app-updates-unmetered-only", true)
  ))
  val state = mutableState.asStateFlow()

  fun update(settings: AppSettings) {
    preferences.edit()
      .putBoolean("automatic-updates", settings.automaticUpdates)
      .putBoolean("unmetered-only", settings.unmeteredOnly)
      .putBoolean("reduce-motion", settings.reduceMotion)
      .putBoolean("check-app-updates-on-launch", settings.checkAppUpdatesOnLaunch)
      .putBoolean("automatic-app-updates", settings.automaticAppUpdates)
      .putBoolean("app-updates-unmetered-only", settings.appUpdatesUnmeteredOnly)
      .apply()
    mutableState.value = settings
  }
}
