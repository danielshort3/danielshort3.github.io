package me.danielshort.wayfarers

import android.app.Application
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import me.danielshort.app.BuildConfig
import me.danielshort.app.data.AppSettingsStore
import me.danielshort.app.updates.AppUpdateCoordinator
import me.danielshort.app.updates.AppUpdateManager
import me.danielshort.app.updates.AutomaticAppInstaller
import me.danielshort.app.updates.UpdateNetworkMonitor

class WayfarersApplication : Application() {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
  lateinit var settings: AppSettingsStore
    private set
  lateinit var updates: AppUpdateManager
    private set
  lateinit var installer: AutomaticAppInstaller
    private set
  lateinit var coordinator: AppUpdateCoordinator
    private set

  override fun onCreate() {
    super.onCreate()
    settings = AppSettingsStore(this)
    updates = AppUpdateManager(this, BuildConfig.APP_UPDATE_URL)
    installer = AutomaticAppInstaller(this, updates)
    coordinator = AppUpdateCoordinator(updates, settings.state, UpdateNetworkMonitor(this, scope).state,
      scope, installAutomatically = installer::attemptIfEligible)
  }
}
