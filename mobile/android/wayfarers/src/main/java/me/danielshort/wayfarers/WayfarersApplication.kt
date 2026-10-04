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
import me.danielshort.wayfarers.content.GuildContentVerifier
import me.danielshort.wayfarers.content.GuildContentStore
import me.danielshort.wayfarers.content.GuildContentUpdateManager
import me.danielshort.wayfarers.content.GuildContentUpdateState
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import kotlinx.coroutines.launch
import java.io.File

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
  lateinit var contentStore: GuildContentStore
    private set
  lateinit var contentUpdates: GuildContentUpdateManager
    private set

  override fun onCreate() {
    super.onCreate()
    settings = AppSettingsStore(this)
    updates = AppUpdateManager(this, BuildConfig.APP_UPDATE_URL)
    installer = AutomaticAppInstaller(this, updates)
    coordinator = AppUpdateCoordinator(updates, settings.state, UpdateNetworkMonitor(this, scope).state,
      scope, installAutomatically = installer::attemptIfEligible)
    contentStore = GuildContentStore(File(filesDir, "guild-content"), GuildContentVerifier(
      BuildConfig.CONTENT_PUBLIC_KEY, packageName, BuildConfig.CONTENT_NATIVE_API,
      BuildConfig.VERSION_CODE, BuildConfig.CONTENT_SAVE_SCHEMA), BuildConfig.BUNDLED_CONTENT_VERSION)
    contentUpdates = GuildContentUpdateManager(contentStore, BuildConfig.CONTENT_UPDATE_URL)
    if (settings.state.value.checkAppUpdatesOnLaunch) contentUpdates.check()
    scope.launch {
      contentUpdates.state.collect { state ->
        val preferences = settings.state.value
        if (state is GuildContentUpdateState.Available && state.autoDownloadAllowed && preferences.automaticAppUpdates) {
          val network = getSystemService(ConnectivityManager::class.java)
          val unmetered = network.getNetworkCapabilities(network.activeNetwork)?.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) == true
          if (!preferences.appUpdatesUnmeteredOnly || unmetered) contentUpdates.download()
        }
      }
    }
  }
}
