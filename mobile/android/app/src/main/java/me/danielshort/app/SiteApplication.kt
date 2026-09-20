package me.danielshort.app

import android.app.Application
import android.content.Context
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import me.danielshort.app.data.ContentRepository
import me.danielshort.app.data.AppSettingsStore
import me.danielshort.app.updates.AppUpdateManager
import me.danielshort.app.updates.AppUpdateCoordinator
import me.danielshort.app.updates.AutomaticAppInstaller
import me.danielshort.app.updates.UpdateNetworkMonitor
import me.danielshort.app.nativefeatures.recording.ScreenRecording
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.distinctUntilChangedBy
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onEach
import java.util.concurrent.TimeUnit

class SiteApplication : Application() {
  lateinit var contentRepository: ContentRepository
    private set
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
  val appUpdates by lazy { AppUpdateManager(this) }
  val automaticAppInstaller by lazy { AutomaticAppInstaller(this, appUpdates) }
  val appUpdateCoordinator by lazy {
    AppUpdateCoordinator(appUpdates, contentRepository.settings.state, UpdateNetworkMonitor(this, scope).state, scope,
      installAutomatically = { eligible -> automaticAppInstaller.attemptIfEligible(eligible) },
      protectedWorkChanges = ScreenRecording.state.map { it.active }.distinctUntilChanged(),
      hasProtectedWork = { ScreenRecording.state.value.active })
  }
  override fun onCreate() {
    super.onCreate()
    val settings = AppSettingsStore(this)
    contentRepository = ContentRepository(this, settings)
    settings.state.distinctUntilChangedBy { it.automaticUpdates to it.unmeteredOnly }.onEach { options ->
      val work = WorkManager.getInstance(this)
      if (!options.automaticUpdates) work.cancelUniqueWork("website-content-v1")
      else {
        val network = if (options.unmeteredOnly) NetworkType.UNMETERED else NetworkType.CONNECTED
        val sync = PeriodicWorkRequestBuilder<ContentSyncWorker>(6, TimeUnit.HOURS)
          .setConstraints(Constraints.Builder().setRequiredNetworkType(network).build()).build()
        work.enqueueUniquePeriodicWork("website-content-v1", ExistingPeriodicWorkPolicy.UPDATE, sync)
      }
    }.launchIn(scope)
  }
}

class ContentSyncWorker(context: Context, parameters: WorkerParameters) : CoroutineWorker(context, parameters) {
  override suspend fun doWork(): Result {
    val repository = (applicationContext as SiteApplication).contentRepository
    return if (repository.refresh()) Result.success() else Result.retry()
  }
}
