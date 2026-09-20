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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import java.util.concurrent.TimeUnit

class SiteApplication : Application() {
  lateinit var contentRepository: ContentRepository
    private set
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
  val appUpdates by lazy { AppUpdateManager(this) }
  override fun onCreate() {
    super.onCreate()
    val settings = AppSettingsStore(this)
    contentRepository = ContentRepository(this, settings)
    settings.state.onEach { options ->
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
