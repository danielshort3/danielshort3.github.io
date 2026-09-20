package me.danielshort.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import me.danielshort.app.ui.DanielShortApp
import me.danielshort.app.ui.AppUpdateNotice

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    val app = application as SiteApplication
    setContent {
      DanielShortApp(
        repository = app.contentRepository,
        onSafeToInstall = app.appUpdateCoordinator::setSafeToInstall,
        globalNotice = { onOpenSettings -> AppUpdateNotice(app.appUpdates, onOpenSettings) }
      )
    }
  }
  override fun onStart() {
    super.onStart()
    (application as SiteApplication).appUpdateCoordinator.onForeground()
    lifecycleScope.launch { (application as SiteApplication).automaticAppInstaller.recover() }
    lifecycleScope.launch { (application as SiteApplication).contentRepository.refresh() }
  }

  override fun onStop() {
    if (!isChangingConfigurations) (application as SiteApplication).appUpdateCoordinator.onBackground()
    super.onStop()
  }
}
