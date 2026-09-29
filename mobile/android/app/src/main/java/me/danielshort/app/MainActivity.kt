package me.danielshort.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import me.danielshort.app.ui.DanielShortApp

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    val app = application as SiteApplication
    setContent {
      DanielShortApp(
        repository = app.contentRepository,
        onSafeToInstall = AppUpdateRuntime::setSafeToInstall,
        globalNotice = AppUpdateRuntime::notice
      )
    }
  }
  override fun onStart() {
    super.onStart()
    AppUpdateRuntime.onForeground()
    lifecycleScope.launch { (application as SiteApplication).contentRepository.refresh() }
  }

  override fun onStop() {
    if (!isChangingConfigurations) AppUpdateRuntime.onBackground()
    super.onStop()
  }
}
