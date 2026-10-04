package me.danielshort.app.updates

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import me.danielshort.app.AppUpdateRuntime

/** Only PackageInstaller's explicit PendingIntent can reach this unexported receiver. */
class AutomaticInstallReceiver : BroadcastReceiver() {
  override fun onReceive(context: Context, intent: Intent) {
    if (intent.action != AutomaticAppInstaller.callbackAction(context)) return
    val pending = goAsync()
    CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
      try {
        // Failed persistence leaves the pre-commit journal for foreground recovery.
        // Do not turn a transient storage error into an application crash.
        runCatching { AppUpdateRuntime.installer.receive(intent) }
      } finally {
        pending.finish()
      }
    }
  }
}
