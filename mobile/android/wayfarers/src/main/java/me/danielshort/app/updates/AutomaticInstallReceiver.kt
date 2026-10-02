package me.danielshort.app.updates

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import me.danielshort.wayfarers.WayfarersApplication

class AutomaticInstallReceiver : BroadcastReceiver() {
  override fun onReceive(context: Context, intent: Intent) {
    if (intent.action != AutomaticAppInstaller.callbackAction(context)) return
    val app = context.applicationContext as? WayfarersApplication ?: return
    val pending = goAsync()
    CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
      try { runCatching { app.installer.receive(intent) } }
      finally { pending.finish() }
    }
  }
}
