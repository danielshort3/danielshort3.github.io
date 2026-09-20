package me.danielshort.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import me.danielshort.app.updates.AppUpdateManager
import me.danielshort.app.updates.AppUpdateState

/** Quiet, inline, and actionable. Startup errors and routine checks stay in Settings. */
@Composable
fun AppUpdateNotice(manager: AppUpdateManager, onOpenSettings: () -> Unit, modifier: Modifier = Modifier) {
  val state by manager.state.collectAsStateWithLifecycle()
  AppUpdateNoticeContent(state, onOpenSettings, modifier)
}

@Composable
internal fun AppUpdateNoticeContent(state: AppUpdateState, onOpenSettings: () -> Unit, modifier: Modifier = Modifier) {
  val message = when (state) {
    is AppUpdateState.Available -> "App update available · ${state.offer.versionName}"
    is AppUpdateState.Downloading -> "Preparing app update · ${state.offer.versionName}"
    is AppUpdateState.Ready -> "App update ready · ${state.offer.versionName}"
    else -> return
  }
  Surface(modifier.fillMaxWidth(), color = MaterialTheme.colorScheme.surfaceContainer) {
    Row(Modifier.padding(start = 16.dp, end = 8.dp, top = 4.dp, bottom = 4.dp),
      horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
      Column(Modifier.weight(1f)) {
        Text(message, style = MaterialTheme.typography.bodySmall)
      }
      TextButton(onClick = onOpenSettings) { Text("View update") }
    }
  }
}
