package me.danielshort.app.ui

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import me.danielshort.app.BuildConfig
import me.danielshort.app.updates.AppUpdateManager
import me.danielshort.app.updates.AppUpdateState
import me.danielshort.app.updates.UpdateFailure
import me.danielshort.app.updates.UpdateRetryAction
import java.util.Locale

@Composable
fun AppUpdateSection(manager: AppUpdateManager) {
  val state by manager.state.collectAsStateWithLifecycle()
  val context = LocalContext.current
  val scope = rememberCoroutineScope()
  var installerNotice by remember { mutableStateOf("") }
  var openingInstaller by remember { mutableStateOf(false) }
  val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
    installerNotice = if (context.packageManager.canRequestPackageInstalls()) {
      "Ready to install. Tap Install update to continue."
    } else {
      "Installation permission wasn’t enabled. Your verified update is still ready."
    }
  }
  val installerLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
    // A successful self-update normally replaces this process. Returning without
    // replacement must keep the verified download available, including Cancel.
    installerNotice = "If installation wasn’t completed, you can try again."
  }
  AppUpdateSectionContent(
    state = state,
    installerNotice = installerNotice,
    openingInstaller = openingInstaller,
    onCheck = { installerNotice = ""; manager.check() },
    onDownload = { installerNotice = ""; manager.download() },
    onCancel = { installerNotice = ""; manager.cancel() },
    onInstall = {
      if (!openingInstaller) scope.launch {
        openingInstaller = true
        installerNotice = ""
        try {
          val apk = manager.verifiedApkForInstall()
          if (!context.packageManager.canRequestPackageInstalls()) {
            permissionLauncher.launch(Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${context.packageName}")))
          } else {
            val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", apk)
            installerLauncher.launch(Intent(Intent.ACTION_VIEW).apply {
              setDataAndType(uri, "application/vnd.android.package-archive")
              addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
              putExtra(Intent.EXTRA_RETURN_RESULT, true)
            })
          }
        } catch (cancelled: CancellationException) {
          throw cancelled
        } catch (error: Exception) {
          installerNotice = if (error is UpdateFailure) error.message.orEmpty()
            else "Android couldn’t open the installer. Please try again from Settings."
        } finally {
          openingInstaller = false
        }
      }
    }
  )
}

@Composable
internal fun AppUpdateSectionContent(
  state: AppUpdateState,
  installerNotice: String = "",
  openingInstaller: Boolean = false,
  onCheck: () -> Unit,
  onDownload: () -> Unit,
  onCancel: () -> Unit,
  onInstall: () -> Unit
) {
  Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
    Text("App updates", Modifier.semantics { heading() }, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
    Text("Installed · ${BuildConfig.VERSION_NAME}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    val status = when (state) {
      is AppUpdateState.Idle -> "Check for a new version of the app."
      is AppUpdateState.Checking -> "Checking the release and verifying this app…"
      is AppUpdateState.UpToDate -> "You’re up to date. Installed app verified."
      is AppUpdateState.Available -> "${state.offer.versionName} is available · ${formatUpdateBytes(state.offer.downloadBytes)}${if (state.offer.usingPatch) " patch" else " download"}"
      is AppUpdateState.Downloading -> when {
        state.progress == null -> "Verifying app files…"
        state.usingPatch -> "Downloading patch · ${formatUpdateBytes(state.offer.downloadBytes)}…"
        else -> "Downloading full app · ${formatUpdateBytes(state.offer.downloadBytes)}…"
      }
      is AppUpdateState.Ready -> "${state.offer.versionName} is verified and ready to install. Android will ask you to confirm."
      is AppUpdateState.Error -> state.message
    }
    Text(status, Modifier.semantics { liveRegion = LiveRegionMode.Polite }, color = MaterialTheme.colorScheme.onSurfaceVariant)
    when (state) {
      is AppUpdateState.Checking -> {
        LinearProgressIndicator(Modifier.fillMaxWidth())
        TextButton(onClick = onCancel) { Text("Cancel") }
      }
      is AppUpdateState.Downloading -> {
        if (state.progress == null) LinearProgressIndicator(Modifier.fillMaxWidth())
        else LinearProgressIndicator(progress = { state.progress.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
        TextButton(onClick = onCancel) { Text("Cancel download") }
      }
      is AppUpdateState.Available -> Button(onClick = onDownload) { Text("Download update") }
      is AppUpdateState.Ready -> Button(onClick = onInstall, enabled = !openingInstaller) {
        Text(if (openingInstaller) "Verifying…" else "Install update")
      }
      is AppUpdateState.Error -> if (state.retryAction == UpdateRetryAction.DOWNLOAD) {
        OutlinedButton(onClick = onDownload) { Text("Retry download") }
      } else OutlinedButton(onClick = onCheck) { Text("Check again") }
      else -> OutlinedButton(onClick = onCheck) { Text("Check for updates") }
    }
    if (installerNotice.isNotBlank()) Text(installerNotice, Modifier.semantics { liveRegion = LiveRegionMode.Polite }, style = MaterialTheme.typography.bodySmall)
  }
}

private fun formatUpdateBytes(bytes: Long): String = String.format(Locale.getDefault(), "%.1f MB", bytes / (1024.0 * 1024.0))
