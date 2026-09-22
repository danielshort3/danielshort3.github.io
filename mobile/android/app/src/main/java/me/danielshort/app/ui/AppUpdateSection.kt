package me.danielshort.app.ui

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import me.danielshort.app.SiteApplication
import me.danielshort.app.updates.AppUpdateManager
import me.danielshort.app.updates.AutomaticAppInstaller
import me.danielshort.app.updates.AutomaticInstallStatus
import me.danielshort.app.updates.AppUpdateState
import me.danielshort.app.updates.UpdateFailure
import me.danielshort.app.updates.UpdateRetryAction
import java.util.Locale

@Composable
fun AppUpdateSection(
  manager: AppUpdateManager,
  automaticInstaller: AutomaticAppInstaller? = null,
  automaticUpdatesEnabled: Boolean = false,
  preferences: @Composable () -> Unit = {}
) {
  val state by manager.state.collectAsStateWithLifecycle()
  val context = LocalContext.current
  val scope = rememberCoroutineScope()
  var installerNotice by remember { mutableStateOf("") }
  var openingInstaller by remember { mutableStateOf(false) }
  var installationAllowed by remember { mutableStateOf(context.packageManager.canRequestPackageInstalls()) }
  LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
    installationAllowed = context.packageManager.canRequestPackageInstalls()
  }
  val automaticStatus = automaticInstaller?.status?.collectAsStateWithLifecycle()?.value ?: AutomaticInstallStatus.IDLE
  val automaticInstallBusy = automaticStatus == AutomaticInstallStatus.STAGING || automaticStatus == AutomaticInstallStatus.INSTALLING
  val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
    installationAllowed = context.packageManager.canRequestPackageInstalls()
    installerNotice = if (installationAllowed) {
      if (state is AppUpdateState.Ready) "Ready to install. Tap Install update to continue."
      else "App installation is allowed. Android may still ask you to confirm an update."
    } else {
      "Installation permission wasn’t enabled. You can allow it when you install an update."
    }
  }
  val installerLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
    // A successful self-update normally replaces this process. Returning without
    // replacement must keep the verified download available, including Cancel.
    installerNotice = "If installation wasn’t completed, you can try again."
  }
  AppUpdateSectionContent(
    state = state,
    installerNotice = installerNotice.ifBlank {
      when (automaticStatus) {
        AutomaticInstallStatus.STAGING, AutomaticInstallStatus.INSTALLING -> "Android is preparing the automatic update."
        AutomaticInstallStatus.NEEDS_PERMISSION -> "Allow app installation to use automatic updates."
        AutomaticInstallStatus.MANUAL_REQUIRED -> "Android needs your confirmation. Tap Install update when you’re ready."
        AutomaticInstallStatus.FAILED -> "The automatic installation didn’t finish. You can install the verified update manually."
        else -> ""
      }
    },
    openingInstaller = openingInstaller || automaticInstallBusy,
    onCheck = { installerNotice = ""; manager.check() },
    onDownload = { installerNotice = ""; manager.download() },
    onCancel = { installerNotice = ""; manager.cancel() },
    onInstall = {
      if (!openingInstaller && !automaticInstallBusy) scope.launch {
        openingInstaller = true
        installerNotice = ""
        try {
          val apk = manager.verifiedApkForInstall()
          if (automaticInstaller?.markManualInstallRequested() == false) {
            installerNotice = "Android is already preparing an update. Please wait."
            return@launch
          }
          (context.applicationContext as? SiteApplication)?.appUpdateCoordinator?.suppressAutomaticInstall()
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
            else "Android couldn’t open the installer. Please try again from Updates."
        } finally {
          openingInstaller = false
        }
      }
    },
    preferences = {
      preferences()
      if (automaticUpdatesEnabled && !installationAllowed) {
        Text("Allow installation from this app so Android can apply automatic updates when permitted.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        TextButton(onClick = {
          permissionLauncher.launch(Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${context.packageName}")))
        }) { Text("Allow app installation") }
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
  onInstall: () -> Unit,
  preferences: @Composable () -> Unit = {}
) {
  Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
    Text("App updates", Modifier.semantics { heading() }, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
    val status = when (state) {
      is AppUpdateState.Idle -> "Check for an app update"
      is AppUpdateState.Checking -> "Checking for updates…"
      is AppUpdateState.UpToDate -> "You’re up to date"
      is AppUpdateState.Available -> "Update available · ${state.offer.versionName} · ${formatUpdateBytes(state.offer.downloadBytes)}"
      is AppUpdateState.Downloading -> if (state.progress == null) "Verifying update files…"
        else "Downloading update… ${formatUpdateBytes(state.offer.downloadBytes)}"
      is AppUpdateState.Ready -> "Ready to install · ${state.offer.versionName}"
      is AppUpdateState.Error -> state.message
    }
    Text(status, Modifier.testTag("app-update-status").semantics { liveRegion = LiveRegionMode.Polite },
      color = if (state is AppUpdateState.Error) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
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
      is AppUpdateState.UpToDate -> OutlinedButton(onClick = onCheck) { Text("Check again") }
      is AppUpdateState.Idle -> OutlinedButton(onClick = onCheck) { Text("Check now") }
    }
    if (installerNotice.isNotBlank()) Text(installerNotice, Modifier.semantics { liveRegion = LiveRegionMode.Polite }, style = MaterialTheme.typography.bodySmall)
    Spacer(Modifier.height(8.dp))
    preferences()
  }
}

private fun formatUpdateBytes(bytes: Long): String = String.format(Locale.getDefault(), "%.1f MB", bytes / (1024.0 * 1024.0))
