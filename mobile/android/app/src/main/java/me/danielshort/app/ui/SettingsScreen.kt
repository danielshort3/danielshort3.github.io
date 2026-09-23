package me.danielshort.app.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.imageLoader
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import me.danielshort.app.BuildConfig
import me.danielshort.app.SiteApplication
import me.danielshort.app.data.AppSettings
import me.danielshort.app.data.ContentRepository
import me.danielshort.app.data.appUpdateMode
import me.danielshort.app.updates.AppUpdateState
import java.text.DateFormat
import java.util.Date

internal enum class SettingsPage(val title: String) {
  OVERVIEW("Settings"), UPDATES("Updates"), STORAGE("Storage"), INFORMATION("App information")
}

@Composable
internal fun SettingsTheme(content: @Composable () -> Unit) {
  val blue = Color(0xFF155DFC)
  val navy = Color(0xFF091F3B)
  MaterialTheme(colorScheme = MaterialTheme.colorScheme.copy(
    primary = blue, onPrimary = Color.White, primaryContainer = Color(0xFFE8EFFF), onPrimaryContainer = navy,
    secondary = blue, onSecondary = Color.White, secondaryContainer = Color(0xFFE8EFFF), onSecondaryContainer = navy,
    surfaceTint = blue
  ), content = content)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SettingsScreen(repository: ContentRepository, onBack: () -> Unit, initialPage: SettingsPage = SettingsPage.OVERVIEW) {
  val options by repository.settings.state.collectAsStateWithLifecycle()
  val application = LocalContext.current.applicationContext as SiteApplication
  val updateState by application.appUpdates.state.collectAsStateWithLifecycle()
  var pageName by rememberSaveable(initialPage) { mutableStateOf(initialPage.name) }
  val page = SettingsPage.valueOf(pageName)
  val snackbar = remember { SnackbarHostState() }
  val scope = rememberCoroutineScope()
  val goBack = {
    if (page == SettingsPage.OVERVIEW) onBack() else pageName = SettingsPage.OVERVIEW.name
  }
  val showNotice: (String) -> Unit = { message ->
    scope.launch { snackbar.currentSnackbarData?.dismiss(); snackbar.showSnackbar(message) }
  }
  BackHandler(onBack = goBack)
  SettingsTheme {
    Scaffold(
      modifier = Modifier.fillMaxSize().safeDrawingPadding(),
      contentWindowInsets = WindowInsets(0, 0, 0, 0),
      topBar = {
        TopAppBar(title = { Text(page.title) }, navigationIcon = {
          IconButton(onClick = goBack, modifier = Modifier.testTag("settings-back")) {
            Icon(Icons.AutoMirrored.Outlined.ArrowBack, if (page == SettingsPage.OVERVIEW) "Back to app" else "Back to settings")
          }
        }, windowInsets = WindowInsets(0, 0, 0, 0), colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface))
      },
      snackbarHost = { SnackbarHost(snackbar, Modifier.testTag("settings-feedback")) }
    ) { padding ->
      key(page) {
        Column(Modifier.fillMaxSize().padding(padding).verticalScroll(rememberScrollState())
          .padding(horizontal = 20.dp, vertical = 12.dp).testTag("settings-body"),
          verticalArrangement = Arrangement.spacedBy(16.dp)) {
          when (page) {
            SettingsPage.OVERVIEW -> SettingsOverview(options, settingsUpdateSummary(updateState, options),
              onChange = repository.settings::update, onOpen = { pageName = it.name })
            SettingsPage.UPDATES -> {
              AppUpdateSection(application.appUpdates, application.automaticAppInstaller, options.automaticAppUpdates) {
                AppUpdatePreferences(options, repository.settings::update)
              }
              HorizontalDivider(Modifier.padding(vertical = 8.dp))
              ContentRefreshSettings(repository, onNotice = showNotice)
            }
            SettingsPage.STORAGE -> StorageSettings(onNotice = showNotice)
            SettingsPage.INFORMATION -> AppInformation()
          }
        }
      }
    }
  }
}

internal fun settingsUpdateSummary(state: AppUpdateState, options: AppSettings): String = when (state) {
  is AppUpdateState.Available -> "Update available"
  is AppUpdateState.Ready -> "Ready to install"
  is AppUpdateState.Checking -> "Checking for updates…"
  is AppUpdateState.Downloading -> "Downloading update…"
  is AppUpdateState.UpToDate -> "You’re up to date"
  is AppUpdateState.Error -> "Update needs attention"
  is AppUpdateState.Idle -> appUpdateChoices.first { it.value == options.appUpdateMode }.title
}

@Composable
internal fun SettingsOverview(options: AppSettings, updateSummary: String, onChange: (AppSettings) -> Unit, onOpen: (SettingsPage) -> Unit) {
  val systemReduceMotion = rememberSystemReduceMotion()
  Column {
    SettingsNavigationRow("Updates", updateSummary, Modifier.testTag("settings-updates")) { onOpen(SettingsPage.UPDATES) }
    HorizontalDivider()
    SettingSwitch("Reduce motion", if (systemReduceMotion) "Android is already reducing motion. Also reduce it when system animations are on." else "Fewer animations and game effects.",
      options.reduceMotion, Modifier.testTag("reduce-motion")) { onChange(options.copy(reduceMotion = it)) }
    HorizontalDivider()
    SettingsNavigationRow("Storage", "Cached images", Modifier.testTag("settings-storage")) { onOpen(SettingsPage.STORAGE) }
    HorizontalDivider()
    SettingsNavigationRow("App information", "Privacy and version", Modifier.testTag("settings-information")) { onOpen(SettingsPage.INFORMATION) }
  }
}

@Composable
private fun ContentRefreshSettings(repository: ContentRepository, onNotice: (String) -> Unit) {
  val options by repository.settings.state.collectAsStateWithLifecycle()
  val content by repository.state.collectAsStateWithLifecycle()
  val scope = rememberCoroutineScope()
  var error by rememberSaveable { mutableStateOf("") }
  var refreshing by remember { mutableStateOf(false) }
  SettingsHeading("Website content")
  Text("Refreshes website information and listings, not the installed app.",
    style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
  ContentUpdatePreferences(options, repository.settings::update)
  OutlinedButton(onClick = {
    if (!refreshing) scope.launch {
      refreshing = true
      error = ""
      try {
        if (repository.refresh(force = true)) onNotice(repository.state.value.message.ifBlank { "Content refreshed" })
        else error = if (repository.state.value.content != null) "Couldn’t refresh. Your saved content is still available. Check your connection and try again."
          else "Couldn’t reach the content feed. Check your connection and try again."
      } finally { refreshing = false }
    }
  }, enabled = !content.refreshing && !refreshing, modifier = Modifier.testTag("refresh-content")) {
    Text(if (content.refreshing || refreshing) "Refreshing…" else "Refresh now")
  }
  if (error.isNotBlank()) SettingsError(error)
  Text(if (content.lastChecked == 0L) "Using bundled content" else "Last checked ${DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(content.lastChecked))}",
    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun StorageSettings(onNotice: (String) -> Unit) {
  val context = LocalContext.current
  val scope = rememberCoroutineScope()
  var clearing by remember { mutableStateOf(false) }
  var error by rememberSaveable { mutableStateOf("") }
  SettingsHeading("Cached images")
  Text("Images are saved temporarily so they load faster. Clearing them does not remove bookmarks, game progress, or generated files.",
    color = MaterialTheme.colorScheme.onSurfaceVariant)
  OutlinedButton(onClick = {
    if (!clearing) scope.launch {
      clearing = true
      error = ""
      try {
        context.imageLoader.memoryCache?.clear()
        withContext(Dispatchers.IO) { context.imageLoader.diskCache?.clear() }
        onNotice("Cached images cleared")
      } catch (cancelled: CancellationException) { throw cancelled }
      catch (_: Exception) { error = "Couldn’t clear cached images. Please try again." }
      finally { clearing = false }
    }
  }, enabled = !clearing, modifier = Modifier.testTag("clear-cached-images")) {
    Text(if (clearing) "Clearing…" else "Clear cached images")
  }
  Text("Images will download again when needed.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
  if (error.isNotBlank()) SettingsError(error)
}

@Composable
private fun AppInformation() {
  Text("Daniel Short", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
  Text("Version ${BuildConfig.VERSION_NAME}", color = MaterialTheme.colorScheme.onSurfaceVariant)
  HorizontalDivider()
  SettingsHeading("On this device")
  Text("Bookmarks, native game progress, and files created by native tools stay on your device. You choose when to export or share them.",
    color = MaterialTheme.colorScheme.onSurfaceVariant)
  SettingsHeading("AI demos")
  Text("AI demos use the website’s inference services. Submitted text and drawings, or generation settings, are sent to those services. Some demos connect or load generated examples when opened. See the notice beside each submit control.",
    color = MaterialTheme.colorScheme.onSurfaceVariant)
  SettingsHeading("Content and app updates")
  Text("Website information follows your content-refresh preference. New native features require an app update.",
    color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@Composable
private fun SettingsHeading(title: String) {
  Text(title, Modifier.semantics { heading() }, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
}

@Composable
private fun SettingsError(message: String) {
  Text(message, Modifier.semantics { liveRegion = LiveRegionMode.Polite }, color = MaterialTheme.colorScheme.error,
    style = MaterialTheme.typography.bodyMedium)
}
