package me.danielshort.app.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.imageLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import me.danielshort.app.BuildConfig
import me.danielshort.app.SiteApplication
import me.danielshort.app.data.ContentRepository
import me.danielshort.app.nativefeatures.NativeFeatureHeader
import java.text.DateFormat
import java.util.Date

@Composable
fun SettingsScreen(repository: ContentRepository, onBack: () -> Unit) {
  val options by repository.settings.state.collectAsStateWithLifecycle()
  val content by repository.state.collectAsStateWithLifecycle()
  val saved by repository.favorites.collectAsStateWithLifecycle()
  val context = LocalContext.current
  val scope = rememberCoroutineScope()
  var notice by remember { mutableStateOf("") }
  var confirmClear by remember { mutableStateOf(false) }
  Column(Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
    NativeFeatureHeader("Settings", "Make the app work your way.", onBack)
    AppUpdateSection((context.applicationContext as SiteApplication).appUpdates)
    HorizontalDivider()
    Text("Content updates", modifier = Modifier.semantics { heading() }, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
    SettingSwitch("Automatic content updates", "Check for published content on launch and in the background.", options.automaticUpdates) {
      repository.settings.update(options.copy(automaticUpdates = it))
    }
    SettingSwitch("Save mobile data", "Use unmetered networks for automatic content updates.", options.unmeteredOnly, options.automaticUpdates) {
      repository.settings.update(options.copy(unmeteredOnly = it))
    }
    OutlinedButton(onClick = { scope.launch { repository.refresh(force = true); notice = repository.state.value.message } }, enabled = !content.refreshing) {
      Text(if (content.refreshing) "Refreshing…" else "Refresh now")
    }
    Text(if (content.lastChecked == 0L) "Using bundled content" else "Last checked ${DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(content.lastChecked))}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    HorizontalDivider()
    SettingSwitch("Reduce motion", "Limit interface motion and extra game effects.", options.reduceMotion) {
      repository.settings.update(options.copy(reduceMotion = it))
    }
    HorizontalDivider()
    Text("On this device", modifier = Modifier.semantics { heading() }, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
    Text("Your text, generated files, bookmarks, and game progress stay on this device. AI demos send the input you submit to their existing inference services.", color = MaterialTheme.colorScheme.onSurfaceVariant)
    OutlinedButton(onClick = { scope.launch {
      notice = try {
        context.imageLoader.memoryCache?.clear()
        withContext(Dispatchers.IO) { context.imageLoader.diskCache?.clear() }
        "Image cache cleared. Images will load again when needed."
      } catch (cancelled: CancellationException) { throw cancelled }
      catch (_: Exception) { "Couldn’t clear the image cache. Please try again." }
    } }) { Text("Clear image cache") }
    TextButton(onClick = { confirmClear = true }, enabled = saved.isNotEmpty()) { Text("Clear saved projects (${saved.size})") }
    if (notice.isNotBlank()) Text(notice, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite }, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
    HorizontalDivider()
    Text("Daniel Short · ${BuildConfig.VERSION_NAME}", fontWeight = FontWeight.SemiBold)
    Text("Website content follows your update preferences. New native features arrive through app updates.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    Spacer(Modifier.height(12.dp))
  }
  if (confirmClear) AlertDialog(onDismissRequest = { confirmClear = false }, title = { Text("Clear saved projects?") }, text = { Text("This removes your local bookmarks. Your game progress and generated files are kept.") }, confirmButton = { TextButton(onClick = { repository.clearSavedProjects(); confirmClear = false; notice = "Saved projects cleared" }) { Text("Clear bookmarks") } }, dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("Cancel") } })
}

@Composable
private fun SettingSwitch(title: String, description: String, checked: Boolean, enabled: Boolean = true, onChange: (Boolean) -> Unit) {
  Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).toggleable(checked, enabled = enabled, role = Role.Switch, onValueChange = onChange).padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
    Column(Modifier.weight(1f)) {
      Text(title, fontWeight = FontWeight.SemiBold)
      Text(description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    Switch(checked, onCheckedChange = null, enabled = enabled)
  }
}
