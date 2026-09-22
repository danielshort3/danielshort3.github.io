package me.danielshort.app.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.KeyboardArrowRight
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import me.danielshort.app.data.*

internal data class SettingsChoice<T>(val value: T, val title: String, val description: String)

internal val appUpdateChoices = listOf(
  SettingsChoice(AppUpdateMode.CHECK_AUTOMATICALLY, "Check automatically", "Check when the app opens. You choose when to download and install."),
  SettingsChoice(AppUpdateMode.AUTOMATIC, "Automatic", "Download automatically and install when the app is idle and Android permits. Confirmation may still be required."),
  SettingsChoice(AppUpdateMode.MANUAL, "Manual", "Check only when you choose Check now.")
)
internal val contentRefreshChoices = listOf(
  SettingsChoice(ContentRefreshMode.AUTOMATIC, "Automatic", "Refresh on launch and in the background using any connection."),
  SettingsChoice(ContentRefreshMode.UNMETERED_ONLY, "Unmetered connections only", "Refresh automatically on connections Android considers unmetered, usually Wi-Fi."),
  SettingsChoice(ContentRefreshMode.MANUAL, "Manual", "Refresh only when you choose Refresh now.")
)

@Composable
internal fun SettingsNavigationRow(title: String, summary: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
  Row(modifier.fillMaxWidth().heightIn(min = 72.dp).clickable(role = Role.Button, onClick = onClick)
    .padding(vertical = 14.dp), verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(16.dp)) {
    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
      Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
      Text(summary, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    Icon(Icons.AutoMirrored.Outlined.KeyboardArrowRight, contentDescription = null,
      tint = MaterialTheme.colorScheme.onSurfaceVariant)
  }
}

@Composable
internal fun SettingSwitch(title: String, description: String, checked: Boolean, modifier: Modifier = Modifier, onChange: (Boolean) -> Unit) {
  Row(modifier.fillMaxWidth().heightIn(min = 72.dp)
    .toggleable(checked, role = Role.Switch, onValueChange = onChange).padding(vertical = 14.dp),
    verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
      Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
      Text(description, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    Switch(checked, onCheckedChange = null)
  }
}

@Composable
internal fun AppUpdatePreferences(options: AppSettings, onChange: (AppSettings) -> Unit) {
  var chooseMode by rememberSaveable { mutableStateOf(false) }
  var chooseNetwork by rememberSaveable { mutableStateOf(false) }
  val selected = appUpdateChoices.first { it.value == options.appUpdateMode }
  SettingsNavigationRow("Update mode", selected.title, Modifier.testTag("app-update-mode")) { chooseMode = true }
  Text(selected.description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
  if (options.appUpdateMode == AppUpdateMode.AUTOMATIC) {
    SettingsNavigationRow("Automatic downloads", if (options.appUpdatesUnmeteredOnly) "Unmetered connections" else "Any connection",
      Modifier.testTag("app-update-network")) { chooseNetwork = true }
    Text("Manual downloads work on any connection.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
  }
  if (chooseMode) SettingsChoiceDialog("App updates", appUpdateChoices, options.appUpdateMode,
    onDismiss = { chooseMode = false }, onSelect = { onChange(options.withAppUpdateMode(it)); chooseMode = false })
  if (chooseNetwork && options.appUpdateMode == AppUpdateMode.AUTOMATIC) SettingsChoiceDialog(
    "Automatic downloads",
    listOf(
      SettingsChoice(true, "Unmetered connections", "Usually Wi-Fi. Android decides whether a connection is metered."),
      SettingsChoice(false, "Any connection", "May use mobile data or a metered Wi-Fi connection.")
    ), options.appUpdatesUnmeteredOnly, onDismiss = { chooseNetwork = false },
    onSelect = { onChange(options.copy(appUpdatesUnmeteredOnly = it)); chooseNetwork = false }
  )
}

@Composable
internal fun ContentUpdatePreferences(options: AppSettings, onChange: (AppSettings) -> Unit) {
  var choosing by rememberSaveable { mutableStateOf(false) }
  SettingsNavigationRow("Content refresh", contentRefreshChoices.first { it.value == options.contentRefreshMode }.title,
    Modifier.testTag("content-refresh-mode")) { choosing = true }
  if (choosing) SettingsChoiceDialog("Content refresh", contentRefreshChoices, options.contentRefreshMode,
    onDismiss = { choosing = false }, onSelect = { onChange(options.withContentRefreshMode(it)); choosing = false })
}

@Composable
internal fun <T> SettingsChoiceDialog(title: String, choices: List<SettingsChoice<T>>, selected: T, onDismiss: () -> Unit, onSelect: (T) -> Unit) {
  AlertDialog(onDismissRequest = onDismiss, title = { Text(title) }, text = {
    Column(Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState()).selectableGroup()) {
      choices.forEach { choice ->
        Row(Modifier.fillMaxWidth().heightIn(min = 64.dp).testTag("choice-${choice.value}")
          .selectable(selected = selected == choice.value, role = Role.RadioButton, onClick = { onSelect(choice.value) })
          .padding(vertical = 12.dp), verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(12.dp)) {
          RadioButton(selected = selected == choice.value, onClick = null)
          Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(choice.title, fontWeight = FontWeight.SemiBold)
            Text(choice.description, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
          }
        }
      }
    }
  }, confirmButton = { TextButton(onClick = onDismiss) { Text("Cancel") } })
}
