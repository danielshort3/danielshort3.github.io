package me.danielshort.app.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag

@Composable
internal fun BrowseOverflowMenu(refreshing: Boolean, onRefresh: () -> Unit) {
  var expanded by remember { mutableStateOf(false) }
  Box {
    IconButton(onClick = { expanded = true }) { Icon(Icons.Outlined.MoreVert, "More options") }
    DropdownMenu(expanded, onDismissRequest = { expanded = false }) {
      DropdownMenuItem(text = { Text(if (refreshing) "Refreshing…" else "Refresh website content") },
        leadingIcon = { Icon(Icons.Outlined.Refresh, contentDescription = null) }, enabled = !refreshing,
        onClick = { expanded = false; onRefresh() })
    }
  }
}

@Composable
internal fun SavedProjectsMenu(savedCount: Int, onRemoveAll: () -> Unit) {
  var expanded by remember { mutableStateOf(false) }
  var confirming by rememberSaveable { mutableStateOf(false) }
  Box {
    IconButton(onClick = { expanded = true }, enabled = savedCount > 0) {
      Icon(Icons.Outlined.MoreVert, "Saved project options")
    }
    DropdownMenu(expanded, onDismissRequest = { expanded = false }) {
      DropdownMenuItem(text = { Text("Remove all saved projects") }, enabled = savedCount > 0,
        onClick = { expanded = false; confirming = true })
    }
  }
  if (confirming) AlertDialog(onDismissRequest = { confirming = false },
    title = { Text("Remove all saved projects?") },
    text = { Text("This removes your $savedCount bookmarks. Project content, game progress, and generated files are kept.") },
    confirmButton = {
      TextButton(onClick = { confirming = false; onRemoveAll() }, enabled = savedCount > 0,
        modifier = Modifier.testTag("confirm-remove-saved")) { Text("Remove all saved projects") }
    }, dismissButton = { TextButton(onClick = { confirming = false }) { Text("Cancel") } })
}
