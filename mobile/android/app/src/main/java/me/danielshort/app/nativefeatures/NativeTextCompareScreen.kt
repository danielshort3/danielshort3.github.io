package me.danielshort.app.nativefeatures

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@Composable
fun NativeTextCompareScreen(onBack: () -> Unit) {
  var before by rememberSaveable { mutableStateOf("") }
  var after by rememberSaveable { mutableStateOf("") }
  var requested by rememberSaveable { mutableStateOf(false) }
  var result by remember { mutableStateOf<TextDiff?>(null) }
  var copied by remember { mutableStateOf(false) }
  val clipboard = LocalClipboardManager.current
  val focusManager = LocalFocusManager.current
  val usingExample = before.isEmpty() && after.isEmpty()
  LaunchedEffect(before, after, requested) {
    result = null
    if (requested) {
      val (left, right) = comparisonInputs(before, after)
      result = withContext(Dispatchers.Default) { compareText(left, right) }
    }
  }

  Column(
    Modifier.fillMaxSize()
      .background(MaterialTheme.colorScheme.background)
      .safeDrawingPadding()
      .imePadding()
      .verticalScroll(rememberScrollState())
      .padding(horizontal = 20.dp),
    verticalArrangement = Arrangement.spacedBy(16.dp)
  ) {
    NativeFeatureHeader("Text Compare", "Spot changes between two pieces of text.", onBack)
    OutlinedTextField(
      value = before,
      onValueChange = {
        before = it.take(MAX_COMPARE_LENGTH)
        requested = false
        result = null
        copied = false
      },
      label = { Text("Before") },
      placeholder = { Text(if (usingExample) EXAMPLE_BEFORE else "Enter text") },
      minLines = 3,
      maxLines = 7,
      modifier = Modifier.fillMaxWidth()
    )
    OutlinedTextField(
      value = after,
      onValueChange = {
        after = it.take(MAX_COMPARE_LENGTH)
        requested = false
        result = null
        copied = false
      },
      label = { Text("After") },
      placeholder = { Text(if (usingExample) EXAMPLE_AFTER else "Enter text") },
      minLines = 3,
      maxLines = 7,
      modifier = Modifier.fillMaxWidth()
    )
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(onClick = {
        focusManager.clearFocus()
        copied = false
        requested = true
      }) { Text("Compare") }
      TextButton(onClick = {
        before = ""
        after = ""
        requested = false
        result = null
        copied = false
        focusManager.clearFocus()
      }, enabled = before.isNotEmpty() || after.isNotEmpty() || requested) { Text("Clear") }
    }
    if (usingExample && !requested) {
      Text("Compare the example, or enter your own text.", style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    if (before.length == MAX_COMPARE_LENGTH || after.length == MAX_COMPARE_LENGTH) {
      Text("Each field supports up to 50,000 characters.", style = MaterialTheme.typography.bodySmall)
    }
    if (requested) {
      val currentResult = result
      if (currentResult == null) CircularProgressIndicator(Modifier.size(24.dp))
      else {
        HorizontalDivider()
        Text(
          if (!currentResult.hasChanges) "No differences" else if (usingExample) "Example comparison" else "Comparison",
          style = MaterialTheme.typography.titleMedium,
          modifier = Modifier.semantics { heading(); liveRegion = LiveRegionMode.Polite }
        )
        if (currentResult.hasChanges) {
          Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Added", color = Color(0xFF12613A), style = MaterialTheme.typography.labelMedium)
            Text("Removed", color = Color(0xFF93430A), style = MaterialTheme.typography.labelMedium,
              textDecoration = TextDecoration.LineThrough)
          }
        }
        SelectionContainer {
          Surface(color = MaterialTheme.colorScheme.surfaceContainerLow,
            shape = MaterialTheme.shapes.medium, modifier = Modifier.fillMaxWidth()) {
            Text(
              text = buildAnnotatedString {
                currentResult.spans.forEach { span ->
                  val style = when (span.kind) {
                    DiffKind.Added -> SpanStyle(color = Color(0xFF12613A), background = Color(0xFFE2F3E8),
                      textDecoration = TextDecoration.Underline)
                    DiffKind.Removed -> SpanStyle(color = Color(0xFF93430A), background = Color(0xFFFFEDD9),
                      textDecoration = TextDecoration.LineThrough)
                    DiffKind.Unchanged -> SpanStyle()
                  }
                  withStyle(style) { append(span.text) }
                }
              },
              modifier = Modifier.padding(16.dp).semantics {
                contentDescription = currentResult.spans.joinToString(" ") { span ->
                  when (span.kind) {
                    DiffKind.Added -> "Added: ${span.text}."
                    DiffKind.Removed -> "Removed: ${span.text}."
                    DiffKind.Unchanged -> span.text
                  }
                }
              },
              style = MaterialTheme.typography.bodyLarge
            )
          }
        }
        if (currentResult.simplified) Text("Large changes are grouped into passages.",
          style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        TextButton(onClick = {
          clipboard.setText(AnnotatedString(comparisonInputs(before, after).second))
          copied = true
        }) {
          Icon(Icons.Default.ContentCopy, contentDescription = null, modifier = Modifier.size(18.dp))
          Text(if (copied) "  Copied" else "  Copy After", modifier = Modifier.semantics {
            liveRegion = LiveRegionMode.Polite
          })
        }
      }
    }
    Spacer(Modifier.height(16.dp))
  }
}

@Composable
internal fun NativeFeatureHeader(title: String, subtitle: String, onBack: () -> Unit) {
  Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
    IconButton(onClick = onBack) {
      Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
    }
    Text(title, style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold,
      modifier = Modifier.semantics { heading() })
    Text(subtitle, style = MaterialTheme.typography.bodyMedium,
      color = MaterialTheme.colorScheme.onSurfaceVariant)
    HorizontalDivider(Modifier.padding(top = 8.dp), thickness = 2.dp, color = MaterialTheme.colorScheme.primary)
  }
}
