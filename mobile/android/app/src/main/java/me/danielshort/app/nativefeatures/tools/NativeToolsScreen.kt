package me.danielshort.app.nativefeatures.tools

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import me.danielshort.app.nativefeatures.NativeFeatureHeader

val NATIVE_TOOL_IDS = setOf("nbsp-cleaner", "word-frequency", "point-of-view-checker", "oxford-comma-checker", "utm-batch-builder", "qr-code-generator", "image-optimizer", "background-remover")

@Composable
fun NativeToolsScreen(toolId: String, onBack: () -> Unit) {
  key(toolId) {
    when (toolId) {
      "utm-batch-builder" -> UtmScreen(onBack)
      "qr-code-generator" -> NativeQrScreen(onBack)
      "image-optimizer", "background-remover" -> NativeImageScreen(toolId == "background-remover", onBack)
      else -> TextUtilityScreen(toolId, onBack)
    }
  }
}

@Composable
internal fun ToolPage(title: String, subtitle: String, onBack: () -> Unit, body: @Composable ColumnScope.() -> Unit) {
  Column(
    Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).safeDrawingPadding().imePadding()
      .verticalScroll(rememberScrollState()).padding(horizontal = 20.dp),
    verticalArrangement = Arrangement.spacedBy(14.dp)
  ) {
    NativeFeatureHeader(title, subtitle, onBack)
    body()
    Spacer(Modifier.height(24.dp))
  }
}

@Composable
internal fun ToolToggle(label: String, value: Boolean, onChange: (Boolean) -> Unit, enabled: Boolean = true) {
  Row(Modifier.fillMaxWidth().toggleable(value, enabled = enabled, role = Role.Checkbox, onValueChange = onChange), verticalAlignment = Alignment.CenterVertically) {
    Checkbox(checked = value, onCheckedChange = null, enabled = enabled)
    Text(label, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
  }
}

@Composable
internal fun CopyResult(text: String, label: String = "Copy result") {
  val clipboard = LocalClipboardManager.current
  var copied by remember(text) { mutableStateOf(false) }
  TextButton(onClick = { clipboard.setText(AnnotatedString(text)); copied = true }) { Text(if (copied) "Copied" else label) }
}

@Composable
internal fun ToolStatus(message: String, error: Boolean = false) {
  if (message.isNotEmpty()) Text(message, color = if (error) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
    style = MaterialTheme.typography.bodySmall, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
}

private data class TextResult(val summary: String, val text: AnnotatedString, val copyText: String)

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TextUtilityScreen(id: String, onBack: () -> Unit) {
  var input by rememberSaveable { mutableStateOf("") }
  var option by rememberSaveable { mutableStateOf(id != "point-of-view-checker") }
  var secondOption by rememberSaveable { mutableStateOf(false) }
  var phraseLength by rememberSaveable { mutableStateOf(1) }
  var result by remember { mutableStateOf<TextResult?>(null) }
  var busy by remember { mutableStateOf(false) }
  val scope = rememberCoroutineScope()
  val focus = LocalFocusManager.current
  val title = when (id) {
    "nbsp-cleaner" -> "NBSP Cleaner"
    "word-frequency" -> "Word Frequency"
    "point-of-view-checker" -> "Point of View Checker"
    else -> "Oxford Comma Checker"
  }
  val subtitle = when (id) {
    "nbsp-cleaner" -> "Replace hard spaces while keeping your text intact."
    "word-frequency" -> "Find the words and phrases that appear most often."
    "point-of-view-checker" -> "Highlight first, second, and third person references."
    else -> "Find candidate lists and review their final comma."
  }
  ToolPage(title, subtitle, onBack) {
    OutlinedTextField(input, { input = it.take(MAX_TOOL_TEXT); result = null }, label = { Text("Text") }, minLines = 5, maxLines = 9,
      enabled = !busy, modifier = Modifier.fillMaxWidth())
    when (id) {
      "nbsp-cleaner" -> {
        ToolToggle("Replace non-breaking and hard spaces", option, { option = it; result = null }, !busy)
        ToolToggle("Remove non-ASCII characters", secondOption, { secondOption = it; result = null }, !busy)
        if (secondOption) ToolStatus("Removes accented letters, emoji, and other non-ASCII characters.")
      }
      "word-frequency" -> {
        ToolToggle("Skip common English words", option, { option = it; result = null }, !busy)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          listOf("Words", "2-word phrases", "3-word phrases").forEachIndexed { index, label ->
            FilterChip(selected = phraseLength == index + 1, onClick = { phraseLength = index + 1; result = null }, label = { Text(label) }, enabled = !busy)
          }
        }
      }
      "point-of-view-checker" -> ToolToggle("Include it, its, and itself", option, { option = it; result = null }, !busy)
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Button(enabled = input.isNotBlank() && !busy, onClick = {
        focus.clearFocus()
        busy = true
        scope.launch {
          try {
            result = withContext(Dispatchers.Default) { analyzeText(id, input, option, secondOption, phraseLength) }
          } finally { busy = false }
        }
      }) { Text(if (busy) "Working…" else if (id == "nbsp-cleaner") "Clean text" else if (id == "word-frequency") "Analyze text" else "Check text") }
      TextButton(enabled = !busy && input.isNotEmpty(), onClick = { input = ""; result = null }) { Text("Clear") }
    }
    if (id == "oxford-comma-checker" || id == "point-of-view-checker") ToolStatus("Pattern-based suggestions; review the surrounding context.")
    result?.let { output ->
      HorizontalDivider(color = MaterialTheme.colorScheme.primary, thickness = 2.dp)
      Text(output.summary, fontWeight = FontWeight.SemiBold, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
      SelectionContainer { Text(output.text, style = MaterialTheme.typography.bodyLarge) }
      CopyResult(output.copyText)
    }
  }
}

private fun analyzeText(id: String, text: String, option: Boolean, secondOption: Boolean, phraseLength: Int): TextResult = when (id) {
  "nbsp-cleaner" -> cleanSpaces(text, option, secondOption).let {
    TextResult("${it.replacedSpaces} spaces replaced · ${it.removedCharacters} characters removed", AnnotatedString(it.text), it.text)
  }
  "word-frequency" -> wordFrequency(text, option, phraseLength = phraseLength).let {
    val output = it.words.take(50).joinToString("\n") { word -> "${word.count}  ${word.word}" }.ifEmpty { "No matching words." }
    TextResult("${it.totalWords} words · ${it.words.size} unique results" + if (it.words.size > 50) " · Top 50 shown" else "",
      AnnotatedString(output), it.words.joinToString("\n") { word -> "${word.word}\t${word.count}" })
  }
  "point-of-view-checker" -> pointOfView(text, option).let { matches ->
    val counts = (1..3).map { person -> matches.count { it.person == person } }
    val colors = listOf(Color(0xFFBBE7ED), Color(0xFFFFE6B0), Color(0xFFBCECD3))
    val annotated = buildAnnotatedString {
      append(text)
      matches.forEach { addStyle(SpanStyle(background = colors[it.person - 1], color = Color(0xFF10253F), fontWeight = FontWeight.SemiBold), it.start, it.end) }
    }
    TextResult("First ${counts[0]} · Second ${counts[1]} · Third ${counts[2]}", annotated, text)
  }
  else -> oxfordCandidates(text).let { candidates ->
    val output = candidates.joinToString("\n\n") { "${if (it.present) "Present" else "Absent"}: ${it.text}" }.ifEmpty { "No candidate lists found." }
    TextResult("${candidates.size} candidates · ${candidates.count { !it.present }} without a final comma", AnnotatedString(output), output)
  }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun UtmScreen(onBack: () -> Unit) {
  var pages by rememberSaveable { mutableStateOf("") }
  var source by rememberSaveable { mutableStateOf("") }
  var medium by rememberSaveable { mutableStateOf("") }
  var campaign by rememberSaveable { mutableStateOf("") }
  var content by rememberSaveable { mutableStateOf("") }
  var term by rememberSaveable { mutableStateOf("") }
  var advanced by rememberSaveable { mutableStateOf(false) }
  var cartesian by rememberSaveable { mutableStateOf(true) }
  var normalize by rememberSaveable { mutableStateOf(true) }
  var overrideExisting by rememberSaveable { mutableStateOf(true) }
  var output by remember { mutableStateOf<List<String>>(emptyList()) }
  var error by remember { mutableStateOf("") }
  var busy by remember { mutableStateOf(false) }
  val scope = rememberCoroutineScope()
  val focus = LocalFocusManager.current
  fun invalidate() { output = emptyList(); error = "" }
  ToolPage("UTM Batch Builder", "Build campaign links from one value or a list per field.", onBack) {
    listOf(Triple("Landing pages", pages, { value: String -> pages = value }), Triple("Source", source, { value: String -> source = value }),
      Triple("Medium", medium, { value: String -> medium = value }), Triple("Campaign", campaign, { value: String -> campaign = value })).forEach { (label, value, change) ->
      OutlinedTextField(value, { change(it.take(10_000)); invalidate() }, label = { Text(label) }, maxLines = 4, enabled = !busy, modifier = Modifier.fillMaxWidth())
    }
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      FilterChip(cartesian, { cartesian = true; invalidate() }, label = { Text("All combinations") }, enabled = !busy)
      FilterChip(!cartesian, { cartesian = false; invalidate() }, label = { Text("Match rows") }, enabled = !busy)
    }
    TextButton(onClick = { advanced = !advanced }) { Text(if (advanced) "Hide options" else "Optional parameters & formatting") }
    if (advanced) {
      OutlinedTextField(content, { content = it.take(10_000); invalidate() }, label = { Text("Content (optional)") }, maxLines = 3, enabled = !busy, modifier = Modifier.fillMaxWidth())
      OutlinedTextField(term, { term = it.take(10_000); invalidate() }, label = { Text("Term (optional)") }, maxLines = 3, enabled = !busy, modifier = Modifier.fillMaxWidth())
      ToolToggle("Lowercase with underscores for spaces", normalize, { normalize = it; invalidate() }, !busy)
      ToolToggle("Replace existing UTM parameters", overrideExisting, { overrideExisting = it; invalidate() }, !busy)
    }
    Button(enabled = !busy, onClick = {
      focus.clearFocus(); error = ""; busy = true
      scope.launch {
        try { output = withContext(Dispatchers.Default) { buildUtmLinks(pages, source, medium, campaign, content, term, cartesian, normalize, overrideExisting) } }
        catch (cancelled: CancellationException) { throw cancelled }
        catch (problem: IllegalArgumentException) { error = problem.message ?: "Check the landing pages and campaign values." }
        finally { busy = false }
      }
    }) { Text(if (busy) "Building…" else "Build links") }
    ToolStatus(error, error = true)
    if (output.isNotEmpty()) {
      Text("${output.size} links", fontWeight = FontWeight.SemiBold)
      CopyResult(output.joinToString("\n"), "Copy all links")
      SelectionContainer { Text(output.take(20).joinToString("\n\n"), style = MaterialTheme.typography.bodyMedium) }
      if (output.size > 20) ToolStatus("Showing the first 20. Copy all links to use the full batch.")
    }
  }
}
