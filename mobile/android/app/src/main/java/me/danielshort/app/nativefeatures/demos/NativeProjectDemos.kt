@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package me.danielshort.app.nativefeatures.demos

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Paint
import android.util.Base64
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import me.danielshort.app.nativefeatures.NativeFeatureHeader
import me.danielshort.app.R
import me.danielshort.app.ui.ScrollChromeLayout
import me.danielshort.app.ui.LocalChromeInputFocus
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.Locale
import kotlin.random.Random

val NATIVE_DEMO_IDS = setOf("handwritingRating", "shapeClassifier", "digitGenerator", "smartSentence", "chatbotLora", "babynames", "covidAnalysis", "retailStore", "targetEmptyPackage", "nonogram", "pizza", "pizzaDashboard", "ufoDashboard")

@Composable
fun NativeProjectDemoScreen(projectId: String, onBack: () -> Unit) {
  when (projectId) {
    "handwritingRating", "shapeClassifier" -> DrawingDemo(projectId == "shapeClassifier", onBack)
    "digitGenerator" -> DigitDemo(onBack)
    "smartSentence" -> LanguageDemo(false, onBack)
    "chatbotLora" -> LanguageDemo(true, onBack)
    "nonogram" -> NonogramDemo(onBack)
    "pizza" -> PizzaTipsDemo(onBack)
    else -> NativeDashboardDemo(projectId, onBack)
  }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
internal fun DemoPage(title: String, description: String, onBack: () -> Unit, content: @Composable ColumnScope.() -> Unit) {
  ScrollChromeLayout(screenKey = title, topBar = {
    Column(Modifier.background(MaterialTheme.colorScheme.surface)) {
      TopAppBar(title = {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
          Image(painterResource(R.drawable.brand_mark), contentDescription = null, modifier = Modifier.size(30.dp))
          Text("Daniel Short", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        }
      }, navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back to project") } },
        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.surface))
      HorizontalDivider(thickness = 2.dp, color = MaterialTheme.colorScheme.primary.copy(alpha = .35f))
    }
  }) { padding ->
    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)
      .verticalScroll(rememberScrollState()).padding(start = 20.dp, end = 20.dp,
        top = padding.calculateTopPadding() + 20.dp, bottom = padding.calculateBottomPadding() + 20.dp),
      verticalArrangement = Arrangement.spacedBy(16.dp)) {
      NativeFeatureHeader(title, description, onBack, showBack = false)
      content()
    }
  }
}

@Composable
internal fun DemoStatus(busy: Boolean, status: String, onRetry: (() -> Unit)? = null) {
  Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    if (busy) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
    Text(status, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    if (onRetry != null && !busy) TextButton(onClick = onRetry) { Text("Retry") }
  }
}

/** A scroll inside an editor must not be interpreted as reading further down the page. */
@Composable
internal fun Modifier.protectChromeWhileEditing(): Modifier {
  val onFocus = LocalChromeInputFocus.current
  var focused by remember { mutableStateOf(false) }
  DisposableEffect(onFocus) {
    onDispose { if (focused) onFocus(false) }
  }
  return onFocusChanged {
    focused = it.isFocused
    onFocus(it.isFocused)
  }
}

@Composable
internal fun DemoSelect(label: String, options: List<String>, selected: String, onSelect: (String) -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
  var expanded by remember { mutableStateOf(false) }
  Box(modifier) {
    OutlinedButton(onClick = { expanded = true }, enabled = enabled, modifier = Modifier.fillMaxWidth()) { Text("$label: $selected") }
    DropdownMenu(expanded, onDismissRequest = { expanded = false }, modifier = Modifier.heightIn(max = 360.dp)) {
      options.forEach { option -> DropdownMenuItem(text = { Text(option) }, onClick = { onSelect(option); expanded = false }) }
    }
  }
}

@Composable
internal fun ScoreRow(label: String, value: Double, caption: String = percent(value)) {
  Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
    Row(Modifier.fillMaxWidth()) { Text(label, Modifier.weight(1f)); Text(caption, fontWeight = FontWeight.SemiBold) }
    LinearProgressIndicator(progress = { value.toFloat().coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth().height(5.dp))
  }
}

internal fun percent(value: Double) = String.format(Locale.US, "%.1f%%", value * 100)

@Composable
private fun DrawingDemo(shape: Boolean, onBack: () -> Unit) {
  val strokes = remember { mutableStateListOf<List<Offset>>() }
  var sample by remember { mutableStateOf<Bitmap?>(null) }
  var predictions by remember { mutableStateOf<List<Prediction>>(emptyList()) }
  var busy by remember { mutableStateOf(false) }
  var status by remember { mutableStateOf("AWS · Checking connection") }
  val scope = rememberCoroutineScope()
  val service = if (shape) "shape" else "handwriting"
  suspend fun health() {
    try { DemoApi.json("$SITE/api/demos/$service/health"); status = "AWS · Connected" }
    catch (e: CancellationException) { throw e }
    catch (_: Exception) { status = "AWS · Unavailable. You can retry when connected." }
  }
  LaunchedEffect(service) { health() }
  fun score() {
    busy = true; predictions = emptyList(); status = "AWS · Reading your drawing"
    val snapshot = strokes.toList()
    val original = sample
    scope.launch {
      try {
        val b64 = withContext(Dispatchers.Default) { encodeDrawing(snapshot, original) }
        predictions = parsePredictions(DemoApi.json("$SITE/api/demos/$service/${if (shape) "predict" else "score"}", JSONObject().put("b64", b64)), shape)
        status = "AWS · Connected"
      } catch (e: CancellationException) { throw e }
      catch (e: Exception) { status = e.message ?: "Could not read the drawing. Please retry." }
      finally { busy = false }
    }
  }
  DemoPage(if (shape) "Shape Classifier" else "Handwriting Rating",
    if (shape) "Draw a circle, triangle, square, hexagon, or octagon." else "Draw one digit to see how confidently the model reads it.", onBack) {
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally,
      verticalArrangement = Arrangement.spacedBy(14.dp)) {
    Row(Modifier.widthIn(max = 320.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
      horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      if (busy) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
      Text(status, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
      TextButton(onClick = { strokes.clear(); sample = null; predictions = emptyList() },
        enabled = !busy && (strokes.isNotEmpty() || sample != null)) { Text("Clear") }
    }
    Box(Modifier.widthIn(max = 320.dp).fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(12.dp)).background(Color.Black)) {
      sample?.let { Image(it.asImageBitmap(), null, Modifier.fillMaxSize()) }
      Canvas(Modifier.fillMaxSize().semantics { contentDescription = if (shape) "Black shape drawing canvas" else "Black digit drawing canvas" }
        .pointerInput(busy) {
          if (!busy) detectDragGestures(onDragStart = { p ->
            predictions = emptyList()
            strokes.add(listOf(Offset((p.x / size.width).coerceIn(0f, 1f), (p.y / size.height).coerceIn(0f, 1f))))
          }, onDrag = { change, _ ->
            change.consume()
            if (strokes.isNotEmpty()) strokes[strokes.lastIndex] = strokes.last() + Offset((change.position.x / size.width).coerceIn(0f, 1f), (change.position.y / size.height).coerceIn(0f, 1f))
          })
        }) {
        strokes.forEach { points ->
          points.zipWithNext().forEach { (a, b) -> drawLine(Color.White, Offset(a.x * size.width, a.y * size.height), Offset(b.x * size.width, b.y * size.height), size.width * .04f, StrokeCap.Round) }
          points.firstOrNull()?.let { drawCircle(Color.White, size.width * .02f, Offset(it.x * size.width, it.y * size.height)) }
        }
      }
    }
      Text("Your drawing is sent to the website’s AWS model only when you submit it.",
        modifier = Modifier.widthIn(max = 320.dp), style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant)
      Button(onClick = ::score, enabled = !busy && (strokes.isNotEmpty() || sample != null),
        modifier = Modifier.widthIn(min = 168.dp).heightIn(min = 48.dp), shape = RoundedCornerShape(10.dp)) {
        Text(if (shape) "Classify shape" else "Rate digit")
      }
    }
    if (!shape) {
      Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
      Text("Try a handwriting sample", style = MaterialTheme.typography.labelLarge)
      (0..9).toList().chunked(5).forEach { row ->
      Row(Modifier.widthIn(max = 360.dp).fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        row.forEach { digit ->
          OutlinedButton(onClick = {
            busy = true; predictions = emptyList(); status = "Loading sample $digit"
            scope.launch {
              try {
                val bytes = DemoApi.bytes("$SITE/img/demos/handwriting-samples/digit-$digit.jpg")
                sample = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: error("Sample image is unavailable.")
                strokes.clear(); status = "Sample $digit loaded"
              } catch (e: CancellationException) { throw e }
              catch (e: Exception) { status = e.message ?: "Could not load sample." }
              finally { busy = false }
            }
          }, enabled = !busy, modifier = Modifier.weight(1f).height(48.dp), shape = RoundedCornerShape(10.dp),
            contentPadding = PaddingValues(0.dp)) { Text(digit.toString()) }
        }
      }
      }
      }
    }
    predictions.firstOrNull()?.let { top ->
      Text(top.label.replaceFirstChar { it.uppercase() }, style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
      Text("${percent(top.confidence)} model confidence", color = MaterialTheme.colorScheme.onSurfaceVariant)
      predictions.forEach { ScoreRow(it.label, it.confidence) }
    }
  }
}

internal fun encodeDrawing(strokes: List<List<Offset>>, sample: Bitmap?): String {
  val bitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
  val canvas = android.graphics.Canvas(bitmap)
  canvas.drawColor(android.graphics.Color.BLACK)
  if (sample != null) canvas.drawBitmap(sample, null, android.graphics.Rect(0, 0, 256, 256), Paint(Paint.ANTI_ALIAS_FLAG))
  val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = android.graphics.Color.WHITE; strokeWidth = 10.24f; strokeCap = Paint.Cap.ROUND }
  strokes.forEach { points ->
    points.firstOrNull()?.let { canvas.drawCircle(it.x * 256, it.y * 256, 5.12f, paint) }
    points.zipWithNext().forEach { (a, b) -> canvas.drawLine(a.x * 256, a.y * 256, b.x * 256, b.y * 256, paint) }
  }
  return ByteArrayOutputStream().use { output -> bitmap.compress(Bitmap.CompressFormat.PNG, 100, output); bitmap.recycle(); Base64.encodeToString(output.toByteArray(), Base64.NO_WRAP) }
}

@Composable
private fun DigitDemo(onBack: () -> Unit) {
  var digit by rememberSaveable { mutableStateOf("Auto") }
  var grid by rememberSaveable { mutableStateOf(6) }
  var advanced by rememberSaveable { mutableStateOf(false) }
  var seed by rememberSaveable { mutableStateOf("") }
  var dimension by rememberSaveable { mutableStateOf(0) }
  var distortion by rememberSaveable { mutableStateOf(5f) }
  var imageRows by remember { mutableStateOf<List<List<Bitmap>>>(emptyList()) }
  var busy by remember { mutableStateOf(false) }
  var status by remember { mutableStateOf("AWS · Checking connection") }
  val scope = rememberCoroutineScope()
  fun generate() {
    busy = true; status = "AWS · Generating"
    scope.launch {
      try {
        val payload = digitRequest(digit.toIntOrNull(), grid, seed.toLongOrNull() ?: Random.nextLong(2_147_483_647), dimension, distortion)
        val response = DemoApi.json("$SITE/api/demos/digit-generator/generate", payload)
        imageRows = withContext(Dispatchers.Default) {
          val array = response.getJSONArray("images")
          val encoded = (0 until array.length()).flatMap { index ->
            val row = array.optJSONArray(index)
            if (row == null) listOf(array.getString(index)) else row.strings()
          }
          require(encoded.size == grid * grid) { "The model returned an incomplete grid. Please retry." }
          encoded.map { val bytes = Base64.decode(it, Base64.DEFAULT); BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: error("Invalid digit image.") }.chunked(grid)
        }
        status = "AWS · Connected"
      } catch (e: CancellationException) { throw e }
      catch (e: Exception) { status = e.message ?: "Could not generate digits. Please retry." }
      finally { busy = false }
    }
  }
  LaunchedEffect(Unit) {
    try { DemoApi.json("$SITE/api/demos/digit-generator/health"); generate() }
    catch (e: CancellationException) { throw e }
    catch (_: Exception) { status = "AWS · Unavailable. Tap Regenerate to retry." }
  }
  DemoPage("Digit Generator", "Explore new handwritten digits generated by the VAE model.", onBack) {
    Surface(shape = RoundedCornerShape(16.dp), border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)) {
      Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(status, Modifier.align(Alignment.End), style = MaterialTheme.typography.bodySmall)
        Text("Digit settings are sent to the website’s AWS model. The first grid is generated automatically.",
          style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
          DemoSelect("Digit", listOf("Auto") + (0..9).map(Int::toString), digit, { digit = it }, Modifier.weight(1f), !busy)
          Button(onClick = ::generate, enabled = !busy) { Text("Regenerate") }
        }
      }
    }
    if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
    if (imageRows.isEmpty()) Box(Modifier.fillMaxWidth().aspectRatio(1f).background(Color.Black, RoundedCornerShape(14.dp)), contentAlignment = Alignment.Center) {
      Text(if (busy) "Generating digits…" else "Your digit collection appears here.", color = Color.White)
    } else Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Color.Black).padding(5.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
      imageRows.forEach { row -> Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        row.forEach { bitmap -> Image(bitmap.asImageBitmap(), "Generated digit", Modifier.weight(1f).aspectRatio(1f)) }
      } }
    }
    TextButton(onClick = { advanced = !advanced }) { Text(if (advanced) "Hide advanced settings" else "Advanced settings") }
    if (advanced) {
      DemoSelect("Grid", (2..8).map { "$it × $it" }, "$grid × $grid", { grid = it.substringBefore(' ').toInt() }, enabled = !busy)
      OutlinedTextField(seed, { seed = it.filter(Char::isDigit).take(10).takeIf { value -> (value.toLongOrNull() ?: 0L) <= 2_147_483_647 } ?: seed }, label = { Text("Seed (optional)") }, singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number), modifier = Modifier.fillMaxWidth().protectChromeWhileEditing(), enabled = !busy)
      DemoSelect("Latent dimension", (0..19).map(Int::toString), dimension.toString(), { dimension = it.toInt() }, enabled = !busy)
      Text("Distortion: ${String.format(Locale.US, "%.1f", distortion)}")
      Slider(distortion, { distortion = it }, valueRange = 0f..20f, enabled = !busy)
    }
  }
}

@Composable
private fun LanguageDemo(chat: Boolean, onBack: () -> Unit) {
  var query by rememberSaveable { mutableStateOf("") }
  var top by rememberSaveable { mutableStateOf(5) }
  var busy by remember { mutableStateOf(false) }
  var status by remember { mutableStateOf("AWS · Ready to connect") }
  var results by remember { mutableStateOf<List<Pair<String, Double>>>(emptyList()) }
  var answer by rememberSaveable { mutableStateOf("") }
  var answerQuery by rememberSaveable { mutableStateOf("") }
  var sources by remember { mutableStateOf<List<String>>(emptyList()) }
  val scope = rememberCoroutineScope()
  DemoPage(if (chat) "Grand Junction Travel Chat" else "Smart Sentence Retriever",
    if (chat) "Ask the website’s Bedrock assistant about Grand Junction." else "Find related sentences in Alice in Wonderland.", onBack) {
    DemoStatus(busy, status)
    OutlinedTextField(query, { query = it.take(1200) }, label = { Text(if (chat) "Your question" else "Search phrase") },
      placeholder = { Text(if (chat) "What can I do in Grand Junction?" else "She wonders about things.") }, minLines = 3, maxLines = 8,
      modifier = Modifier.fillMaxWidth().protectChromeWhileEditing(), enabled = !busy)
    if (!chat) DemoSelect("Results", listOf("3", "5", "10", "20"), top.toString(), { top = it.toInt() }, enabled = !busy)
    Text("${if (chat) "Questions" else "Search phrases"} are sent to the website’s AWS service. ${if (chat) "AI answers can be mistaken; verify current details." else "Similarity scores describe related wording, not certainty."}",
      style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    Button(onClick = {
      val submitted = query.trim().ifBlank { if (chat) "What can I do in Grand Junction?" else "She wonders about things." }
      busy = true; results = emptyList(); answer = ""; sources = emptyList(); status = "AWS · ${if (chat) "Thinking" else "Searching"}"
      scope.launch {
        try {
          if (chat) {
            val accepted = DemoApi.json("$CHAT/submit", JSONObject().put("prompt", submitted))
            var response = accepted
            val jobId = accepted.optString("jobId")
            if (!accepted.optString("status").equals("READY", true)) {
              require(jobId.isNotBlank()) { "The assistant is not available. Please try again." }
              var count = 0
              while (!response.optString("status").equals("READY", true) && count++ < 80) {
                require(!response.optString("status").equals("FAILED", true)) { "The assistant could not finish. Please retry." }
                delay(1500)
                response = DemoApi.json("$CHAT/result?jobId=${java.net.URLEncoder.encode(jobId, "UTF-8")}")
              }
              require(response.optString("status").equals("READY", true)) { "The assistant took too long. Please retry." }
            }
            val data = response.optJSONObject("data") ?: response
            answer = data.optString("generated_text").ifBlank { data.optString("answer") }.also { require(it.isNotBlank()) { "No answer was returned." } }
            sources = data.optJSONArray("sources")?.let { list -> (0 until list.length()).mapNotNull { index ->
              val item = list.opt(index); if (item is String) item else (item as? JSONObject)?.optString("url")
            } } ?: emptyList()
            answerQuery = submitted
          } else {
            val data = DemoApi.json("$SITE/api/demos/smart-sentence/rank", JSONObject().put("query", submitted).put("top", top))
            results = data.getJSONArray("top").objects().map { it.getString("sentence") to it.getDouble("score").coerceIn(0.0, 1.0) }
          }
          status = "AWS · Connected"
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) {
          status = if (!chat && e is DemoServiceException && e.statusCode == 504) {
            "AWS is still preparing. Wait about a minute, then tap Search again."
          } else e.message ?: "The request failed. Please retry."
        }
        finally { busy = false }
      }
    }, enabled = !busy) { Text(if (chat) "Ask" else "Search") }
    if (answer.isNotBlank()) {
      Text(answerQuery, fontWeight = FontWeight.SemiBold)
      SelectionContainer { Text(answer) }
      if (sources.isNotEmpty()) Text("Sources\n" + sources.joinToString("\n"), style = MaterialTheme.typography.bodySmall)
    }
    results.forEachIndexed { index, item ->
      Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        ScoreRow("Result ${index + 1}", item.second, "Similarity ${percent(item.second)}")
        SelectionContainer { Text(item.first) }
        HorizontalDivider()
      }
    }
  }
}

@Composable
private fun NonogramDemo(onBack: () -> Unit) {
  val lifecycleOwner = LocalLifecycleOwner.current
  var puzzle by remember { mutableStateOf<JSONObject?>(null) }
  var visibleSteps by remember { mutableStateOf(0) }
  var showSolution by remember { mutableStateOf(false) }
  var busy by remember { mutableStateOf(false) }
  var playing by remember { mutableStateOf(false) }
  var status by remember { mutableStateOf("Load a puzzle to watch the trained agent solve it.") }
  val scope = rememberCoroutineScope()
  val steps = puzzle?.optJSONArray("steps")?.objects() ?: emptyList()
  LaunchedEffect(playing, lifecycleOwner) {
    if (playing) lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
      while (playing && visibleSteps < steps.size) { delay(180); visibleSteps++ }
      playing = false
    }
  }
  DemoPage("Nonogram Solver", "Follow the real reinforcement-learning agent’s decisions.", onBack) {
    DemoStatus(busy, status)
    Text("Loads a puzzle from the website’s AWS service.", style = MaterialTheme.typography.bodySmall,
      color = MaterialTheme.colorScheme.onSurfaceVariant)
    Button(onClick = {
      busy = true; playing = false; status = "AWS · Loading puzzle"
      scope.launch {
        try {
          val data = DemoApi.json("$SITE/api/demos/nonogram/solve", JSONObject())
          require(data.getInt("grid") in 2..10 && data.getJSONArray("steps").length() <= 2000)
          puzzle = data; visibleSteps = 0; showSolution = false; status = "AWS · Connected"
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { status = e.message ?: "Could not load a puzzle. Please retry." }
        finally { busy = false }
      }
    }, enabled = !busy) { Text("New puzzle") }
    puzzle?.let { data ->
      val size = data.getInt("grid")
      val rows = data.getJSONArray("row_clues")
      val cols = data.getJSONArray("col_clues")
      val current = Array(size) { IntArray(size) { -1 } }
      val mistakes = mutableSetOf<Pair<Int, Int>>()
      steps.take(if (showSolution) steps.size else visibleSteps).forEach { step ->
        val r = step.optInt("row", -1); val c = step.optInt("col", -1)
        if (r in 0 until size && c in 0 until size) { current[r][c] = step.optInt("actual", 0); if (!showSolution && !nonogramPredictionMatches(step)) mistakes.add(r to c) }
      }
      Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Row { Spacer(Modifier.width(44.dp)); (0 until size).forEach { c -> Text((0 until cols.getJSONArray(c).length()).joinToString("\n") { cols.getJSONArray(c).getInt(it).toString() }, Modifier.weight(1f), textAlign = androidx.compose.ui.text.style.TextAlign.Center) } }
        (0 until size).forEach { r -> Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(3.dp)) {
          Text((0 until rows.getJSONArray(r).length()).joinToString(" ") { rows.getJSONArray(r).getInt(it).toString() }, Modifier.width(41.dp), style = MaterialTheme.typography.bodySmall)
          (0 until size).forEach { c -> Box(Modifier.weight(1f).aspectRatio(1f).background(if (current[r][c] == 1) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant).border(if (r to c in mistakes) 2.dp else 1.dp, if (r to c in mistakes) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.outlineVariant), contentAlignment = Alignment.Center) {
            if (current[r][c] == 0) Text("×")
          } }
        } }
      }
      FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = { if (visibleSteps >= steps.size) visibleSteps = 0; showSolution = false; playing = !playing }, enabled = !busy) { Text(if (playing) "Pause" else "Watch AI solve") }
        TextButton(onClick = { playing = false; showSolution = !showSolution }) { Text(if (showSolution) "Hide solution" else "Show solution") }
      }
      Text("Step $visibleSteps of ${steps.size}")
      if (visibleSteps > 0) {
        val shown = steps.take(visibleSteps)
        Text("Prediction accuracy: ${shown.count(::nonogramPredictionMatches)} / ${shown.size}")
        val latest = shown.last()
        Text("Row ${latest.optInt("row") + 1}, column ${latest.optInt("col") + 1}: ${if (latest.optInt("predicted") == 1) "fill" else "empty"}. ${if (nonogramPredictionMatches(latest)) "Correct" else "Incorrect"}.")
      }
      Text("Red borders mark incorrect decisions. Cells show the puzzle’s actual answer, matching the website’s replay.", style = MaterialTheme.typography.bodySmall)
    }
  }
}

internal fun nonogramPredictionMatches(step: JSONObject): Boolean {
  val predicted = step.optInt("predicted", -1)
  val actual = step.optInt("actual", -1)
  return predicted in 0..1 && actual in 0..1 && predicted == actual
}
