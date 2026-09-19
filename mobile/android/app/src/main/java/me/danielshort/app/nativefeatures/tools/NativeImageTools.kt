package me.danielshort.app.nativefeatures.tools

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

@Composable
internal fun ImagePreview(bitmap: android.graphics.Bitmap, description: String, checkerboard: Boolean = false) {
  BoxWithConstraints(Modifier.fillMaxWidth()) {
    val aspect = (bitmap.width.toFloat() / bitmap.height).coerceIn(.8f, 2.5f)
    val previewHeight = (maxWidth / aspect).coerceAtMost(340.dp)
    Box(Modifier.fillMaxWidth().height(previewHeight).clipToBounds().background(Color.White), contentAlignment = Alignment.Center) {
      if (checkerboard) Canvas(Modifier.matchParentSize()) {
        val cell = 12.dp.toPx()
        for (x in 0..(size.width / cell).toInt()) for (y in 0..(size.height / cell).toInt()) {
          drawRect(if ((x + y) % 2 == 0) Color(0xFFE4E9EE) else Color.White, Offset(x * cell, y * cell), Size(cell, cell))
        }
      }
      Image(bitmap.asImageBitmap(), description, Modifier.matchParentSize(), contentScale = ContentScale.Fit)
    }
  }
}

@Composable
private fun SaveImage(output: ImageExport, fileName: String) {
  val context = LocalContext.current
  val scope = rememberCoroutineScope()
  var status by remember(output) { mutableStateOf("") }
  var error by remember(output) { mutableStateOf(false) }
  var saving by remember { mutableStateOf(false) }
  val save = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument(output.mimeType)) { uri ->
    if (uri != null) scope.launch {
      saving = true; error = false
      try {
        withContext(Dispatchers.IO) {
          val stream = context.contentResolver.openOutputStream(uri, "wt") ?: throw IllegalStateException("Could not open the selected location.")
          stream.use { it.write(output.bytes) }
        }
        status = "Saved to the location you selected."
      } catch (cancelled: CancellationException) { throw cancelled }
      catch (_: Exception) { status = "Could not save this file. Choose another location and try again."; error = true }
      finally { saving = false }
    }
  }
  Button(enabled = !saving, onClick = { save.launch("$fileName.${output.extension}") }) { Text(if (saving) "Saving…" else "Save ${output.extension.uppercase(Locale.ROOT)}") }
  ToolStatus(status, error)
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun NativeQrScreen(onBack: () -> Unit) {
  var mode by rememberSaveable { mutableStateOf("Text or link") }
  var input by rememberSaveable { mutableStateOf("") }
  var ssid by rememberSaveable { mutableStateOf("") }
  // Wi-Fi passwords are deliberately not put in saved instance state.
  var password by remember { mutableStateOf("") }
  var open by rememberSaveable { mutableStateOf(false) }
  var hidden by rememberSaveable { mutableStateOf(false) }
  var output by remember { mutableStateOf<ImageExport?>(null) }
  var busy by remember { mutableStateOf(false) }
  var error by remember { mutableStateOf("") }
  val scope = rememberCoroutineScope()
  val focus = LocalFocusManager.current
  fun invalidate() { output = null; error = "" }
  ToolPage("QR Code Generator", "Create a scannable code on your device.", onBack) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      listOf("Text or link", "Wi-Fi").forEach { label -> FilterChip(mode == label, { mode = label; invalidate() }, label = { Text(label) }, enabled = !busy) }
    }
    if (mode == "Wi-Fi") {
      OutlinedTextField(ssid, { ssid = it.take(100); invalidate() }, label = { Text("Network name") }, enabled = !busy, modifier = Modifier.fillMaxWidth())
      if (!open) OutlinedTextField(password, { password = it.take(200); invalidate() }, label = { Text("Password") },
        visualTransformation = PasswordVisualTransformation(), keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, autoCorrectEnabled = false),
        singleLine = true, enabled = !busy, modifier = Modifier.fillMaxWidth())
      ToolToggle("Open network (no password)", open, { open = it; invalidate() }, !busy)
      ToolToggle("Hidden network", hidden, { hidden = it; invalidate() }, !busy)
    } else OutlinedTextField(input, { input = it.take(1500); invalidate() }, label = { Text("Text or link") }, minLines = 3, maxLines = 6,
      enabled = !busy, modifier = Modifier.fillMaxWidth())
    Button(enabled = !busy && if (mode == "Wi-Fi") ssid.isNotBlank() else input.isNotBlank(), onClick = {
      focus.clearFocus(); error = ""; busy = true
      scope.launch {
        try { output = withContext(Dispatchers.Default) { generateNativeQr(if (mode == "Wi-Fi") wifiQrPayload(ssid, password, open, hidden) else input) } }
        catch (cancelled: CancellationException) { throw cancelled }
        catch (problem: IllegalArgumentException) { error = problem.message ?: "Check the QR code contents." }
        catch (_: Exception) { error = "This content is too long for a QR code. Shorten it and try again." }
        finally { busy = false }
      }
    }) { Text(if (busy) "Creating…" else "Create QR code") }
    ToolStatus(error, true)
    output?.let {
      ImagePreview(it.bitmap, "Generated QR code")
      SaveImage(it, "qr-code")
    }
  }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
internal fun NativeImageScreen(removeBackground: Boolean, onBack: () -> Unit) {
  val context = LocalContext.current
  val scope = rememberCoroutineScope()
  val remover = remember(removeBackground) { if (removeBackground) OnDeviceBackgroundRemover(context.applicationContext) else null }
  DisposableEffect(remover) { onDispose { remover?.close() } }
  var image by remember { mutableStateOf<NativeImage?>(null) }
  var output by remember { mutableStateOf<ImageExport?>(null) }
  var busy by remember { mutableStateOf(false) }
  var status by remember { mutableStateOf("") }
  var error by remember { mutableStateOf(false) }
  var format by rememberSaveable { mutableStateOf("JPEG") }
  var quality by rememberSaveable { mutableStateOf(85f) }
  var maxSide by rememberSaveable { mutableStateOf(1600f) }
  fun invalidate() { output = null; status = ""; error = false }
  fun load(uri: Uri) {
    scope.launch {
      busy = true; invalidate(); image = null
      try { image = withContext(Dispatchers.IO) { loadNativeImage(context, uri) } }
      catch (cancelled: CancellationException) { throw cancelled }
      catch (problem: Exception) { status = problem.message ?: "This image could not be opened."; error = true }
      finally { busy = false }
    }
  }
  val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> if (uri != null) load(uri) }
  ToolPage(if (removeBackground) "Background Remover" else "Image Optimizer",
    if (removeBackground) "Keep the subject and export a transparent PNG." else "Resize and compress an image on your device.", onBack) {
    OutlinedButton(enabled = !busy, onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }) {
      Text(if (image == null) "Choose image" else "Change image")
    }
    image?.let { source ->
      Text("${source.originalWidth} × ${source.originalHeight}", style = MaterialTheme.typography.bodyMedium)
      if (output == null) ImagePreview(source.bitmap, "Selected image", checkerboard = source.bitmap.hasAlpha())
      if (!removeBackground) {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          listOf("JPEG", "PNG", "WebP").forEach { label -> FilterChip(format == label, { format = label; invalidate() }, label = { Text(label) }, enabled = !busy) }
        }
        Text("Maximum edge: ${maxSide.toInt()} px", fontWeight = FontWeight.Medium)
        Slider(maxSide, { maxSide = it; invalidate() }, valueRange = 320f..2048f, enabled = !busy)
        if (format != "PNG") {
          Text("Quality: ${quality.toInt()}%", fontWeight = FontWeight.Medium)
          Slider(quality, { quality = it; invalidate() }, valueRange = 10f..100f, enabled = !busy)
        }
      }
      ToolStatus("Processes images up to 2,048 px on the longest edge.")
      Button(enabled = !busy, onClick = {
        busy = true; invalidate()
        scope.launch {
          try {
            output = if (removeBackground) remover!!.remove(source.bitmap) { status = it }
            else withContext(Dispatchers.Default) { optimizeNativeImage(source.bitmap, maxSide.toInt(), quality.toInt(), format) }
            status = ""
          } catch (_: TimeoutCancellationException) { status = "The model is still downloading. Keep an internet connection and try again shortly."; error = true }
          catch (cancelled: CancellationException) { throw cancelled }
          catch (problem: Exception) { status = if (removeBackground) "Background removal could not finish. Check Google Play services and your first-use internet connection, then retry." else problem.message ?: "This image could not be processed."; error = true }
          finally { busy = false }
        }
      }) { Text(if (busy) "Processing…" else if (removeBackground) "Remove background" else "Optimize image") }
    }
    if (busy) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) { CircularProgressIndicator() }
    ToolStatus(status, error)
    output?.let { result ->
      ImagePreview(result.bitmap, if (removeBackground) "Subject with transparent background" else "Optimized image", checkerboard = removeBackground || format != "JPEG")
      Text("${result.bitmap.width} × ${result.bitmap.height} · ${String.format(Locale.ROOT, "%.1f", result.bytes.size / 1024.0)} KB",
        style = MaterialTheme.typography.bodyMedium)
      SaveImage(result, if (removeBackground) "transparent-subject" else "optimized-image")
    }
    if (removeBackground && image == null) ToolStatus("Google Play services downloads the model on first use. Photos are processed on your device.")
  }
}
