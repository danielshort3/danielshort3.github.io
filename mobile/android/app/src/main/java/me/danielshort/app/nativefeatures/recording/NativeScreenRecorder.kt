package me.danielshort.app.nativefeatures.recording

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.ClipData
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.SystemClock
import android.provider.DocumentsContract
import android.widget.MediaController
import android.widget.VideoView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import me.danielshort.app.nativefeatures.NativeFeatureHeader
import java.io.File

@Composable
fun NativeScreenRecorder(onBack: () -> Unit) {
  val context = LocalContext.current
  val state by ScreenRecording.state.collectAsStateWithLifecycle()
  val scope = rememberCoroutineScope()
  var microphone by rememberSaveable { mutableStateOf(false) }
  var notice by rememberSaveable { mutableStateOf("") }
  var elapsed by remember { mutableLongStateOf(0L) }
  var exporting by remember { mutableStateOf(false) }
  var choosingClip by remember { mutableStateOf(false) }
  var confirmDelete by remember { mutableStateOf(false) }
  var selectedPath by rememberSaveable { mutableStateOf<String?>(null) }
  var exportPath by rememberSaveable { mutableStateOf<String?>(null) }
  val lifecycle = LocalLifecycleOwner.current.lifecycle
  var preview by remember { mutableStateOf<VideoView?>(null) }
  DisposableEffect(lifecycle) {
    val observer = LifecycleEventObserver { _, event -> if (event == Lifecycle.Event.ON_STOP) preview?.pause() }
    lifecycle.addObserver(observer)
    onDispose { lifecycle.removeObserver(observer); preview?.stopPlayback() }
  }
  val selectedFile = state.clips.find { it.absolutePath == selectedPath } ?: state.file
  LaunchedEffect(Unit) { ScreenRecording.restore(context) }
  LaunchedEffect(state.active) { if (state.active) selectedPath = null }
  val consent = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
    val data = result.data
    if (result.resultCode == Activity.RESULT_OK && data != null) {
      runCatching { ContextCompat.startForegroundService(context, Intent(context, ScreenRecordingService::class.java)
        .putExtra("consent", data).putExtra("microphone", microphone)) }
        .onFailure { notice = "Couldn’t start recording. Please try again." }
    } else notice = "Recording cancelled"
  }
  val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
    if (granted) consent.launch(context.getSystemService(MediaProjectionManager::class.java).createScreenCaptureIntent())
    else notice = "Microphone access was declined. Turn Microphone off to record video only."
  }
  val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("video/mp4")) { uri ->
    // Keep the requested clip stable even if the system recreates the activity in the picker.
    val file = exportPath?.let(::File)?.takeIf {
      it.parentFile == File(context.filesDir, "native-recordings") && it.isFile && it.extension == "mp4"
    }
    exportPath = null
    if (uri != null && file != null) scope.launch {
      exporting = true
      notice = try {
        withContext(Dispatchers.IO) {
          context.contentResolver.openOutputStream(uri)?.use { target ->
            file.inputStream().use { source ->
              val buffer = ByteArray(64 * 1024)
              while (true) {
                currentCoroutineContext().ensureActive()
                val count = source.read(buffer)
                if (count == -1) break
                target.write(buffer, 0, count)
              }
            }
          } ?: error("No output")
        }
        "Recording saved"
      } catch (error: Exception) {
        withContext(NonCancellable + Dispatchers.IO) {
          runCatching { DocumentsContract.deleteDocument(context.contentResolver, uri) }
        }
        if (error is CancellationException) throw error
        "Couldn’t save the recording. Please try another location."
      }
      finally { exporting = false }
    } else if (uri != null) scope.launch {
      withContext(Dispatchers.IO) { runCatching { DocumentsContract.deleteDocument(context.contentResolver, uri) } }
      notice = "This recording is no longer available. Choose another clip."
    }
  }
  LaunchedEffect(state.active, state.startedAt) {
    while (state.active) { elapsed = (SystemClock.elapsedRealtime() - state.startedAt) / 1000; delay(500) }
  }
  Column(Modifier.fillMaxSize().safeDrawingPadding().imePadding().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
    NativeFeatureHeader("Screen recorder", "Capture a screen or app and save an MP4.", onBack)
    Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).toggleable(microphone, enabled = !state.active, role = Role.Switch, onValueChange = { microphone = it }), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
      Text("Microphone")
      Switch(microphone, onCheckedChange = null, enabled = !state.active)
    }
    if (state.active) {
      Text("Recording · ${elapsed / 60}:${(elapsed % 60).toString().padStart(2, '0')}", color = MaterialTheme.colorScheme.error)
      Button(onClick = { context.startService(Intent(context, ScreenRecordingService::class.java).setAction("STOP")) }, Modifier.fillMaxWidth()) { Text("Stop recording") }
    } else Button(onClick = {
      notice = ""
      if (microphone && ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) permission.launch(Manifest.permission.RECORD_AUDIO)
      else consent.launch(context.getSystemService(MediaProjectionManager::class.java).createScreenCaptureIntent())
    }, Modifier.fillMaxWidth(), enabled = !exporting) { Text("Start recording") }
    if (!state.active && selectedFile != null) {
      if (state.clips.size > 1) Box {
        TextButton(onClick = { choosingClip = true }, enabled = !exporting) { Text("Recordings (${state.clips.size})") }
        DropdownMenu(choosingClip, onDismissRequest = { choosingClip = false }) {
          state.clips.forEach { clip -> DropdownMenuItem(text = { Text(java.text.DateFormat.getDateTimeInstance().format(java.util.Date(clip.lastModified()))) }, onClick = { selectedPath = clip.absolutePath; choosingClip = false }) }
        }
      }
      key(selectedFile.absolutePath) {
        AndroidView(factory = { viewContext -> VideoView(viewContext).apply {
          preview = this
          setVideoPath(selectedFile.absolutePath)
          setMediaController(MediaController(viewContext).also { it.setAnchorView(this) })
          setOnPreparedListener { seekTo(1) }
        } }, modifier = Modifier.fillMaxWidth().height(300.dp), onRelease = { it.stopPlayback(); if (preview === it) preview = null })
      }
      Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Button(onClick = { exportPath = selectedFile.absolutePath; export.launch(selectedFile.name) }, enabled = !exporting) { Text(if (exporting) "Saving…" else "Save video") }
        OutlinedButton(onClick = {
          val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", selectedFile)
          runCatching { context.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "video/mp4"; putExtra(Intent.EXTRA_STREAM, uri)
            clipData = ClipData.newRawUri("Screen recording", uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
          }, "Share recording")) }.onFailure { notice = "No sharing app is available." }
        }) { Text("Share") }
      }
      TextButton(onClick = { confirmDelete = true }, enabled = !exporting) { Text("Delete recording") }
    }
    val message = notice.ifBlank { state.message }
    if (message.isNotBlank()) Text(message, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite }, color = MaterialTheme.colorScheme.onSurfaceVariant)
    Text("Recordings stay in the app until you delete them. Save video exports a copy. Microphone is optional; device audio isn’t captured. Up to 15 minutes per clip.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    Spacer(Modifier.height(12.dp))
  }
  if (confirmDelete && selectedFile != null) AlertDialog(onDismissRequest = { confirmDelete = false }, title = { Text("Delete this recording?") }, text = { Text("This removes the copy inside the app. Exported copies are kept.") }, confirmButton = { TextButton(onClick = {
    preview?.stopPlayback()
    if (selectedFile.delete()) { ScreenRecording.restore(context); selectedPath = null; notice = "Recording deleted" }
    else notice = "Couldn’t delete this recording. Please try again."
    confirmDelete = false
  }) { Text("Delete") } }, dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } })
}
