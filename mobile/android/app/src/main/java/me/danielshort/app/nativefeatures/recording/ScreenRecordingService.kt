package me.danielshort.app.nativefeatures.recording

import android.app.*
import android.content.Intent
import android.content.Context
import android.content.pm.ServiceInfo
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.*
import android.view.WindowManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import kotlin.math.roundToInt

data class RecordingState(val active: Boolean = false, val startedAt: Long = 0, val file: File? = null, val message: String = "", val clips: List<File> = emptyList())
object ScreenRecording {
  internal val mutableState = MutableStateFlow(RecordingState())
  val state = mutableState.asStateFlow()
  fun restore(context: Context) {
    if (state.value.active) return
    val clips = RecordingFiles.completed(File(context.filesDir, "native-recordings"))
    mutableState.value = state.value.copy(file = clips.firstOrNull(), clips = clips)
  }
}

/** One consent token and one virtual display per session, including Android 14+. */
class ScreenRecordingService : Service() {
  private var projection: MediaProjection? = null
  private var display: VirtualDisplay? = null
  private var recorder: MediaRecorder? = null
  private var output: File? = null
  private var started = false
  private var stopping = false
  private val callback = object : MediaProjection.Callback() {
    override fun onStop() { finishRecording() }
  }

  override fun onBind(intent: Intent?) = null
  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    if (intent?.action == "STOP") { finishRecording(); return START_NOT_STICKY }
    if (started || stopping || intent == null) return START_NOT_STICKY
    try {
      val manager = getSystemService(NotificationManager::class.java)
      manager.createNotificationChannel(NotificationChannel("screen-recording", "Screen recording", NotificationManager.IMPORTANCE_LOW))
      val stop = PendingIntent.getService(this, 0, Intent(this, javaClass).setAction("STOP"), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
      val open = PendingIntent.getActivity(this, 1, packageManager.getLaunchIntentForPackage(packageName), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
      val notification = Notification.Builder(this, "screen-recording")
        .setSmallIcon(android.R.drawable.presence_video_online).setContentTitle("Screen recording")
        .setContentText("Tap Stop when you’re finished.").setContentIntent(open).setOngoing(true)
        .addAction(Notification.Action.Builder(null, "Stop", stop).build()).build()
      val microphone = intent.getBooleanExtra("microphone", false)
      if (Build.VERSION.SDK_INT >= 29) {
        val types = ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION or
          (if (microphone && Build.VERSION.SDK_INT >= 30) ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE else 0)
        startForeground(201, notification, types)
      } else startForeground(201, notification)
      @Suppress("DEPRECATION")
      val consent = if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra("consent", Intent::class.java) else intent.getParcelableExtra<Intent>("consent")
      requireNotNull(consent) { "Recording permission was not granted." }
      val projectionManager = getSystemService(MediaProjectionManager::class.java)
      projection = projectionManager.getMediaProjection(Activity.RESULT_OK, consent)
      projection!!.registerCallback(callback, Handler(Looper.getMainLooper()))
      val metrics = resources.displayMetrics
      val bounds = if (Build.VERSION.SDK_INT >= 30) getSystemService(WindowManager::class.java).maximumWindowMetrics.bounds else null
      val sourceWidth = bounds?.width() ?: metrics.widthPixels
      val sourceHeight = bounds?.height() ?: metrics.heightPixels
      val scale = minOf(1f, 1920f / maxOf(sourceWidth, sourceHeight))
      val width = ((sourceWidth * scale).roundToInt() / 2 * 2).coerceAtLeast(2)
      val height = ((sourceHeight * scale).roundToInt() / 2 * 2).coerceAtLeast(2)
      val directory = File(filesDir, "native-recordings").apply { mkdirs() }
      directory.listFiles()?.filter { it.extension == "part" }?.forEach { it.delete() }
      require(directory.usableSpace > 100L * 1024 * 1024) { "Not enough free space." }
      output = File(directory, "screen-${System.currentTimeMillis()}.part")
      @Suppress("DEPRECATION")
      val capture = if (Build.VERSION.SDK_INT >= 31) MediaRecorder(this) else MediaRecorder()
      recorder = capture
      if (microphone) capture.setAudioSource(MediaRecorder.AudioSource.MIC)
      capture.setVideoSource(MediaRecorder.VideoSource.SURFACE)
      capture.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
      capture.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
      capture.setVideoSize(width, height)
      capture.setVideoFrameRate(30)
      capture.setVideoEncodingBitRate(6_000_000)
      if (microphone) { capture.setAudioEncoder(MediaRecorder.AudioEncoder.AAC); capture.setAudioEncodingBitRate(128_000); capture.setAudioSamplingRate(44_100) }
      capture.setOutputFile(output!!.absolutePath)
      capture.setMaxDuration(15 * 60 * 1000)
      capture.setMaxFileSize(512L * 1024 * 1024)
      capture.setOnInfoListener { _, what, _ ->
        if (what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_DURATION_REACHED || what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_FILESIZE_REACHED) finishRecording("Recording limit reached. Your clip is ready.")
      }
      capture.setOnErrorListener { _, _, _ -> finishRecording("Recording stopped. Try a shorter clip.") }
      capture.prepare()
      display = projection!!.createVirtualDisplay("DanielShortRecording", width, height, metrics.densityDpi,
        DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, capture.surface, null, null)
      capture.start()
      started = true
      ScreenRecording.mutableState.value = ScreenRecording.state.value.copy(active = true, startedAt = SystemClock.elapsedRealtime(), message = "")
    } catch (_: Exception) {
      finishRecording("Couldn’t start recording. Grant screen access and try again.")
    }
    return START_NOT_STICKY
  }

  private fun finishRecording(message: String = "") {
    if (stopping) return
    stopping = true
    val valid = if (started) runCatching { recorder?.stop(); true }.getOrDefault(false) else false
    started = false
    runCatching { recorder?.reset() }
    runCatching { recorder?.release() }; recorder = null
    runCatching { display?.release() }; display = null
    runCatching { projection?.unregisterCallback(callback) }
    runCatching { projection?.stop() }; projection = null
    val clip = output?.let { RecordingFiles.finalize(it, valid) }
    if (clip == null) output?.delete()
    ScreenRecording.mutableState.value = ScreenRecording.state.value.copy(active = false, message = message.ifBlank { if (clip == null) "The recording was too short. Please try again." else "Recording ready" })
    ScreenRecording.restore(this)
    stopForeground(STOP_FOREGROUND_REMOVE)
    stopSelf()
  }

  override fun onDestroy() { if (!stopping) finishRecording(); super.onDestroy() }
}
