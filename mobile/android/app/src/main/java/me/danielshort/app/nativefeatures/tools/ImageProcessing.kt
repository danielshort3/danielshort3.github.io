package me.danielshort.app.nativefeatures.tools

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.google.android.gms.common.ConnectionResult
import com.google.android.gms.common.GoogleApiAvailability
import com.google.android.gms.common.moduleinstall.ModuleInstall
import com.google.android.gms.common.moduleinstall.ModuleInstallRequest
import com.google.android.gms.tasks.Task
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions
import com.google.zxing.BarcodeFormat
import com.google.zxing.EncodeHintType
import com.google.zxing.common.BitMatrix
import com.google.zxing.qrcode.QRCodeWriter
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel
import java.io.ByteArrayOutputStream
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout

data class NativeImage(val bitmap: Bitmap, val originalWidth: Int, val originalHeight: Int)
data class ImageExport(val bitmap: Bitmap, val bytes: ByteArray, val mimeType: String, val extension: String)

fun loadNativeImage(context: Context, uri: Uri): NativeImage {
  val resolver = context.contentResolver
  val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
  resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
  require(options.outWidth > 0 && options.outHeight > 0) { "Choose a supported photograph or image." }
  require(options.outWidth.toLong() * options.outHeight <= 250_000_000) { "This image is too large. Choose an image under 250 megapixels." }
  val sourceWidth = options.outWidth
  val sourceHeight = options.outHeight
  var sample = 1
  // Decode just above the target, then resize exactly. Rounding the sample up would
  // reduce a 3,000 px photo to 1,500 px even when the user requests 2,048 px.
  while (maxOf(sourceWidth, sourceHeight) / (sample * 2L) >= 2048) sample *= 2
  options.inJustDecodeBounds = false
  options.inSampleSize = sample
  options.inPreferredConfig = Bitmap.Config.ARGB_8888
  val decoded = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
    ?: throw IllegalArgumentException("This image could not be opened.")
  val orientation = runCatching {
    resolver.openInputStream(uri)?.use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
  }.getOrNull() ?: ExifInterface.ORIENTATION_NORMAL
  val matrix = Matrix().apply {
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> setScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> setRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> setScale(1f, -1f)
      ExifInterface.ORIENTATION_TRANSPOSE -> { setRotate(90f); postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_90 -> setRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> { setRotate(-90f); postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_270 -> setRotate(-90f)
    }
  }
  val (boundedWidth, boundedHeight) = scaledImageSize(decoded.width, decoded.height, 2048)
  val bounded = Bitmap.createScaledBitmap(decoded, boundedWidth, boundedHeight, true)
  if (bounded !== decoded) decoded.recycle()
  val upright = if (matrix.isIdentity) bounded else Bitmap.createBitmap(bounded, 0, 0, bounded.width, bounded.height, matrix, true)
  if (upright !== bounded) bounded.recycle()
  val swap = orientation in setOf(ExifInterface.ORIENTATION_TRANSPOSE, ExifInterface.ORIENTATION_ROTATE_90, ExifInterface.ORIENTATION_TRANSVERSE, ExifInterface.ORIENTATION_ROTATE_270)
  return NativeImage(upright, if (swap) sourceHeight else sourceWidth, if (swap) sourceWidth else sourceHeight)
}

@Suppress("DEPRECATION")
fun optimizeNativeImage(source: Bitmap, maxSide: Int, quality: Int, format: String): ImageExport {
  require(format in setOf("JPEG", "PNG", "WebP")) { "Choose JPEG, PNG, or WebP." }
  val (width, height) = scaledImageSize(source.width, source.height, maxSide)
  val resized = Bitmap.createScaledBitmap(source, width, height, true)
  val bitmap = if (format == "JPEG") Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
    Canvas(it).apply { drawColor(Color.WHITE); drawBitmap(resized, 0f, 0f, null) }
  } else resized
  val compressFormat = when (format) { "PNG" -> Bitmap.CompressFormat.PNG; "WebP" -> Bitmap.CompressFormat.WEBP; else -> Bitmap.CompressFormat.JPEG }
  val mime = when (format) { "PNG" -> "image/png"; "WebP" -> "image/webp"; else -> "image/jpeg" }
  val extension = when (format) { "PNG" -> "png"; "WebP" -> "webp"; else -> "jpg" }
  val bytes = ByteArrayOutputStream().use { buffer ->
    check(bitmap.compress(compressFormat, quality.coerceIn(1, 100), buffer)) { "The image could not be exported." }
    buffer.toByteArray()
  }
  val preview = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: bitmap
  if (resized !== source && resized !== bitmap && resized !== preview) resized.recycle()
  if (bitmap !== source && bitmap !== preview) bitmap.recycle()
  return ImageExport(preview, bytes, mime, extension)
}

fun encodeQrMatrix(payload: String, size: Int = 768): BitMatrix {
  require(payload.isNotEmpty() && payload.length <= 1500) { "Enter up to 1,500 characters." }
  return QRCodeWriter().encode(payload, BarcodeFormat.QR_CODE, size, size, mapOf(
    EncodeHintType.CHARACTER_SET to "UTF-8", EncodeHintType.ERROR_CORRECTION to ErrorCorrectionLevel.M,
    EncodeHintType.MARGIN to 4
  ))
}

fun generateNativeQr(payload: String): ImageExport {
  val matrix = encodeQrMatrix(payload)
  val pixels = IntArray(matrix.width * matrix.height) { index -> if (matrix[index % matrix.width, index / matrix.width]) Color.rgb(11, 34, 60) else Color.WHITE }
  val bitmap = Bitmap.createBitmap(pixels, matrix.width, matrix.height, Bitmap.Config.ARGB_8888)
  return optimizeNativeImage(bitmap, matrix.width, 100, "PNG")
}

private suspend fun <T> Task<T>.awaitResult(): T = suspendCancellableCoroutine { continuation ->
  addOnSuccessListener { if (continuation.isActive) continuation.resume(it) }
  addOnFailureListener { if (continuation.isActive) continuation.resumeWithException(it) }
  addOnCanceledListener { continuation.cancel() }
}

class OnDeviceBackgroundRemover(private val context: Context) : AutoCloseable {
  private val segmenter = SubjectSegmentation.getClient(SubjectSegmenterOptions.Builder().enableForegroundBitmap().build())

  suspend fun remove(bitmap: Bitmap, onStatus: (String) -> Unit): ImageExport {
    require(GoogleApiAvailability.getInstance().isGooglePlayServicesAvailable(context) == ConnectionResult.SUCCESS) {
      "Update Google Play services to use on-device background removal."
    }
    val modules = ModuleInstall.getClient(context)
    if (!modules.areModulesAvailable(segmenter).awaitResult().areModulesAvailable()) {
      onStatus("Downloading the on-device model. Your photo stays on this device.")
      modules.installModules(ModuleInstallRequest.newBuilder().addApi(segmenter).build()).awaitResult()
      withTimeout(120_000) {
        while (!modules.areModulesAvailable(segmenter).awaitResult().areModulesAvailable()) delay(800)
      }
    }
    onStatus("Separating the subject…")
    val result = segmenter.process(InputImage.fromBitmap(bitmap, 0)).awaitResult()
    val foreground = result.foregroundBitmap ?: throw IllegalStateException("No foreground was found. Try a photo with a clearer subject.")
    return withContext(Dispatchers.Default) { optimizeNativeImage(foreground, 2048, 100, "PNG") }
  }

  override fun close() { segmenter.close() }
}
