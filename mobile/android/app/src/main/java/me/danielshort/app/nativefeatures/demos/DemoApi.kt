package me.danielshort.app.nativefeatures.demos

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import kotlin.coroutines.coroutineContext

internal const val SITE = "https://www.danielshort.me"
internal const val CHAT = "https://k8bys9gicf.execute-api.us-east-2.amazonaws.com/prod/bedrock"

internal class DemoServiceException(val statusCode: Int, message: String) : IllegalStateException(message)

/** The public website's JSON contracts; never downloads or executes web code. */
internal object DemoApi {
  suspend fun json(url: String, body: JSONObject? = null, maxBytes: Int = 4 * 1024 * 1024): JSONObject =
    JSONObject(String(bytes(url, body, maxBytes), Charsets.UTF_8))

  suspend fun bytes(url: String, body: JSONObject? = null, maxBytes: Int = 4 * 1024 * 1024): ByteArray = withContext(Dispatchers.IO) {
    val uri = URI(url)
    require(uri.scheme == "https" && uri.host in setOf("www.danielshort.me", "k8bys9gicf.execute-api.us-east-2.amazonaws.com"))
    val connection = URL(url).openConnection() as HttpURLConnection
    try {
      connection.connectTimeout = 15_000
      connection.readTimeout = if (body == null) 30_000 else 125_000
      connection.instanceFollowRedirects = false
      connection.setRequestProperty("Accept", "application/json")
      if (body != null) {
        val encoded = body.toString().toByteArray(Charsets.UTF_8)
        require(encoded.size <= 500_000) { "Input is too large." }
        connection.requestMethod = "POST"
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/json")
        connection.setFixedLengthStreamingMode(encoded.size)
        connection.outputStream.use { it.write(encoded) }
      }
      val status = connection.responseCode
      if (status !in 200..299) {
        throw DemoServiceException(status, when (status) {
          429 -> "Please wait a moment before trying again."
          401, 403 -> "The demo service did not accept this request."
          502, 503, 504 -> "AWS is unavailable or still preparing. Please try again."
          else -> "The demo could not load (HTTP $status). Please retry."
        })
      }
      connection.inputStream.use { input ->
        val output = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(8192)
        while (true) {
          coroutineContext.ensureActive()
          val count = input.read(buffer)
          if (count < 0) break
          require(output.size() + count <= maxBytes) { "The demo response is too large." }
          output.write(buffer, 0, count)
        }
        output.toByteArray()
      }
    } finally { connection.disconnect() }
  }
}

internal fun JSONArray.objects(): List<JSONObject> = (0 until length()).mapNotNull { optJSONObject(it) }
internal fun JSONArray.strings(): List<String> = (0 until length()).map { getString(it) }
internal fun JSONObject.firstArray(vararg keys: String): JSONArray? = keys.firstNotNullOfOrNull { optJSONArray(it) }

internal data class Prediction(val label: String, val confidence: Double)

internal fun parsePredictions(raw: JSONObject, shape: Boolean): List<Prediction> {
  val payload = raw.optJSONObject("body") ?: raw.optJSONObject("result") ?: raw
  val keys = if (shape) listOf("shape_confidences", "shape_scores", "shapeScores", "confidences", "probabilities", "probs", "scores")
    else listOf("digit_confidences", "digitConfidences", "confidences", "per_digit_confidence", "probabilities", "scores", "per_digit")
  val values = keys.firstNotNullOfOrNull { payload.opt(it).takeUnless { v -> v == null || v == JSONObject.NULL } }
  val labels = if (shape) listOf("circle", "triangle", "square", "hexagon", "octagon") else (0..9).map(Int::toString)
  val pairs = when (values) {
    is JSONObject -> values.keys().asSequence().map { it to values.optDouble(it, Double.NaN) }.toList()
    is JSONArray -> (0 until values.length()).map { index ->
      val item = values.optJSONObject(index)
      if (item == null) labels.getOrElse(index) { index.toString() } to values.optDouble(index, Double.NaN)
      else (item.optString("digit").ifBlank { item.optString("class").ifBlank { item.optString("label").ifBlank { item.optString("shape") } } }) to
        listOf("confidence", "score", "prob", "probability").firstNotNullOfOrNull { key -> item.optDouble(key, Double.NaN).takeIf(Double::isFinite) }.orNaN()
    }
    else -> emptyList()
  }
  if (pairs.isNotEmpty()) {
    require(pairs.all { it.first in labels && it.second.isFinite() && it.second in 0.0..100.0 }) { "The model returned invalid confidence values." }
    require(shape || pairs.map { it.first }.toSet().size == 10) { "The model did not return all digit scores." }
    val scale = if (pairs.any { it.second > 1 }) 100.0 else 1.0
    return pairs.map { Prediction(it.first, it.second / scale) }.sortedByDescending { it.confidence }
  }
  val label = listOf("class", "cls", "label", "shape", "prediction").firstNotNullOfOrNull { payload.optString(it).takeIf(String::isNotBlank) }
  val confidence = payload.optDouble("confidence", payload.optDouble("conf", Double.NaN))
  require(shape && label in labels && confidence.isFinite() && confidence in 0.0..100.0) { "The model did not return a prediction." }
  return listOf(Prediction(label!!, if (confidence > 1) confidence / 100 else confidence))
}

private fun Double?.orNaN() = this ?: Double.NaN

internal fun digitRequest(digit: Int?, grid: Int, seed: Long, dimension: Int, distortion: Float): JSONObject {
  require(digit == null || digit in 0..9)
  require(grid in 2..8 && seed in 0..2_147_483_647 && dimension in 0..19 && distortion.isFinite() && distortion in 0f..20f)
  return JSONObject().put("seed", seed).put("mode", if (digit == null) "random" else "cluster")
    .put("cluster_digit", digit ?: JSONObject.NULL).put("dim", dimension).put("value", distortion.toDouble()).put("rows", grid).put("cols", grid)
}
