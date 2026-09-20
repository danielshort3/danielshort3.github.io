package me.danielshort.app.updates

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class UpdateTransferFailure(message: String, val notFound: Boolean = false, cause: Throwable? = null) : IOException(message, cause)

interface UpdateTransport {
  suspend fun manifest(url: String): String
  suspend fun download(artifact: UpdateArtifact, destination: File, progress: (Long, Long) -> Unit)
}

/** Every redirect, full response length and wall-clock transfer deadline is checked. */
class HttpsUpdateTransport : UpdateTransport {
  override suspend fun manifest(url: String): String = withContext(Dispatchers.IO) {
    val output = java.io.ByteArrayOutputStream()
    transfer(UpdateUrlPolicy.requireFeedUrl(url), true, MAX_MANIFEST_BYTES, null) { bytes, size -> output.write(bytes, 0, size) }
    output.toString(Charsets.UTF_8.name())
  }

  override suspend fun download(artifact: UpdateArtifact, destination: File, progress: (Long, Long) -> Unit) = withContext(Dispatchers.IO) {
    require(artifact.size in 1..MAX_UPDATE_BYTES)
    try {
      destination.outputStream().buffered(64 * 1024).use { output ->
        var transferred = 0L
        transfer(UpdateUrlPolicy.requirePublishedUrl(artifact.url), false, artifact.size, artifact.size) { bytes, size ->
          output.write(bytes, 0, size)
          transferred += size
          progress(transferred, artifact.size)
        }
      }
    } catch (failure: Throwable) {
      destination.delete()
      throw failure
    }
  }

  private suspend fun transfer(startedAt: URI, feed: Boolean, maximumBytes: Long, expectedBytes: Long?, consume: (ByteArray, Int) -> Unit) {
    val coroutine = currentCoroutineContext()
    val deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(if (feed) 30 else 180)
    var endpoint = startedAt
    var connection: HttpURLConnection? = null
    val activeConnection = AtomicReference<HttpURLConnection?>(null)
    val expired = AtomicBoolean(false)
    val timer = Executors.newSingleThreadScheduledExecutor { action -> Thread(action, "app-update-deadline").apply { isDaemon = true } }
    // Disconnecting also bounds a stalled body or TLS exchange, instead of relying on read timeout alone.
    val guard = timer.scheduleAtFixedRate({
      if (System.nanoTime() >= deadlineNanos) expired.set(true)
      if (expired.get() || !coroutine.isActive) activeConnection.get()?.disconnect()
    }, 100, 100, TimeUnit.MILLISECONDS)
    try {
      var redirects = 0
      while (true) {
        coroutine.ensureActive()
        if (expired.get() || System.nanoTime() >= deadlineNanos) throw UpdateTransferFailure("The update download timed out. Please try again.")
        connection = endpoint.toURL().openConnection() as HttpURLConnection
        activeConnection.set(connection)
        val active = connection
        active.instanceFollowRedirects = false
        active.connectTimeout = 12_000
        active.readTimeout = 15_000
        active.useCaches = false
        active.setRequestProperty("Accept", if (feed) "application/json" else "application/octet-stream")
        active.setRequestProperty("Accept-Encoding", "identity")
        active.setRequestProperty("User-Agent", "DanielShort-Android-Updater/1")
        val response = active.responseCode
        if (response in setOf(301, 302, 303, 307, 308)) {
          if (++redirects > 5) throw UpdateTransferFailure("The update server redirected too many times.")
          val location = active.getHeaderField("Location") ?: throw UpdateTransferFailure("The update download address was missing.")
          endpoint = UpdateUrlPolicy.requireRedirect(endpoint, location, startedAt, feed)
          active.disconnect()
          continue
        }
        if (response != 200) throw UpdateTransferFailure(if (feed && response == 404) "App updates have not been published yet. Please check again later." else "The update server could not be reached. Please try again.", notFound = response == 404)
        if (active.contentEncoding?.let { it != "identity" } == true) throw UpdateFailure("The update server returned an unsupported download encoding.")
        val length = active.contentLengthLong
        if (length > maximumBytes || (expectedBytes != null && length >= 0 && length != expectedBytes)) throw UpdateFailure("The update download size did not match the published release.")
        var received = 0L
        active.inputStream.use { input ->
          val buffer = ByteArray(64 * 1024)
          while (true) {
            coroutine.ensureActive()
            if (expired.get() || System.nanoTime() >= deadlineNanos) throw UpdateTransferFailure("The update download timed out. Please try again.")
            val count = input.read(buffer)
            if (count < 0) break
            if (count.toLong() > maximumBytes - received) throw UpdateFailure("The update download exceeded its published size.")
            received += count
            consume(buffer, count)
          }
        }
        if (expectedBytes != null && received != expectedBytes) throw UpdateTransferFailure("The update download was interrupted. Please try again.")
        coroutine.ensureActive()
        if (expired.get()) throw UpdateTransferFailure("The update download timed out. Please try again.")
        return
      }
    } catch (failure: CancellationException) {
      throw failure
    } catch (failure: IOException) {
      coroutine.ensureActive()
      if (failure is UpdateTransferFailure) throw failure
      throw UpdateTransferFailure(if (expired.get()) "The update download timed out. Please try again." else "The update download was interrupted. Check your connection and try again.", cause = failure)
    } finally {
      guard.cancel(true)
      timer.shutdownNow()
      connection?.disconnect()
      activeConnection.set(null)
    }
  }
}
