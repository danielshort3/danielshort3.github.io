package me.danielshort.wayfarers.content

import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.security.MessageDigest

interface GuildContentTransport {
  fun manifest(url: String, cancel: () -> Unit = {}): String
  fun download(archive: GuildContentArchive, target: File, progress: (Long, Long) -> Unit, cancel: () -> Unit = {})
}

/** TLS, exact first-party release URLs, bounded redirects and streaming limits apply before signature parsing. */
class GuildContentHttpsTransport(private val connectionFactory: (URL) -> HttpURLConnection = { it.openConnection() as HttpURLConnection }) : GuildContentTransport {
  override fun manifest(url: String, cancel: () -> Unit): String {
    val bytes = transfer(url, GuildContentLimits.MAX_ENVELOPE.toLong(), null, cancel) { input, _ ->
      val output = java.io.ByteArrayOutputStream()
      copy(input, output, GuildContentLimits.MAX_ENVELOPE.toLong(), cancel)
      output.toByteArray()
    }
    return GuildContentLimits.utf8(bytes)
  }

  override fun download(archive: GuildContentArchive, target: File, progress: (Long, Long) -> Unit, cancel: () -> Unit) {
    require(archive.size in 1..GuildContentLimits.MAX_ARCHIVE)
    require(target.parentFile.mkdirs() || target.parentFile.isDirectory)
    try {
      transfer(archive.url, archive.size, archive.size, cancel) { input, _ ->
        val digest = MessageDigest.getInstance("SHA-256")
        var received = 0L
        FileOutputStream(target).use { output ->
          val buffer = ByteArray(32 * 1024)
          while (true) {
            cancel()
            val count = input.read(buffer)
            if (count < 0) break
            received += count
            require(received <= archive.size)
            output.write(buffer, 0, count)
            digest.update(buffer, 0, count)
            progress(received, archive.size)
          }
          require(received == archive.size)
          val hash = digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
          require(hash == archive.sha256) { "Downloaded content checksum mismatch" }
          output.flush()
          output.fd.sync()
        }
        Unit
      }
    } catch (failure: Throwable) {
      target.delete()
      throw failure
    }
  }

  private fun <T> transfer(url: String, limit: Long, exactSize: Long?, cancel: () -> Unit, consume: (InputStream, Long) -> T): T {
    GuildContentUrlPolicy.requireRelease(url)
    var current = url
    repeat(5) { redirect ->
      cancel()
      GuildContentUrlPolicy.requireRedirect(current)
      val connection = connectionFactory(URL(current))
      connection.instanceFollowRedirects = false
      connection.connectTimeout = 15000
      connection.readTimeout = 15000
      connection.setRequestProperty("Accept-Encoding", "identity")
      connection.setRequestProperty("Cache-Control", "no-cache")
      connection.setRequestProperty("User-Agent", "Wayfarers-Guild-Content/1")
      try {
        val status = connection.responseCode
        cancel()
        if (status in setOf(301, 302, 303, 307, 308)) {
          require(redirect < 4)
          val location = connection.getHeaderField("Location") ?: throw GuildContentFailure("The update server did not provide a download.")
          current = URI(current).resolve(location).toString()
          GuildContentUrlPolicy.requireRedirect(current)
        } else {
          if (status != 200) throw GuildContentFailure("The game update is unavailable. Try again when you are online.")
          require(connection.contentEncoding == null || connection.contentEncoding.equals("identity", true))
          val length = connection.getHeaderFieldLong("Content-Length", -1)
          require(length == -1L || length in 1..limit)
          if (exactSize != null) require(length == -1L || length == exactSize)
          return connection.inputStream.use { consume(it, length) }
        }
      } finally {
        connection.disconnect()
      }
    }
    throw GuildContentFailure("The update server redirected too many times.")
  }

  private fun copy(input: InputStream, output: java.io.OutputStream, limit: Long, cancel: () -> Unit) {
    var received = 0L
    val buffer = ByteArray(32 * 1024)
    while (true) {
      cancel()
      val count = input.read(buffer)
      if (count < 0) break
      received += count
      require(received <= limit)
      output.write(buffer, 0, count)
    }
    require(received > 0)
  }
}
