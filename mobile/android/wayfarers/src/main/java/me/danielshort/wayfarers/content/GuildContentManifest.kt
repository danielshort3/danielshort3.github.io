package me.danielshort.wayfarers.content

import org.json.JSONObject
import java.net.URI
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.security.KeyFactory
import java.security.MessageDigest
import java.security.Signature
import java.security.interfaces.ECPublicKey
import java.security.spec.X509EncodedKeySpec
import java.util.Base64

data class GuildContentRecord(val path: String, val sha256: String, val size: Long)
data class GuildContentArchive(val url: String, val sha256: String, val size: Long)
data class GuildContentManifest(
  val contentVersion: Long,
  val label: String,
  val minAppVersionCode: Int,
  val archive: GuildContentArchive,
  val records: List<GuildContentRecord>,
  val envelope: String,
) {
  val id: String get() = "v$contentVersion-${archive.sha256.take(16)}"
}

open class GuildContentFailure(message: String, cause: Throwable? = null) : Exception(message, cause)
class GuildContentCompatibilityFailure(val contentVersion: Long) : GuildContentFailure("This game update requires a compatible app version. Check for an app update first.")

object GuildContentLimits {
  const val MAX_ENVELOPE = 512 * 1024
  const val MAX_PAYLOAD = 256 * 1024
  const val MAX_ARCHIVE = 32L * 1024 * 1024
  const val MAX_EXTRACTED = 64L * 1024 * 1024
  const val MAX_FILE = 8L * 1024 * 1024
  const val MAX_FILES = 512
  const val MAX_STORAGE = 4 * 1024 * 1024
  const val MAX_CHECKPOINT = 2 * 1024 * 1024 + 4096
  val RESERVED = setOf("wayfarers/native-checkpoint.js", "wayfarers/checkpoint.js", "wayfarers/android.js", "wayfarers/bundle-manifest.json")

  fun validPath(path: String): Boolean {
    if (path.length !in 1..200 || path in RESERVED || ".." in path || '%' in path || '\\' in path) return false
    val game = Regex("wayfarers/[A-Za-z0-9][A-Za-z0-9._-]*\\.(js|css|json)")
    val image = Regex("img/wayfarers-guild/[A-Za-z0-9][A-Za-z0-9._-]*\\.(png|webp|jpg|jpeg|gif|json)")
    return path == "wayfarers/index.html" || game.matches(path) || image.matches(path)
  }

  fun digest(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }

  fun utf8(bytes: ByteArray): String = Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
    .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString()
}

/** Only an authenticated, bounded manifest can authorize executable game content. */
class GuildContentVerifier(
  publicKeyBase64: String,
  private val packageName: String,
  private val nativeApi: Int,
  private val appVersionCode: Int,
  private val saveSchema: Int,
) {
  private val key = (KeyFactory.getInstance("EC").generatePublic(X509EncodedKeySpec(Base64.getDecoder().decode(publicKeyBase64))) as ECPublicKey).also {
    require(it.params.curve.field.fieldSize == 256) { "A pinned P-256 signing key is required." }
  }

  fun verify(envelope: String): GuildContentManifest {
    try {
      require(envelope.toByteArray(Charsets.UTF_8).size in 1..GuildContentLimits.MAX_ENVELOPE)
      val signed = JSONObject(envelope)
      require(signed.length() == 2 && signed.has("payload") && signed.has("signature"))
      val payload = Base64.getDecoder().decode(signed.getString("payload"))
      val signature = Base64.getDecoder().decode(signed.getString("signature"))
      require(payload.size in 1..GuildContentLimits.MAX_PAYLOAD && signature.size in 64..80)
      val verifier = Signature.getInstance("SHA256withECDSA")
      verifier.initVerify(key)
      verifier.update(payload)
      require(verifier.verify(signature)) { "Signature mismatch" }
      val manifest = JSONObject(GuildContentLimits.utf8(payload))
      require(manifest.length() == 9)
      require(integer(manifest, "schemaVersion") == 1L)
      require(manifest.getString("packageName") == packageName)
      val signedNativeApi = integer(manifest, "nativeApi")
      val signedSaveSchema = integer(manifest, "saveSchema")
      val minimum = integer(manifest, "minAppVersionCode")
      require(minimum in 1..Int.MAX_VALUE.toLong())
      val version = integer(manifest, "contentVersion")
      require(version in 1..Int.MAX_VALUE.toLong())
      val label = manifest.getString("label")
      require(label.length in 1..80 && label.none { it.isISOControl() })
      val archiveJson = manifest.getJSONObject("archive")
      require(archiveJson.length() == 3)
      val archive = GuildContentArchive(archiveJson.getString("url"), hash(archiveJson, "sha256"), integer(archiveJson, "size"))
      require(archive.size in 1..GuildContentLimits.MAX_ARCHIVE)
      GuildContentUrlPolicy.requireRelease(archive.url)
      val recordsJson = manifest.getJSONArray("records")
      require(recordsJson.length() in 2..GuildContentLimits.MAX_FILES)
      val records = (0 until recordsJson.length()).map { index ->
        val item = recordsJson.getJSONObject(index)
        require(item.length() == 3)
        val path = item.getString("path")
        require(GuildContentLimits.validPath(path)) { "Forbidden content path" }
        val size = integer(item, "size")
        require(size in 1..GuildContentLimits.MAX_FILE)
        GuildContentRecord(path, hash(item, "sha256"), size)
      }
      require(records.map { it.path }.toSet().size == records.size) { "Duplicate content path" }
      require(records.sumOf { it.size } <= GuildContentLimits.MAX_EXTRACTED)
      require(records.any { it.path == "wayfarers/index.html" } && records.any { it.path == "wayfarers/game.css" })
      if (signedNativeApi != nativeApi.toLong() || signedSaveSchema != saveSchema.toLong() || minimum > appVersionCode) {
        throw GuildContentCompatibilityFailure(version)
      }
      return GuildContentManifest(version, label, minimum.toInt(), archive, records, envelope)
    } catch (failure: GuildContentFailure) {
      throw failure
    } catch (failure: Exception) {
      throw GuildContentFailure("This game update could not be verified or is incompatible with this app.", failure)
    }
  }

  private fun integer(json: JSONObject, field: String): Long {
    val number = json.get(field)
    require(number is Int || number is Long) { "Expected an integer" }
    return (number as Number).toLong()
  }
  private fun hash(json: JSONObject, field: String): String = json.getString(field).also { require(it.matches(Regex("[0-9a-f]{64}"))) }
}

object GuildContentUrlPolicy {
  private const val RELEASE_PREFIX = "/danielshort3/danielshort3.github.io/releases/download/"
  private val redirectHosts = setOf("release-assets.githubusercontent.com", "objects.githubusercontent.com")

  fun requireRelease(value: String) {
    val uri = secure(value)
    require(uri.host == "github.com" && uri.path.startsWith(RELEASE_PREFIX) && uri.rawQuery == null)
    val suffix = uri.path.removePrefix(RELEASE_PREFIX)
    require(suffix.matches(Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,100}/[A-Za-z0-9][A-Za-z0-9._-]{0,180}")))
    require(uri.rawPath == uri.path && ".." !in suffix)
  }

  fun requireRedirect(value: String) {
    val uri = secure(value)
    if (uri.host == "github.com") requireRelease(value) else require(uri.host in redirectHosts)
  }

  private fun secure(value: String): URI = URI(value).also {
    require(it.scheme == "https" && it.port == -1 && it.userInfo == null && it.rawFragment == null && !it.isOpaque)
    require(it.host != null && it.host == it.host.lowercase() && '\\' !in value && it.path.split('/').none { part -> part == "." || part == ".." })
  }
}
