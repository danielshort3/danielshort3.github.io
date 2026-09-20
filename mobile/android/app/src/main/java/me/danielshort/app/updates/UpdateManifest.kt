package me.danielshort.app.updates

import org.json.JSONObject
import java.net.URI

const val MAX_UPDATE_BYTES = 256L * 1024 * 1024
const val MAX_MANIFEST_BYTES = 512L * 1024

class UpdateFailure(message: String, cause: Throwable? = null) : Exception(message, cause)

data class UpdateArtifact(val url: String, val sha256: String, val size: Long)
data class PublishedRelease(val versionCode: Long, val sha256: String, val size: Long, val signerSha256: String)
data class LatestRelease(
  val versionCode: Long,
  val versionName: String,
  val minSdk: Int,
  val apk: UpdateArtifact,
  val signerSha256: String,
)
data class UpdatePatch(val fromSha256: String, val toSha256: String, val artifact: UpdateArtifact, val format: String)
data class UpdateManifest(
  val packageName: String,
  val latest: LatestRelease,
  val releases: List<PublishedRelease>,
  val patches: List<UpdatePatch>,
)

/** Manifest hashes locate an exact release; APK signing certificates remain the trust anchor. */
object UpdateManifestParser {
  fun parse(text: String): UpdateManifest {
    try {
      require(text.toByteArray(Charsets.UTF_8).size <= MAX_MANIFEST_BYTES)
      val root = JSONObject(text)
      require(root.strictLong("schemaVersion") == 1L)
      val packageName = root.getString("packageName")
      require(packageName.matches(Regex("[A-Za-z][A-Za-z0-9_]*(\\.[A-Za-z][A-Za-z0-9_]*)+")))
      if (root.has("channel")) require(root.getString("channel") == if (packageName.endsWith(".debug")) "review" else "stable")
      val latestJson = root.getJSONObject("latest")
      val latest = LatestRelease(
        latestJson.positiveLong("versionCode"),
        latestJson.getString("versionName").also { require(it.isNotBlank() && it.length <= 80 && it.none(Char::isISOControl)) },
        latestJson.positiveLong("minSdk").also { require(it <= 1000) }.toInt(),
        artifact(latestJson.getJSONObject("apk")),
        latestJson.hash("signerSha256"),
      )
      val releaseJson = root.getJSONArray("releases")
      require(releaseJson.length() in 1..200)
      val releases = (0 until releaseJson.length()).map { index ->
        val item = releaseJson.getJSONObject(index)
        PublishedRelease(item.positiveLong("versionCode"), item.hash("sha256"), item.size(), item.hash("signerSha256"))
      }
      require(releases.map { it.sha256 }.distinct().size == releases.size)
      val patchJson = root.getJSONArray("patches")
      require(patchJson.length() <= 200)
      val patches = (0 until patchJson.length()).map { index ->
        val item = patchJson.getJSONObject(index)
        UpdatePatch(item.hash("fromSha256"), item.hash("toSha256"), artifact(item), item.getString("format").also { require(it == "dsupd1-gzip") })
      }
      require(patches.map { it.fromSha256 to it.toSha256 }.distinct().size == patches.size)
      require(releases.any { it.versionCode == latest.versionCode && it.sha256 == latest.apk.sha256 && it.size == latest.apk.size && it.signerSha256 == latest.signerSha256 })
      return UpdateManifest(packageName, latest, releases, patches)
    } catch (failure: Exception) {
      throw UpdateFailure("The update information could not be verified. Please try again later.", failure)
    }
  }

  private fun artifact(item: JSONObject) = UpdateArtifact(item.getString("url").also { UpdateUrlPolicy.requirePublishedUrl(it) }, item.hash("sha256"), item.size())
  private fun JSONObject.strictLong(key: String): Long {
    val value = get(key)
    require(value is Int || value is Long)
    return (value as Number).toLong()
  }
  private fun JSONObject.positiveLong(key: String) = strictLong(key).also { require(it > 0) }
  private fun JSONObject.size() = positiveLong("size").also { require(it <= MAX_UPDATE_BYTES) }
  private fun JSONObject.hash(key: String) = getString(key).lowercase().also { require(it.matches(Regex("[0-9a-f]{64}"))) }
}

/** CDN URLs are accepted only after a redirect from this repository's release download URL. */
object UpdateUrlPolicy {
  private val siteHosts = setOf("danielshort.me", "www.danielshort.me")
  private val releaseCdns = setOf("release-assets.githubusercontent.com", "objects.githubusercontent.com")
  private const val RELEASE_PATH = "/danielshort3/danielshort3.github.io/releases/download/"

  fun requireFeedUrl(value: String): URI = validated(value).also {
    require(it.host.lowercase() in siteHosts && safePath(it).startsWith("/app-updates/") && it.rawQuery == null) { "Unsupported update feed" }
  }

  fun requirePublishedUrl(value: String): URI = validated(value).also {
    val host = it.host.lowercase()
    val path = safePath(it)
    require((host in siteHosts && path.startsWith("/app-updates/")) || (host == "github.com" && path.startsWith(RELEASE_PATH))) { "Unsupported update download" }
    require(it.rawQuery == null) { "Unexpected download query" }
  }

  fun requireRedirect(current: URI, location: String, startedAt: URI, feed: Boolean): URI {
    val next = validated(current.resolve(location).toString())
    val startedAtGitHub = startedAt.host.equals("github.com", true) && safePath(startedAt).startsWith(RELEASE_PATH)
    if (!feed && startedAtGitHub && next.host.lowercase() in releaseCdns) {
      require(safePath(next).isNotEmpty())
      return next
    }
    return if (feed) requireFeedUrl(next.toString()) else requirePublishedUrl(next.toString())
  }

  private fun validated(value: String): URI {
    require(value.length <= 8192)
    val uri = URI(value)
    require(uri.scheme == "https" && !uri.host.isNullOrBlank() && uri.userInfo == null && uri.fragment == null && (uri.port == -1 || uri.port == 443)) { "Updates require a trusted HTTPS address" }
    safePath(uri)
    return uri
  }

  private fun safePath(uri: URI): String {
    val path = uri.path.orEmpty()
    require(path.startsWith("/") && '\\' !in path && path.split('/').none { it == "." || it == ".." } && !uri.rawPath.orEmpty().contains(Regex("%2f|%5c|%25", RegexOption.IGNORE_CASE)))
    return path
  }
}
