package me.danielshort.wayfarers

import java.net.URI

/** Stable origin is part of the save-format contract; never use a versioned origin. */
object GuildContentPolicy {
  const val ORIGIN = "https://appassets.androidplatform.net"
  const val GAME_URL = "$ORIGIN/assets/wayfarers/index.html"
  const val MAX_SAVE_BYTES = 1024 * 1024

  fun isGame(value: String): Boolean = uri(value)?.let {
    it.path == "/assets/wayfarers/index.html" && it.rawQuery == null
  } == true

  fun isAsset(value: String): Boolean = uri(value)?.let {
    it.path.startsWith("/assets/wayfarers/") || it.path.startsWith("/assets/img/wayfarers-guild/") || it.path.startsWith("/img/wayfarers-guild/")
  } == true

  private fun uri(value: String): URI? = runCatching {
    URI(value).also {
      require(it.scheme == "https" && it.host == "appassets.androidplatform.net" && it.port == -1 && it.userInfo == null)
      require(it.path.split('/').none { part -> part == "." || part == ".." })
      require(!it.rawPath.orEmpty().contains(Regex("%2f|%5c|%25", RegexOption.IGNORE_CASE)) && '\\' !in it.path)
    }
  }.getOrNull()
}
