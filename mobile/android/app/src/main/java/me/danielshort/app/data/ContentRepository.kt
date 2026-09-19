package me.danielshort.app.data

import android.content.Context
import android.net.ConnectivityManager
import android.util.AtomicFile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import me.danielshort.app.BuildConfig
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URI

data class ContentState(val content: SiteContent? = null, val refreshing: Boolean = false, val message: String = "", val lastChecked: Long = 0L, val hasRemoteCopy: Boolean = false)

class ContentRepository(private val context: Context, val settings: AppSettingsStore) {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
  private val stateFlow = MutableStateFlow(ContentState())
  val state = stateFlow.asStateFlow()
  private val preferences = context.getSharedPreferences("native-content", Context.MODE_PRIVATE)
  private val cache = AtomicFile(File(context.filesDir, "site-content-v1.json"))
  private val mutex = Mutex()
  private val favoritesFlow = MutableStateFlow(preferences.getStringSet("saved-projects", emptySet()).orEmpty().toSet())
  val favorites = favoritesFlow.asStateFlow()

  init {
    scope.launch {
      loadLocal()
      refresh()
    }
  }

  fun toggleSaved(id: String) {
    val next = favoritesFlow.value.toMutableSet().apply { if (!remove(id)) add(id) }.toSet()
    preferences.edit().putStringSet("saved-projects", next).apply()
    favoritesFlow.value = next
  }

  fun clearSavedProjects() {
    preferences.edit().remove("saved-projects").apply()
    favoritesFlow.value = emptySet()
  }

  private suspend fun loadLocal() = withContext(Dispatchers.IO) {
    mutex.withLock {
      if (stateFlow.value.content != null) return@withLock
      val cached = runCatching { cache.openRead().bufferedReader().use { SiteContentParser.parse(it.readText()) } }.getOrNull()
      val bundled = cached ?: runCatching { context.assets.open("catalog.json").bufferedReader().use { SiteContentParser.parse(it.readText()) } }.getOrNull()
      stateFlow.value = ContentState(content = bundled, message = if (bundled == null) "Content could not be loaded. Try refreshing." else "", hasRemoteCopy = cached != null, lastChecked = preferences.getLong("last-checked", 0L))
    }
  }

  suspend fun refresh(force: Boolean = false): Boolean = withContext(Dispatchers.IO) {
    mutex.withLock {
      val previous = stateFlow.value
      if (!force) {
        val options = settings.state.value
        if (!options.automaticUpdates) return@withLock true
        val connectivity = context.getSystemService(ConnectivityManager::class.java)
        if (options.unmeteredOnly && connectivity.isActiveNetworkMetered) return@withLock true
      }
      if (!force && System.currentTimeMillis() - previous.lastChecked < 15 * 60_000L) return@withLock true
      stateFlow.value = previous.copy(refreshing = true, message = "")
      var connection: HttpURLConnection? = null
      try {
        val endpoint = URI(BuildConfig.CONTENT_URL)
        require(endpoint.scheme == "https" || (BuildConfig.DEBUG && endpoint.scheme == "http" && endpoint.host in setOf("10.0.2.2", "127.0.0.1", "localhost")))
        connection = endpoint.toURL().openConnection() as HttpURLConnection
        connection.connectTimeout = 12_000
        connection.readTimeout = 15_000
        connection.instanceFollowRedirects = false
        connection.setRequestProperty("Accept", "application/json")
        connection.setRequestProperty("User-Agent", "DanielShort-Android/${BuildConfig.VERSION_NAME}")
        if (previous.hasRemoteCopy) preferences.getString("etag", null)?.let { connection.setRequestProperty("If-None-Match", it) }
        val checkedAt = System.currentTimeMillis()
        if (connection.responseCode == HttpURLConnection.HTTP_NOT_MODIFIED && previous.hasRemoteCopy) {
          preferences.edit().putLong("last-checked", checkedAt).apply()
          stateFlow.value = previous.copy(refreshing = false, lastChecked = checkedAt, message = if (force) "Content is up to date" else "")
          return@withLock true
        }
        check(connection.responseCode == HttpURLConnection.HTTP_OK) { "HTTP ${connection.responseCode}" }
        require(connection.contentLengthLong <= 2_000_000) { "Content exceeds supported size" }
        val bytes = connection.inputStream.use { input ->
          val output = ByteArrayOutputStream()
          val buffer = ByteArray(8192)
          while (true) {
            val size = input.read(buffer)
            if (size < 0) break
            require(output.size() + size <= 2_000_000) { "Content exceeds supported size" }
            output.write(buffer, 0, size)
          }
          output.toByteArray()
        }
        val content = SiteContentParser.parse(bytes.toString(Charsets.UTF_8))
        val stream = cache.startWrite()
        try { stream.write(bytes); cache.finishWrite(stream) } catch (error: Exception) { cache.failWrite(stream); throw error }
        preferences.edit().putString("etag", connection.getHeaderField("ETag")).putLong("last-checked", checkedAt).apply()
        stateFlow.value = ContentState(content, false, if (force) { if (previous.content?.revision == content.revision) "Content is up to date" else "New website content loaded" } else "", checkedAt, true)
        true
      } catch (cancelled: CancellationException) {
        stateFlow.value = previous.copy(refreshing = false)
        throw cancelled
      } catch (_: Exception) {
        stateFlow.value = previous.copy(refreshing = false, message = if (previous.content != null) "Saved content is available. Updates will retry when reachable." else "Couldn't reach the content feed. Try again when connected.")
        false
      } finally { connection?.disconnect() }
    }
  }
}
