package me.danielshort.wayfarers

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.SystemBarStyle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import me.danielshort.app.BuildConfig
import me.danielshort.app.updates.*
import org.json.JSONObject
import org.json.JSONArray
import java.io.ByteArrayInputStream
import java.util.Locale

/** A standalone game shell. Only signed, installed APKs change executable game assets. */
class MainActivity : ComponentActivity() {
  private val app get() = application as WayfarersApplication
  private val checkpoints by lazy { GuildCheckpointStore(java.io.File(filesDir, "guild-checkpoint.json")) }
  private var game: WebView? = null
  private var options by mutableStateOf(false)
  private var rendererFailed by mutableStateOf(false)
  private var rendererGeneration by mutableIntStateOf(0)
  private var installNotice by mutableStateOf("")
  private var openingInstaller by mutableStateOf(false)
  private var pendingExport: String? = null
  private var pendingImport: String? = null
  private var importing = false
  private var resumed = false
  private var lifecycleEpoch = 0L

  private val exportDocument = registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
    val text = pendingExport
    pendingExport = null
    java.io.File(cacheDir, "guild-pending-export.json").delete()
    if (uri != null && text != null) lifecycleScope.launch {
      val success = withContext(Dispatchers.IO) { runCatching {
        contentResolver.openOutputStream(uri, "wt")?.use { it.write(text.toByteArray(Charsets.UTF_8)) }
          ?: error("No destination")
      }.isSuccess }
      notice(if (success) "Guild backup saved." else "Couldn’t save that backup. Try another destination.")
    }
  }
  private val importDocument = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
    if (uri != null) reviewFile(uri)
  }
  private val installationPermission = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
    installNotice = if (packageManager.canRequestPackageInstalls()) "Permission enabled. Tap Install update to continue."
      else "Installation permission wasn’t enabled. Your guild is unchanged."
  }
  private val installActivity = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
    installNotice = "If installation didn’t finish, you can try again."
  }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge(
      statusBarStyle = SystemBarStyle.dark(android.graphics.Color.rgb(20, 43, 67)),
      navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.rgb(20, 43, 67))
    )
    options = savedInstanceState?.getBoolean("options") ?: false
    pendingExport = restorePending(savedInstanceState, "export")
    pendingImport = restorePending(savedInstanceState, "import")
    if (savedInstanceState == null) receiveImport(intent)
    setContent {
      MaterialTheme(colorScheme = lightColorScheme(primary = Color(0xFF234C69), secondary = Color(0xFFA77813),
        background = Color(0xFFFAF5E7), surface = Color(0xFFFAF5E7))) {
        Box(Modifier.fillMaxSize().safeDrawingPadding()) {
          key(rendererGeneration) {
            if (!rendererFailed) AndroidView(
              modifier = Modifier.fillMaxSize(),
              factory = { createGame() },
              onRelease = { released ->
                if (game === released) game = null
                lifecycleEpoch++
                app.coordinator.setSafeToInstall(false)
                if (WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) WebViewCompat.removeWebMessageListener(released, "WayfarersAndroid")
                released.stopLoading()
                released.destroy()
              }
            )
          }
          if (rendererFailed) Surface(Modifier.fillMaxSize()) {
            Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
              Text("The game couldn’t open", style = MaterialTheme.typography.headlineSmall)
              Text("Your stored guild has been kept. Reopen the game to try again.")
              Button(onClick = { rendererFailed = false; rendererGeneration++ }) { Text("Reopen game") }
              OutlinedButton(onClick = { showOptions() }) { Text("App updates") }
            }
          }
          if (options) OptionsScreen()
        }
        BackHandler {
          if (options) closeOptions() else game?.evaluateJavascript("Boolean(window.WayfarersUI?.handleBack())") { handled ->
            if (handled != "true") flushGame { safe -> if (safe) moveTaskToBack(true) }
          }
        }
      }
    }
  }

  override fun onNewIntent(intent: Intent) { super.onNewIntent(intent); setIntent(intent); receiveImport(intent) }
  override fun onSaveInstanceState(outState: Bundle) {
    outState.putBoolean("options", options)
    persistPending(outState, "export", pendingExport)
    persistPending(outState, "import", pendingImport)
    super.onSaveInstanceState(outState)
  }
  override fun onStart() { super.onStart(); app.coordinator.onForeground() }
  override fun onResume() {
    super.onResume()
    resumed = true
    lifecycleEpoch++
    game?.onResume()
    lifecycleScope.launch { app.installer.recover() }
  }
  override fun onPause() {
    resumed = false
    val epoch = ++lifecycleEpoch
    val retiring = game
    app.coordinator.setSafeToInstall(false)
    flushGame { safe ->
      if (epoch != lifecycleEpoch || resumed || game !== retiring) return@flushGame
      app.coordinator.setSafeToInstall(safe && options && !importing && pendingExport == null)
      retiring?.onPause()
    }
    super.onPause()
  }
  override fun onStop() { app.coordinator.onBackground(); super.onStop() }

  private fun showOptions() {
    flushGame { _ -> options = true }
  }
  private fun closeOptions() {
    app.coordinator.setSafeToInstall(false)
    options = false
    game?.onResume()
  }
  private fun flushGame(complete: (Boolean) -> Unit) {
    val current = game
    if (current == null || !GuildContentPolicy.isGame(current.url.orEmpty())) { complete(false); return }
    current.evaluateJavascript("window.WayfarersAndroidUI?.durableSnapshot() || ''") { result ->
      val text = runCatching { JSONArray("[$result]").getString(0) }.getOrNull()
      complete(!text.isNullOrEmpty() && checkpoints.write(text))
    }
  }
  private fun notice(message: String) { Toast.makeText(this, message, Toast.LENGTH_LONG).show() }

  @SuppressLint("SetJavaScriptEnabled")
  private fun createGame(): WebView {
    val assets = WebViewAssetLoader.Builder()
      .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
      .addPathHandler("/", WebViewAssetLoader.AssetsPathHandler(this))
      .build()
    return WebView(this).apply {
      lifecycleEpoch++
      game = this
      layoutParams = android.view.ViewGroup.LayoutParams(android.view.ViewGroup.LayoutParams.MATCH_PARENT, android.view.ViewGroup.LayoutParams.MATCH_PARENT)
      contentDescription = "Wayfarers Guild game"
      setBackgroundColor(android.graphics.Color.rgb(20, 43, 67))
      settings.javaScriptEnabled = true
      settings.domStorageEnabled = true
      settings.useWideViewPort = true
      settings.loadWithOverviewMode = true
      settings.allowFileAccess = false
      settings.allowContentAccess = false
      settings.mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
      settings.javaScriptCanOpenWindowsAutomatically = false
      settings.setSupportMultipleWindows(false)
      WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG)
      webViewClient = object : WebViewClient() {
        override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse {
          if (!GuildContentPolicy.isAsset(request.url.toString()) || request.method != "GET") return denied()
          if (request.url.toString() == GuildContentPolicy.ORIGIN + "/assets/wayfarers/native-checkpoint.js") {
            return WebResourceResponse("application/javascript", "UTF-8", 200, "OK", mapOf("Cache-Control" to "no-store"),
              ByteArrayInputStream(checkpoints.bootstrapScript().toByteArray(Charsets.UTF_8)))
          }
          return assets.shouldInterceptRequest(request.url) ?: denied()
        }
        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean =
          !GuildContentPolicy.isGame(request.url.toString())
        override fun onPageFinished(view: WebView, url: String) {
          if (!GuildContentPolicy.isGame(url)) return
          pendingImport?.let { text ->
            pendingImport = null
            java.io.File(cacheDir, "guild-pending-import.json").delete()
            view.evaluateJavascript("window.WayfarersAndroidUI?.reviewImport(${JSONObject.quote(text)})", null)
          }
        }
        override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
          rendererFailed = true
          app.coordinator.setSafeToInstall(false)
          return true
        }
      }
      if (WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) {
        WebViewCompat.addWebMessageListener(this, "WayfarersAndroid", setOf(GuildContentPolicy.ORIGIN)) { view, message, origin, mainFrame, reply ->
          if (!mainFrame || origin.toString().trimEnd('/') != GuildContentPolicy.ORIGIN ||
            !GuildContentPolicy.isGame(view.url.orEmpty())) return@addWebMessageListener
          val value = message.data ?: return@addWebMessageListener
          if (value.toByteArray(Charsets.UTF_8).size > GuildContentPolicy.MAX_SAVE_BYTES * 2 + 4096) return@addWebMessageListener
          val request = runCatching { JSONObject(value) }.getOrNull() ?: return@addWebMessageListener
          when (request.optString("type")) {
            "checkpoint" -> {
              val replacement = if (request.has("replacesCreatedAt")) request.optDouble("replacesCreatedAt") else null
              val success = checkpoints.write(request.optString("text"), replacement)
              reply.postMessage(JSONObject().put("type", "checkpoint").put("requestId", request.optLong("requestId")).put("ok", success).toString())
            }
            "options" -> showOptions()
            "import" -> importDocument.launch(arrayOf("application/json", "text/plain", "application/octet-stream"))
            "export" -> {
              val text = request.optString("text")
              if (pendingExport == null && text.toByteArray(Charsets.UTF_8).size in 1 until GuildContentPolicy.MAX_SAVE_BYTES) {
                pendingExport = text
                exportDocument.launch("wayfarers-guild-backup.json")
              }
            }
          }
        }
      } else {
        notice("Update Android System WebView to enable app options and save files.")
        options = true
      }
      loadUrl(GuildContentPolicy.GAME_URL)
    }
  }

  private fun denied() = WebResourceResponse("text/plain", "UTF-8", 403, "Blocked", emptyMap(), ByteArrayInputStream(byteArrayOf()))

  private fun receiveImport(received: Intent?) {
    if (received == null) return
    val uri = when (received.action) {
      Intent.ACTION_VIEW -> received.data
      Intent.ACTION_SEND -> @Suppress("DEPRECATION") received.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
      else -> null
    }
    if (uri?.scheme == "content") reviewFile(uri)
  }
  private fun reviewFile(uri: Uri) {
    if (importing) { notice("Wait for the current backup to open."); return }
    importing = true
    app.coordinator.setSafeToInstall(false)
    lifecycleScope.launch {
      val text = withContext(Dispatchers.IO) { runCatching {
        contentResolver.openInputStream(uri)?.use { input ->
          val bytes = input.readBytesBounded(GuildContentPolicy.MAX_SAVE_BYTES)
          String(bytes, Charsets.UTF_8)
        } ?: error("Missing file")
      }.getOrNull() }
      importing = false
      if (text == null) notice("Choose a readable Guild backup smaller than 1 MB.")
      else {
        closeOptions()
        val current = game
        if (current == null) pendingImport = text
        else current.evaluateJavascript("Boolean(window.WayfarersAndroidUI)") { ready ->
          if (ready == "true") current.evaluateJavascript("window.WayfarersAndroidUI.reviewImport(${JSONObject.quote(text)})", null)
          else pendingImport = text
        }
      }
    }
  }

  @Composable
  private fun OptionsScreen() {
    val state by app.updates.state.collectAsStateWithLifecycle()
    val preferences by app.settings.state.collectAsStateWithLifecycle()
    val automaticStatus by app.installer.status.collectAsStateWithLifecycle()
    Surface(Modifier.fillMaxSize()) {
      Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
          TextButton(onClick = { closeOptions() }) { Text("Back to guild") }
          Text("${BuildConfig.VERSION_NAME}", Modifier.padding(top = 12.dp), style = MaterialTheme.typography.labelLarge)
        }
        Text("App updates", style = MaterialTheme.typography.headlineSmall)
        Text(updateStatus(state), color = if (state is AppUpdateState.Error) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
        when (val current = state) {
          is AppUpdateState.Idle, is AppUpdateState.UpToDate -> OutlinedButton(onClick = { installNotice = ""; app.updates.check() }) { Text("Check now") }
          is AppUpdateState.Checking -> { LinearProgressIndicator(Modifier.fillMaxWidth()); TextButton(onClick = { app.updates.cancel() }) { Text("Cancel") } }
          is AppUpdateState.Available -> Button(onClick = { app.updates.download() }) { Text("Download update") }
          is AppUpdateState.Downloading -> {
            if (current.progress == null) LinearProgressIndicator(Modifier.fillMaxWidth())
            else LinearProgressIndicator(progress = { current.progress!!.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
            TextButton(onClick = { app.updates.cancel() }) { Text("Cancel download") }
          }
          is AppUpdateState.Ready -> Button(onClick = { installUpdate() }, enabled = !openingInstaller) { Text(if (openingInstaller) "Verifying…" else "Install update") }
          is AppUpdateState.Error -> OutlinedButton(onClick = {
            if (current.retryAction == UpdateRetryAction.DOWNLOAD) app.updates.download() else app.updates.check()
          }) { Text(if (current.retryAction == UpdateRetryAction.DOWNLOAD) "Retry download" else "Check again") }
        }
        if (installNotice.isNotBlank()) Text(installNotice)
        if (automaticStatus == AutomaticInstallStatus.MANUAL_REQUIRED) Text("Android needs your confirmation. Tap Install update.")
        HorizontalDivider()
        Text("Update mode", style = MaterialTheme.typography.titleMedium)
        val mode = if (preferences.automaticAppUpdates) 2 else if (preferences.checkAppUpdatesOnLaunch) 1 else 0
        listOf("Manual", "Check automatically", "Automatic updates").forEachIndexed { index, label ->
          Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
            RadioButton(selected = mode == index, onClick = {
              app.settings.update(preferences.copy(checkAppUpdatesOnLaunch = index > 0, automaticAppUpdates = index == 2))
            })
            TextButton(onClick = { app.settings.update(preferences.copy(checkAppUpdatesOnLaunch = index > 0, automaticAppUpdates = index == 2)) }) { Text(label) }
          }
        }
        if (preferences.automaticAppUpdates) {
          Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Unmetered connections only", Modifier.weight(1f).padding(top = 12.dp))
            Switch(checked = preferences.appUpdatesUnmeteredOnly, onCheckedChange = { app.settings.update(preferences.copy(appUpdatesUnmeteredOnly = it)) })
          }
        }
        Text("Updates preserve your guild. Patches save download size; every APK is verified before Android installs it. Automatic installation waits until you leave the app with App updates open and may need Android’s confirmation.", style = MaterialTheme.typography.bodySmall)
        HorizontalDivider()
        Text("Guild backups", style = MaterialTheme.typography.titleMedium)
        OutlinedButton(onClick = { importDocument.launch(arrayOf("application/json", "text/plain", "application/octet-stream")) }) { Text("Open save file") }
        Text("For an export, return to your guild and open Settings → Download save. Importing always shows a review before replacing a guild.", style = MaterialTheme.typography.bodySmall)
        Text("Wayfarers’ Guild · separate app and saves · offline play", style = MaterialTheme.typography.labelMedium)
      }
    }
  }

  private fun installUpdate() {
    if (openingInstaller) return
    if (importing || pendingExport != null || pendingImport != null) {
      installNotice = "Finish opening or saving your backup before installing."
      return
    }
    openingInstaller = true
    flushGame { safe ->
      if (!safe && !rendererFailed) {
        installNotice = "Save your guild before installing. Return to the game and export a backup if a storage warning appears."
        openingInstaller = false
        return@flushGame
      }
      lifecycleScope.launch {
        try {
          val apk = app.updates.verifiedApkForInstall()
          if (!app.installer.markManualInstallRequested()) { installNotice = "Android is already preparing this update."; return@launch }
          app.coordinator.suppressAutomaticInstall()
          if (!packageManager.canRequestPackageInstalls()) {
            installationPermission.launch(Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:$packageName")))
          } else {
            val uri = FileProvider.getUriForFile(this@MainActivity, "$packageName.files", apk)
            installActivity.launch(Intent(Intent.ACTION_VIEW).apply {
              setDataAndType(uri, "application/vnd.android.package-archive")
              addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
              putExtra(Intent.EXTRA_RETURN_RESULT, true)
            })
          }
        } catch (cancelled: CancellationException) { throw cancelled }
        catch (failure: Exception) { installNotice = failure.message ?: "Android couldn’t open this update. Try again." }
        finally { openingInstaller = false }
      }
    }
  }

  private fun updateStatus(state: AppUpdateState): String = when (state) {
    is AppUpdateState.Idle -> "Check for a new version"
    is AppUpdateState.Checking -> "Checking for updates…"
    is AppUpdateState.UpToDate -> "You’re up to date · ${state.versionName}"
    is AppUpdateState.Available -> "${state.offer.versionName} available · ${downloadLabel(state.offer)}"
    is AppUpdateState.Downloading -> if (state.progress == null) "Verifying update files…" else "Downloading ${if (state.usingPatch) "patch" else "app"} · ${(state.progress!! * 100).toInt()}%"
    is AppUpdateState.Ready -> "Ready to install · ${state.offer.versionName} · ${downloadLabel(state.offer)}"
    is AppUpdateState.Error -> state.message
  }
  private fun downloadLabel(offer: UpdateOffer) = String.format(Locale.getDefault(), "%.2f MB %s", offer.downloadBytes / (1024.0 * 1024.0), if (offer.usingPatch) "patch" else "APK")

  /** Store large backups privately, not in Android's size-limited activity parcel. */
  private fun persistPending(state: Bundle, kind: String, text: String?) {
    val file = java.io.File(cacheDir, "guild-pending-$kind.json")
    if (text == null) { file.delete(); return }
    runCatching { file.writeText(text, Charsets.UTF_8); state.putBoolean("pending-$kind", true) }
  }
  private fun restorePending(state: Bundle?, kind: String): String? {
    if (state?.getBoolean("pending-$kind") != true) return null
    val file = java.io.File(cacheDir, "guild-pending-$kind.json")
    return runCatching {
      require(file.length() in 1 until GuildContentPolicy.MAX_SAVE_BYTES.toLong())
      file.readText(Charsets.UTF_8)
    }.getOrNull()
  }
}

private fun java.io.InputStream.readBytesBounded(limit: Int): ByteArray {
  val output = java.io.ByteArrayOutputStream()
  val buffer = ByteArray(8192)
  while (true) {
    val count = read(buffer)
    if (count < 0) break
    require(output.size() + count < limit) { "Save too large" }
    output.write(buffer, 0, count)
  }
  return output.toByteArray()
}
