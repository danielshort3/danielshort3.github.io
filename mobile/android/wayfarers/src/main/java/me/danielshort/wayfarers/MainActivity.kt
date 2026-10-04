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
import androidx.compose.ui.Alignment
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
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import me.danielshort.app.BuildConfig
import me.danielshort.app.updates.*
import org.json.JSONObject
import org.json.JSONArray
import java.io.ByteArrayInputStream
import java.util.Locale
import java.util.UUID
import java.io.File
import me.danielshort.wayfarers.content.*

/** Offline shell with authenticated game content and a separate native APK updater. */
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
  private var applyingContent by mutableStateOf(false)
  private var contentNotice by mutableStateOf("")
  private var contentSession: ContentSession? = null
  private var contentRecovery: Recovery? = null
  private var applyingSessionId: String? = null
  private var documentToken = ""
  private var contentTimeout: Job? = null
  private var expectedGuildId: Double? = null
  private var committedUpdateDocument: String? = null

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
              if (contentNotice.isNotBlank()) Text(contentNotice)
              Button(onClick = { rendererFailed = false; rendererGeneration++ }) { Text("Reopen game") }
              OutlinedButton(onClick = { showOptions() }) { Text("App updates") }
            }
          }
          if (options && !applyingContent) OptionsScreen()
          if (applyingContent) Surface(Modifier.fillMaxSize(), color = Color(0xFF142B43)) {
            Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.Center,
              horizontalAlignment = Alignment.CenterHorizontally) {
              CircularProgressIndicator(color = Color(0xFFFFDA67))
              Spacer(Modifier.height(20.dp))
              Text("Applying game update…", color = Color.White, style = MaterialTheme.typography.titleLarge)
              Text("Your guild will resume here.", Modifier.padding(top = 8.dp), color = Color(0xFFBDD7E4))
            }
          }
        }
        BackHandler {
          if (applyingContent) return@BackHandler
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
    if (applyingContent) return
    flushGame { _ -> options = true }
  }
  private fun closeOptions() {
    app.coordinator.setSafeToInstall(false)
    options = false
    game?.onResume()
  }
  private fun flushGame(complete: (Boolean) -> Unit) {
    val current = game
    val token = documentToken
    if (applyingContent || token.isBlank() || current == null || !GuildContentPolicy.isGame(current.url.orEmpty())) { complete(false); return }
    current.evaluateJavascript("JSON.stringify({text:window.WayfarersAndroidUI?.durableSnapshot() || '',generation:window.WayfarersCheckpoint?.generation() || ''})") { result ->
      if (isDestroyed || applyingContent || game !== current || documentToken != token) { complete(false); return@evaluateJavascript }
      val snapshot = runCatching { JSONObject(JSONArray("[$result]").getString(0)) }.getOrNull()
      val text = snapshot?.optString("text")
      complete(!text.isNullOrEmpty() && checkpoints.write(text, generation = snapshot.optString("generation")))
    }
  }
  private fun notice(message: String) { Toast.makeText(this, message, Toast.LENGTH_LONG).show() }

  @SuppressLint("SetJavaScriptEnabled")
  private fun createGame(): WebView {
    val prepared = runCatching {
      val recovery = if (applyingSessionId == null) app.contentStore.startupRecovery() else null
      if (recovery != null) check(checkpoints.restoreForContentUpdate(recovery.checkpointBytes))
      recovery to app.contentStore.session()
    }
    if (prepared.isFailure) {
      contentStorageFailure()
      return WebView(this).apply { game = this }
    }
    val (recovery, session) = prepared.getOrThrow()
    contentRecovery = recovery
    contentSession = session
    expectedGuildId = checkpoints.read()?.createdAt
    val token = UUID.randomUUID().toString()
    documentToken = token
    contentTimeout?.cancel()
    if (applyingSessionId != null || recovery != null) {
      applyingContent = !rendererFailed
      contentTimeout = lifecycleScope.launch {
        delay(30000)
        if (documentToken == token && applyingContent) recoverContent("The game update did not finish opening.")
      }
    }
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
      settings.cacheMode = WebSettings.LOAD_NO_CACHE
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
              ByteArrayInputStream(contentBootstrap(session, token, recovery).toByteArray(Charsets.UTF_8)))
          }
          val path = request.url.path.orEmpty().removePrefix("/assets/").removePrefix("/")
          val nativeOwned = path in setOf("wayfarers/checkpoint.js", "wayfarers/android.js")
          if (!session.isBundled && !nativeOwned) {
            val input = session.open(path) ?: return denied()
            val mime = when (path.substringAfterLast('.')) {
              "js" -> "application/javascript"; "css" -> "text/css"; "html" -> "text/html"
              "png" -> "image/png"; "webp" -> "image/webp"; "json" -> "application/json"
              else -> "application/octet-stream"
            }
            return WebResourceResponse(mime, if (mime.startsWith("image/")) null else "UTF-8", 200, "OK",
              mapOf("Cache-Control" to "no-store"), input)
          }
          return assets.shouldInterceptRequest(request.url) ?: denied()
        }
        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean =
          !GuildContentPolicy.isGame(request.url.toString())
        override fun onPageFinished(view: WebView, url: String) {
          if (game !== view || applyingContent || !GuildContentPolicy.isGame(url)) return
          pendingImport?.let { text ->
            pendingImport = null
            java.io.File(cacheDir, "guild-pending-import.json").delete()
            view.evaluateJavascript("window.WayfarersAndroidUI?.reviewImport(${JSONObject.quote(text)})", null)
          }
        }
        override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
          if (game !== view) return true
          if (applyingContent) { recoverContent("The game update could not open."); return true }
          rendererFailed = true
          app.coordinator.setSafeToInstall(false)
          return true
        }
      }
      if (WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_LISTENER)) {
        WebViewCompat.addWebMessageListener(this, "WayfarersAndroid", setOf(GuildContentPolicy.ORIGIN)) { view, message, origin, mainFrame, reply ->
          if (game !== view || isDestroyed || !mainFrame || origin.toString().trimEnd('/') != GuildContentPolicy.ORIGIN ||
            !GuildContentPolicy.isGame(view.url.orEmpty())) return@addWebMessageListener
          val value = message.data ?: return@addWebMessageListener
          if (value.toByteArray(Charsets.UTF_8).size > GuildContentPolicy.MAX_SAVE_BYTES * 2 + 4096) return@addWebMessageListener
          val request = runCatching { JSONObject(value) }.getOrNull() ?: return@addWebMessageListener
          if (request.optString("documentToken") != token || documentToken != token) return@addWebMessageListener
          when (request.optString("type")) {
            "content-ready" -> acknowledgeContent(view, session, request)
            "checkpoint" -> {
              val replacement = if (request.has("replacesCreatedAt")) request.optDouble("replacesCreatedAt") else null
              val success = checkpoints.write(request.optString("text"), replacement, request.optString("generation", ""))
              reply.postMessage(JSONObject().put("type", "checkpoint").put("requestId", request.optLong("requestId")).put("ok", success).toString())
            }
            "reset-guild" -> {
              val success = checkpoints.reset(request.optString("text"), request.optString("previousGeneration"), request.optString("generation"), request.optString("updateId"))
              if (success) expectedGuildId = checkpoints.read()?.createdAt
              reply.postMessage(JSONObject().put("type", "reset-guild").put("requestId", request.optLong("requestId")).put("ok", success).toString())
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
      if (!rendererFailed) loadUrl(GuildContentPolicy.GAME_URL)
    }
  }

  private fun contentBootstrap(session: ContentSession, token: String, recovery: Recovery?): String {
    val metadata = JSONObject().put("version", session.contentVersion).put("label", session.label)
      .put("apkVersion", BuildConfig.VERSION_CODE)
      .put("documentToken", token).put("recoveryToken", recovery?.token ?: "")
    val restore = recovery?.let {
      """
        try {
          var restored=JSON.parse(${JSONObject.quote(it.localStorageJson)});
          for(var i=localStorage.length-1;i>=0;i--){var k=localStorage.key(i);if(k&&k.startsWith('wayfarers-guild-'))localStorage.removeItem(k);}
          Object.keys(restored).forEach(function(k){localStorage.setItem(k,restored[k]);});
        }catch(error){window.WayfarersContent.restoreFailed=true;}
      """.trimIndent()
    } ?: ""
    return "window.WayfarersContent=Object.assign(window.WayfarersContent||{},$metadata);$restore${checkpoints.bootstrapScript()}"
  }

  private fun applyContentUpdate() {
    if (applyingContent || openingInstaller || importing || pendingImport != null || pendingExport != null) return
    if (app.contentStore.stagedManifest() == null) return
    val current = game ?: return
    val token = documentToken
    applyingContent = true
    contentNotice = ""
    app.coordinator.setSafeToInstall(false)
    current.evaluateJavascript("JSON.stringify(window.WayfarersAndroidUI?.prepareContentUpdate() || null)") { result ->
      if (isDestroyed || game !== current || documentToken != token) return@evaluateJavascript
      val snapshot = runCatching { JSONObject(JSONArray("[$result]").getString(0)) }.getOrNull()
      val text = snapshot?.optString("text").orEmpty()
      val safe = text.isNotBlank() && checkpoints.write(text, generation = snapshot?.optString("generation").orEmpty())
      val backup = if (safe) checkpoints.backupForContentUpdate() else null
      documentToken = "" // Fence every asynchronous message from the retired document.
      if (snapshot == null || backup == null) {
        applyingContent = false
        contentNotice = "Finish the current action and save your guild, then try applying again."
        rendererGeneration++
        return@evaluateJavascript
      }
      lifecycleScope.launch {
        try {
          val next = withContext(Dispatchers.IO) { app.contentStore.beginApply(backup, snapshot.getJSONObject("storage").toString()) }
          applyingSessionId = next.id
          options = false
          rendererGeneration++
        } catch (failure: Exception) {
          if (failure is CancellationException) throw failure
          applyingContent = false
          contentNotice = "The update could not be applied. Your guild has been kept."
          rendererGeneration++
        }
      }
    }
  }

  private fun acknowledgeContent(view: WebView, session: ContentSession, request: JSONObject) {
    if (request.optLong("version") != session.contentVersion || contentSession?.id != session.id) return
    val text = request.optString("text")
    val identity = runCatching { JSONObject(text).getJSONObject("state").getDouble("createdAt") }.getOrNull() ?: return
    if (expectedGuildId != null && identity != expectedGuildId) return
    if (!checkpoints.write(text, generation = request.optString("generation"))) return
    val committed = runCatching { applyingSessionId != session.id || app.contentStore.commitApply(session.id) }
    if (committed.isFailure) { contentStorageFailure(); return }
    if (!committed.getOrThrow()) {
      recoverContent("The update could not be confirmed.")
      return
    }
    val recovery = contentRecovery
    if (recovery != null) {
      val consumed = runCatching { request.optString("recoveryToken") == recovery.token && app.contentStore.consumeRecovery(recovery.token) }
      if (consumed.isFailure) { contentStorageFailure(); return }
      if (!consumed.getOrThrow()) return
    }
    contentTimeout?.cancel()
    applyingSessionId = null
    contentRecovery = null
    val wasApplying = applyingContent
    applyingContent = false
    app.contentUpdates.refresh()
    if (wasApplying && game === view) {
      contentNotice = if (recovery == null) "Game update applied · ${session.label}" else "Previous game restored. Your guild is ready."
      notice(contentNotice)
    }
    // Debugging resets may observe only a committed, durable and rendered game.
    // Restored content consumes the identity without resetting the recovered guild.
    val documentTime = request.optDouble("documentTimeOrigin", 0.0)
    val committedDocument = "$documentToken:${session.id}:${if (documentTime.isFinite()) documentTime else 0.0}"
    if (game === view && committedUpdateDocument != committedDocument) {
      committedUpdateDocument = committedDocument
      val identity = JSONObject().put("apkVersion", BuildConfig.VERSION_CODE).put("contentVersion", session.contentVersion)
        .put("recovered", recovery != null).put("documentToken", documentToken)
      view.evaluateJavascript("window.dispatchEvent(new CustomEvent('wayfarers-update-committed',{detail:$identity}));", null)
    }
  }

  private fun recoverContent(reason: String) {
    val id = applyingSessionId
    documentToken = ""
    contentTimeout?.cancel()
    lifecycleScope.launch {
      val recovered = withContext(Dispatchers.IO) { runCatching {
        if (id != null) app.contentStore.rollbackApply(id, reason) else app.contentStore.startupRecovery()
      } }
      if (recovered.isFailure) { contentStorageFailure(); return@launch }
      val recovery = recovered.getOrNull()
      applyingSessionId = null
      if (recovery == null || !checkpoints.restoreForContentUpdate(recovery.checkpointBytes) || contentRecovery?.token == recovery.token) {
        applyingContent = false
        rendererFailed = true
        contentNotice = "Recovery is waiting. Your saved update backup is safe; reopen the game to retry."
        options = true
      } else {
        contentRecovery = recovery
        contentNotice = reason
        rendererFailed = false
        rendererGeneration++
      }
    }
  }

  private fun contentStorageFailure() {
    contentTimeout?.cancel()
    documentToken = ""
    applyingSessionId = null
    applyingContent = false
    rendererFailed = true
    options = false
    contentNotice = "Game update recovery is waiting for storage. Reopen to retry, or export your saved guild from App updates. The update backup has been kept."
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
    if (applyingContent) { notice("Finish applying the game update, then open your backup again."); return }
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
        val token = documentToken
        if (current == null) pendingImport = text
        else current.evaluateJavascript("Boolean(window.WayfarersAndroidUI)") { ready ->
          if (ready == "true" && !applyingContent && game === current && documentToken == token) current.evaluateJavascript("window.WayfarersAndroidUI.reviewImport(${JSONObject.quote(text)})", null)
          else pendingImport = text
        }
      }
    }
  }

  @Composable
  private fun OptionsScreen() {
    val state by app.updates.state.collectAsStateWithLifecycle()
    val contentState by app.contentUpdates.state.collectAsStateWithLifecycle()
    val preferences by app.settings.state.collectAsStateWithLifecycle()
    val automaticStatus by app.installer.status.collectAsStateWithLifecycle()
    Surface(Modifier.fillMaxSize()) {
      Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
          TextButton(onClick = { closeOptions() }) { Text("Back to guild") }
          Text("${BuildConfig.VERSION_NAME}", Modifier.padding(top = 12.dp), style = MaterialTheme.typography.labelLarge)
        }
        Text("Game updates", style = MaterialTheme.typography.headlineSmall)
        Text("Game content v${contentSession?.contentVersion ?: BuildConfig.BUNDLED_CONTENT_VERSION}", style = MaterialTheme.typography.labelLarge)
        when (val current = contentState) {
          is GuildContentUpdateState.Idle, is GuildContentUpdateState.Current -> {
            Text(if (current is GuildContentUpdateState.Current) "Your game content is up to date." else "Check for new game content.")
            OutlinedButton(onClick = { app.contentUpdates.check() }) { Text("Check game updates") }
          }
          is GuildContentUpdateState.Checking -> { LinearProgressIndicator(Modifier.fillMaxWidth()); TextButton(onClick = { app.contentUpdates.cancel() }) { Text("Cancel game check") } }
          is GuildContentUpdateState.Available -> {
            Text("${current.manifest.label} available")
            Button(onClick = { app.contentUpdates.download() }) { Text("Download game update") }
          }
          is GuildContentUpdateState.Downloading -> {
            Text("Downloading game content · ${(current.progress * 100).toInt()}%")
            LinearProgressIndicator(progress = { current.progress.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
            TextButton(onClick = { app.contentUpdates.cancel() }) { Text("Cancel game download") }
          }
          is GuildContentUpdateState.Ready -> {
            Text("${current.manifest.label} ready to apply")
            Button(onClick = { applyContentUpdate() }, enabled = !openingInstaller) { Text("Apply game update") }
          }
          is GuildContentUpdateState.Error -> {
            Text(current.message, color = MaterialTheme.colorScheme.error)
            OutlinedButton(onClick = { app.contentUpdates.check() }) { Text("Check game updates") }
          }
        }
        if (contentNotice.isNotBlank()) Text(contentNotice)
        Text("Keep playing while content downloads. Applying briefly reloads your game here and preserves your guild.", style = MaterialTheme.typography.bodySmall)
        HorizontalDivider()
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
        Text("Game content applies here. Native app updates use Android’s installer. Automatic downloads follow this mode; applying game content waits for your tap.", style = MaterialTheme.typography.bodySmall)
        HorizontalDivider()
        Text("Guild backups", style = MaterialTheme.typography.titleMedium)
        if (rendererFailed && checkpoints.read() != null) OutlinedButton(onClick = {
          pendingExport = checkpoints.read()?.text
          if (pendingExport != null) exportDocument.launch("wayfarers-guild-backup.json")
        }) { Text("Export saved guild") }
        OutlinedButton(onClick = { importDocument.launch(arrayOf("application/json", "text/plain", "application/octet-stream")) }) { Text("Open save file") }
        Text("For an export, return to your guild and open Settings → Download save. Importing always shows a review before replacing a guild.", style = MaterialTheme.typography.bodySmall)
        Text("Wayfarers’ Guild · separate app and saves · offline play", style = MaterialTheme.typography.labelMedium)
      }
    }
  }

  private fun installUpdate() {
    if (openingInstaller || applyingContent) return
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
