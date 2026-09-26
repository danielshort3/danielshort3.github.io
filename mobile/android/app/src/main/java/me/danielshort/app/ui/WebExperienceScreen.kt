package me.danielshort.app.ui

import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.Message
import android.view.View
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import android.webkit.RenderProcessGoneDetail
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.OpenInNew
import androidx.compose.material.icons.outlined.BookmarkBorder
import androidx.compose.material.icons.outlined.BookmarkAdded
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlinx.coroutines.delay

/** The website is the source of behavior for browser-ready games, demos, and project pages. */
@OptIn(ExperimentalMaterial3Api::class)
@SuppressLint("SetJavaScriptEnabled")
@Composable
internal fun WebExperienceScreen(
  experience: WebExperience,
  feedUrl: String,
  onBack: () -> Unit,
  onOpenNative: (() -> Unit)? = null,
  nativeLabel: String = "Native version",
  isSaved: Boolean = false,
  onToggleSaved: (() -> Unit)? = null,
  onShare: (() -> Unit)? = null,
  linkedDemo: WebExperience? = null,
  onOpenLinkedDemo: (() -> Unit)? = null,
  linkedProjectIds: Set<String> = emptySet(),
  onOpenLinkedProject: ((String) -> Unit)? = null
) {
  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current
  val url = trustedWebExperienceUrl(experience, feedUrl) ?: canonicalWebExperienceUrl(experience)
  var rendererVersion by remember(url) { mutableIntStateOf(0) }
  var rendererGone by remember(url) { mutableStateOf(false) }
  val webView = remember(context, url, rendererVersion) { WebView(context) }
  val pendingLayout = remember(webView) { arrayOfNulls<Runnable>(1) }
  var loading by remember(url, rendererVersion) { mutableStateOf(true) }
  var prepared by remember(url, rendererVersion) { mutableStateOf(false) }
  var error by remember(url, rendererVersion) { mutableStateOf(false) }
  var fullscreenView by remember(url, rendererVersion) { mutableStateOf<View?>(null) }
  var fullscreenCallback by remember(url, rendererVersion) { mutableStateOf<WebChromeClient.CustomViewCallback?>(null) }
  var actionsExpanded by remember(url) { mutableStateOf(false) }
  var playOptionsExpanded by remember(url) { mutableStateOf(false) }
  var loadAttempt by remember(url, rendererVersion) { mutableIntStateOf(0) }
  var showProjectLoadingDetails by remember(url, rendererVersion) { mutableStateOf(false) }
  val hasStarfallPlayOptions = experience.canonicalPath == WebExperience.STARFALL.canonicalPath && onOpenNative != null

  LaunchedEffect(url, rendererVersion, loadAttempt, prepared, error) {
    showProjectLoadingDetails = false
    if (experience.kind == WebExperienceKind.PROJECT && !prepared && !error) {
      delay(1_200L)
      // A finished page only needs the short layout preparation, not a message flash.
      if (loading && !prepared && !error) showProjectLoadingDetails = true
    }
  }

  fun closeFullscreen() {
    fullscreenCallback?.onCustomViewHidden()
    fullscreenCallback = null
    fullscreenView = null
  }

  BackHandler(fullscreenView != null) { closeFullscreen() }
  DisposableEffect(webView, lifecycleOwner) {
    val observer = LifecycleEventObserver { _, event ->
      when (event) {
        Lifecycle.Event.ON_RESUME -> webView.onResume()
        Lifecycle.Event.ON_PAUSE -> webView.onPause()
        else -> Unit
      }
    }
    lifecycleOwner.lifecycle.addObserver(observer)
    onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
  }

  Box(Modifier.fillMaxSize().testTag("web-experience-panel")) {
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
      if (fullscreenView == null) {
        TopAppBar(
          title = { Text(experience.title, maxLines = 1, overflow = TextOverflow.Ellipsis) },
          navigationIcon = { IconButton(onClick = onBack) {
            Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back")
          } },
          actions = {
            if (onToggleSaved != null) {
              IconButton(onClick = onToggleSaved) {
                Icon(if (isSaved) Icons.Outlined.BookmarkAdded else Icons.Outlined.BookmarkBorder,
                  if (isSaved) "Unsave project" else "Save project")
              }
              Box {
                IconButton(onClick = { actionsExpanded = true }) {
                  Icon(Icons.Outlined.MoreVert, "Project actions")
                }
                DropdownMenu(expanded = actionsExpanded, onDismissRequest = { actionsExpanded = false }) {
                  if (onOpenNative != null) DropdownMenuItem(text = { Text(nativeLabel) }, onClick = {
                    actionsExpanded = false
                    onOpenNative()
                  })
                  if (onShare != null) DropdownMenuItem(text = { Text("Share project") }, onClick = {
                    actionsExpanded = false
                    onShare()
                  })
                  DropdownMenuItem(text = { Text("Open in browser") }, onClick = {
                    actionsExpanded = false
                    openExternalExperience(context, Uri.parse(url))
                  })
                }
              }
            } else {
              if (onOpenNative != null) TextButton(onClick = {
                if (hasStarfallPlayOptions) playOptionsExpanded = true else onOpenNative()
              }) { Text(if (hasStarfallPlayOptions) "Play options" else nativeLabel) }
              IconButton(onClick = { openExternalExperience(context, Uri.parse(url)) }) {
                Icon(Icons.AutoMirrored.Outlined.OpenInNew, "Open in browser")
              }
            }
          }
        )
        if (!error && (loading || !prepared)) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
      }
      Box(Modifier.fillMaxWidth().weight(1f)) {
        key(webView) {
        if (!rendererGone) AndroidView(
          factory = {
            webView.apply {
              setBackgroundColor(android.graphics.Color.WHITE)
              settings.javaScriptEnabled = true
              settings.domStorageEnabled = true
              settings.allowFileAccess = false
              settings.allowContentAccess = false
              settings.allowFileAccessFromFileURLs = false
              settings.allowUniversalAccessFromFileURLs = false
              settings.mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
              settings.safeBrowsingEnabled = true
              settings.javaScriptCanOpenWindowsAutomatically = false
              settings.setSupportMultipleWindows(true)
              webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                  if (!request.isForMainFrame) return false
                  if (trustedWebExperienceUrl(experience, request.url.toString()) != null) return false
                  if (linkedDemo != null && onOpenLinkedDemo != null &&
                    trustedWebExperienceUrl(linkedDemo, request.url.toString()) != null) {
                    onOpenLinkedDemo()
                    return true
                  }
                  val linkedProjectId = if (experience.kind == WebExperienceKind.PROJECT)
                    trustedLinkedProjectId(request.url.toString(), linkedProjectIds) else null
                  if (linkedProjectId != null && onOpenLinkedProject != null) {
                    onOpenLinkedProject(linkedProjectId)
                    return true
                  }
                  openExternalExperience(context, request.url)
                  return true
                }

                override fun doUpdateVisitedHistory(view: WebView, pageUrl: String?, isReload: Boolean) {
                  super.doUpdateVisitedHistory(view, pageUrl, isReload)
                  if (pageUrl == null || pageUrl == "about:blank" ||
                    trustedWebExperienceUrl(experience, pageUrl) != null) return
                  if (linkedDemo != null && onOpenLinkedDemo != null &&
                    trustedWebExperienceUrl(linkedDemo, pageUrl) != null) {
                    view.post { onOpenLinkedDemo() }
                    return
                  }
                  val linkedProjectId = if (experience.kind == WebExperienceKind.PROJECT)
                    trustedLinkedProjectId(pageUrl, linkedProjectIds) else null
                  if (linkedProjectId != null && onOpenLinkedProject != null) {
                    view.post { onOpenLinkedProject(linkedProjectId) }
                    return
                  }
                  // The website can change routes with history.pushState without a WebView navigation.
                  openExternalExperience(context, Uri.parse(pageUrl))
                  view.post { onBack() }
                }

                override fun onPageStarted(view: WebView, pageUrl: String?, favicon: Bitmap?) {
                  pendingLayout[0]?.let { view.removeCallbacks(it) }
                  pendingLayout[0] = null
                  loadAttempt += 1
                  showProjectLoadingDetails = false
                  loading = true
                  prepared = false
                  error = false
                }

                override fun onPageFinished(view: WebView, pageUrl: String?) {
                  loading = false
                  pendingLayout[0]?.let { view.removeCallbacks(it) }
                  pendingLayout[0] = null
                  if (pageUrl != null && trustedWebExperienceUrl(experience, pageUrl) != null) {
                    val isStarfall = experience.canonicalPath == WebExperience.STARFALL.canonicalPath
                    val selector = when (experience.kind) {
                      WebExperienceKind.GAME -> if (isStarfall) "[data-starfall-root]" else "main"
                      WebExperienceKind.DEMO -> ".project-demo-wrapper-frame"
                      WebExperienceKind.PROJECT -> ".project-main"
                    }
                    val startCss = if (isStarfall)
                      ".project-starfall-canvas-wrap:has(.project-starfall-start-screen:not([hidden])) { min-height: 300px !important; }"
                    else ""
                    val projectCss = if (experience.kind == WebExperienceKind.PROJECT)
                      ".project-parent-link, .project-question-dock { display: none !important; }"
                    else ""
                    // On the Android WebView device, CSS vh/dvh measured zero although innerHeight was valid.
                    val gameCss = if (experience.canonicalPath == "/games/stellar-dogfight")
                      "body.is-playing { height: var(--android-webview-height) !important; min-height: var(--android-webview-height) !important; } body.is-playing .site-frame__stage { height: var(--android-webview-height) !important; } body.personal-accordion-page[data-personal-item=stellar-dogfight].is-playing .site-frame__stage { grid-template-rows: 0px minmax(0, 1fr) !important; }"
                    else ""
                    val featureLayout = when {
                      isStarfall -> """
                        document.querySelector('.personal-game-header')?.style.setProperty('display', 'none', 'important');
                        document.querySelector('.project-starfall-start-screen')?.style.setProperty('overflow-y', 'auto', 'important');
                      """.trimIndent()
                      experience.kind == WebExperienceKind.DEMO -> """
                        document.querySelector('.project-demo-wrapper-header')?.style.setProperty('display', 'none', 'important');
                        if (!window.androidFeatureFit) {
                          window.androidFeatureFit = () => {
                            const height = window.innerHeight + 'px';
                            document.querySelectorAll('.project-demo-wrapper-main, .project-demo-wrapper-frame, .project-demo-wrapper-iframe').forEach(element => {
                              element.style.setProperty('height', height, 'important');
                            });
                          };
                          window.addEventListener('resize', window.androidFeatureFit);
                        }
                        window.androidFeatureFit();
                      """.trimIndent()
                      experience.canonicalPath == "/games/stellar-dogfight" -> """
                        if (!window.androidStellarFit) {
                          window.androidStellarFit = () => {
                            const height = window.innerHeight + 'px';
                            document.documentElement.style.setProperty('--android-webview-height', height);
                            document.documentElement.style.setProperty('--viewport-height', height);
                          };
                          window.addEventListener('resize', window.androidStellarFit);
                        }
                        window.androidStellarFit();
                      """.trimIndent()
                      else -> ""
                    }
                    val layoutTask = Runnable {
                      pendingLayout[0] = null
                      if (trustedWebExperienceUrl(experience, view.url.orEmpty()) != null) {
                        view.evaluateJavascript("""
                          (() => {
                            if (!document.getElementById('android-feature-chrome')) {
                              const style = document.createElement('style');
                              style.id = 'android-feature-chrome';
                              style.textContent = '[data-site-shell-header], .mobile-site-masthead, .mobile-section-nav, .mobile-site-dock, .site-frame__tab, [data-site-shell-footer] { display: none !important; } .site-frame__stage { grid-template-rows: 0px auto !important; } .site-frame__slot { padding-top: 0 !important; } $startCss $projectCss $gameCss';
                              document.head.appendChild(style);
                            }
                            document.body.style.setProperty('padding-top', '0', 'important');
                            document.body.style.setProperty('padding-bottom', '0', 'important');
                            document.documentElement.style.setProperty('--mobile-site-masthead-height', '0px');
                            document.documentElement.style.setProperty('--mobile-section-nav-height', '0px');
                            document.querySelector('.site-frame__stage')?.style.setProperty('grid-template-rows', '0px auto', 'important');
                            $featureLayout
                            document.querySelector('$selector')?.scrollIntoView({block: 'start'});
                            window.dispatchEvent(new Event('resize'));
                            return !!document.querySelector('$selector');
                          })();
                        """.trimIndent()) { result ->
                          if (result == "true") prepared = true
                          else { error = true; loading = false }
                        }
                      }
                    }
                    pendingLayout[0] = layoutTask
                    view.postDelayed(layoutTask, 450)
                  }
                }

                override fun onReceivedError(view: WebView, request: WebResourceRequest, webError: WebResourceError) {
                  if (request.isForMainFrame || isPrimaryDemoFrameUrl(experience, request.url.toString())) {
                    error = true
                    loading = false
                  }
                }

                override fun onReceivedHttpError(view: WebView, request: WebResourceRequest, response: WebResourceResponse) {
                  if (response.statusCode >= 400 &&
                    (request.isForMainFrame || isPrimaryDemoFrameUrl(experience, request.url.toString()))) {
                    error = true
                    loading = false
                  }
                }

                override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
                  rendererGone = true
                  error = true
                  loading = false
                  prepared = false
                  return true
                }
              }
              webChromeClient = object : WebChromeClient() {
                override fun onCreateWindow(view: WebView, isDialog: Boolean,
                  isUserGesture: Boolean, resultMsg: Message): Boolean {
                  if (!isUserGesture) return false
                  val transport = resultMsg.obj as? WebView.WebViewTransport ?: return false
                  val popup = WebView(context)
                  val handler = Handler(Looper.getMainLooper())
                  var opened = false
                  var closed = false
                  fun closePopup() {
                    if (closed) return
                    closed = true
                    popup.stopLoading()
                    popup.destroy()
                  }
                  fun openOnce(uri: Uri) {
                    if (opened || closed) return
                    opened = true
                    openExternalExperience(context, uri)
                    handler.post { closePopup() }
                  }
                  popup.webViewClient = object : WebViewClient() {
                    override fun shouldOverrideUrlLoading(view: WebView,
                      request: WebResourceRequest): Boolean {
                      if (request.isForMainFrame) openOnce(request.url)
                      return true
                    }

                    override fun onPageStarted(view: WebView, url: String?, favicon: Bitmap?) {
                      if (url != null && url != "about:blank") openOnce(Uri.parse(url))
                    }
                  }
                  handler.postDelayed({ closePopup() }, 10_000L)
                  transport.webView = popup
                  resultMsg.sendToTarget()
                  return true
                }

                override fun onShowCustomView(view: View, callback: CustomViewCallback) {
                  if (fullscreenView != null) { callback.onCustomViewHidden(); return }
                  fullscreenView = view
                  fullscreenCallback = callback
                }

                override fun onHideCustomView() {
                  fullscreenView = null
                  fullscreenCallback = null
                }
              }
              loadUrl(url)
            }
          },
          modifier = Modifier.fillMaxSize().testTag("website-feature-view"),
          onRelease = { view ->
            pendingLayout[0]?.let { view.removeCallbacks(it) }
            pendingLayout[0] = null
            if (fullscreenView != null) closeFullscreen()
            view.stopLoading()
            view.destroy()
          }
        )
        }
        if (!prepared && !error && fullscreenView == null) {
          Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
            BoxWithConstraints(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
              // Keep taps off the WebView while leaving the fallback action clickable above it.
              Box(Modifier.matchParentSize().pointerInput(Unit) {
                awaitPointerEventScope { while (true) awaitPointerEvent().changes.forEach { it.consume() } }
              })
              CircularProgressIndicator()
              if (showProjectLoadingDetails) Column(
                Modifier.align(Alignment.TopCenter)
                  .padding(top = maxHeight / 2 + 22.dp, start = 24.dp, end = 24.dp)
                  .widthIn(max = 460.dp)
                  .verticalScroll(rememberScrollState()),
                horizontalAlignment = Alignment.CenterHorizontally
              ) {
                Text("WEBSITE PROJECT", color = MaterialTheme.colorScheme.primary,
                  style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(8.dp))
                Text("Opening ${experience.title}", style = MaterialTheme.typography.titleLarge,
                  fontSize = 20.sp, fontWeight = FontWeight.Bold, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                Spacer(Modifier.height(8.dp))
                Text("The full case study is loading.", color = MaterialTheme.colorScheme.onSurfaceVariant,
                  style = MaterialTheme.typography.bodyMedium, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                if (onOpenNative != null) {
                  Spacer(Modifier.height(20.dp))
                  OutlinedButton(onClick = onOpenNative, modifier = Modifier.heightIn(min = 48.dp)
                    .testTag("project-loading-offline-summary"), shape = RoundedCornerShape(8.dp),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = .45f))) {
                    Text("Read offline summary")
                  }
                  Spacer(Modifier.height(8.dp))
                  Text("Available without a connection", color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodySmall)
                }
              }
            }
          }
        }
        if (error && fullscreenView == null) {
          Surface(Modifier.fillMaxSize().pointerInput(Unit) {
            awaitPointerEventScope { while (true) awaitPointerEvent().changes.forEach { if (!it.isConsumed) it.consume() } }
          }, color = MaterialTheme.colorScheme.surface) {
            Column(Modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally,
              verticalArrangement = Arrangement.Center) {
              Text(if (rendererGone) "${experience.title} stopped" else "Couldn’t load ${experience.title}",
                style = MaterialTheme.typography.titleLarge)
              Spacer(Modifier.height(8.dp))
              Text(if (rendererGone) "Restart the page to continue." else "Check your connection and try again.")
              Spacer(Modifier.height(16.dp))
              Button(onClick = {
                error = false
                loading = true
                prepared = false
                if (rendererGone) {
                  rendererVersion += 1
                  rendererGone = false
                } else webView.loadUrl(url)
              }) { Text(if (rendererGone) "Restart" else "Retry") }
              if (onOpenNative != null) TextButton(onClick = onOpenNative) {
                Text(if (hasStarfallPlayOptions) "Open Offline Expedition" else "Open $nativeLabel")
              }
              TextButton(onClick = { openExternalExperience(context, Uri.parse(url)) }) { Text("Open in browser") }
            }
          }
        }
      }
    }
    val customView = fullscreenView
    if (customView != null) {
      AndroidView(factory = { customView }, modifier = Modifier.fillMaxSize().testTag("website-fullscreen-view"))
    }
  }
  if (hasStarfallPlayOptions && playOptionsExpanded) {
    ModalBottomSheet(onDismissRequest = { playOptionsExpanded = false },
      scrimColor = Color.Transparent, containerColor = MaterialTheme.colorScheme.surface,
      shape = RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)) {
      Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, bottom = 32.dp)) {
        Text("Play options", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(14.dp))
        val accent = MaterialTheme.colorScheme.primary
        OutlinedCard(onClick = { playOptionsExpanded = false },
          modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp).testTag("starfall-website-choice"),
          shape = RoundedCornerShape(8.dp), border = BorderStroke(1.dp, accent.copy(alpha = .45f)),
          colors = CardDefaults.outlinedCardColors(containerColor = Color(0xFFFFF8F3))) {
          Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
              Text("Website game", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
              Text("Full experience in this app", style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text("✓ Current", color = accent, fontWeight = FontWeight.Bold,
              style = MaterialTheme.typography.labelLarge)
          }
        }
        Spacer(Modifier.height(8.dp))
        OutlinedCard(onClick = { playOptionsExpanded = false; onOpenNative?.invoke() },
          modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp).testTag("starfall-offline-choice"),
          shape = RoundedCornerShape(8.dp), border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline)) {
          Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
              Text("Offline Expedition", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
              Text("Compact Android game · progress on this device",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text("→", color = accent, style = MaterialTheme.typography.titleMedium)
          }
        }
        Spacer(Modifier.height(12.dp))
        Text("Progress stays separate between versions.",
          style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
      }
    }
  }
}

private fun openExternalExperience(context: Context, uri: Uri) {
  if (uri.scheme !in setOf("https", "http", "mailto", "tel")) return
  val intent = Intent(Intent.ACTION_VIEW, uri).addCategory(Intent.CATEGORY_BROWSABLE)
  try {
    context.startActivity(intent)
  } catch (_: ActivityNotFoundException) {
    Toast.makeText(context, "No app can open this link", Toast.LENGTH_SHORT).show()
  }
}
