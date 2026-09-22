package me.danielshort.app.ui

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.automirrored.outlined.ArrowForward
import androidx.compose.material.icons.automirrored.outlined.OpenInNew
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil.compose.AsyncImage
import kotlinx.coroutines.launch
import me.danielshort.app.R
import me.danielshort.app.data.*
import me.danielshort.app.nativefeatures.NativeGameScreen
import me.danielshort.app.nativefeatures.NativeTextCompareScreen
import me.danielshort.app.nativefeatures.tools.NativeToolsScreen
import me.danielshort.app.nativefeatures.tools.NATIVE_TOOL_IDS
import me.danielshort.app.nativefeatures.games.NativeGamesScreen
import me.danielshort.app.nativefeatures.games.NATIVE_GAME_IDS
import me.danielshort.app.nativefeatures.demos.NativeProjectDemoScreen
import me.danielshort.app.nativefeatures.demos.NATIVE_DEMO_IDS
import me.danielshort.app.nativefeatures.recording.NativeScreenRecorder

private val Navy = Color(0xFF091F3B)
private val Blue = Color(0xFF155DFC)
private val Teal = Color(0xFF087F8C)
private val Orange = Color(0xFFC94B0A)
private val Slate = Color(0xFF334155)
private val Muted = Color(0xFF586B82)
private val Line = Color(0xFFDCE4ED)
private val Paper = Color(0xFFF5F8FB)
internal enum class Section(val label: String, val color: Color, val icon: ImageVector) {
  ABOUT("About", Navy, Icons.Outlined.Person),
  PROJECTS("Projects", Blue, Icons.Outlined.FolderOpen),
  TOOLS("Tools", Teal, Icons.Outlined.Build),
  GAMES("Games", Orange, Icons.Outlined.SportsEsports),
  CONTACT("Contact", Slate, Icons.Outlined.ChatBubbleOutline)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DanielShortApp(
  repository: ContentRepository,
  onSafeToInstall: (Boolean) -> Unit = {},
  globalNotice: @Composable (onOpenSettings: () -> Unit) -> Unit = {}
) {
  val state by repository.state.collectAsStateWithLifecycle()
  val saved by repository.favorites.collectAsStateWithLifecycle()
  val settings by repository.settings.state.collectAsStateWithLifecycle()
  val reduceMotion = effectiveReduceMotion(settings.reduceMotion, rememberSystemReduceMotion())
  var selectedName by rememberSaveable { mutableStateOf(Section.ABOUT.name) }
  var projectId by rememberSaveable { mutableStateOf<String?>(null) }
  var nativeFeature by rememberSaveable { mutableStateOf<String?>(null) }
  val section = Section.valueOf(selectedName)
  val scope = rememberCoroutineScope()
  val snackbar = remember { SnackbarHostState() }
  val context = LocalContext.current
  val screenState = rememberSaveableStateHolder()
  val content = state.content
  val project = content?.projects?.find { it.id == projectId }
  val safeToInstallCallback by rememberUpdatedState(onSafeToInstall)
  DisposableEffect(nativeFeature, projectId) {
    safeToInstallCallback(nativeFeature == null && projectId == null)
    onDispose { safeToInstallCallback(false) }
  }
  // Settings owns feedback for its own actions; do not replay that feedback on return.
  LaunchedEffect(state.message) {
    if (nativeFeature == null && state.message.isNotBlank()) snackbar.showSnackbar(state.message)
  }
  MaterialTheme(colorScheme = lightColorScheme(primary = section.color, onPrimary = Color.White, primaryContainer = section.color.copy(alpha = .10f), onPrimaryContainer = Navy, secondary = Teal, secondaryContainer = section.color.copy(alpha = .10f), onSecondaryContainer = Navy, background = Color.White, surface = Color.White, surfaceTint = section.color, surfaceContainer = Paper, surfaceContainerLow = Paper, onSurface = Navy, onBackground = Navy, surfaceVariant = Paper, onSurfaceVariant = Muted, outline = Line, outlineVariant = Line)) {
    CompositionLocalProvider(LocalNativeReduceMotion provides reduceMotion) {
    Surface(modifier = Modifier.fillMaxSize(), color = Color.White, contentColor = Navy) {
    AdaptiveSiteLayout(
      selected = section,
      onSection = { selectedName = it.name; projectId = null; nativeFeature = null }
    ) { wide ->
    if (nativeFeature != null) {
      BackHandler { nativeFeature = null }
      val back = { nativeFeature = null }
      val feature = nativeFeature!!
      val featureWidth = when {
        feature == "settings" || feature == "settings:updates" -> 760.dp
        feature.startsWith("game:") || feature == "roulette" -> 1200.dp
        else -> 1000.dp
      }
      Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
      Box(Modifier.widthIn(max = featureWidth).fillMaxSize().testTag("native-feature-panel")) {
      screenState.SaveableStateProvider("native:$feature") {
      when {
        feature == "settings" || feature == "settings:updates" -> SettingsScreen(repository, back,
          initialPage = if (feature == "settings:updates") SettingsPage.UPDATES else SettingsPage.OVERVIEW)
        nativeFeature == "tool:text-compare" || nativeFeature == "text-compare" -> NativeTextCompareScreen(back)
        nativeFeature == "tool:screen-recorder" -> NativeScreenRecorder(back)
        nativeFeature == "game:roulette" || nativeFeature == "roulette" -> NativeGameScreen(back)
        nativeFeature!!.startsWith("tool:") -> NativeToolsScreen(nativeFeature!!.substringAfter(":"), back)
        nativeFeature!!.startsWith("game:") -> NativeGamesScreen(nativeFeature!!.substringAfter(":"), back, reduceMotion)
        nativeFeature!!.startsWith("demo:") -> NativeProjectDemoScreen(nativeFeature!!.substringAfter(":"), back)
      }
      }
      }
      }
    } else {
    BackHandler(projectId != null) { projectId = null }
    ScrollChromeLayout(
      screenKey = "${section.name}:${project?.id.orEmpty()}",
      snackbar = { SnackbarHost(snackbar) },
      topBar = {
        Column(Modifier.background(Color.White)) {
          TopAppBar(title = {
            if (project != null) Text("Project", style = MaterialTheme.typography.titleLarge)
            else Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
              Image(painterResource(R.drawable.brand_mark), contentDescription = null, modifier = Modifier.size(34.dp))
              Text("Daniel Short", fontWeight = FontWeight.Bold, fontSize = 20.sp)
            }
          }, navigationIcon = {
            if (project != null) IconButton(onClick = { projectId = null }) { Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back to projects") }
          }, actions = {
            IconButton(onClick = { nativeFeature = "settings" }) { Icon(Icons.Outlined.Settings, "Settings") }
            if (project != null) {
              IconButton(onClick = { repository.toggleSaved(project.id) }) { Icon(if (project.id in saved) Icons.Outlined.BookmarkAdded else Icons.Outlined.BookmarkBorder, if (project.id in saved) "Unsave project" else "Save project") }
              IconButton(onClick = { share(context, project.title, project.url) }) { Icon(Icons.Outlined.Share, "Share project") }
            } else {
              BrowseOverflowMenu(state.refreshing) { scope.launch { repository.refresh(force = true) } }
            }
          }, colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.White))
          globalNotice { nativeFeature = "settings:updates" }
          HorizontalDivider(thickness = 2.dp, color = section.color.copy(alpha = .35f))
        }
      },
      bottomBar = if (wide) null else {
        {
        Column(Modifier.background(Color.White)) {
        HorizontalDivider(color = Line)
        NavigationBar(containerColor = Color.White, tonalElevation = 0.dp) {
          Section.entries.forEach { tab ->
            NavigationBarItem(selected = section == tab, onClick = { selectedName = tab.name; projectId = null }, icon = { Icon(tab.icon, contentDescription = null) }, label = { Text(tab.label, fontSize = 11.sp, fontWeight = if (section == tab) FontWeight.Bold else FontWeight.Medium) }, colors = NavigationBarItemDefaults.colors(selectedIconColor = tab.color, selectedTextColor = tab.color, indicatorColor = Color.Transparent, unselectedIconColor = Muted, unselectedTextColor = Muted))
          }
        }
        }
      }
      }
    ) { padding ->
      val reading = project != null || section == Section.ABOUT || section == Section.CONTACT
      Box(Modifier.fillMaxSize(), contentAlignment = Alignment.TopCenter) {
      val body = Modifier.widthIn(max = if (reading) 760.dp else 1080.dp).fillMaxSize()
      val contentPadding = PaddingValues(start = 22.dp, end = 22.dp,
        top = padding.calculateTopPadding() + 22.dp, bottom = padding.calculateBottomPadding() + 22.dp)
      if (content == null) {
        Box(body, contentAlignment = Alignment.Center) { if (state.refreshing) CircularProgressIndicator() else Text("Preparing your content…", color = Muted) }
      } else if (project != null) {
        screenState.SaveableStateProvider("project:${project.id}") {
        ProjectDetail(project, content.site, body, contentPadding, onDemo = { nativeFeature = "demo:${project.id}" })
        }
      } else screenState.SaveableStateProvider(section.name) { when (section) {
        Section.ABOUT -> AboutScreen(content, body, contentPadding, onProjects = { selectedName = Section.PROJECTS.name })
        Section.PROJECTS -> ProjectsScreen(content.projects, saved, body, contentPadding,
          onProject = { projectId = it }, onRemoveSaved = {
            repository.clearSavedProjects()
            scope.launch { snackbar.showSnackbar("Saved projects removed") }
          })
        Section.TOOLS -> CatalogScreen("Useful little utilities", "Practical tools for everyday tasks.", content.tools, Teal, body, contentPadding, NATIVE_TOOL_IDS + setOf("text-compare", "screen-recorder"), "Open tool", onNative = { nativeFeature = "tool:$it" })
        Section.GAMES -> CatalogScreen("Play and explore", "Games, simulations, and small experiments.", content.games, Orange, body, contentPadding, NATIVE_GAME_IDS + "roulette", "Play in app", onNative = { nativeFeature = "game:$it" })
        Section.CONTACT -> ContactScreen(content.site, content.about.location, body, contentPadding)
      } }
      }
    }
    }
    }
    }
    }
  }
}

@Composable
private fun Heading(title: String, subtitle: String = "") {
  Column {
  Text(title, style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, color = Navy)
  if (subtitle.isNotBlank()) { Spacer(Modifier.height(8.dp)); Text(subtitle, style = MaterialTheme.typography.bodyLarge, color = Muted) }
  HorizontalDivider(Modifier.padding(top = 18.dp), thickness = 2.dp, color = MaterialTheme.colorScheme.primary)
  }
}

@Composable
private fun AboutScreen(content: SiteContent, modifier: Modifier, contentPadding: PaddingValues, onProjects: () -> Unit) {
  val about = content.about
  val context = LocalContext.current
  LazyColumn(modifier, contentPadding = contentPadding, verticalArrangement = Arrangement.spacedBy(24.dp)) {
    item {
      Row(horizontalArrangement = Arrangement.spacedBy(16.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(86.dp).clip(RoundedCornerShape(18.dp)).background(Paper), contentAlignment = Alignment.Center) {
          Icon(Icons.Outlined.Person, null, Modifier.size(40.dp), tint = Navy)
          AsyncImage(about.portraitUrl, "Daniel Short", Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
        }
        Column(Modifier.weight(1f)) { Text(about.greeting.ifBlank { "Hi, I’m Daniel." }, fontSize = 27.sp, lineHeight = 32.sp, fontWeight = FontWeight.Bold); Spacer(Modifier.height(6.dp)); Text(about.location, color = Muted, style = MaterialTheme.typography.bodyMedium) }
      }
      Spacer(Modifier.height(20.dp))
      Text(about.intro, style = MaterialTheme.typography.bodyLarge, lineHeight = 25.sp)
      Spacer(Modifier.height(18.dp))
      Button(onClick = onProjects, shape = RoundedCornerShape(10.dp)) { Text("Explore projects"); Spacer(Modifier.width(8.dp)); Icon(Icons.AutoMirrored.Outlined.ArrowForward, null, Modifier.size(18.dp)) }
    }
    if (about.interests.isNotEmpty()) item {
      SectionTitle("Life shapes what I build")
      about.interests.forEach { interest ->
        Row(Modifier.padding(top = 18.dp), horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.CenterVertically) {
          NativeThumbnail(interest.imageUrl, interest.title, Teal)
          Column(Modifier.weight(1f)) { Text(interest.title, fontWeight = FontWeight.SemiBold); if (interest.body.isNotBlank()) Text(interest.body, color = Muted, modifier = Modifier.padding(top = 4.dp)) }
        }
      }
    }
    if (about.experience.isNotEmpty()) item { Milestones("Experience", about.experience, context) }
    if (about.education.isNotEmpty()) item { Milestones("Education", about.education, context) }
    if (about.credentials.isNotEmpty()) item { Milestones("Credentials", about.credentials, context) }
    item { Text("Content is saved on your device and refreshed from the website.", color = Muted, style = MaterialTheme.typography.bodySmall) }
  }
}

@Composable
private fun SectionTitle(title: String) {
  Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
  HorizontalDivider(Modifier.padding(top = 10.dp), color = Line)
}

@Composable
private fun Milestones(title: String, entries: List<Milestone>, context: Context) {
  SectionTitle(title)
  entries.forEach { entry ->
    Column(Modifier.fillMaxWidth().then(if (entry.url.isNotBlank()) Modifier.clickable { openWeb(context, entry.url) } else Modifier).padding(vertical = 13.dp)) {
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(entry.title, Modifier.weight(1f), fontWeight = FontWeight.SemiBold, color = if (entry.url.isNotBlank()) Blue else Navy)
        if (entry.url.isNotBlank()) Icon(Icons.AutoMirrored.Outlined.OpenInNew, "Open credential in browser", Modifier.size(16.dp), tint = Blue)
      }
      if (entry.organization.isNotBlank()) Text(entry.organization, color = Muted, modifier = Modifier.padding(top = 4.dp))
      if (entry.date.isNotBlank()) Text(entry.date, color = Muted, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 5.dp))
    }
  }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ProjectsScreen(projects: List<Project>, saved: Set<String>, modifier: Modifier, contentPadding: PaddingValues, onProject: (String) -> Unit, onRemoveSaved: () -> Unit) {
  var query by rememberSaveable { mutableStateOf("") }
  var savedOnly by rememberSaveable { mutableStateOf(false) }
  val inputFocus = LocalChromeInputFocus.current
  val filtered = projects.filter { (!savedOnly || it.id in saved) && (it.title + " " + it.summary + " " + it.tags.joinToString(" ")).contains(query.trim(), ignoreCase = true) }
  LazyVerticalGrid(columns = GridCells.Adaptive(adaptiveCardMinWidth()), modifier = modifier.testTag("projects-list"),
    contentPadding = contentPadding, verticalArrangement = Arrangement.spacedBy(14.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
    item(key = "heading", span = { GridItemSpan(maxLineSpan) }) { Heading("Projects", "A collection of ideas put into practice.") }
    item(key = "filters", span = { GridItemSpan(maxLineSpan) }) {
      Column {
      OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth().testTag("project-search").onFocusChanged { inputFocus(it.isFocused) }, label = { Text("Search projects") }, leadingIcon = { Icon(Icons.Outlined.Search, null) }, trailingIcon = { if (query.isNotEmpty()) IconButton(onClick = { query = "" }) { Icon(Icons.Outlined.Close, "Clear search") } }, singleLine = true, shape = RoundedCornerShape(12.dp))
      Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        FlowRow(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          FilterChip(selected = !savedOnly, onClick = { savedOnly = false }, label = { Text("All ${projects.size}") })
          FilterChip(selected = savedOnly, onClick = { savedOnly = true }, label = { Text("Saved ${saved.count { id -> projects.any { it.id == id } }}") }, leadingIcon = { Icon(Icons.Outlined.BookmarkBorder, null, Modifier.size(16.dp)) })
        }
        if (savedOnly && saved.isNotEmpty()) SavedProjectsMenu(saved.size, onRemoveSaved)
      }
      }
    }
    if (filtered.isEmpty()) item(key = "empty", span = { GridItemSpan(maxLineSpan) }) { EmptyResult(if (savedOnly) "No saved projects yet" else "No projects found", if (savedOnly) "Open a project and use the bookmark to keep it here." else "Try a different title, topic, or tool.") }
    items(filtered, key = { it.id }) { project ->
      OutlinedCard(onClick = { onProject(project.id) }, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), border = BorderStroke(1.dp, Line), colors = CardDefaults.outlinedCardColors(containerColor = Color.White)) {
        Row(Modifier.padding(17.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
          NativeThumbnail(project.iconUrl, project.title, Blue)
          Column(Modifier.weight(1f)) {
            Text(project.title, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(5.dp)); Text(project.summary, color = Muted, style = MaterialTheme.typography.bodyMedium, maxLines = 3, overflow = TextOverflow.Ellipsis)
            Spacer(Modifier.height(10.dp)); Text(if (project.id in saved) "Saved project" else "View project  →", color = Blue, fontWeight = FontWeight.SemiBold, style = MaterialTheme.typography.labelLarge)
          }
        }
      }
    }
  }
}

@Composable
private fun NativeThumbnail(url: String, title: String, color: Color) {
  Box(Modifier.size(58.dp).clip(RoundedCornerShape(12.dp)).background(color.copy(alpha = .07f)).clearAndSetSemantics { }, contentAlignment = Alignment.Center) {
    Text(title.take(1), fontSize = 24.sp, color = color, fontWeight = FontWeight.Bold)
    if (url.isNotBlank()) AsyncImage(url, null, Modifier.fillMaxSize().padding(7.dp), contentScale = ContentScale.Fit)
  }
}

@Composable
private fun CatalogScreen(title: String, subtitle: String, entries: List<CatalogItem>, accent: Color, modifier: Modifier, contentPadding: PaddingValues, nativeIds: Set<String>, nativeLabel: String, onNative: (String) -> Unit) {
  val context = LocalContext.current
  var query by rememberSaveable(title) { mutableStateOf("") }
  val inputFocus = LocalChromeInputFocus.current
  val filtered = entries.filter { (it.title + " " + it.summary + " " + it.category).contains(query.trim(), ignoreCase = true) }
  LazyVerticalGrid(columns = GridCells.Adaptive(adaptiveCardMinWidth()), modifier = modifier.testTag("catalog-list"),
    contentPadding = contentPadding, verticalArrangement = Arrangement.spacedBy(12.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
    item(key = "heading", span = { GridItemSpan(maxLineSpan) }) { Heading(title, subtitle) }
    item(key = "search", span = { GridItemSpan(maxLineSpan) }) { OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth().testTag("catalog-search").onFocusChanged { inputFocus(it.isFocused) }, label = { Text("Search") }, leadingIcon = { Icon(Icons.Outlined.Search, null) }, singleLine = true, shape = RoundedCornerShape(12.dp)) }
    if (filtered.isEmpty()) item(key = "empty", span = { GridItemSpan(maxLineSpan) }) { EmptyResult("No matches", "Try another name or topic.") }
    items(filtered, key = { it.id }) { entry ->
      val isNative = entry.id in nativeIds
      OutlinedCard(onClick = { if (isNative) onNative(entry.id) else openWeb(context, entry.url) }, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp), border = BorderStroke(1.dp, Line), colors = CardDefaults.outlinedCardColors(containerColor = Color.White)) {
          Row(Modifier.padding(16.dp), horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.CenterVertically) {
            NativeThumbnail(entry.iconUrl, entry.title, accent)
            Column(Modifier.weight(1f)) {
              Text(entry.title, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
              Spacer(Modifier.height(5.dp))
              val summary = when (entry.id) {
                "screen-recorder" -> "Record a screen or app to MP4 with an optional microphone."
                "ocean-wave-simulation" -> "Adjust waves, wind, and daylight in a native ocean simulation."
                else -> entry.summary
              }
              Text(summary, color = Muted, style = MaterialTheme.typography.bodyMedium)
              Text(if (isNative) nativeLabel else "Open in browser", color = accent, style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold, modifier = Modifier.padding(top = 9.dp))
            }
            Icon(if (isNative) Icons.AutoMirrored.Outlined.ArrowForward else Icons.AutoMirrored.Outlined.OpenInNew,
              contentDescription = null, modifier = Modifier.size(18.dp), tint = accent)
          }
      }
    }
  }
}

@Composable
private fun ProjectDetail(project: Project, site: SiteInfo, modifier: Modifier, contentPadding: PaddingValues, onDemo: () -> Unit) {
  val context = LocalContext.current
  LazyColumn(modifier, contentPadding = contentPadding, verticalArrangement = Arrangement.spacedBy(22.dp)) {
    item { Heading(project.title, project.summary) }
    if (project.id in NATIVE_DEMO_IDS) item {
      Button(onClick = onDemo, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) { Text("Open demo") }
    }
    if (project.imageUrl.isNotBlank()) item {
      Box(Modifier.fillMaxWidth().aspectRatio(1.6f).clip(RoundedCornerShape(14.dp)).background(Paper), contentAlignment = Alignment.Center) {
        Icon(Icons.Outlined.Photo, null, Modifier.size(42.dp), tint = Line)
        AsyncImage(project.imageUrl, "Preview of ${project.title}", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
      }
    }
    if (project.situation.isNotBlank()) item { TextSection("The idea", project.situation) }
    if (project.task.isNotBlank()) item { TextSection("The goal", project.task) }
    if (project.actions.isNotEmpty()) item { BulletSection("What I built", project.actions) }
    if (project.results.isNotEmpty()) item { BulletSection("The result", project.results) }
    if (project.demoUrl.isNotBlank() && project.id !in NATIVE_DEMO_IDS) item {
      OutlinedButton(onClick = { openWeb(context, project.demoUrl) }, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) { Icon(Icons.AutoMirrored.Outlined.OpenInNew, null, Modifier.size(18.dp)); Spacer(Modifier.width(8.dp)); Text("Open web demo") }
    }
    if (project.resources.isNotEmpty()) item {
      SectionTitle("Resources")
      project.resources.filterNot {
        project.id in NATIVE_DEMO_IDS && (it.url.removeSuffix(".html") == project.demoUrl.removeSuffix(".html") || it.label.equals("Live Demo", ignoreCase = true))
      }.forEach { resource -> TextButton(onClick = { openWeb(context, resource.url) }) { Text(resource.label, Modifier.weight(1f)); Icon(Icons.AutoMirrored.Outlined.OpenInNew, null, Modifier.size(16.dp)) } }
    }
    item { Button(onClick = { email(context, site.email, "Question about ${project.title}") }, Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) { Icon(Icons.Outlined.ChatBubbleOutline, null, Modifier.size(18.dp)); Spacer(Modifier.width(8.dp)); Text("Ask about this project") } }
  }
}

@Composable
private fun TextSection(title: String, body: String) { SectionTitle(title); Text(body, Modifier.padding(top = 12.dp), style = MaterialTheme.typography.bodyLarge, lineHeight = 25.sp) }
@Composable
private fun BulletSection(title: String, lines: List<String>) { SectionTitle(title); lines.forEach { Row(Modifier.padding(top = 12.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) { Text("•", color = Blue); Text(it, Modifier.weight(1f), lineHeight = 24.sp) } } }
@Composable
private fun EmptyResult(title: String, body: String) { Column(Modifier.fillMaxWidth().padding(vertical = 32.dp), horizontalAlignment = Alignment.CenterHorizontally) { Icon(Icons.Outlined.Search, null, Modifier.size(34.dp), tint = Muted); Text(title, Modifier.padding(top = 12.dp), fontWeight = FontWeight.Bold); Text(body, Modifier.padding(top = 8.dp), color = Muted) } }

@Composable
private fun ContactScreen(site: SiteInfo, location: String, modifier: Modifier, contentPadding: PaddingValues) {
  val context = LocalContext.current
  LazyColumn(modifier, contentPadding = contentPadding, verticalArrangement = Arrangement.spacedBy(20.dp)) {
    item { Heading("Say hello", "A project, a question, or an idea — I’d like to hear it.") }
    item { ContactCard("Send a message", site.email, Icons.Outlined.MailOutline) { email(context, site.email, "Hello from the Android app") } }
    if (site.githubUrl.isNotBlank()) item { ContactCard("GitHub", "Explore the code behind the projects", Icons.Outlined.Code) { openWeb(context, site.githubUrl) } }
    item { Text(location, color = Muted, style = MaterialTheme.typography.bodyLarge) }
    if (site.privacyUrl.isNotBlank()) item { TextButton(onClick = { openWeb(context, site.privacyUrl) }) { Text("Website privacy details"); Spacer(Modifier.width(7.dp)); Icon(Icons.AutoMirrored.Outlined.OpenInNew, null, Modifier.size(16.dp)) } }
  }
}

@Composable
private fun ContactCard(title: String, body: String, icon: ImageVector, onClick: () -> Unit) {
  OutlinedCard(onClick = onClick, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(16.dp), border = BorderStroke(1.dp, Line), colors = CardDefaults.outlinedCardColors(containerColor = Color.White)) {
    Row(Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) { Icon(icon, null, Modifier.size(28.dp), tint = Slate); Column(Modifier.weight(1f)) { Text(title, fontWeight = FontWeight.Bold); Text(body, Modifier.padding(top = 5.dp), color = Muted) }; Icon(Icons.AutoMirrored.Outlined.ArrowForward, null, Modifier.size(18.dp)) }
  }
}

private fun openWeb(context: Context, value: String) {
  val safe = SiteContentParser.safeUrl(value)
  if (safe.isBlank()) return
  launch(context, Intent(Intent.ACTION_VIEW, Uri.parse(safe)))
}
private fun email(context: Context, address: String, subject: String) {
  launch(context, Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:$address?subject=${Uri.encode(subject)}")))
}
private fun share(context: Context, title: String, url: String) {
  launch(context, Intent.createChooser(Intent(Intent.ACTION_SEND).apply { type = "text/plain"; putExtra(Intent.EXTRA_SUBJECT, title); putExtra(Intent.EXTRA_TEXT, "$title\n$url") }, "Share project"))
}
private fun launch(context: Context, intent: Intent) {
  try { context.startActivity(intent) } catch (_: ActivityNotFoundException) { Toast.makeText(context, "No app is available for this action.", Toast.LENGTH_SHORT).show() }
}
