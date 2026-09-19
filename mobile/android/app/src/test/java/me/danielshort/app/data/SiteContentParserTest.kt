package me.danielshort.app.data

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class SiteContentParserTest {
  private fun repositoryRoot(): File = generateSequence(File(System.getProperty("user.dir") ?: ".").absoluteFile) { it.parentFile }
    .firstOrNull { File(it, "content/audiences/personal.json").isFile }
    ?: error("Run the native unit tests from this repository")

  private fun fixture(): JSONObject = JSONObject(File(repositoryRoot(), "dist/app-content/v1/catalog.json").readText())

  private fun parse(document: JSONObject): SiteContent = SiteContentParser.parse(document.toString())

  private fun assertRejected(document: JSONObject) {
    try {
      parse(document)
      fail("Expected the malformed content document to be rejected")
    } catch (expected: IllegalArgumentException) {
      // Parser validation failed before the new content could replace the cache.
    } catch (expected: org.json.JSONException) {
      // Missing or incorrectly typed required JSON fields are also invalid.
    }
  }

  @Test fun realGeneratedCatalogIncludesEveryPublishedPersonalProject() {
    val content = parse(fixture())
    val sourceProjects = File(repositoryRoot(), "content/projects").listFiles().orEmpty()
      .filter { it.extension == "json" }
      .map { JSONObject(it.readText()) }
      .filter { it.optBoolean("published", true) && !it.optBoolean("hidden") && !it.optBoolean("noindex") }
    assertEquals(sourceProjects.map { it.getString("id") }.toSet(), content.projects.map { it.id }.toSet())
    assertEquals(sourceProjects.size, content.projects.size)
    assertFalse(content.projects.any { it.id == "minesweeper" })
    assertTrue(content.projects.all { it.url == "https://www.danielshort.me/portfolio/${it.id}" })
    assertTrue(content.projects.all { it.imageUrl.startsWith("https://www.danielshort.me/img/") && it.imageUrl.contains("?v=") })
    assertEquals("Daniel Short", content.site.name)
    assertEquals("https://www.danielshort.me/", content.site.url)
    assertTrue(content.tools.isNotEmpty())
    assertFalse(content.tools.any { it.id in setOf("transcribe", "job-application-tracker", "short-links") })
    assertTrue(content.games.isNotEmpty())
  }

  @Test fun nativeAboutPreservesThePersonalHomepageCopyAndMilestones() {
    val source = JSONObject(File(repositoryRoot(), "content/audiences/personal.json").readText())
    val sections = source.getJSONObject("page").getJSONArray("sections")
    val categories = (0 until sections.length()).flatMap { index ->
      val items = sections.getJSONObject(index).optJSONObject("props")?.optJSONArray("categories") ?: JSONArray()
      (0 until items.length()).map(items::getJSONObject)
    }
    val original = categories.first { it.getString("id") == "about" }
    val about = parse(fixture()).about
    assertEquals(original.getString("title"), about.greeting)
    assertEquals(original.getString("lead"), about.intro)
    assertEquals(original.getString("context"), about.location)
    assertEquals("Playing for 20 years.", about.interests.first { it.title == "French horn" }.body)
    assertEquals("A central part of my life.", about.interests.first { it.title == "Family" }.body)
    assertEquals("Visit Grand Junction", about.experience.first().organization)
    assertEquals("Feb 2024–present", about.experience.first().date)
    assertTrue(about.education.first { it.title == "B.S. Data Analytics" }.url.startsWith("https://www.credential.net/"))
  }

  @Test fun incompatibleOrMissingSchemaIsRejected() {
    listOf(0, 2, -1, 1.5, "1", true, "unsupported", JSONObject.NULL).forEach { schema ->
      assertRejected(fixture().put("schemaVersion", schema))
    }
    assertRejected(fixture().apply { remove("schemaVersion") })
  }

  @Test fun malformedRevisionIsRejected() {
    listOf("", "a".repeat(63), "a".repeat(65), "A".repeat(64), "g".repeat(64), "revision-1", JSONObject.NULL).forEach { revision ->
      assertRejected(fixture().put("revision", revision))
    }
    assertRejected(fixture().apply { remove("revision") })
  }

  @Test fun duplicateIdsAreRejectedWithinEveryCatalog() {
    listOf("projects", "tools", "games").forEach { key ->
      val source = fixture()
      val items = source.getJSONArray(key)
      items.put(JSONObject(items.getJSONObject(0).toString()))
      assertRejected(source)
    }
  }

  @Test fun identifiersCannotBeRoutesOrExecutableCommands() {
    listOf("", "../private", "name/route", "javascript:alert(1)", "has space", "a".repeat(82)).forEach { id ->
      assertRejected(fixture().apply { getJSONArray("projects").getJSONObject(0).put("id", id) })
    }
  }

  @Test fun insecureRequiredLinksRejectContentInsteadOfReplacingValidCache() {
    val unsafe = listOf("http://www.danielshort.me/", "javascript:alert(1)", "file:///private", "data:text/html,hello", "intent://settings", "//example.com/path", "/portfolio/example", "https://user:password@example.com/", "https://example.com/\nprivate", "https://localhost/", "https://127.0.0.1/", "https://10.0.0.1/", "https://[::1]/", "https://private.internal/", "https://example.com:8443/")
    unsafe.forEach { url ->
      assertEquals("Unsafe URL must be rejected: $url", "", SiteContentParser.safeUrl(url))
      assertRejected(fixture().apply { getJSONArray("projects").getJSONObject(0).put("url", url) })
    }
  }

  @Test fun insecureOptionalLinksAreRemovedWithoutLosingPublicText() {
    val source = fixture()
    source.getJSONObject("site").put("githubUrl", "javascript:alert(1)").put("email", "bad@example.com?subject=attack")
    source.getJSONObject("about").put("portraitUrl", "http://example.com/avatar.jpg")
    val project = source.getJSONArray("projects").getJSONObject(0)
    project.put("demoUrl", "intent://settings").put("iconUrl", "file:///private")
    project.put("resources", JSONArray().put(JSONObject().put("label", "Unsafe").put("url", "javascript:alert(1)")))
    val parsed = parse(source)
    assertEquals("", parsed.site.githubUrl)
    assertEquals("", parsed.site.email)
    assertEquals("", parsed.about.portraitUrl)
    assertEquals("", parsed.projects.first().demoUrl)
    assertEquals("", parsed.projects.first().iconUrl)
    assertTrue(parsed.projects.first().resources.isEmpty())
    assertEquals(project.getString("title"), parsed.projects.first().title)
  }

  @Test fun imagesCanOnlyLoadFromTheSitesPublicImageDirectory() {
    listOf("https://example.com/profile.png", "https://www.danielshort.me/api/private", "https://www.danielshort.me/documents/file.pdf").forEach { url ->
      val source = fixture()
      source.getJSONObject("about").put("portraitUrl", url)
      source.getJSONObject("about").getJSONArray("interests").getJSONObject(0).put("imageUrl", url)
      source.getJSONArray("projects").getJSONObject(0).put("imageUrl", url).put("iconUrl", url)
      source.getJSONArray("tools").getJSONObject(0).put("iconUrl", url)
      val parsed = parse(source)
      assertEquals("", parsed.about.portraitUrl)
      assertEquals("", parsed.about.interests.first().imageUrl)
      assertEquals("", parsed.projects.first().imageUrl)
      assertEquals("", parsed.projects.first().iconUrl)
      assertEquals("", parsed.tools.first().iconUrl)
    }
    assertEquals("https://github.com/danielshort3", SiteContentParser.safeUrl("https://github.com/danielshort3"))
  }

  @Test fun dataCannotInstallScriptsHtmlOrAnExecutableInterface() {
    val source = fixture()
    val control = parse(source)
    val code = "java.lang.System.setProperty('native-content-executed', 'true')"
    source.put("script", code).put("html", "<script>$code</script>")
    source.put("evaluate", code).put("screen", JSONObject().put("type", "WebView").put("url", "javascript:alert(1)"))
    source.getJSONArray("projects").getJSONObject(0).put("onClick", code).put("javascript", code)
    assertEquals("Unknown executable-looking fields must have no effect on the native data model", control, parse(source))
    assertEquals(null, System.getProperty("native-content-executed"))

    // HTML-looking copy is just a Kotlin string consumed by Compose Text; it
    // must never be evaluated, loaded into a WebView, or converted to handlers.
    val literal = "<script>alert('not executable')</script> & visible text"
    source.getJSONObject("about").put("intro", literal)
    assertEquals(literal, parse(source).about.intro)
  }

  @Test fun malformedDocumentsAndUnboundedCollectionsAreRejected() {
    assertRejected(fixture().put("projects", JSONArray()))
    assertRejected(fixture().put("projects", JSONArray().put("not an object")))
    assertRejected(fixture().apply { getJSONArray("projects").getJSONObject(0).remove("title") })
    assertRejected(fixture().put("tools", JSONArray().apply { repeat(301) { put(JSONObject().put("id", "tool-$it").put("title", "Tool").put("url", "https://www.danielshort.me/tools/tool-$it")) } }))
    try {
      SiteContentParser.parse(" ".repeat(2_000_001))
      fail("Oversized remote responses must be rejected")
    } catch (expected: IllegalArgumentException) {
      assertTrue(expected.message.orEmpty().contains("size"))
    }
  }
}
