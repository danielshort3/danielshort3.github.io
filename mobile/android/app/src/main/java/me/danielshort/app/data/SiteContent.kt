package me.danielshort.app.data

import org.json.JSONArray
import org.json.JSONObject
import java.net.URI

data class SiteInfo(val name: String, val description: String, val url: String, val email: String, val githubUrl: String, val privacyUrl: String)
data class Interest(val title: String, val body: String, val imageUrl: String)
data class Milestone(val title: String, val organization: String, val date: String, val url: String)
data class AboutInfo(val greeting: String, val location: String, val intro: String, val portraitUrl: String, val interests: List<Interest>, val experience: List<Milestone>, val education: List<Milestone>, val credentials: List<Milestone>)
data class ResourceLink(val label: String, val url: String)
data class Project(val id: String, val title: String, val summary: String, val imageUrl: String, val iconUrl: String, val url: String, val tags: List<String>, val situation: String, val task: String, val actions: List<String>, val results: List<String>, val resources: List<ResourceLink>, val demoUrl: String)
data class CatalogItem(val id: String, val title: String, val summary: String, val iconUrl: String, val url: String, val category: String)
data class SiteContent(val revision: String, val site: SiteInfo, val about: AboutInfo, val projects: List<Project>, val tools: List<CatalogItem>, val games: List<CatalogItem>)

/** The remote document supplies content, never executable UI, HTML or code. */
object SiteContentParser {
  fun parse(source: String): SiteContent {
    require(source.toByteArray(Charsets.UTF_8).size <= 2_000_000) { "Content exceeds the supported size" }
    val root = JSONObject(source)
    require(root.opt("schemaVersion") == 1) { "This content needs a newer app version" }
    val revision = root.getString("revision")
    require(Regex("[a-f0-9]{64}").matches(revision)) { "Invalid content revision" }
    val site = root.getJSONObject("site")
    val about = root.getJSONObject("about")
    val projects = root.getJSONArray("projects").objects().map { project ->
      Project(
        id(project), requiredText(project, "title"), project.text("summary"), project.image("imageUrl"), project.image("iconUrl"), requiredUrl(project, "url"),
        project.optJSONArray("tags").strings(), project.text("situation"), project.text("task"), project.optJSONArray("actions").strings(), project.optJSONArray("results").strings(),
        project.optJSONArray("resources").objects().mapNotNull { resource -> resource.url("url").takeIf(String::isNotEmpty)?.let { ResourceLink(resource.text("label"), it) } }, project.url("demoUrl")
      )
    }
    require(projects.isNotEmpty() && projects.map { it.id }.distinct().size == projects.size) { "Invalid project catalog" }
    return SiteContent(revision,
      SiteInfo(requiredText(site, "name"), site.text("description"), requiredUrl(site, "url"), site.text("email").takeIf { Regex("[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}").matches(it) }.orEmpty(), site.url("githubUrl"), site.url("privacyUrl")),
      AboutInfo(about.text("greeting"), about.text("location"), about.text("intro"), about.image("portraitUrl"),
        about.optJSONArray("interests").objects().map { Interest(it.text("title"), it.text("body"), it.image("imageUrl")) },
        milestones(about.optJSONArray("experience")), milestones(about.optJSONArray("education")), milestones(about.optJSONArray("credentials"))),
      projects, catalog(root.optJSONArray("tools")), catalog(root.optJSONArray("games")))
  }

  fun safeUrl(value: String): String = runCatching {
    val uri = URI(value)
    val host = uri.host?.lowercase().orEmpty()
    val validHost = host.contains('.') && !host.matches(Regex("[0-9.]+")) && !host.contains(':') &&
      listOf("localhost", "local", "internal", "test", "invalid").none { host == it || host.endsWith(".$it") }
    if (uri.scheme == "https" && validHost && uri.userInfo == null && uri.port in setOf(-1, 443)) value else ""
  }.getOrDefault("")

  private fun safeImage(value: String): String = safeUrl(value).takeIf { url ->
    url.isNotEmpty() && URI(url).let { uri ->
      uri.host.lowercase() in setOf("www.danielshort.me", "danielshort.me", "www.dshort.me", "dshort.me") && uri.path.startsWith("/img/")
    }
  }.orEmpty()

  private fun catalog(array: JSONArray?): List<CatalogItem> = array.objects().map {
    CatalogItem(id(it), requiredText(it, "title"), it.text("summary"), it.image("iconUrl"), requiredUrl(it, "url"), it.text("category"))
  }.also { list -> require(list.map { it.id }.distinct().size == list.size) { "Duplicate catalog entry" } }
  private fun milestones(array: JSONArray?): List<Milestone> = array.objects().map { Milestone(it.text("title"), it.text("organization"), it.text("date"), it.url("url")) }
  private fun id(value: JSONObject) = requiredText(value, "id").also { require(Regex("[A-Za-z0-9][A-Za-z0-9_-]{0,80}").matches(it)) }
  private fun requiredText(value: JSONObject, key: String) = value.text(key).also { require(it.isNotBlank()) { "Missing $key" } }
  private fun requiredUrl(value: JSONObject, key: String) = value.url(key).also { require(it.isNotEmpty()) { "Invalid $key" } }
  private fun JSONObject.text(key: String): String = if (isNull(key)) "" else optString(key, "").trim().take(30_000)
  private fun JSONObject.url(key: String) = safeUrl(text(key))
  private fun JSONObject.image(key: String) = safeImage(text(key))
  private fun JSONArray?.strings(): List<String> = if (this == null) emptyList() else (0 until length().coerceAtMost(200)).mapNotNull { opt(it) as? String }.map { it.take(30_000) }
  private fun JSONArray?.objects(): List<JSONObject> {
    if (this == null) return emptyList()
    require(length() <= 300) { "Catalog exceeds the supported entry count" }
    return (0 until length()).map { getJSONObject(it) }
  }
}
