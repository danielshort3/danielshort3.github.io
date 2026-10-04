package me.danielshort.wayfarers.content

import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

/** A page captures this immutable session; applying an update never mixes versions in an existing page. */
data class ContentSession(val id: String, val contentVersion: Long, val label: String, val directory: File?) {
  val isBundled: Boolean get() = directory == null
  fun open(path: String): InputStream? {
    if (isBundled || !GuildContentLimits.validPath(path)) return null
    val file = File(directory, path)
    if (!file.canonicalPath.startsWith(directory!!.canonicalPath + File.separator) || !file.isFile || Files.isSymbolicLink(file.toPath())) return null
    return FileInputStream(file)
  }
}

data class Recovery(val token: String, val checkpointBytes: ByteArray, val localStorageJson: String, val rejectedVersion: Long, val reason: String)

/** Atomic journal and durable paired backups separate downloading from the reversible apply transaction. */
class GuildContentStore(private val root: File, private val verifier: GuildContentVerifier, private val bundledVersion: Long = 1) {
  private val bundles = File(root, "bundles")
  private val transactions = File(root, "transactions")
  private val journalFile = File(root, "journal.json")

  init { require(root.mkdirs() || root.isDirectory); require(!Files.isSymbolicLink(root.toPath())); require(bundles.mkdirs() || bundles.isDirectory); require(transactions.mkdirs() || transactions.isDirectory) }

  @Synchronized fun currentVersion(): Long = session().contentVersion

  fun verifyManifest(envelope: String): GuildContentManifest = verifier.verify(envelope)
  fun downloadFile(): File = File(root, "content-download.part")

  @Synchronized fun startupRecovery(): Recovery? {
    var journal = readJournal()
    if (journal.has("pending")) {
      rollback(journal, "The previous game update did not finish. Your guild was restored.")
      journal = readJournal()
    }
    val active = optional(journal, "active")
    val activeFailure = active?.let { runCatching { loadSession(it) }.exceptionOrNull() }
    if (active != null && activeFailure != null) {
      // After a committed update, keep the player's current save when selecting compatible older content.
      versionFromId(active)?.let { rejectVersion(journal, it) }
      val previous = optional(journal, "previous")?.takeIf { runCatching { loadSession(it) }.isSuccess }
      val appUpgrade = activeFailure is GuildContentCompatibilityFailure
      if (appUpgrade) {
        // Retain the old immutable directory for recovery/export. Exact manifest
        // compatibility still prevents this app from executing its older schema.
        journal.put("appUpgradeBackup", active)
      }
      putOptional(journal, "active", previous)
      journal.remove("previous")
      journal.put("notice", if (appUpgrade) "The updated game included with this app is active. Your guild was kept."
        else "Game content was damaged. The previous verified version is active.")
      writeJournal(journal)
    }
    journal = readJournal()
    pruneUnreferencedBundles(journal)
    return recovery(journal)
  }

  @Synchronized fun session(): ContentSession {
    val journal = readJournal()
    val selected = if (journal.has("pending")) journal.getJSONObject("pending").getString("target") else optional(journal, "active")
    return if (selected == null) bundledSession() else loadSession(selected)
  }

  @Synchronized fun stagedManifest(): GuildContentManifest? = optional(readJournal(), "staged")?.let { id ->
    runCatching { loadManifest(id).also { GuildContentArchiveReader.verifyDirectory(File(bundles, id), it) } }.getOrNull()
  }

  @Synchronized fun accept(manifest: GuildContentManifest) {
    val journal = readJournal()
    require(manifest.contentVersion > journal.getLong("highest")) { "This game update is older than an already applied update." }
    require(!contains(journal.getJSONArray("rejected"), manifest.contentVersion)) { "This game update previously failed and cannot be reapplied." }
    require(!journal.has("pending") && !journal.has("recovery")) { "Finish restoring your guild before checking for updates." }
  }

  @Synchronized fun stage(manifest: GuildContentManifest, archive: File, cancel: () -> Unit = {}) {
    // Never trust a caller-created model: authenticate the exact signed envelope again.
    val authenticated = verifier.verify(manifest.envelope)
    require(authenticated == manifest)
    accept(manifest)
    val target = File(bundles, manifest.id)
    if (target.exists()) {
      loadSession(manifest.id)
    } else {
      val temporary = File(bundles, "staging-${UUID.randomUUID()}")
      try {
        GuildContentArchiveReader.extract(archive, temporary, manifest, cancel)
        atomicWrite(File(temporary, "manifest-envelope.json"), manifest.envelope.toByteArray(Charsets.UTF_8))
        cancel()
        Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
      } finally {
        if (temporary.exists()) temporary.deleteRecursively()
      }
    }
    val journal = readJournal()
    journal.put("staged", manifest.id)
    writeJournal(journal)
  }

  @Synchronized fun beginApply(checkpointBytes: ByteArray, localStorageJson: String): ContentSession {
    validateCheckpoint(checkpointBytes)
    validateStorage(localStorageJson)
    val journal = readJournal()
    require(!journal.has("pending") && !journal.has("recovery"))
    val id = optional(journal, "staged") ?: throw GuildContentFailure("Download the game update first.")
    val manifest = loadManifest(id)
    accept(manifest)
    val candidate = loadSession(id)
    val token = UUID.randomUUID().toString()
    val backup = File(transactions, token)
    require(backup.mkdirs())
    try {
      atomicWrite(File(backup, "checkpoint.json"), checkpointBytes)
      atomicWrite(File(backup, "storage.json"), localStorageJson.toByteArray(Charsets.UTF_8))
      val pending = JSONObject().put("token", token).put("target", id)
        .put("version", manifest.contentVersion).put("checkpointSha256", GuildContentLimits.digest(checkpointBytes))
        .put("storageSha256", GuildContentLimits.digest(localStorageJson.toByteArray(Charsets.UTF_8)))
      putOptional(pending, "previous", optional(journal, "active"))
      journal.put("pending", pending)
      writeJournal(journal)
    } catch (failure: Throwable) {
      backup.deleteRecursively()
      throw failure
    }
    return candidate
  }

  @Synchronized fun commitApply(sessionId: String): Boolean {
    val journal = readJournal()
    if (!journal.has("pending")) return optional(journal, "active") == sessionId
    val pending = journal.getJSONObject("pending")
    if (pending.getString("target") != sessionId) return false
    loadSession(sessionId)
    putOptional(journal, "previous", optional(pending, "previous"))
    journal.put("active", sessionId).put("highest", maxOf(journal.getLong("highest"), pending.getLong("version")))
    journal.remove("pending")
    journal.remove("staged")
    journal.remove("notice")
    writeJournal(journal)
    File(transactions, pending.getString("token")).deleteRecursively()
    return true
  }

  @Synchronized fun rollbackApply(sessionId: String, reason: String): Recovery? {
    val journal = readJournal()
    if (!journal.has("pending") || journal.getJSONObject("pending").getString("target") != sessionId) return recovery(journal)
    rollback(journal, reason.take(240))
    return recovery(readJournal())
  }

  @Synchronized fun consumeRecovery(token: String): Boolean {
    val journal = readJournal()
    if (!journal.has("recovery") || journal.getJSONObject("recovery").getString("token") != token) return false
    journal.remove("recovery")
    writeJournal(journal)
    File(transactions, token).deleteRecursively()
    return true
  }

  @Synchronized fun notice(): String? = optional(readJournal(), "notice")

  private fun rollback(journal: JSONObject, reason: String) {
    val pending = journal.getJSONObject("pending")
    // Read and authenticate both snapshots before changing the active pointer.
    backup(pending, reason)
    val previous = optional(pending, "previous")?.takeIf { runCatching { loadSession(it) }.isSuccess }
    optional(pending, "previous")?.takeIf { (versionFromId(it) ?: Long.MAX_VALUE) < bundledVersion }
      ?.let { journal.put("appUpgradeBackup", it) }
    putOptional(journal, "active", previous)
    val version = pending.getLong("version")
    rejectVersion(journal, version)
    journal.put("highest", maxOf(journal.getLong("highest"), version))
    journal.put("recovery", JSONObject(pending.toString()).put("reason", reason))
    journal.remove("pending")
    journal.remove("staged")
    writeJournal(journal)
  }

  private fun recovery(journal: JSONObject): Recovery? = if (journal.has("recovery")) {
    val record = journal.getJSONObject("recovery")
    backup(record, record.getString("reason"))
  } else null

  private fun backup(record: JSONObject, reason: String): Recovery {
    val token = record.getString("token")
    require(token.matches(Regex("[a-f0-9-]{36}")))
    val checkpoint = boundedRead(File(transactions, "$token/checkpoint.json"), GuildContentLimits.MAX_CHECKPOINT)
    val storage = boundedRead(File(transactions, "$token/storage.json"), GuildContentLimits.MAX_STORAGE)
    require(GuildContentLimits.digest(checkpoint) == record.getString("checkpointSha256"))
    require(GuildContentLimits.digest(storage) == record.getString("storageSha256"))
    validateCheckpoint(checkpoint)
    val text = GuildContentLimits.utf8(storage)
    validateStorage(text)
    return Recovery(token, checkpoint, text, record.getLong("version"), reason)
  }

  private fun loadSession(id: String): ContentSession {
    val manifest = loadManifest(id)
    val directory = File(bundles, id)
    GuildContentArchiveReader.verifyDirectory(directory, manifest)
    return ContentSession(id, manifest.contentVersion, manifest.label, directory)
  }

  private fun loadManifest(id: String): GuildContentManifest {
    require(id.matches(Regex("v[0-9]{1,10}-[a-f0-9]{16}")))
    val envelope = GuildContentLimits.utf8(boundedRead(File(bundles, "$id/manifest-envelope.json"), GuildContentLimits.MAX_ENVELOPE))
    return verifier.verify(envelope).also { require(it.id == id) }
  }

  private fun bundledSession() = ContentSession("apk", bundledVersion, "Included with app", null)

  private fun readJournal(): JSONObject {
    if (!journalFile.exists()) return JSONObject().put("schemaVersion", 1).put("highest", bundledVersion).put("rejected", org.json.JSONArray())
    val journal = JSONObject(GuildContentLimits.utf8(boundedRead(journalFile, 64 * 1024)))
    require(journal.getInt("schemaVersion") == 1 && journal.getLong("highest") >= 1)
    require(journal.getJSONArray("rejected").length() <= 512)
    // An APK may carry a newer baseline than the previous content high-water
    // mark. Advance it atomically without discarding save recovery transactions.
    if (journal.getLong("highest") < bundledVersion) {
      journal.put("highest", bundledVersion)
      writeJournal(journal)
    }
    return journal
  }

  private fun writeJournal(journal: JSONObject) = atomicWrite(journalFile, journal.toString().toByteArray(Charsets.UTF_8))
  private fun optional(json: JSONObject, field: String): String? = if (json.has(field) && !json.isNull(field)) json.getString(field) else null
  private fun putOptional(json: JSONObject, field: String, value: String?) { if (value == null) json.remove(field) else json.put(field, value) }
  private fun contains(array: org.json.JSONArray, version: Long) = (0 until array.length()).any { array.getLong(it) == version }
  private fun versionFromId(id: String) = id.substringBefore('-').removePrefix("v").toLongOrNull()

  private fun rejectVersion(journal: JSONObject, version: Long) {
    val rejected = journal.getJSONArray("rejected")
    if (!contains(rejected, version)) rejected.put(version)
    // The high-water mark also rejects older versions; bound the diagnostic rejection history for long-lived installs.
    while (rejected.length() > 512) rejected.remove(0)
  }

  private fun pruneUnreferencedBundles(journal: JSONObject) {
    val keep = mutableSetOf<String>()
    listOf("active", "previous", "staged", "appUpgradeBackup").forEach { optional(journal, it)?.let(keep::add) }
    listOf("pending", "recovery").forEach { field ->
      if (journal.has(field)) {
        val transaction = journal.getJSONObject(field)
        keep.add(transaction.getString("target"))
        optional(transaction, "previous")?.let(keep::add)
      }
    }
    bundles.listFiles()?.forEach { directory ->
      if (directory.name !in keep && directory.isDirectory && !Files.isSymbolicLink(directory.toPath()) &&
        directory.canonicalPath.startsWith(bundles.canonicalPath + File.separator)) directory.deleteRecursively()
    }
  }

  private fun validateCheckpoint(bytes: ByteArray) {
    require(bytes.size in 1..GuildContentLimits.MAX_CHECKPOINT)
    val checkpoint = JSONObject(GuildContentLimits.utf8(bytes))
    require(checkpoint.getInt("version") == 1)
    val text = checkpoint.getString("text")
    require(checkpoint.getString("sha256") == GuildContentLimits.digest(text.toByteArray(Charsets.UTF_8)))
    val save = JSONObject(text)
    require(save.getString("format") == "wayfarers-guild-save" && save.getInt("version") in 1..8)
    require(save.getJSONObject("state").getInt("schemaVersion") == save.getInt("version"))
  }

  private fun validateStorage(text: String) {
    require(text.toByteArray(Charsets.UTF_8).size in 2..GuildContentLimits.MAX_STORAGE)
    val snapshot = JSONObject(text)
    require(snapshot.length() <= 128)
    snapshot.keys().forEach { key -> require(key.startsWith("wayfarers-guild-") && key.length <= 160 && snapshot.get(key) is String) }
  }

  private fun boundedRead(file: File, limit: Int): ByteArray {
    require(file.isFile && !Files.isSymbolicLink(file.toPath()) && file.length() in 1..limit.toLong())
    return file.readBytes().also { require(it.size <= limit) }
  }

  private fun atomicWrite(file: File, bytes: ByteArray) {
    val parent = requireNotNull(file.parentFile)
    require(parent.mkdirs() || parent.isDirectory)
    val pending = File(parent, file.name + ".pending")
    FileOutputStream(pending).use { output -> output.write(bytes); output.flush(); output.fd.sync() }
    Files.move(pending.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
  }
}
