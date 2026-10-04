package me.danielshort.wayfarers

import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/** A private, fsynced copy of the validated game envelope, never a purchase wallet. */
class GuildCheckpointStore(private val file: File) {
  data class Checkpoint(val text: String, val createdAt: Double, val savedAt: Double, val lastUpdate: Double, val replacesCreatedAt: Double?, val generation: String = "", val previousGeneration: String? = null, val updateId: String? = null)

  @Synchronized fun read(): Checkpoint? = runCatching {
    if (!file.isFile || file.length() !in 1..MAX_RECORD_BYTES.toLong()) return null
    decodeRecord(file.readText(Charsets.UTF_8))
  }.getOrNull()

  private fun decodeRecord(raw: String): Checkpoint {
    require(raw.toByteArray(Charsets.UTF_8).size in 1..MAX_RECORD_BYTES)
    val record = JSONObject(raw)
    require(record.getInt("version") == 1)
    val text = record.getString("text")
    require(record.getString("sha256") == digest(text))
    return parse(text, if (record.has("replacesCreatedAt")) record.getDouble("replacesCreatedAt") else null).copy(
      generation = record.optString("generation", ""), previousGeneration = if (record.has("previousGeneration")) record.getString("previousGeneration") else null,
      updateId = if (record.has("updateId")) record.getString("updateId").also { require(it.matches(Regex("[a-zA-Z0-9-]{16,100}"))) } else null)
  }

  /** Only the native content transaction can restore its verified pre-apply record. */
  @Synchronized fun restoreForContentUpdate(bytes: ByteArray): Boolean = runCatching {
    val checkpoint = decodeRecord(bytes.toString(Charsets.UTF_8))
    persist(checkpoint)
    read()?.text == checkpoint.text && read()?.generation == checkpoint.generation
  }.getOrDefault(false)

  @Synchronized fun backupForContentUpdate(): ByteArray? =
    if (read() != null) file.readBytes() else null

  @Synchronized fun write(text: String, replacesCreatedAt: Double? = null, generation: String = ""): Boolean = runCatching {
    val incoming = parse(text, replacesCreatedAt)
    val previous = read()
    require(generation == (previous?.generation ?: "")) { "Stale reset generation" }
    if (previous != null) {
      require(incoming.savedAt >= previous.savedAt) { "Older checkpoint" }
      require(incoming.createdAt == previous.createdAt || replacesCreatedAt == previous.createdAt) { "Unreviewed guild replacement" }
      require(JSONObject(incoming.text).getInt("version") >= JSONObject(previous.text).getInt("version")) { "Older save schema" }
      if (incoming.text == previous.text) return true
    }
    val replacement = replacesCreatedAt ?: previous?.takeIf { it.createdAt == incoming.createdAt }?.replacesCreatedAt
    // Keep the exact last good older-schema record before adopting a migration.
    // It is a recovery/export backup, never an automatic downgrade of a new save.
    if (previous != null && incoming.createdAt == previous.createdAt &&
      JSONObject(previous.text).getInt("version") < JSONObject(incoming.text).getInt("version")) {
      val backup = File(file.parentFile, file.name + ".before-schema-${JSONObject(incoming.text).getInt("version")}")
      if (!backup.exists()) atomicWrite(backup, file.readBytes())
      decodeRecord(backup.readText(Charsets.UTF_8))
    }
    persist(incoming.copy(replacesCreatedAt = replacement, generation = generation, previousGeneration = previous?.previousGeneration,
      updateId = previous?.takeIf { it.createdAt == incoming.createdAt }?.updateId))
    true
  }.getOrDefault(false)

  @Synchronized fun reset(text: String, previousGeneration: String, generation: String, updateId: String = ""): Boolean = runCatching {
    require(generation.matches(Regex("[a-zA-Z0-9-]{16,100}")) && generation != previousGeneration)
    require(updateId.isEmpty() || updateId.matches(Regex("[a-zA-Z0-9-]{16,100}")))
    val incoming = parse(text, null)
    val previous = read()
    // A lost acknowledgment retries the same transaction without rolling back
    // progress already made by the new guild.
    if (previous?.generation == generation) return incoming.createdAt == previous.createdAt && updateId == previous.updateId.orEmpty()
    require(previousGeneration == (previous?.generation ?: "")) { "Stale reset generation" }
    if (updateId.isNotEmpty()) {
      require(previous != null) { "An automatic reset requires a prior guild" }
      atomicWrite(File(file.parentFile, file.name + ".before-update-reset"), file.readBytes())
    }
    persist(incoming.copy(replacesCreatedAt = previous?.createdAt, generation = generation, previousGeneration = previousGeneration,
      updateId = updateId.takeIf { it.isNotEmpty() }))
    true
  }.getOrDefault(false)

  private fun persist(checkpoint: Checkpoint) {
    val record = JSONObject().put("version", 1).put("text", checkpoint.text).put("sha256", digest(checkpoint.text)).put("generation", checkpoint.generation)
    checkpoint.replacesCreatedAt?.let { record.put("replacesCreatedAt", it) }
    checkpoint.previousGeneration?.let { record.put("previousGeneration", it) }
    checkpoint.updateId?.let { record.put("updateId", it) }
    val bytes = record.toString().toByteArray(Charsets.UTF_8)
    require(bytes.size <= MAX_RECORD_BYTES)
    atomicWrite(file, bytes)
  }

  private fun atomicWrite(destination: File, bytes: ByteArray) {
    destination.parentFile?.mkdirs()
    val temporary = File(destination.parentFile, destination.name + ".pending")
    FileOutputStream(temporary).use { stream -> stream.write(bytes); stream.flush(); stream.fd.sync() }
    Files.move(temporary.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
  }

  /** Executed as a same-origin external script before the game creates its store. */
  fun bootstrapScript(): String {
    val checkpoint = read()
    val value = checkpoint?.let {
      JSONObject().put("text", it.text).put("generation", it.generation).apply {
        if (it.replacesCreatedAt != null) put("replacesCreatedAt", it.replacesCreatedAt)
        if (it.previousGeneration != null) put("previousGeneration", it.previousGeneration)
        if (it.updateId != null) put("updateId", it.updateId)
      }.toString()
    } ?: "null"
    return "window.WayfarersNativeCheckpoint=$value;"
  }

  private fun parse(text: String, replacesCreatedAt: Double?): Checkpoint {
    require(text.toByteArray(Charsets.UTF_8).size in 1 until GuildContentPolicy.MAX_SAVE_BYTES)
    val payload = JSONObject(text)
    val gameVersion = payload.getInt("version")
    require(payload.getString("format") == "wayfarers-guild-save" && gameVersion in 1..8)
    val state = payload.getJSONObject("state")
    require(state.getInt("schemaVersion") == gameVersion)
    require(state.has("resources") && state.has("upgrades") && state.has("rooms"))
    // Keep older envelopes readable so the canonical parser can migrate them.
    // The checkpoint record version is independent of the game schema. Full
    // semantic validation runs before sending and restoring this envelope.
    val createdAt = state.getDouble("createdAt")
    val savedAt = payload.getDouble("savedAt")
    val lastUpdate = state.getDouble("lastUpdate")
    listOf(createdAt, savedAt, lastUpdate).forEach { require(it.isFinite() && it in 0.0..8.64e15) }
    if (replacesCreatedAt != null) require(replacesCreatedAt.isFinite() && replacesCreatedAt in 0.0..8.64e15)
    return Checkpoint(text, createdAt, savedAt, lastUpdate, replacesCreatedAt)
  }

  private fun digest(text: String) = MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8))
    .joinToString("") { "%02x".format(it.toInt() and 255) }

  companion object { private const val MAX_RECORD_BYTES = GuildContentPolicy.MAX_SAVE_BYTES * 2 + 4096 }
}
