package me.danielshort.wayfarers.content

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.security.KeyPairGenerator
import java.security.Signature
import java.util.Base64
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class GuildContentUpdateTest {
  private val key = KeyPairGenerator.getInstance("EC").apply { initialize(256) }.generateKeyPair()
  private val publicKey get() = Base64.getEncoder().encodeToString(key.public.encoded)
  private val url = "https://github.com/danielshort3/danielshort3.github.io/releases/download/guild-content-2/content.zip"
  private fun verifier(version: Int = 16, native: Int = 1, schema: Int = 7) = GuildContentVerifier(publicKey, "me.danielshort.wayfarers", native, version, schema)
  private data class Fixture(val bytes: ByteArray, val json: JSONObject, val manifest: GuildContentManifest)

  private fun sign(json: JSONObject): String {
    val payload = json.toString().toByteArray(Charsets.UTF_8)
    val signer = Signature.getInstance("SHA256withECDSA").apply { initSign(key.private); update(payload) }
    return JSONObject().put("payload", Base64.getEncoder().encodeToString(payload)).put("signature", Base64.getEncoder().encodeToString(signer.sign())).toString()
  }
  private fun fixture(version: Int = 2, transform: (JSONObject) -> Unit = {}, archiveTransform: (ByteArray) -> ByteArray = { it }): Fixture {
    val files = linkedMapOf("wayfarers/index.html" to "<html>Version $version</html>", "wayfarers/game.css" to "body{color:#fff}")
    val zipped = ByteArrayOutputStream()
    ZipOutputStream(zipped).use { zip -> files.forEach { (path, text) -> zip.putNextEntry(ZipEntry(path)); zip.write(text.toByteArray()); zip.closeEntry() } }
    val bytes = archiveTransform(zipped.toByteArray())
    val records = JSONArray()
    files.forEach { (path, text) -> records.put(JSONObject().put("path", path).put("sha256", GuildContentLimits.digest(text.toByteArray())).put("size", text.toByteArray().size)) }
    val json = JSONObject().put("schemaVersion", 1).put("packageName", "me.danielshort.wayfarers").put("contentVersion", version)
      .put("label", "Game $version").put("nativeApi", 1).put("minAppVersionCode", 16).put("saveSchema", 7)
      .put("archive", JSONObject().put("url", url).put("sha256", GuildContentLimits.digest(bytes)).put("size", bytes.size)).put("records", records)
    transform(json)
    return Fixture(bytes, json, verifier().verify(sign(json)))
  }
  private fun temporary(): File = Files.createTempDirectory("guild-content-test").toFile().also { it.deleteOnExit() }
  private fun stage(store: GuildContentStore, fixture: Fixture) {
    val archive = File(temporary(), "content.zip").apply { writeBytes(fixture.bytes) }
    store.stage(fixture.manifest, archive)
  }
  private fun checkpoint(): ByteArray {
    val save = JSONObject().put("format", "wayfarers-guild-save").put("version", 7).put("savedAt", 123)
      .put("state", JSONObject().put("schemaVersion", 7).put("createdAt", 10).put("lastUpdate", 123).put("resources", JSONObject()).put("upgrades", JSONObject()).put("rooms", JSONArray())).toString()
    return JSONObject().put("version", 1).put("text", save).put("sha256", GuildContentLimits.digest(save.toByteArray())).put("generation", "").toString().toByteArray()
  }
  private val storage = "{\"wayfarers-guild-save\":\"retained guild\",\"wayfarers-guild-generation\":\"old generation\"}"
  private fun fails(work: () -> Unit) { try { work(); fail("Expected rejection") } catch (_: Exception) { } }

  @Test fun nodeSignedEnvelopeAndZipInteroperateWithJava() {
    val ancestors = generateSequence(File(requireNotNull(System.getProperty("user.dir"))).canonicalFile) { it.parentFile }.toList()
    val fixtureFile = ancestors.map { File(it, "mobile/android/scripts/guild-content-protocol-fixture.json") }.first { it.isFile }
    val protocol = JSONObject(fixtureFile.readText())
    val native = GuildContentVerifier(protocol.getString("publicKeySpkiBase64"), "me.danielshort.wayfarers", 1, 16, 7)
    val envelope = String(Base64.getDecoder().decode(protocol.getString("envelopeBase64")), Charsets.UTF_8)
    val manifest = native.verify(envelope)
    val archive = File(temporary(), "fixture.zip").apply { writeBytes(Base64.getDecoder().decode(protocol.getString("archiveBase64"))) }
    val destination = File(temporary(), "extracted")
    GuildContentArchiveReader.extract(archive, destination, manifest)
    val files = protocol.getJSONArray("files")
    repeat(files.length()) { index -> val item = files.getJSONObject(index); assertArrayEquals(Base64.getDecoder().decode(item.getString("bytesBase64")), File(destination, item.getString("path")).readBytes()) }
  }

  @Test fun signatureAndPayloadTamperingFailClosed() {
    val valid = fixture()
    val envelope = JSONObject(valid.manifest.envelope)
    envelope.put("payload", Base64.getEncoder().encodeToString(valid.json.put("contentVersion", 99).toString().toByteArray()))
    fails { verifier().verify(envelope.toString()) }
    val otherKey = KeyPairGenerator.getInstance("EC").apply { initialize(256) }.generateKeyPair()
    fails { GuildContentVerifier(Base64.getEncoder().encodeToString(otherKey.public.encoded), "me.danielshort.wayfarers", 1, 16, 7).verify(valid.manifest.envelope) }
  }
  @Test fun incompatiblePackageApiAppAndSaveAreRejected() {
    val valid = fixture()
    fails { verifier(version = 15).verify(valid.manifest.envelope) }
    fails { verifier(native = 2).verify(valid.manifest.envelope) }
    fails { verifier(schema = 8).verify(valid.manifest.envelope) }
    fails { verifier().verify(sign(valid.json.put("packageName", "other.game"))) }
  }
  @Test fun traversalReservedFilesDuplicatesAndNumericCoercionAreRejected() {
    for (path in listOf("wayfarers/../secret.js", "wayfarers/%2e.js", "wayfarers/native-checkpoint.js", "wayfarers/checkpoint.js", "wayfarers/android.js", "/wayfarers/a.js", "wayfarers/a/b.js", "img/wayfarers-guild/../a.png")) {
      val json = fixture().json
      json.getJSONArray("records").getJSONObject(0).put("path", path)
      fails { verifier().verify(sign(json)) }
    }
    val duplicate = fixture().json
    duplicate.getJSONArray("records").put(duplicate.getJSONArray("records").getJSONObject(0))
    fails { verifier().verify(sign(duplicate)) }
    val coercion = fixture().json.put("contentVersion", "2")
    fails { verifier().verify(sign(coercion)) }
  }
  @Test fun limitsAndUnexpectedSignedFieldsAreRejected() {
    val huge = fixture().json
    huge.getJSONArray("records").getJSONObject(0).put("size", GuildContentLimits.MAX_FILE + 1)
    fails { verifier().verify(sign(huge)) }
    val extra = fixture().json.put("redirect", "https://other.test")
    fails { verifier().verify(sign(extra)) }
    fails { verifier().verify(" ".repeat(GuildContentLimits.MAX_ENVELOPE + 1)) }
  }
  @Test fun releaseAndRedirectUrlsAreRestricted() {
    GuildContentUrlPolicy.requireRelease(url)
    GuildContentUrlPolicy.requireRedirect("https://release-assets.githubusercontent.com/github-production-release-asset/123?token=abc")
    for (bad in listOf("http://github.com/danielshort3/danielshort3.github.io/releases/download/v/file.zip", "https://github.com/other/repo/releases/download/v/file.zip", "$url?x=1", "https://github.com:443/danielshort3/danielshort3.github.io/releases/download/v/file.zip", "https://user@github.com/danielshort3/danielshort3.github.io/releases/download/v/file.zip", "https://github.com/danielshort3/danielshort3.github.io/releases/download/v/%2e%2e.zip")) fails { GuildContentUrlPolicy.requireRelease(bad) }
    fails { GuildContentUrlPolicy.requireRedirect("https://evil.test/content.zip") }
  }

  @Test fun downloadedHashMismatchAndInventoryMismatchDoNotStage() {
    val valid = fixture()
    val store = GuildContentStore(temporary(), verifier())
    val archive = File(temporary(), "bad.zip").apply { writeBytes(valid.bytes.copyOf().also { it[10] = (it[10].toInt() xor 1).toByte() }) }
    fails { store.stage(valid.manifest, archive) }
    assertTrue(store.session().isBundled)
    assertNull(store.stagedManifest())
    val json = valid.json
    json.getJSONArray("records").getJSONObject(0).put("sha256", "0".repeat(64))
    val wrong = valid.copy(manifest = verifier().verify(sign(json)))
    fails { stage(store, wrong) }
    assertNull(store.stagedManifest())
  }
  @Test fun unixSymlinksAreRejectedBeforeExtraction() {
    val linked = fixture(archiveTransform = { bytes ->
      val central = (0 until bytes.size - 46).first { bytes[it] == 0x50.toByte() && bytes[it + 1] == 0x4b.toByte() && bytes[it + 2] == 1.toByte() && bytes[it + 3] == 2.toByte() }
      bytes[central + 40] = 0xff.toByte(); bytes[central + 41] = 0xa1.toByte(); bytes
    })
    val store = GuildContentStore(temporary(), verifier())
    fails { stage(store, linked) }
    assertTrue(store.session().isBundled)
  }
  @Test fun stageDoesNotChangeRunningSessionAndCommitPinsEveryPage() {
    val store = GuildContentStore(temporary(), verifier())
    val old = store.session()
    stage(store, fixture())
    assertEquals(1, store.currentVersion())
    val candidate = store.beginApply(checkpoint(), storage)
    assertEquals(2, candidate.contentVersion)
    assertTrue(old.isBundled)
    assertFalse(store.commitApply("wrong-session"))
    assertTrue(store.commitApply(candidate.id))
    assertEquals(2, store.currentVersion())
    assertNull(candidate.open("wayfarers/android.js"))
    assertNull(candidate.open("wayfarers/../secret.js"))
    assertTrue(candidate.open("wayfarers/index.html")!!.bufferedReader().use { it.readText() }.contains("Version 2"))
    assertTrue(GuildContentStore(storeRoot(candidate), verifier()).session().isBundled) // A separate store cannot adopt a folder by existence alone.
  }
  private fun storeRoot(session: ContentSession): File = File(session.directory, "unrelated-store")

  @Test fun processDeathRollsBackBothSnapshotsUntilAcknowledgedAndRejectsReplay() {
    val root = temporary()
    val store = GuildContentStore(root, verifier())
    val valid = fixture()
    stage(store, valid)
    store.beginApply(checkpoint(), storage)
    val restarted = GuildContentStore(root, verifier())
    val recovery = restarted.startupRecovery()!!
    assertArrayEquals(checkpoint(), recovery.checkpointBytes)
    assertEquals(JSONObject(storage).toString(), JSONObject(recovery.localStorageJson).toString())
    assertTrue(restarted.session().isBundled)
    assertNotNull(GuildContentStore(root, verifier()).startupRecovery())
    assertFalse(restarted.consumeRecovery("wrong-token"))
    assertTrue(restarted.consumeRecovery(recovery.token))
    assertNull(restarted.startupRecovery())
    fails { stage(restarted, valid) }
    fails { restarted.accept(fixture(version = 1).manifest) }
    stage(restarted, fixture(version = 3))
    assertEquals(3, restarted.stagedManifest()!!.contentVersion)
  }
  @Test fun explicitFailureRestoresPreviousBundleAndDoesNotCommitWrongSession() {
    val store = GuildContentStore(temporary(), verifier())
    stage(store, fixture())
    val first = store.beginApply(checkpoint(), storage)
    assertTrue(store.commitApply(first.id))
    stage(store, fixture(version = 3))
    val next = store.beginApply(checkpoint(), storage)
    assertNull(store.rollbackApply("other", "failed"))
    assertEquals(3, store.currentVersion())
    val restored = store.rollbackApply(next.id, "Could not load update")!!
    assertEquals(2, store.currentVersion())
    assertEquals("Could not load update", restored.reason)
    assertFalse(store.commitApply(next.id))
  }
  @Test fun damagedCommittedBundleFallsBackWithoutRestoringOldSave() {
    val root = temporary()
    val store = GuildContentStore(root, verifier())
    stage(store, fixture()); val first = store.beginApply(checkpoint(), storage); store.commitApply(first.id)
    stage(store, fixture(version = 3)); val second = store.beginApply(checkpoint(), storage); store.commitApply(second.id)
    File(second.directory, "wayfarers/index.html").appendText("damage")
    val restart = GuildContentStore(root, verifier())
    assertNull(restart.startupRecovery())
    assertEquals(2, restart.currentVersion())
    assertNotNull(restart.notice())
    fails { restart.accept(fixture(version = 3).manifest) }
  }
  @Test fun startupRetainsOnlyActivePreviousAndStagedVerifiedBundles() {
    val root = temporary()
    val store = GuildContentStore(root, verifier())
    for (version in 2..5) { stage(store, fixture(version)); val candidate = store.beginApply(checkpoint(), storage); store.commitApply(candidate.id) }
    stage(store, fixture(6))
    File(root, "bundles/staging-abandoned").mkdirs()
    GuildContentStore(root, verifier()).startupRecovery()
    val retained = File(root, "bundles").listFiles()!!.map { it.name.substringBefore('-') }.toSet()
    assertEquals(setOf("v4", "v5", "v6"), retained)
    assertEquals(5, store.currentVersion())
    assertEquals(6, store.stagedManifest()!!.contentVersion)
  }
  @Test fun corruptPairedBackupAndUnsafeStorageBlockApplyingOrRollback() {
    val root = temporary()
    val store = GuildContentStore(root, verifier())
    stage(store, fixture())
    fails { store.beginApply(checkpoint(), "{\"account-token\":\"secret\"}") }
    fails { store.beginApply(checkpoint(), "{\"wayfarers-guild-x\":4}") }
    fails { store.beginApply("bad checkpoint".toByteArray(), storage) }
    store.beginApply(checkpoint(), storage)
    File(root, "transactions").walkTopDown().first { it.name == "storage.json" }.appendText("damage")
    fails { GuildContentStore(root, verifier()).startupRecovery() }
  }
  @Test fun cancellationDuringExtractionKeepsOriginalBundle() {
    val store = GuildContentStore(temporary(), verifier())
    val valid = fixture()
    val archive = File(temporary(), "ok.zip").apply { writeBytes(valid.bytes) }
    var calls = 0
    fails { store.stage(valid.manifest, archive) { if (++calls > 2) throw IllegalStateException("cancel") } }
    assertTrue(store.session().isBundled)
    assertNull(store.stagedManifest())
  }

  private class FakeConnection(url: URL, private val response: Int, private val bytes: ByteArray, private val headers: Map<String, String> = emptyMap()) : HttpURLConnection(url) {
    var disconnected = false
    override fun connect() { }
    override fun disconnect() { disconnected = true }
    override fun usingProxy() = false
    override fun getResponseCode() = response
    override fun getInputStream() = ByteArrayInputStream(bytes)
    override fun getHeaderField(name: String): String? = headers[name]
    override fun getContentEncoding(): String? = headers["Content-Encoding"]
    override fun getHeaderFieldLong(name: String, default: Long): Long = headers[name]?.toLongOrNull() ?: default
  }
  @Test fun realTransportStreamsChecksHashesAndRejectsExtraOrTruncatedBytes() {
    val valid = fixture()
    val transport = GuildContentHttpsTransport { request -> FakeConnection(request, 200, valid.bytes, mapOf("Content-Length" to valid.bytes.size.toString())) }
    val target = File(temporary(), "download.zip")
    transport.download(valid.manifest.archive, target, { received, total -> assertTrue(received <= total) })
    assertArrayEquals(valid.bytes, target.readBytes())
    for (bytes in listOf(valid.bytes.copyOf(valid.bytes.size - 1), valid.bytes + byteArrayOf(1), valid.bytes.copyOf().also { it[4] = 99 })) {
      val broken = GuildContentHttpsTransport { request -> FakeConnection(request, 200, bytes) }
      fails { broken.download(valid.manifest.archive, target, { _, _ -> }) }
      assertFalse(target.exists())
    }
  }
  @Test fun transportRejectsCrossHostRedirectsHugeBodiesAndCompressedResponses() {
    val crossed = GuildContentHttpsTransport { request -> FakeConnection(request, 302, byteArrayOf(), mapOf("Location" to "https://evil.test/a")) }
    fails { crossed.manifest(url) }
    val oversized = GuildContentHttpsTransport { request -> FakeConnection(request, 200, ByteArray(GuildContentLimits.MAX_ENVELOPE + 1)) }
    fails { oversized.manifest(url) }
    val compressed = GuildContentHttpsTransport { request -> FakeConnection(request, 200, "{}".toByteArray(), mapOf("Content-Encoding" to "gzip")) }
    fails { compressed.manifest(url) }
    val endlesslyRedirecting = GuildContentHttpsTransport { request -> FakeConnection(request, 302, byteArrayOf(), mapOf("Location" to url)) }
    fails { endlesslyRedirecting.manifest(url) }
  }
  @Test fun managerCheckAndDownloadRemainSeparateFromActivation() {
    val valid = fixture()
    val store = GuildContentStore(temporary(), verifier())
    val fake = object : GuildContentTransport {
      override fun manifest(url: String, cancel: () -> Unit) = valid.manifest.envelope
      override fun download(archive: GuildContentArchive, target: File, progress: (Long, Long) -> Unit, cancel: () -> Unit) { target.writeBytes(valid.bytes); progress(valid.bytes.size.toLong(), valid.bytes.size.toLong()) }
    }
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
    val manager = GuildContentUpdateManager(store, url, fake, scope)
    manager.check(); assertTrue(manager.state.value is GuildContentUpdateState.Available)
    manager.download(); assertTrue(manager.state.value is GuildContentUpdateState.Ready)
    assertTrue(store.session().isBundled)
    val reopened = GuildContentUpdateManager(store, url, fake, scope)
    reopened.refresh(); assertTrue(reopened.state.value is GuildContentUpdateState.Ready)
    scope.cancel()
  }

  @Test fun availableCollectorCanImmediatelyStartDownloadWithoutLosingTheOperation() {
    val valid = fixture()
    val store = GuildContentStore(temporary(), verifier())
    var downloads = 0
    val fake = object : GuildContentTransport {
      override fun manifest(url: String, cancel: () -> Unit) = valid.manifest.envelope
      override fun download(archive: GuildContentArchive, target: File, progress: (Long, Long) -> Unit, cancel: () -> Unit) {
        downloads++
        target.writeBytes(valid.bytes)
        progress(valid.bytes.size.toLong(), valid.bytes.size.toLong())
      }
    }
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Unconfined)
    val manager = GuildContentUpdateManager(store, url, fake, scope)
    scope.launch(start = CoroutineStart.UNDISPATCHED) {
      manager.state.collect { state -> if (state is GuildContentUpdateState.Available) manager.download() }
    }
    manager.check()
    assertEquals(1, downloads)
    assertTrue(manager.state.value is GuildContentUpdateState.Ready)
    assertTrue(store.session().isBundled)
    scope.cancel()
  }

  @Test fun cancelStopsAutomaticDownloadUntilFreshCheckAndStillAllowsManualDownload() {
    val valid = fixture()
    val store = GuildContentStore(temporary(), verifier())
    val downloading = CountDownLatch(1)
    val releaseTransfer = CountDownLatch(1)
    val cancellationPublished = CountDownLatch(1)
    val manualFinished = CountDownLatch(1)
    val downloads = AtomicInteger()
    val fake = object : GuildContentTransport {
      override fun manifest(url: String, cancel: () -> Unit) = valid.manifest.envelope
      override fun download(archive: GuildContentArchive, target: File, progress: (Long, Long) -> Unit, cancel: () -> Unit) {
        if (downloads.incrementAndGet() == 1) {
          downloading.countDown()
          check(releaseTransfer.await(10, TimeUnit.SECONDS))
        }
        cancel()
        target.writeBytes(valid.bytes)
        progress(valid.bytes.size.toLong(), valid.bytes.size.toLong())
      }
    }
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val manager = GuildContentUpdateManager(store, url, fake, scope)
    val collector = scope.launch(start = CoroutineStart.UNDISPATCHED) {
      manager.state.collect { state ->
        if (state is GuildContentUpdateState.Available) {
          if (state.autoDownloadAllowed) manager.download() else cancellationPublished.countDown()
        }
        if (state is GuildContentUpdateState.Ready) manualFinished.countDown()
      }
    }
    try {
      manager.check()
      assertTrue(downloading.await(10, TimeUnit.SECONDS))
      manager.cancel()
      releaseTransfer.countDown()
      assertTrue(cancellationPublished.await(10, TimeUnit.SECONDS))
      assertEquals(1, downloads.get())
      assertFalse((manager.state.value as GuildContentUpdateState.Available).autoDownloadAllowed)
      assertTrue(store.session().isBundled)
      // The player can explicitly resume the same offer without re-enabling automatic continuation.
      manager.download()
      assertTrue(manualFinished.await(10, TimeUnit.SECONDS))
      assertTrue(manager.state.value is GuildContentUpdateState.Ready)
      assertEquals(2, downloads.get())
      assertTrue(store.session().isBundled)
    } finally {
      releaseTransfer.countDown()
      scope.cancel()
    }
  }
}
