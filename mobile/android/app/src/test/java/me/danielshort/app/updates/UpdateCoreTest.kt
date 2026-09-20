package me.danielshort.app.updates

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.file.Files
import java.security.MessageDigest
import java.util.Base64
import java.util.concurrent.atomic.AtomicInteger
import java.util.zip.GZIPOutputStream

class UpdateCoreTest {
  private lateinit var directory: File
  private lateinit var base: File
  private lateinit var target: ByteArray
  private lateinit var verifier: FakeVerifier
  private lateinit var transport: FakeTransport
  private val supervisor = SupervisorJob()
  private val scope = CoroutineScope(supervisor + Dispatchers.Default)
  private val signer = "ab".repeat(32)
  private val feed = "https://www.danielshort.me/app-updates/review/latest.json"
  private val apkUrl = "https://github.com/danielshort3/danielshort3.github.io/releases/download/v0.3.0/app.apk"
  private val patchUrl = "https://www.danielshort.me/app-updates/review/from-2.patch.gz"

  @Before fun setup() {
    directory = Files.createTempDirectory("android-updater-test").toFile()
    base = File(directory, "installed.apk").apply { writeBytes(ByteArray(8192) { (it % 127).toByte() }) }
    target = base.readBytes() + "new version".toByteArray()
    verifier = FakeVerifier()
    transport = FakeTransport()
    transport.manifestText = manifest()
    transport.artifacts[apkUrl] = target
  }

  @After fun cleanup() {
    runBlocking { supervisor.cancelAndJoin() }
    directory.deleteRecursively()
  }

  @Test fun recognizesPublishedBaseAndOffersNewerVersion() = runBlocking {
    val manager = manager()
    manager.check()
    val result = awaitState<AppUpdateState.Available>(manager)
    assertEquals(3L, result.offer.versionCode)
    assertFalse(result.offer.usingPatch)
    assertEquals(target.size.toLong(), result.offer.downloadBytes)
  }

  @Test fun fullApkIsVerifiedBeforeReadyAndAgainBeforeInstall() = runBlocking {
    val manager = available()
    manager.download()
    awaitState<AppUpdateState.Ready>(manager)
    val before = verifier.archiveChecks
    val apk = manager.verifiedApkForInstall()
    assertArrayEquals(target, apk.readBytes())
    assertEquals(before + 1, verifier.archiveChecks)
    assertEquals("ready", apk.parentFile?.name)
  }

  @Test fun automatedOptOutOnlyCancelsOwnedDownloads() = runBlocking {
    transport.downloadGate = CompletableDeferred()
    val manager = available()
    manager.download()
    awaitState<AppUpdateState.Downloading>(manager)
    manager.cancelAutomaticDownload()
    transport.downloadGate!!.complete(Unit)
    awaitState<AppUpdateState.Ready>(manager)
    assertFalse(manager.automaticDownloadsBlocked.value)
  }

  @Test fun canceledAutomaticDownloadKeepsVerifiedOfferAndManualRetry(): Unit = runBlocking {
    transport.downloadGate = CompletableDeferred()
    val manager = available()
    manager.download(automated = true)
    awaitState<AppUpdateState.Downloading>(manager)
    manager.cancelAutomaticDownload()
    awaitState<AppUpdateState.Available>(manager)
    transport.downloadGate!!.complete(Unit)
    manager.download()
    awaitState<AppUpdateState.Ready>(manager)
  }

  @Test fun explicitCancelBlocksAutomaticDownloadButAllowsManualRetry(): Unit = runBlocking {
    transport.downloadGate = CompletableDeferred()
    val manager = available()
    manager.download(automated = true)
    awaitState<AppUpdateState.Downloading>(manager)
    manager.cancel()
    awaitState<AppUpdateState.Available>(manager)
    assertTrue(manager.automaticDownloadsBlocked.value)
    assertFalse(manager.download(automated = true))
    transport.downloadGate!!.complete(Unit)
    manager.download()
    awaitState<AppUpdateState.Ready>(manager)
  }

  @Test fun exactPatchReconstructsTarget() = runBlocking {
    val patch = patchBytes { output ->
      output.writeByte(0); output.writeLong(0); output.writeInt(base.length().toInt())
      output.writeByte(1); output.writeInt(11); output.write("new version".toByteArray())
      output.writeByte(255)
    }
    transport.manifestText = manifest(patch = patch)
    transport.artifacts[patchUrl] = patch
    val manager = available()
    assertTrue((manager.state.value as AppUpdateState.Available).offer.usingPatch)
    manager.download()
    awaitState<AppUpdateState.Ready>(manager)
    assertArrayEquals(target, manager.verifiedApkForInstall().readBytes())
    assertEquals(listOf(patchUrl), transport.requests)
  }

  @Test fun interruptedPatchFallsBackToVerifiedFullApk() = runBlocking {
    val patch = validPatch()
    transport.manifestText = manifest(patch = patch)
    transport.artifacts[patchUrl] = patch
    transport.failPatch = true
    val manager = available()
    manager.download()
    val ready = awaitState<AppUpdateState.Ready>(manager)
    assertFalse(ready.offer.usingPatch)
    assertEquals(listOf(patchUrl, apkUrl), transport.requests)
  }

  @Test fun corruptPatchDoesNotFallBackOrProduceReadyApk() = runBlocking {
    val patch = validPatch()
    transport.manifestText = manifest(patch = patch)
    transport.artifacts[patchUrl] = patch.copyOf().apply { this[20] = (this[20].toInt() xor 1).toByte() }
    val manager = available()
    manager.download()
    assertTrue(awaitState<AppUpdateState.Error>(manager).message.contains("integrity"))
    assertEquals(listOf(patchUrl), transport.requests)
    assertTrue(File(directory, "updates/ready").listFiles().orEmpty().isEmpty())
  }

  @Test fun unrecognizedBaseFailsClosedBeforeDownload() = runBlocking {
    base.appendText("local build")
    val manager = manager()
    manager.check()
    assertTrue(awaitState<AppUpdateState.Error>(manager).message.contains("not in the published"))
    assertTrue(transport.requests.isEmpty())
  }

  @Test fun installedChangeBetweenCheckAndDownloadFailsClosed() = runBlocking {
    val manager = available()
    base.appendText("changed")
    manager.download()
    assertTrue(awaitState<AppUpdateState.Error>(manager).message.contains("changed"))
    assertTrue(transport.requests.isEmpty())
  }

  @Test fun installedChangeBeforeInstallIsRejected() = runBlocking {
    val manager = available()
    manager.download()
    awaitState<AppUpdateState.Ready>(manager)
    base.appendText("changed")
    expectFailure { manager.verifiedApkForInstall() }
    assertTrue(manager.state.value is AppUpdateState.Error)
  }

  @Test fun readyFileChangedBeforeInstallIsRejected() = runBlocking {
    val manager = available()
    manager.download()
    awaitState<AppUpdateState.Ready>(manager)
    File(directory, "updates/ready").listFiles()!!.single().appendText("changed")
    expectFailure { manager.verifiedApkForInstall() }
  }

  @Test fun wrongSignerOrPackageOrVersionInDownloadedApkIsRejected() = runBlocking {
    for (mutation in listOf<(VerifiedApk) -> VerifiedApk>(
      { it.copy(signerSha256 = "cc".repeat(32)) },
      { it.copy(packageName = "another.app") },
      { it.copy(versionCode = 2) },
      { it.copy(minSdk = 37) },
    )) {
      verifier.archiveMutation = mutation
      val manager = available()
      manager.download()
      awaitState<AppUpdateState.Error>(manager)
    }
  }

  @Test fun noUpdateStillRequiresExactBaseAndSigningIdentity() = runBlocking {
    val current = base.readBytes()
    transport.manifestText = manifest(latestVersion = 2, latestBytes = current)
    val manager = manager()
    manager.check()
    awaitState<AppUpdateState.UpToDate>(manager)
    base.appendText("changed")
    manager.check()
    awaitState<AppUpdateState.Error>(manager)
    Unit
  }

  @Test fun feedSignerCannotOverrideInstalledSigner() = runBlocking {
    val json = JSONObject(manifest())
    json.getJSONObject("latest").put("signerSha256", "cc".repeat(32))
    json.getJSONArray("releases").getJSONObject(1).put("signerSha256", "cc".repeat(32))
    transport.manifestText = json.toString()
    val manager = manager()
    manager.check()
    assertTrue(awaitState<AppUpdateState.Error>(manager).message.contains("signing identity"))
  }

  @Test fun checkIsSingleFlightAndCancellationLeavesNoStaleCompletion() = runBlocking {
    transport.manifestGate = CompletableDeferred()
    val manager = manager()
    manager.check()
    awaitState<AppUpdateState.Checking>(manager)
    withTimeout(5_000) { while (transport.manifestRequests.get() == 0) delay(1) }
    manager.check()
    manager.cancel()
    awaitState<AppUpdateState.Idle>(manager)
    assertEquals(1, transport.manifestRequests.get())
    transport.manifestGate!!.complete(Unit)
    assertEquals(AppUpdateState.Idle, manager.state.value)
  }

  @Test fun cancelDownloadRetainsOfferAndRemovesTemporaryFiles() = runBlocking {
    transport.downloadGate = CompletableDeferred()
    val manager = available()
    manager.download()
    awaitState<AppUpdateState.Downloading>(manager)
    manager.cancel()
    awaitState<AppUpdateState.Available>(manager)
    assertFalse(File(directory, "updates/target.part").exists())
    assertFalse(File(directory, "updates/patch.part").exists())
  }

  @Test fun unpublishedFeedShowsFriendlyRetry() = runBlocking {
    transport.manifestFailure = UpdateTransferFailure("App updates have not been published yet. Please check again later.", true)
    val manager = manager()
    manager.check()
    val error = awaitState<AppUpdateState.Error>(manager)
    assertTrue(error.message.contains("not been published"))
    assertEquals(UpdateRetryAction.CHECK, error.retryAction)
  }

  @Test fun errorStateCanBeRetriedImmediately() = runBlocking {
    transport.manifestFailure = UpdateTransferFailure("Temporary failure")
    val manager = manager()
    manager.check()
    awaitState<AppUpdateState.Error>(manager)
    transport.manifestFailure = null
    manager.check()
    awaitState<AppUpdateState.Available>(manager)
    manager.download()
    awaitState<AppUpdateState.Ready>(manager)
    assertEquals(2, transport.manifestRequests.get())
  }

  @Test fun storageFailureDoesNotCrashOrDownload() = runBlocking {
    val file = File(directory, "not-a-directory").apply { writeText("occupied") }
    val manager = AppUpdateManager(file, feed, verifier, transport, scope)
    manager.check()
    awaitState<AppUpdateState.Error>(manager)
    assertTrue(transport.requests.isEmpty())
  }

  @Test fun prunesOnlyUpdaterApksAtOrBelowInstalledVersion() = runBlocking {
    val ready = File(directory, "updates/ready").apply { mkdirs() }
    val obsolete = File(ready, "2-${"01".repeat(32)}.apk").apply { writeText("old") }
    val future = File(ready, "3-${"02".repeat(32)}.apk").apply { writeText("future") }
    val unrelated = File(ready, "notes.txt").apply { writeText("keep") }
    available()
    assertFalse(obsolete.exists())
    assertTrue(future.exists())
    assertTrue(unrelated.exists())
  }

  @Test fun nodeProtocolFixtureReconstructsExactBytes() {
    val fixturePath = generateSequence(File(requireNotNull(System.getProperty("user.dir")))) { it.parentFile }
      .map { File(it, "scripts/app-update-protocol-fixture.json") }.first { it.isFile }
    val fixture = JSONObject(fixturePath.readText())
    base.writeBytes(Base64.getDecoder().decode(fixture.getString("baseBase64")))
    val expected = Base64.getDecoder().decode(fixture.getString("targetBase64"))
    val patch = File(directory, "node.patch").apply { writeBytes(Base64.getDecoder().decode(fixture.getString("patchBase64"))) }
    val output = File(directory, "node.apk")
    UpdatePatchApplier.apply(base, patch, output, fixture.getString("baseSha256"), fixture.getString("targetSha256"), expected.size.toLong())
    assertArrayEquals(expected, output.readBytes())
  }

  @Test fun patchRejectsMalformedOperationsAndDeletesPartialOutput() {
    target = ByteArray(UpdatePatchApplier.MAX_OPERATIONS + 2) { 3 }
    val cases = listOf<Pair<String, (DataOutputStream) -> Unit>>(
      "negative offset" to { it.writeByte(0); it.writeLong(-1); it.writeInt(1) },
      "overflow offset" to { it.writeByte(0); it.writeLong(Long.MAX_VALUE); it.writeInt(1) },
      "beyond source" to { it.writeByte(0); it.writeLong(base.length()); it.writeInt(1) },
      "zero length" to { it.writeByte(1); it.writeInt(0) },
      "oversized output" to { it.writeByte(1); it.writeInt(target.size + 1) },
      "unknown opcode" to { it.writeByte(7) },
      "early end" to { it.writeByte(255) },
      "truncated literal" to { it.writeByte(1); it.writeInt(target.size); it.writeByte(0) },
      "operation cap" to { stream -> repeat(UpdatePatchApplier.MAX_OPERATIONS + 1) { stream.writeByte(0); stream.writeLong(0); stream.writeInt(1) } },
    )
    for ((label, operations) in cases) {
      val patch = File(directory, "bad.patch").apply { writeBytes(patchBytes(operations)) }
      val output = File(directory, "bad.apk")
      try {
        UpdatePatchApplier.apply(base, patch, output, hash(base.readBytes()), hash(target), target.size.toLong())
        fail(label)
      } catch (_: Exception) { assertFalse("$label must remove its partial output", output.exists()) }
    }
  }

  @Test fun patchRejectsTrailingCompressedOrExpandedData() {
    for (bytes in listOf(validPatch() + byteArrayOf(1), patchBytes { stream ->
      stream.writeByte(1); stream.writeInt(target.size); stream.write(target); stream.writeByte(255); stream.writeByte(1)
    })) {
      val patch = File(directory, "trailing.patch").apply { writeBytes(bytes) }
      assertThrows(Exception::class.java) { UpdatePatchApplier.apply(base, patch, File(directory, "output"), hash(base.readBytes()), hash(target), target.size.toLong()) }
    }
  }

  @Test fun patchChecksItsHeaderBaseAndTargetHashes() {
    val patch = File(directory, "patch").apply { writeBytes(validPatch()) }
    assertThrows(Exception::class.java) { UpdatePatchApplier.apply(base, patch, File(directory, "output"), "00".repeat(32), hash(target), target.size.toLong()) }
    assertThrows(Exception::class.java) { UpdatePatchApplier.apply(base, patch, File(directory, "output"), hash(base.readBytes()), "00".repeat(32), target.size.toLong()) }
  }

  @Test fun manifestRejectsFractionalSizesUntrustedUrlsAndInconsistentLatest() {
    val variants = listOf<(JSONObject) -> Unit>(
      { it.getJSONObject("latest").getJSONObject("apk").put("size", 1.5) },
      { it.getJSONObject("latest").getJSONObject("apk").put("size", MAX_UPDATE_BYTES + 1) },
      { it.getJSONObject("latest").getJSONObject("apk").put("url", "https://evil.example/app.apk") },
      { it.getJSONObject("latest").put("versionCode", 9) },
      { it.put("schemaVersion", 2) },
      { it.put("channel", "stable") },
    )
    variants.forEach { mutation ->
      val json = JSONObject(manifest()).also(mutation)
      assertThrows(UpdateFailure::class.java) { UpdateManifestParser.parse(json.toString()) }
    }
  }

  @Test fun urlPolicyRejectsOriginConfusionTraversalAndUnscopedCdn() {
    val blocked = listOf(
      "http://www.danielshort.me/app-updates/app.apk",
      "https://www.danielshort.me.evil.example/app-updates/app.apk",
      "https://user@www.danielshort.me/app-updates/app.apk",
      "https://www.danielshort.me:8443/app-updates/app.apk",
      "https://www.danielshort.me/app-updates/../app.apk",
      "https://www.danielshort.me/app-updates/%2e%2e/app.apk",
      "https://www.danielshort.me/app-updates/%252e/app.apk",
      "https://github.com/another/repo/releases/download/app.apk",
      "https://release-assets.githubusercontent.com/app.apk",
    )
    blocked.forEach { assertThrows(it, IllegalArgumentException::class.java) { UpdateUrlPolicy.requirePublishedUrl(it) } }
    val start = UpdateUrlPolicy.requirePublishedUrl(apkUrl)
    assertEquals("release-assets.githubusercontent.com", UpdateUrlPolicy.requireRedirect(start, "https://release-assets.githubusercontent.com/github-production-release-asset/id?signature=token", start, false).host)
    assertThrows(IllegalArgumentException::class.java) { UpdateUrlPolicy.requireRedirect(start, "http://release-assets.githubusercontent.com/id", start, false) }
    assertThrows(IllegalArgumentException::class.java) { UpdateUrlPolicy.requireRedirect(java.net.URI(feed), "https://release-assets.githubusercontent.com/id", java.net.URI(feed), true) }
  }

  private fun manager() = AppUpdateManager(File(directory, "updates"), feed, verifier, transport, scope)
  private suspend fun available(): AppUpdateManager = manager().also { it.check(); awaitState<AppUpdateState.Available>(it) }
  private suspend inline fun <reified T : AppUpdateState> awaitState(manager: AppUpdateManager): T = withTimeout(5_000) { manager.state.first { it is T } as T }
  private suspend fun expectFailure(action: suspend () -> Unit) {
    try { action(); fail("Expected update verification to fail") } catch (_: UpdateFailure) { }
  }
  private fun hash(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).hex()

  private fun manifest(patch: ByteArray? = null, latestVersion: Int = 3, latestBytes: ByteArray = target): String {
    fun release(version: Int, bytes: ByteArray) = JSONObject().put("versionCode", version).put("sha256", hash(bytes)).put("size", bytes.size).put("signerSha256", signer)
    val releases = JSONArray().put(release(2, base.readBytes()))
    if (latestVersion != 2) releases.put(release(latestVersion, latestBytes))
    val patches = JSONArray()
    if (patch != null) patches.put(JSONObject().put("fromSha256", hash(base.readBytes())).put("toSha256", hash(latestBytes)).put("url", patchUrl).put("sha256", hash(patch)).put("size", patch.size).put("format", "dsupd1-gzip"))
    return JSONObject().put("schemaVersion", 1).put("channel", "review").put("packageName", "me.danielshort.app.debug")
      .put("latest", JSONObject().put("versionCode", latestVersion).put("versionName", "0.$latestVersion.0").put("minSdk", 26).put("signerSha256", signer)
        .put("apk", JSONObject().put("url", apkUrl).put("sha256", hash(latestBytes)).put("size", latestBytes.size)))
      .put("releases", releases).put("patches", patches).toString()
  }

  private fun validPatch() = patchBytes { output -> output.writeByte(1); output.writeInt(target.size); output.write(target); output.writeByte(255) }
  private fun patchBytes(operations: (DataOutputStream) -> Unit): ByteArray {
    val bytes = ByteArrayOutputStream()
    DataOutputStream(GZIPOutputStream(bytes)).use { output ->
      output.write("DSUPD001".toByteArray())
      output.write(MessageDigest.getInstance("SHA-256").digest(base.readBytes()))
      output.write(MessageDigest.getInstance("SHA-256").digest(target))
      output.writeLong(base.length()); output.writeLong(target.size.toLong()); operations(output)
    }
    return bytes.toByteArray()
  }

  private inner class FakeVerifier : InstalledAppVerifier {
    override val sdkVersion = 36
    var archiveChecks = 0
    var archiveMutation: (VerifiedApk) -> VerifiedApk = { it }
    override fun installed(checkCancelled: () -> Unit): VerifiedApk {
      checkCancelled()
      return VerifiedApk(base, "me.danielshort.app.debug", 2, "0.2.0", sha256(base), base.length(), signer, 26)
    }
    override fun archive(file: File, checkCancelled: () -> Unit): VerifiedApk {
      checkCancelled(); archiveChecks++
      return archiveMutation(VerifiedApk(file, "me.danielshort.app.debug", 3, "0.3.0", sha256(file), file.length(), signer, 26))
    }
  }

  private inner class FakeTransport : UpdateTransport {
    var manifestText = ""
    var manifestFailure: Exception? = null
    var manifestGate: CompletableDeferred<Unit>? = null
    var downloadGate: CompletableDeferred<Unit>? = null
    var failPatch = false
    val manifestRequests = AtomicInteger()
    val artifacts = mutableMapOf<String, ByteArray>()
    val requests = mutableListOf<String>()
    override suspend fun manifest(url: String): String {
      manifestRequests.incrementAndGet()
      manifestGate?.await()
      manifestFailure?.let { throw it }
      return manifestText
    }
    override suspend fun download(artifact: UpdateArtifact, destination: File, progress: (Long, Long) -> Unit) {
      requests += artifact.url
      destination.writeText("partial")
      downloadGate?.await()
      if (failPatch && artifact.url == patchUrl) throw UpdateTransferFailure("Interrupted")
      destination.writeBytes(artifacts.getValue(artifact.url))
      progress(destination.length(), artifact.size)
    }
  }
}
