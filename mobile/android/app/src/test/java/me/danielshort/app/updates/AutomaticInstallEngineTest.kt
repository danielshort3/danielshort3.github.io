package me.danielshort.app.updates

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class AutomaticInstallEngineTest {
  private class MemoryStore : AutomaticInstallStore {
    var record: AutomaticInstallRecord? = null
    override fun read() = record
    override fun write(record: AutomaticInstallRecord) { this.record = record }
  }

  private class FakePlatform : AutomaticInstallPlatform {
    override var sdkVersion = 36
    var permission = true
    var installed = 4L
    var target = 36
    var created = 0
    var committed = 0
    var duringStage: () -> Unit = {}
    var commitFailure = false
    var abandonFailure = false
    var queryFailure = false
    val active = mutableSetOf<Int>()
    val abandoned = mutableSetOf<Int>()
    override fun canInstall() = permission
    override fun installedVersion() = installed
    override fun targetSdk(apk: File) = target
    override fun sessions(): Set<Int> { if (queryFailure) error("fixture query"); return active.toSet() }
    override fun create(apk: File) = (++created).also { active.add(it) }
    override fun stage(sessionId: Int, apk: File, checkEligible: () -> Unit) { duringStage(); checkEligible() }
    override fun commit(sessionId: Int, token: String) { if (commitFailure) error("fixture"); committed++ }
    override fun abandon(sessionId: Int) { if (abandonFailure) error("fixture abandon"); active.remove(sessionId); abandoned.add(sessionId) }
  }

  private class Fixture {
    val store = MemoryStore()
    val platform = FakePlatform()
    var eligible = true
    var ready: Long? = 5
    var verified = 0
    var clock = 100L
    var verification: suspend () -> Unit = {}
    fun engine() = AutomaticInstallEngine(store, platform, { ready }, {
      verified++
      verification()
      File("verified.apk")
    }, { clock })
  }

  @Test fun platformAndTargetSdkGatesAreConservative() {
    for (sdk in 26..30) assertFalse(AutomaticInstallPolicy.supportsUnattended(sdk, 36))
    for ((sdk, target) in mapOf(31 to 29, 32 to 29, 33 to 30, 34 to 31, 35 to 33, 36 to 34, 37 to 35)) {
      assertTrue(AutomaticInstallPolicy.supportsUnattended(sdk, target))
      assertFalse(AutomaticInstallPolicy.supportsUnattended(sdk, target - 1))
    }
    assertFalse(AutomaticInstallPolicy.supportsUnattended(38, 38))
  }

  @Test fun optOutForegroundOrProtectedWorkCannotVerifyOrCreateSession() = runBlocking {
    val fixture = Fixture().apply { eligible = false }
    assertEquals(AutomaticInstallResult.DEFERRED, fixture.engine().attemptIfEligible { fixture.eligible })
    assertEquals(0, fixture.verified)
    assertEquals(0, fixture.platform.created)
    assertNull(fixture.store.record)
  }

  @Test fun onlyReadyUpdateCanStart() = runBlocking {
    val fixture = Fixture().apply { ready = null }
    assertEquals(AutomaticInstallResult.DEFERRED, fixture.engine().attemptIfEligible { true })
    assertEquals(0, fixture.platform.created)
  }

  @Test fun olderAndroidUsesManualInstallWithoutCreatingSession() = runBlocking {
    val fixture = Fixture().apply { platform.sdkVersion = 30 }
    assertEquals(AutomaticInstallResult.MANUAL_REQUIRED, fixture.engine().attemptIfEligible { true })
    assertEquals(AutomaticInstallStatus.MANUAL_REQUIRED, fixture.store.record!!.status)
    assertEquals(0, fixture.verified)
    assertEquals(0, fixture.platform.created)
  }

  @Test fun permissionDenialIsPersistedAndNeverPromptsOrLoopsAfterRestart() = runBlocking {
    val fixture = Fixture().apply { platform.permission = false }
    assertEquals(AutomaticInstallResult.MANUAL_REQUIRED, fixture.engine().attemptIfEligible { true })
    assertEquals(AutomaticInstallStatus.NEEDS_PERMISSION, fixture.store.record!!.status)
    fixture.platform.permission = true
    assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, fixture.engine().attemptIfEligible { true })
    assertEquals(0, fixture.platform.created)
  }

  @Test fun unsupportedTargetStopsBeforeSessionCreation() = runBlocking {
    val fixture = Fixture().apply { platform.target = 33 }
    assertEquals(AutomaticInstallResult.MANUAL_REQUIRED, fixture.engine().attemptIfEligible { true })
    assertEquals(1, fixture.verified)
    assertEquals(0, fixture.platform.created)
  }

  @Test fun verifiesAgainAndJournalsBeforeCommit() = runBlocking {
    val fixture = Fixture()
    val engine = fixture.engine()
    assertEquals(AutomaticInstallResult.STARTED, engine.attemptIfEligible { true })
    assertEquals(1, fixture.verified)
    assertEquals(1, fixture.platform.committed)
    assertEquals(AutomaticInstallStatus.INSTALLING, engine.status.value)
    assertTrue(fixture.store.record!!.token.isNotBlank())
    assertEquals(1, fixture.store.record!!.sessionId)
  }

  @Test fun failedVerificationNeverCreatesSessionAndRemainsBlockedOnRestart() = runBlocking {
    val fixture = Fixture().apply { verification = { throw UpdateFailure("bad APK") } }
    assertEquals(AutomaticInstallResult.FAILED, fixture.engine().attemptIfEligible { true })
    assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, fixture.engine().attemptIfEligible { true })
    assertEquals(0, fixture.platform.created)
  }

  @Test fun concurrentAttemptsProduceExactlyOneCommittedSession() = runBlocking {
    val entered = CompletableDeferred<Unit>()
    val release = CompletableDeferred<Unit>()
    val fixture = Fixture().apply { verification = { entered.complete(Unit); release.await() } }
    val engine = fixture.engine()
    val first = async { engine.attemptIfEligible { true } }
    entered.await()
    val second = async { engine.attemptIfEligible { true } }
    release.complete(Unit)
    assertEquals(AutomaticInstallResult.STARTED, first.await())
    assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, second.await())
    assertEquals(1, fixture.platform.committed)
  }

  @Test fun returningForegroundDuringVerificationDefersBeforeSessionCreation() = runBlocking {
    val fixture = Fixture()
    fixture.verification = { fixture.eligible = false }
    val engine = fixture.engine()
    assertEquals(AutomaticInstallResult.DEFERRED, engine.attemptIfEligible { fixture.eligible })
    assertEquals(0, fixture.platform.created)
    fixture.verification = {}
    fixture.eligible = true
    assertEquals(AutomaticInstallResult.STARTED, engine.attemptIfEligible { fixture.eligible })
  }

  @Test fun returningForegroundDuringStagingAbandonsBeforeCommit() = runBlocking {
    val fixture = Fixture()
    fixture.platform.duringStage = { fixture.eligible = false }
    assertEquals(AutomaticInstallResult.DEFERRED, fixture.engine().attemptIfEligible { fixture.eligible })
    assertEquals(setOf(1), fixture.platform.abandoned)
    assertEquals(0, fixture.platform.committed)
  }

  @Test fun permissionRevocationDuringStagingAbandonsAndRequiresUserAction() = runBlocking {
    val fixture = Fixture()
    fixture.platform.duringStage = { fixture.platform.permission = false }
    assertEquals(AutomaticInstallResult.MANUAL_REQUIRED, fixture.engine().attemptIfEligible { true })
    assertEquals(AutomaticInstallStatus.NEEDS_PERMISSION, fixture.store.record!!.status)
    assertEquals(0, fixture.platform.committed)
  }

  @Test fun forgedOrStaleCallbacksCannotChangeInstallationState() = runBlocking {
    val fixture = Fixture()
    val engine = fixture.engine()
    engine.attemptIfEligible { true }
    val record = fixture.store.record!!
    assertFalse(engine.receive(record.sessionId, "forged", AutomaticInstallCallback.SUCCESS))
    assertFalse(engine.receive(999, record.token, AutomaticInstallCallback.FAILURE))
    assertEquals(AutomaticInstallStatus.INSTALLING, engine.status.value)
    assertTrue(engine.receive(record.sessionId, record.token, AutomaticInstallCallback.FAILURE))
    assertFalse(engine.receive(record.sessionId, record.token, AutomaticInstallCallback.SUCCESS))
    assertEquals(AutomaticInstallStatus.FAILED, engine.status.value)
  }

  @Test fun pendingSystemConfirmationPersistsManualFallbackWithoutRepeatingAttempt() = runBlocking {
    val fixture = Fixture()
    val engine = fixture.engine()
    engine.attemptIfEligible { true }
    val record = fixture.store.record!!
    assertTrue(engine.receive(record.sessionId, record.token, AutomaticInstallCallback.PENDING_USER_ACTION))
    assertEquals(setOf(record.sessionId), fixture.platform.abandoned)
    val restarted = fixture.engine()
    restarted.recover()
    assertEquals(AutomaticInstallStatus.MANUAL_REQUIRED, restarted.status.value)
    assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, restarted.attemptIfEligible { true })
    assertEquals(1, fixture.platform.committed)
  }

  @Test fun failedCommitAbandonsAndCannotLoop() = runBlocking {
    val fixture = Fixture().apply { platform.commitFailure = true }
    assertEquals(AutomaticInstallResult.FAILED, fixture.engine().attemptIfEligible { true })
    assertEquals(setOf(1), fixture.platform.abandoned)
    assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, fixture.engine().attemptIfEligible { true })
  }

  @Test fun processDeathDuringStagingAndUnjournaledSessionAreRecovered() = runBlocking {
    val fixture = Fixture()
    fixture.store.record = AutomaticInstallRecord(5, AutomaticInstallStatus.STAGING, 1, "token", fixture.clock)
    fixture.platform.active.addAll(listOf(1, 2))
    val restarted = fixture.engine()
    restarted.recover()
    assertEquals(setOf(1, 2), fixture.platform.abandoned)
    assertEquals(AutomaticInstallStatus.FAILED, restarted.status.value)
    assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, restarted.attemptIfEligible { true })
  }

  @Test fun committedSessionSurvivesRestartWithoutDuplicateCommit() = runBlocking {
    val fixture = Fixture()
    fixture.engine().attemptIfEligible { true }
    val restarted = fixture.engine()
    restarted.recover()
    assertEquals(AutomaticInstallStatus.INSTALLING, restarted.status.value)
    assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, restarted.attemptIfEligible { true })
    assertEquals(1, fixture.platform.committed)
  }

  @Test fun missingOrExpiredCallbackLeavesManualFallback() = runBlocking {
    for (expire in listOf(false, true)) {
      val fixture = Fixture()
      fixture.engine().attemptIfEligible { true }
      if (expire) fixture.clock += 31 * 60 * 1000L else fixture.platform.active.clear()
      val restarted = fixture.engine()
      restarted.recover()
      assertEquals(AutomaticInstallStatus.MANUAL_REQUIRED, restarted.status.value)
      assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, restarted.attemptIfEligible { true })
    }
  }

  @Test fun successfulReplacementRecoversEvenIfSuccessCallbackWasLost() = runBlocking {
    val fixture = Fixture()
    fixture.engine().attemptIfEligible { true }
    fixture.platform.installed = 5
    fixture.platform.active.clear()
    val restarted = fixture.engine()
    restarted.recover()
    assertEquals(AutomaticInstallStatus.INSTALLED, restarted.status.value)
    fixture.ready = 6
    assertEquals(AutomaticInstallResult.STARTED, restarted.attemptIfEligible { true })
  }

  @Test fun manualHandoffAndCancellationSurviveRestartWithoutAutomaticRetry() = runBlocking {
    val fixture = Fixture()
    assertTrue(fixture.engine().markManualInstallRequested())
    val restarted = fixture.engine()
    assertEquals(AutomaticInstallStatus.MANUAL_REQUIRED, restarted.status.value)
    assertEquals(AutomaticInstallResult.ALREADY_ATTEMPTED, restarted.attemptIfEligible { true })
    assertEquals(0, fixture.platform.committed)
  }

  @Test fun manualHandoffCannotRaceAnAutomaticSession() = runBlocking {
    val fixture = Fixture()
    val engine = fixture.engine()
    engine.attemptIfEligible { true }
    assertFalse(engine.markManualInstallRequested())
  }

  @Test fun unconfirmedAbandonmentDoesNotEnableAnotherInstaller() = runBlocking {
    val fixture = Fixture()
    val engine = fixture.engine()
    engine.attemptIfEligible { true }
    fixture.platform.abandonFailure = true
    val record = fixture.store.record!!
    assertFalse(engine.receive(record.sessionId, record.token, AutomaticInstallCallback.PENDING_USER_ACTION))
    fixture.clock += 31 * 60 * 1000L
    engine.recover()
    assertEquals(AutomaticInstallStatus.INSTALLING, engine.status.value)
    assertFalse(engine.markManualInstallRequested())
    fixture.platform.abandonFailure = false
    engine.recover()
    assertEquals(AutomaticInstallStatus.MANUAL_REQUIRED, engine.status.value)
  }

  @Test fun expectedRecoveryErrorDoesNotCrashOrHidePotentiallyActiveSession() = runBlocking {
    val fixture = Fixture()
    val engine = fixture.engine()
    engine.attemptIfEligible { true }
    fixture.platform.queryFailure = true
    engine.recover()
    assertEquals(AutomaticInstallStatus.INSTALLING, engine.status.value)
    assertFalse(engine.markManualInstallRequested())
    fixture.platform.queryFailure = false
    fixture.platform.active.clear()
    engine.recover()
    assertEquals(AutomaticInstallStatus.MANUAL_REQUIRED, engine.status.value)
  }

  @Test fun ambiguousCommitFailureStaysBusyUntilAndroidSessionCanBeReconciled() = runBlocking {
    val fixture = Fixture().apply { platform.commitFailure = true; platform.abandonFailure = true }
    val engine = fixture.engine()
    assertEquals(AutomaticInstallResult.FAILED, engine.attemptIfEligible { true })
    assertEquals(AutomaticInstallStatus.INSTALLING, engine.status.value)
    assertFalse(engine.markManualInstallRequested())
  }
}
