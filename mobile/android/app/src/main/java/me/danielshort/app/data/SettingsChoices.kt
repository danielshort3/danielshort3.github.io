package me.danielshort.app.data

/** UI choices over the existing preference keys; no storage migration is needed. */
enum class AppUpdateMode { CHECK_AUTOMATICALLY, AUTOMATIC, MANUAL }
enum class ContentRefreshMode { AUTOMATIC, UNMETERED_ONLY, MANUAL }

val AppSettings.appUpdateMode: AppUpdateMode
  get() = when {
    automaticAppUpdates -> AppUpdateMode.AUTOMATIC
    checkAppUpdatesOnLaunch -> AppUpdateMode.CHECK_AUTOMATICALLY
    else -> AppUpdateMode.MANUAL
  }

fun AppSettings.withAppUpdateMode(mode: AppUpdateMode): AppSettings = copy(
  automaticAppUpdates = mode == AppUpdateMode.AUTOMATIC,
  checkAppUpdatesOnLaunch = mode != AppUpdateMode.MANUAL
)

val AppSettings.contentRefreshMode: ContentRefreshMode
  get() = when {
    !automaticUpdates -> ContentRefreshMode.MANUAL
    unmeteredOnly -> ContentRefreshMode.UNMETERED_ONLY
    else -> ContentRefreshMode.AUTOMATIC
  }

fun AppSettings.withContentRefreshMode(mode: ContentRefreshMode): AppSettings = when (mode) {
  ContentRefreshMode.MANUAL -> copy(automaticUpdates = false)
  ContentRefreshMode.AUTOMATIC -> copy(automaticUpdates = true, unmeteredOnly = false)
  ContentRefreshMode.UNMETERED_ONLY -> copy(automaticUpdates = true, unmeteredOnly = true)
}

/** An app preference must never re-enable motion disabled by Android. */
internal fun effectiveReduceMotion(appPreference: Boolean, systemPreference: Boolean): Boolean =
  appPreference || systemPreference
