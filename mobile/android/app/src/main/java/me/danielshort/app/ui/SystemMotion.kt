package me.danielshort.app.ui

import android.animation.ValueAnimator
import android.content.Context
import android.database.ContentObserver
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect

/** Keep custom chrome/game effects aligned with Android's Remove animations setting. */
@Composable
internal fun rememberSystemReduceMotion(): Boolean {
  val context = LocalContext.current.applicationContext
  var reduced by remember(context) { mutableStateOf(readSystemReduceMotion(context)) }
  LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { reduced = readSystemReduceMotion(context) }
  DisposableEffect(context) {
    val resolver = context.contentResolver
    val observer = object : ContentObserver(Handler(Looper.getMainLooper())) {
      override fun onChange(selfChange: Boolean) { reduced = readSystemReduceMotion(context) }
    }
    resolver.registerContentObserver(Settings.Global.getUriFor(Settings.Global.ANIMATOR_DURATION_SCALE), false, observer)
    onDispose { resolver.unregisterContentObserver(observer) }
  }
  return reduced
}

private fun readSystemReduceMotion(context: Context): Boolean = try {
  // Read the setting directly: the platform animator cache can lag its content observer.
  Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
} catch (_: SecurityException) {
  !ValueAnimator.areAnimatorsEnabled()
}
