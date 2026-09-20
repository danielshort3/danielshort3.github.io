package me.danielshort.app.updates

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.stateIn

class UpdateNetworkMonitor(context: Context, scope: CoroutineScope) {
  private val connectivity = context.applicationContext.getSystemService(ConnectivityManager::class.java)
  private fun current(): UpdateNetworkState {
    val capabilities = connectivity.getNetworkCapabilities(connectivity.activeNetwork)
    return UpdateNetworkState(
      connected = capabilities?.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) == true &&
        capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED),
      metered = connectivity.isActiveNetworkMetered
    )
  }

  val state = callbackFlow {
    val callback = object : ConnectivityManager.NetworkCallback() {
      override fun onAvailable(network: Network) { trySend(current()) }
      override fun onLost(network: Network) { trySend(current()) }
      override fun onCapabilitiesChanged(network: Network, capabilities: NetworkCapabilities) { trySend(current()) }
    }
    connectivity.registerDefaultNetworkCallback(callback)
    trySend(current())
    awaitClose { connectivity.unregisterNetworkCallback(callback) }
  }.stateIn(scope, SharingStarted.WhileSubscribed(5_000), current())
}
