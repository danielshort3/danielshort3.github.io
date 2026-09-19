package me.danielshort.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import me.danielshort.app.ui.DanielShortApp

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    setContent { DanielShortApp((application as SiteApplication).contentRepository) }
  }
  override fun onStart() {
    super.onStart()
    lifecycleScope.launch { (application as SiteApplication).contentRepository.refresh() }
  }
}
