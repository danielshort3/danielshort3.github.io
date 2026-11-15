# Watermark Remover on Home GPU

The website administrator opens Watermark Remover from Tools. The website verifies its Tools account session and issues a 60-second launch ticket. The worker accepts it once and sets an HttpOnly session for up to two hours. Close Home GPU session in the app to end that session.

The browser uses the existing React app for Manual Images, Single Song, and Order of Worship. Uploads, live progress, previews, and downloads use the Tailscale connection to the computer. The website receives the launch request only. Remote uploads use the app's multipart format and are capped at 64 MiB per request. The standalone local app retains its own ports and controls.

## Source ownership

- The website owns `pages/watermark-remover.html`, `js/tools/watermark-remover.js`, `content/tools/watermark-remover.json`, and `api/_lib/tools-endpoints/watermark-remover.js`.
- The Watermark Remover Agent repo owns the React app, native processing functions, authenticated wrapper in `src/watermark_remover/server/home_gpu.py`, and the loopback-only entry point in `scripts/run_home_gpu_app.py`.
- Video to Text owns `home_gpu_lease.py` and the Whisper acquire/release hooks. Both workers need the same lease path in the same WSL distro. The lease lives on the Linux filesystem, for example `/home/<user>/.local/state/home-gpu/gpu.lock`.

## Set up the worker

1. Build the React frontend from the Watermark Remover Agent repo with `npm.cmd --prefix frontend run build`. Verify that the repo's WSL `.venv/bin/python` can import Torch with CUDA and that the U-Net and VDSR checkpoints are present. Single Song and Order of Worship also use a reachable local Ollama model and the existing headless Selenium browser. GPU use outside these two workers is independent of the lease.
2. Set `HOME_GPU_WATERMARK_SECRET` to a new random key of at least 32 bytes in the Windows user environment and the matching website deployment environment. Keep the signer's key separate from transcription. Set `HOME_GPU_WATERMARK_ORIGIN` in both places to the existing HTTPS Funnel origin on port 8443. Set `HOME_GPU_WATERMARK_SITE_ORIGIN` on the PC to the exact origin of the deployed website or preview using it.
3. Set `HOME_GPU_LEASE_PATH` once in the Windows user environment and pass it to both workers. The optional `HOME_GPU_WATERMARK_ROOT` selects the private WSL state directory. Without it the launcher uses `~/.local/state/home-gpu/watermark`.
4. Set `HOME_GPU_WATERMARK_OLLAMA_BIN` to an installed WSL Ollama binary to let the same launcher supervise a dedicated loopback instance. Set `HOME_GPU_WATERMARK_OLLAMA_MODELS` to its model directory. The managed instance defaults to port 11435 and is configured to unload models promptly. Alternatively run an existing loopback Ollama service and set `HOME_GPU_WATERMARK_OLLAMA_URL` to it. The default model is `qwen3:8b`. The optional `HOME_GPU_WATERMARK_OLLAMA_MODEL` and `SELENIUM_BINARY` select the locally available model and browser. The desktop app continues to manage its own settings.
5. Test the wrapper privately before exposing the mount. From PowerShell in the Watermark Remover Agent repo, run `.\scripts\run_home_gpu_watermark.ps1`. It binds to loopback port 8788. Try `-Action Install -WhatIf` before registering the logon task with `-Action Install`. The listener needs the Windows account to have logged in after reboot.
6. For activation, let any existing transcription finish and restart its existing logon task with the shared lease configured. Record both existing Funnel mounts and verify transcription health. On port 8443, add the path-specific mount `tailscale funnel --bg --https=8443 --set-path=/home-tools/ http://127.0.0.1:8788`. The existing root mount still serves File Transcriber. Verify unauthenticated worker APIs return 401, the owner can launch, an uploaded PDF job completes, and a Home GPU transcript completes on the existing route. [Tailscale Funnel CLI](https://tailscale.com/docs/reference/tailscale-cli/funnel)

Remote Ollama process, network and filesystem settings are managed on the computer. Those controls remain available through the standalone local app. New tools can use the same website authorization, their own narrowly scoped worker routes, and the shared GPU lease.

## Roll back

Disable ticket issuance by removing the Watermark configuration from the website deployment and retire existing worker sessions by stopping the companion task after draining or canceling its jobs. Remove only the added mount with `tailscale funnel --https=8443 --set-path=/home-tools/ off`. Check the original transcription HTTPS health and complete a test transcript. If removing GPU coordination, stop the Watermark worker before restarting transcription without `HOME_GPU_LEASE_PATH`.
