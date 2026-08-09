:: ============================================
:: LOCAL AI AGENT SETUP - WINDOWS BATCH FILE
:: Run this from YOUR normal terminal (not here)
:: ============================================

@echo off
echo.
echo ================================
echo  MCP Local Agent Setup for Windows
echo ================================
echo.

REM --- Step 1: Install Open WebUI globally ---
echo [1/4] Installing Open WebUI...
npm install -g open-webui || (
    echo Failed to install Open WebUI. Try running as Administrator or use Docker instead.
    pause
    exit /b 1
)
echo OK: Open WebUI installed globally.

REM --- Step 2: Configure Ollama context window ---
echo [2/4] Configuring Ollama for larger context...
powershell -Command "
$ollamaDir = '$env:LOCALAPPDATA\Ollama';
$configPath = Join-Path $ollamaDir 'ollama.json';
if (-not (Test-Path $ollamaDir)) { New-Item -ItemType Directory -Path $ollamaDir -Force | Out-Null };
$cfg = @{};
if (Test-Path $configPath) { $cfg = Get-Content $configPath | ConvertFrom-Json };
$cfg.'num_ctx' = 32768;
$cfg | ConvertTo-Json | Set-Content -Path $configPath;
Write-Host 'OK: Ollama context set to 32K tokens at' $configPath;
"

REM --- Step 3: Start Open WebUI ---
echo [3/4] Starting Open WebUI on port 3000...
start "" cmd /k "title Open WebUI && open-webui --port 3000"
timeout /t 5 >nul
echo OK: Open WebUI launched. Opening browser to http://localhost:3000
start http://localhost:3000

REM --- Step 4: Instructions for connecting MCP ---
echo [4/4] Setup complete!
echo.
echo Next steps in Open WebUI:
echo   1. Click Settings (gear icon) -> Tools -> MCP Servers
echo   2. Add these servers:
echo      - Name: shell, Command: node, Args: C:\Users\clopt\Documents\coding\Personal_Projects\danielshort3.github.io\.mcp\shell-server.js
echo      - Name: filesystem, Command: npx, Args: -y @modelcontextprotocol/server-filesystem C:\Users\clopt\Documents\coding
echo   3. Settings -> Models -> Add Ollama model qwen2.5:27b
echo   4. Start chatting and give it tasks!
echo.
pause
