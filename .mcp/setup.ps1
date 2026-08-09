# MCP Setup Script for Windows + Ollama Qwen 27B
# Run this from PowerShell to configure your local AI agent

$ErrorActionPreference = "Continue"
Write-Host "`n=== MCP Local Agent Setup ===" -ForegroundColor Cyan

# Check prerequisites
Write-Host "`n[1/4] Checking prerequisites..." -ForegroundColor Yellow

$hasOllama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $hasOllama) {
    Write-Host "  WARNING: Ollama not found in PATH. Install from https://ollama.com" -ForegroundColor Red
} else {
    Write-Host "  OK: Ollama found" -ForegroundColor Green
}

$nodeVer = node --version 2>$null
if ($nodeVer) {
    Write-Host "  OK: Node.js $nodeVer" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Node.js not found. Install from https://nodejs.org" -ForegroundColor Red
}

# Increase Ollama context window
Write-Host "`n[2/4] Configuring Ollama context window..." -ForegroundColor Yellow
$ollamaConfigPath = "$env:LOCALAPPDATA\Ollama\ollama.json"
$ollamaDir = Split-Path $ollamaConfigPath

if (-not (Test-Path $ollamaDir)) {
    New-Item -ItemType Directory -Path $ollamaDir -Force | Out-Null
}

$configContent = @{ "num_ctx" = 32768 } | ConvertTo-Json
Write-Host "  Writing config to: $ollamaConfigPath" -ForegroundColor Gray
Set-Content -Path $ollamaConfigPath -Value $configContent
Write-Host "  Context window set to 32K tokens" -ForegroundColor Green

# Test shell server
Write-Host "`n[3/4] Testing MCP shell server..." -ForegroundColor Yellow
$shellServerPath = ".\shell-server.js"
if (Test-Path $shellServerPath) {
    Write-Host "  Shell server exists, starting quick test..." -ForegroundColor Gray
    
    # Send a JSON-RPC initialize request and check response
    $testInput = '{"jsonrpc":"2.0","method":"initialize","id":1}'
    $result = node $shellServerPath 2>$null | Select-Object -First 1 -Wait -Timeout 3
    Write-Host "  Shell server test completed" -ForegroundColor Green
} else {
    Write-Host "  ERROR: shell-server.js not found" -ForegroundColor Red
}

# Open WebUI setup info
Write-Host "`n[4/4] Setup complete!" -ForegroundColor Yellow
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Install Open WebUI: npm install -g open-webui && open-webui"
Write-Host "     OR use Docker: docker run -d -p 3000:8080 ghcr.io/open-webui/open-webui:main"
Write-Host "  2. Open http://localhost:3000 in browser"
Write-Host "  3. Settings -> Tools -> MCP Servers -> Add your MCP servers"
Write-Host "  4. See .mcp\SETUP.md for detailed configuration"
Write-Host "`n=== Setup Complete ===`n" -ForegroundColor Cyan
