@echo off
setlocal EnableExtensions

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

set "PORT=%~1"
if "%PORT%"=="" set "PORT=3000"

for /f "delims=0123456789" %%A in ("%PORT%") do (
  echo [launcher-wsl] Invalid port: %PORT%
  goto :badport
)
if %PORT% LSS 1 goto :badport
if %PORT% GTR 65535 goto :badport

set "DISTRO=%~2"
for %%I in ("%~dp0.") do set "WIN_ROOT=%%~fI"

set "WSL_PATH=%WIN_ROOT:\=/%"
set "DRIVE=%WSL_PATH:~0,1%"
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "$env:DRIVE.ToLower()"`) do set "DRIVE=%%D"
set "WSL_PATH=/mnt/%DRIVE%%WSL_PATH:~2%"

echo [launcher-wsl] Repo path (Windows): %WIN_ROOT%
echo [launcher-wsl] Repo path (WSL): %WSL_PATH%

where wsl >nul 2>nul
if errorlevel 1 (
  echo [launcher-wsl] WSL is not available on PATH.
  goto :error
)

set "WSL_PRECHECK=test -d '%WSL_PATH%'"
if "%DISTRO%"=="" (
  wsl.exe bash -lc "%WSL_PRECHECK%"
) else (
  wsl.exe -d "%DISTRO%" bash -lc "%WSL_PRECHECK%"
)
if errorlevel 1 (
  echo [launcher-wsl] WSL could not access this repo path.
  echo [launcher-wsl] Confirm the distro name and that C: is mounted in WSL.
  goto :error
)

set "WSL_NPM_CHECK=command -v npm >/dev/null 2>&1"
if "%DISTRO%"=="" (
  wsl.exe bash -lc "%WSL_NPM_CHECK%"
) else (
  wsl.exe -d "%DISTRO%" bash -lc "%WSL_NPM_CHECK%"
)
if errorlevel 1 (
  echo [launcher-wsl] npm is not installed in this WSL distro.
  echo [launcher-wsl] Install Node.js in WSL, then run again.
  goto :error
)

set "WSL_SCRIPT=cd '%WSL_PATH%' && if [ ! -d node_modules ] || [ ! -d node_modules/@esbuild/linux-x64 ]; then echo '[launcher-wsl] Installing Linux deps...' && npm install; fi && npm run dev -- --port %PORT%; STATUS=$?; if [ $STATUS -ne 0 ]; then echo '[launcher-wsl] Dev command exited with code' $STATUS; echo '[launcher-wsl] Press Enter to close...'; read _; fi"

echo [launcher-wsl] Starting WSL dev server on http://localhost:%PORT% ...
if "%DISTRO%"=="" (
  start "WSL Local Website Dev Server" wsl.exe bash -lc "%WSL_SCRIPT%"
) else (
  start "WSL Local Website Dev Server" wsl.exe -d "%DISTRO%" bash -lc "%WSL_SCRIPT%"
)

set "LOCAL_DEV_PORT=%PORT%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port=[int]$env:LOCAL_DEV_PORT; $deadline=(Get-Date).AddSeconds(120); while((Get-Date)-lt $deadline){ try { $client=New-Object Net.Sockets.TcpClient; $ar=$client.BeginConnect('127.0.0.1',$port,$null,$null); if($ar.AsyncWaitHandle.WaitOne(250)){ $client.EndConnect($ar); $client.Close(); exit 0 } $client.Close() } catch {} Start-Sleep -Milliseconds 300 }; exit 1"

if errorlevel 1 (
  echo [launcher-wsl] Server was not reachable within 120s.
  echo [launcher-wsl] Check the "WSL Local Website Dev Server" window for errors.
  goto :error
)

echo [launcher-wsl] Opening browser...
start "" "http://localhost:%PORT%/"
goto :done

:badport
echo [launcher-wsl] Port must be a whole number between 1 and 65535.
goto :error

:help
echo Usage:
echo   start-local-dev-wsl.bat [port] [distro]
echo.
echo Examples:
echo   start-local-dev-wsl.bat
echo   start-local-dev-wsl.bat 4173
echo   start-local-dev-wsl.bat 3000 Ubuntu
echo.
echo Notes:
echo   - Runs the dev server inside WSL2.
echo   - Installs Linux dependencies if needed.
exit /b 0

:error
pause
exit /b 1

:done
exit /b 0
