@echo off
setlocal EnableExtensions

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

set "PORT=%~1"
if "%PORT%"=="" set "PORT=3000"

for /f "delims=0123456789" %%A in ("%PORT%") do (
  echo [cms-wsl] Invalid port: %PORT%
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

echo [cms-wsl] Repo path (Windows): %WIN_ROOT%
echo [cms-wsl] Repo path (WSL): %WSL_PATH%

where wsl >nul 2>nul
if errorlevel 1 (
  echo [cms-wsl] WSL is not available on PATH.
  goto :error
)

set "WSL_PRECHECK=test -d '%WSL_PATH%'"
if "%DISTRO%"=="" (
  wsl.exe bash -lc "%WSL_PRECHECK%"
) else (
  wsl.exe -d "%DISTRO%" bash -lc "%WSL_PRECHECK%"
)
if errorlevel 1 (
  echo [cms-wsl] WSL could not access this repo path.
  echo [cms-wsl] Confirm the distro name and that C: is mounted in WSL.
  goto :error
)

set "WSL_NPM_CHECK=command -v npm >/dev/null 2>&1"
if "%DISTRO%"=="" (
  wsl.exe bash -lc "%WSL_NPM_CHECK%"
) else (
  wsl.exe -d "%DISTRO%" bash -lc "%WSL_NPM_CHECK%"
)
if errorlevel 1 (
  echo [cms-wsl] npm is not installed in this WSL distro.
  echo [cms-wsl] Install Node.js in WSL, then run again.
  goto :error
)

set "WSL_IP="
if "%DISTRO%"=="" (
  for /f "tokens=1" %%I in ('wsl.exe bash -lc "hostname -I"') do if not defined WSL_IP set "WSL_IP=%%I"
) else (
  for /f "tokens=1" %%I in ('wsl.exe -d "%DISTRO%" bash -lc "hostname -I"') do if not defined WSL_IP set "WSL_IP=%%I"
)
if not "%WSL_IP%"=="" echo [cms-wsl] WSL IP fallback: %WSL_IP%

set "WSL_SCRIPT=cd '%WSL_PATH%' && if [ ! -d node_modules ] || [ ! -d node_modules/@esbuild/linux-x64 ]; then echo '[cms-wsl] Installing Linux deps...' && npm install; fi && CMS_ALLOW_PRIVATE_HOSTS=1 npm run dev -- --host 0.0.0.0 --port %PORT%; STATUS=$?; if [ $STATUS -ne 0 ]; then echo '[cms-wsl] Dev command exited with code' $STATUS; echo '[cms-wsl] Press Enter to close...'; read _; fi"

echo [cms-wsl] Starting local CMS server on http://localhost:%PORT%/admin ...
if "%DISTRO%"=="" (
  start "WSL Local CMS Server" wsl.exe bash -lc "%WSL_SCRIPT%"
) else (
  start "WSL Local CMS Server" wsl.exe -d "%DISTRO%" bash -lc "%WSL_SCRIPT%"
)

set "LOCAL_CMS_PORT=%PORT%"
set "LOCAL_CMS_WSL_IP=%WSL_IP%"
set "LOCAL_CMS_HOST="
for /f "usebackq delims=" %%H in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$port=[int]$env:LOCAL_CMS_PORT; $hosts=@('localhost','127.0.0.1'); if($env:LOCAL_CMS_WSL_IP){ $hosts += $env:LOCAL_CMS_WSL_IP }; $deadline=(Get-Date).AddSeconds(120); while((Get-Date)-lt $deadline){ foreach($hostName in $hosts){ try { $client=New-Object Net.Sockets.TcpClient; $ar=$client.BeginConnect($hostName,$port,$null,$null); if($ar.AsyncWaitHandle.WaitOne(250)){ $client.EndConnect($ar); $client.Close(); Write-Output $hostName; exit 0 } $client.Close() } catch {} } Start-Sleep -Milliseconds 300 }; exit 1"`) do set "LOCAL_CMS_HOST=%%H"

if "%LOCAL_CMS_HOST%"=="" (
  echo [cms-wsl] Server was not reachable within 120s.
  echo [cms-wsl] Check the "WSL Local CMS Server" window for errors.
  goto :error
)

echo [cms-wsl] Opening CMS in browser at http://%LOCAL_CMS_HOST%:%PORT%/admin ...
start "" "http://%LOCAL_CMS_HOST%:%PORT%/admin"
goto :done

:badport
echo [cms-wsl] Port must be a whole number between 1 and 65535.
goto :error

:help
echo Usage:
echo   start-local-cms-wsl.bat [port] [distro]
echo.
echo Examples:
echo   start-local-cms-wsl.bat
echo   start-local-cms-wsl.bat 4173
echo   start-local-cms-wsl.bat 3000 Ubuntu
echo.
echo Notes:
echo   - Runs the local CMS/dev server inside WSL2.
echo   - Opens localhost when Windows can reach it, otherwise opens the WSL IP.
echo   - Installs Linux dependencies if needed.
exit /b 0

:error
pause
exit /b 1

:done
exit /b 0
