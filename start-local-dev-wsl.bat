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

set "WSL_IP="
if "%DISTRO%"=="" (
  for /f "tokens=1" %%I in ('wsl.exe bash -lc "hostname -I"') do if not defined WSL_IP set "WSL_IP=%%I"
) else (
  for /f "tokens=1" %%I in ('wsl.exe -d "%DISTRO%" bash -lc "hostname -I"') do if not defined WSL_IP set "WSL_IP=%%I"
)
if not "%WSL_IP%"=="" echo [launcher-wsl] WSL IP fallback: %WSL_IP%

set "LOCAL_DEV_PORT="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$start=[int]$env:PORT; $limit=[Math]::Min(65535,$start+50); for($p=$start; $p -le $limit; $p++){ $busy=$false; foreach($hostName in @('localhost','127.0.0.1')){ try { $client=New-Object Net.Sockets.TcpClient; $ar=$client.BeginConnect($hostName,$p,$null,$null); if($ar.AsyncWaitHandle.WaitOne(150)){ $client.EndConnect($ar); $busy=$true }; $client.Close() } catch {} if($busy){ break } }; if(-not $busy){ Write-Output $p; exit 0 } }; exit 1"`) do set "LOCAL_DEV_PORT=%%P"
if "%LOCAL_DEV_PORT%"=="" (
  echo [launcher-wsl] No available local port found from %PORT% through the next 50 ports.
  goto :error
)
if not "%LOCAL_DEV_PORT%"=="%PORT%" echo [launcher-wsl] Port %PORT% is already in use; using %LOCAL_DEV_PORT% instead.
set "PORT=%LOCAL_DEV_PORT%"

set "WSL_SCRIPT=cd '%WSL_PATH%' && if [ ! -d node_modules ] || [ ! -d node_modules/@esbuild/linux-x64 ]; then echo '[launcher-wsl] Installing Linux deps...' && npm install; fi && CMS_ALLOW_PRIVATE_HOSTS=1 npm run dev -- --host 0.0.0.0 --port %PORT%; STATUS=$?; if [ $STATUS -ne 0 ]; then echo '[launcher-wsl] Dev command exited with code' $STATUS; echo '[launcher-wsl] Press Enter to close...'; read _; fi"

echo [launcher-wsl] Starting WSL dev server on http://localhost:%PORT% ...
if "%DISTRO%"=="" (
  start "WSL Local Website Dev Server" wsl.exe bash -lc "%WSL_SCRIPT%"
) else (
  start "WSL Local Website Dev Server" wsl.exe -d "%DISTRO%" bash -lc "%WSL_SCRIPT%"
)

set "LOCAL_DEV_PORT=%PORT%"
set "LOCAL_DEV_WSL_IP=%WSL_IP%"
set "LOCAL_DEV_ENDPOINT="
for /f "usebackq delims=" %%H in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$start=[int]$env:LOCAL_DEV_PORT; $limit=[Math]::Min(65535,$start+50); $hosts=@('localhost','127.0.0.1'); if($env:LOCAL_DEV_WSL_IP){ $hosts += $env:LOCAL_DEV_WSL_IP }; $deadline=(Get-Date).AddSeconds(120); while((Get-Date)-lt $deadline){ for($p=$start; $p -le $limit; $p++){ foreach($hostName in $hosts){ try { $client=New-Object Net.Sockets.TcpClient; $ar=$client.BeginConnect($hostName,$p,$null,$null); if($ar.AsyncWaitHandle.WaitOne(250)){ $client.EndConnect($ar); $client.Close(); Write-Output ($hostName + '|' + $p); exit 0 } $client.Close() } catch {} } } Start-Sleep -Milliseconds 300 }; exit 1"`) do set "LOCAL_DEV_ENDPOINT=%%H"

if "%LOCAL_DEV_ENDPOINT%"=="" (
  echo [launcher-wsl] Server was not reachable within 120s.
  echo [launcher-wsl] Check the "WSL Local Website Dev Server" window for errors.
  goto :error
)
for /f "tokens=1,2 delims=|" %%A in ("%LOCAL_DEV_ENDPOINT%") do (
  set "LOCAL_DEV_HOST=%%A"
  set "LOCAL_DEV_PORT=%%B"
)

echo [launcher-wsl] Opening browser...
start "" "http://%LOCAL_DEV_HOST%:%LOCAL_DEV_PORT%/"
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
echo   - Uses the next available port if the requested port is busy.
echo   - Opens localhost when Windows can reach it, otherwise opens the WSL IP.
echo   - Installs Linux dependencies if needed.
exit /b 0

:error
pause
exit /b 1

:done
exit /b 0
