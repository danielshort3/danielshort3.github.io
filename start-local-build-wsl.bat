@echo off
setlocal EnableExtensions

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

set "DISTRO=%~1"
for %%I in ("%~dp0.") do set "WIN_ROOT=%%~fI"

set "WSL_PATH=%WIN_ROOT:\=/%"
set "DRIVE=%WSL_PATH:~0,1%"
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "$env:DRIVE.ToLower()"`) do set "DRIVE=%%D"
set "WSL_PATH=/mnt/%DRIVE%%WSL_PATH:~2%"

echo [build-wsl] Repo path (Windows): %WIN_ROOT%
echo [build-wsl] Repo path (WSL): %WSL_PATH%

where wsl >nul 2>nul
if errorlevel 1 (
  echo [build-wsl] WSL is not available on PATH.
  goto :error
)

set "WSL_PRECHECK=test -d '%WSL_PATH%'"
if "%DISTRO%"=="" (
  wsl.exe bash -lc "%WSL_PRECHECK%"
) else (
  wsl.exe -d "%DISTRO%" bash -lc "%WSL_PRECHECK%"
)
if errorlevel 1 (
  echo [build-wsl] WSL could not access this repo path.
  echo [build-wsl] Confirm the distro name and that C: is mounted in WSL.
  goto :error
)

set "WSL_NPM_CHECK=command -v npm >/dev/null 2>&1"
if "%DISTRO%"=="" (
  wsl.exe bash -lc "%WSL_NPM_CHECK%"
) else (
  wsl.exe -d "%DISTRO%" bash -lc "%WSL_NPM_CHECK%"
)
if errorlevel 1 (
  echo [build-wsl] npm is not installed in this WSL distro.
  echo [build-wsl] Install Node.js in WSL, then run again.
  goto :error
)

set "WSL_SCRIPT=cd '%WSL_PATH%' && if [ ! -d node_modules ] || [ ! -d node_modules/@esbuild/linux-x64 ]; then echo '[build-wsl] Installing Linux deps...' && npm install; fi && npm run build"

echo [build-wsl] Running npm run build in WSL...
if "%DISTRO%"=="" (
  wsl.exe bash -lc "%WSL_SCRIPT%"
) else (
  wsl.exe -d "%DISTRO%" bash -lc "%WSL_SCRIPT%"
)
if errorlevel 1 (
  echo [build-wsl] Build failed.
  goto :error
)

echo [build-wsl] Build complete. Deployment assets are ready.
goto :done

:help
echo Usage:
echo   start-local-build-wsl.bat [distro]
echo.
echo Examples:
echo   start-local-build-wsl.bat
echo   start-local-build-wsl.bat Ubuntu
echo.
echo Notes:
echo   - Runs npm run build inside WSL2.
echo   - Installs Linux dependencies if needed.
exit /b 0

:error
pause
exit /b 1

:done
exit /b 0
