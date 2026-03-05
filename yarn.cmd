@echo off
setlocal EnableExtensions

if "%~1"=="" goto :install

if /i "%~1"=="install" goto :install
if /i "%~1"=="run" goto :run
if /i "%~1"=="--version" goto :version
if /i "%~1"=="-v" goto :version
if /i "%~1"=="add" goto :add

set "SCRIPT=%~1"
shift
if "%~1"=="" (
  call npm run %SCRIPT%
) else (
  call npm run %SCRIPT% -- %*
)
exit /b %errorlevel%

:install
call npm install
exit /b %errorlevel%

:run
shift
if "%~1"=="" (
  echo yarn shim: missing script name
  exit /b 1
)
set "SCRIPT=%~1"
shift
if "%~1"=="" (
  call npm run %SCRIPT%
) else (
  call npm run %SCRIPT% -- %*
)
exit /b %errorlevel%

:add
shift
if "%~1"=="" (
  echo yarn shim: missing package name
  exit /b 1
)
call npm install %*
exit /b %errorlevel%

:version
echo 1.22.22
exit /b 0
