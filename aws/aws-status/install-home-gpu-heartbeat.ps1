[CmdletBinding()]
param(
  [Parameter(Mandatory)]
  [ValidateSet('Request', 'Install', 'UpdateSchedule')]
  [string] $Phase,

  [string] $ExchangeFolder
)

$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (!([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'Run heartbeat setup in an elevated Windows PowerShell window.'
}

$taskName = 'Daniel Short Home GPU Heartbeat'
$folder = Join-Path $env:ProgramData 'DanielShort\HomeGpuHeartbeat'
$subject = 'CN=Daniel Short Home GPU Status'
$certreq = Join-Path $env:windir 'System32\certreq.exe'
$certutil = Join-Path $env:windir 'System32\certutil.exe'
$source = Join-Path $PSScriptRoot 'home-gpu-heartbeat.ps1'

function New-HeartbeatTrigger {
  $now = [datetime]::UtcNow
  $start = $now.Date.AddHours([Math]::Floor($now.Hour / 6) * 6).AddMinutes(2)
  if ($start -le $now) { $start = $start.AddHours(6) }
  $trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Hours 6)
  # An explicit UTC boundary keeps the heartbeat aligned when local daylight saving time changes.
  $trigger.StartBoundary = $start.ToString("yyyy-MM-ddTHH:mm:ss'Z'", [Globalization.CultureInfo]::InvariantCulture)
  return $trigger
}

if ($Phase -eq 'UpdateSchedule') {
  $previous = Export-ScheduledTask -TaskPath '\' -TaskName $taskName
  $backup = Join-Path $folder ('task-before-update-' + [datetime]::UtcNow.ToString('yyyyMMddTHHmmssfff') + '.xml')
  $previous | Set-Content -LiteralPath $backup -Encoding Unicode
  Set-ScheduledTask -TaskPath '\' -TaskName $taskName -Trigger (New-HeartbeatTrigger) | Out-Null
  Get-ScheduledTaskInfo -TaskPath '\' -TaskName $taskName | Select-Object TaskName, NextRunTime
  return
}

if (!$ExchangeFolder) { throw 'Supply -ExchangeFolder for device enrollment.' }
$workspace = (Resolve-Path -LiteralPath $ExchangeFolder).Path

if ($Phase -eq 'Request') {
  if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'A Home GPU heartbeat task already exists.'
  }
  if (!(Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder | Out-Null }
  & icacls.exe $folder /inheritance:r /grant:r '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' '*S-1-5-19:(OI)(CI)RX' | Out-Null
  if ($LASTEXITCODE -ne 0) { throw 'Could not protect the task directory.' }
  $inf = Join-Path $folder 'device.inf'
  @"
[Version]
Signature="`$Windows NT`$"
[NewRequest]
Subject="$subject"
MachineKeySet=TRUE
Exportable=FALSE
KeyLength=2048
ProviderName="Microsoft Software Key Storage Provider"
HashAlgorithm=sha256
KeyUsage=0x80
RequestType=PKCS10
SecurityDescriptor="D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GR;;;LS)"
"@ | Set-Content -LiteralPath $inf -Encoding Ascii
  $request = Join-Path $workspace 'home-gpu-device.csr'
  & $certreq -q -new -machine $inf $request
  if ($LASTEXITCODE -ne 0) { throw 'Windows could not create the Home GPU certificate request.' }
  $pem = [IO.File]::ReadAllText($request).Replace('NEW CERTIFICATE REQUEST', 'CERTIFICATE REQUEST')
  [IO.File]::WriteAllText($request, $pem, [Text.Encoding]::ASCII)
  return
}

$configuration = Get-Content -LiteralPath (Join-Path $workspace 'home-gpu-device.json') -Raw | ConvertFrom-Json
if ($configuration.endpoint -cnotmatch '^[a-z0-9-]+-ats\.iot\.us-east-2\.amazonaws\.com$' -or
    $configuration.thumbprint -cnotmatch '^[A-F0-9]{40}$' -or
    $configuration.scriptSha256 -cnotmatch '^[A-F0-9]{64}$') {
  throw 'Invalid heartbeat device configuration.'
}
if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -cne $configuration.scriptSha256) {
  throw 'The heartbeat script changed after review.'
}
$signedPath = Join-Path $workspace 'home-gpu-device.cer'
$candidate = [Security.Cryptography.X509Certificates.X509Certificate2]::new($signedPath)
if ($candidate.Thumbprint -cne $configuration.thumbprint -or $candidate.Subject -cne $subject) {
  throw 'The issued certificate does not match the Home GPU request.'
}
$certificatePath = "Cert:\LocalMachine\My\$($configuration.thumbprint)"
if (!(Test-Path -LiteralPath $certificatePath)) {
  # The device issuer is recognized by IoT, so import the pinned leaf only.
  & $certutil -addstore My $signedPath
  if ($LASTEXITCODE -ne 0) { throw 'Windows could not import the Home GPU device certificate.' }
}
$installed = Get-Item -LiteralPath $certificatePath
if (!$installed.HasPrivateKey) {
  & $certutil -repairstore My $configuration.thumbprint
  if ($LASTEXITCODE -ne 0) { throw 'Windows could not find the matching Home GPU machine key.' }
  $installed = Get-Item -LiteralPath $certificatePath
}
if (!$installed.HasPrivateKey) { throw 'The issued certificate was not linked to the machine key.' }

$scriptPath = Join-Path $folder 'home-gpu-heartbeat.ps1'
Copy-Item -LiteralPath $source -Destination $scriptPath
$powerShell = Join-Path $env:windir 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File "{0}" -Endpoint {1} -CertificateThumbprint {2}' -f $scriptPath,$configuration.endpoint,$configuration.thumbprint
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments -WorkingDirectory $folder
$trigger = New-HeartbeatTrigger
$principal = New-ScheduledTaskPrincipal -UserId 'NT AUTHORITY\LOCAL SERVICE' -LogonType ServiceAccount -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 1) `
  -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
  -Description 'Reports Home GPU loopback health to AWS every six hours.' | Out-Null
Start-ScheduledTask -TaskName $taskName
