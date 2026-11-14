[CmdletBinding()]
param(
  [Parameter(Mandatory)]
  [ValidatePattern('^[a-z0-9-]+-ats\.iot\.us-east-2\.amazonaws\.com$')]
  [string] $Endpoint,

  [Parameter(Mandatory)]
  [ValidatePattern('^[a-fA-F0-9]{40}$')]
  [string] $CertificateThumbprint
)

$ErrorActionPreference = 'Stop'
$progressPreference = 'SilentlyContinue'
$certificate = Get-Item -LiteralPath "Cert:\LocalMachine\My\$CertificateThumbprint"
if (!$certificate.HasPrivateKey) { throw 'The Home GPU device certificate is missing its private key.' }

$state = 'unavailable'
try {
  $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/v1/health' -TimeoutSec 4 -MaximumRedirection 0
  $state = 'unknown'
  if ($health.service -ceq 'Home GPU worker' -and $health.ready -is [bool]) {
    $state = 'unavailable'
    if ($health.ok -eq $true -and $health.ready -and $health.status -notin @('offline', 'unavailable', 'error', 'failed')) {
      $state = 'ready'
    }
  }
} catch [System.Net.WebException] {
  # The worker can be closed while Windows is still able to report its own health.
} catch {
  $state = 'unknown'
}

$uri = 'https://' + $Endpoint + ':8443/topics/$aws/rules/website_home_gpu_heartbeat/health'
$body = @{ state = $state } | ConvertTo-Json -Compress
$response = Invoke-WebRequest -UseBasicParsing -Uri $uri -Method Post `
  -Certificate $certificate -ContentType 'application/json' -Body $body -TimeoutSec 8 -MaximumRedirection 0
if ($response.StatusCode -ne 200) { throw 'AWS IoT did not acknowledge the Home GPU heartbeat.' }
