#Requires -Version 5.1

<#
.SYNOPSIS
Runs an authenticated Job Tracker CRUD, attachment, and cross-user isolation canary.

.DESCRIPTION
The script is a dry run unless -Execute is supplied. It never creates Cognito
users or changes authentication settings. Supply two existing users' access
tokens through JOB_TRACKER_TOKEN_A and JOB_TRACKER_TOKEN_B. The tokens are not
printed.

The attachment round trip uploads one small text object. Unless -KeepArtifacts
is supplied, the script deletes the application through the API and deletes the
S3 object with the configured AWS CLI profile. The cleanup profile is verified
before the first API mutation.

.EXAMPLE
$env:JOB_TRACKER_TOKEN_A = '<existing-user-a-access-token>'
$env:JOB_TRACKER_TOKEN_B = '<existing-user-b-access-token>'
./scripts/job-tracker-canary.ps1 `
  -BaseUrl 'https://example.execute-api.us-east-2.amazonaws.com/prod' `
  -CleanupAwsProfile website-operator `
  -Execute
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory)]
  [ValidatePattern('^https://')]
  [string]$BaseUrl,

  [string]$PrimaryToken = $env:JOB_TRACKER_TOKEN_A,

  [string]$SecondaryToken = $env:JOB_TRACKER_TOKEN_B,

  [ValidateNotNullOrEmpty()]
  [string]$Origin = 'https://danielshort.me',

  [ValidateNotNullOrEmpty()]
  [string]$AwsRegion = 'us-east-2',

  [ValidateNotNullOrEmpty()]
  [string]$CleanupAwsProfile = 'website-operator',

  [switch]$Execute,

  [switch]$KeepArtifacts
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$apiBase = $BaseUrl.TrimEnd('/')
$runId = 'CANARY-{0}-{1}' -f (
  [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'),
  [Guid]::NewGuid().ToString('N').Substring(0, 8)
)
$today = [DateTime]::UtcNow.ToString('yyyy-MM-dd')
function Write-RequestPlan {
  Write-Host 'Job Tracker canary request plan (no requests sent):'
  Write-Host '  1. POST   /api/applications'
  Write-Host '     body: { company, title, appliedDate, status, notes, tags[], customFields{} }'
  Write-Host '     expects: 201 application object with applicationId'
  Write-Host '  2. GET    /api/applications?limit=100'
  Write-Host '     expects: 200 { items: application[] } containing the new applicationId'
  Write-Host '  3. PATCH  /api/applications/{encoded applicationId}'
  Write-Host '     body: { status, statusDate, notes, tags[] }'
  Write-Host '     expects: 200 updated application object'
  Write-Host '  4. POST   /api/attachments/presign'
  Write-Host '     body: { applicationId, filename, contentType, size }'
  Write-Host '     expects: 200 { uploadUrl, key, bucket, expiresIn, maxBytes, maxCount }'
  Write-Host '  5. PUT    {uploadUrl} with the exact Content-Type and byte count'
  Write-Host '  6. PATCH  /api/applications/{encoded applicationId}'
  Write-Host '     body: { attachments: [{ key, filename, contentType, kind, size, uploadedAt }] }'
  Write-Host '  7. POST   /api/attachments/download'
  Write-Host '     body: { key }'
  Write-Host '     expects: 200 { downloadUrl, key, expiresIn }, followed by a successful GET'
  Write-Host '  8. User B isolation: list excludes User A item; cross-user attachment download is 403;'
  Write-Host '     cross-user PATCH and DELETE are indistinguishable from missing rows (404) and cannot alter User A.'
  Write-Host '  9. DELETE /api/applications/{encoded applicationId}, then verify it is absent.'
  Write-Host "Run id would be: $runId"
  Write-Host 'Add -Execute only after setting JOB_TRACKER_TOKEN_A and JOB_TRACKER_TOKEN_B.'
}

function Convert-ResponseJson {
  param([AllowNull()][string]$Content)

  if ([string]::IsNullOrWhiteSpace($Content)) {
    return $null
  }
  try {
    return $Content | ConvertFrom-Json
  } catch {
    return $Content
  }
}

function Invoke-JobTrackerRequest {
  param(
    [Parameter(Mandatory)]
    [ValidateSet('GET', 'POST', 'PATCH', 'DELETE')]
    [string]$Method,

    [Parameter(Mandatory)]
    [string]$Path,

    [Parameter(Mandatory)]
    [string]$Token,

    [AllowNull()]
    [object]$Body,

    [int[]]$ExpectedStatus = @()
  )

  $request = @{
    Uri = "$apiBase$Path"
    Method = $Method
    Headers = @{
      Accept = 'application/json'
      Authorization = "Bearer $Token"
      Origin = $Origin
    }
  }
  if ($null -ne $Body) {
    $request.ContentType = 'application/json'
    $request.Body = $Body | ConvertTo-Json -Depth 12 -Compress
  }

  $content = ''
  try {
    $response = Invoke-WebRequest @request -UseBasicParsing
    $status = [int]$response.StatusCode
    $content = [string]$response.Content
  } catch {
    $errorResponse = $_.Exception.Response
    if ($null -eq $errorResponse) {
      throw
    }
    $status = [int]$errorResponse.StatusCode
    $content = [string]$_.ErrorDetails.Message
    if ([string]::IsNullOrWhiteSpace($content)) {
      try {
        $stream = $errorResponse.GetResponseStream()
        if ($null -ne $stream) {
          $reader = [IO.StreamReader]::new($stream)
          $content = $reader.ReadToEnd()
          $reader.Dispose()
        }
      } catch {
        $content = ''
      }
    }
  }
  $parsed = Convert-ResponseJson -Content $content
  if ($ExpectedStatus.Count -and $status -notin $ExpectedStatus) {
    $detail = if ($parsed -is [string]) {
      $parsed
    } elseif ($null -ne $parsed) {
      $parsed | ConvertTo-Json -Depth 6 -Compress
    } else {
      '<empty response>'
    }
    throw "$Method $Path returned $status; expected $($ExpectedStatus -join '/'). Body: $detail"
  }

  return [pscustomobject]@{
    Status = $status
    Json = $parsed
  }
}

function Get-ApplicationItems {
  param([Parameter(Mandatory)][string]$Token)

  $response = Invoke-JobTrackerRequest `
    -Method GET `
    -Path '/api/applications?limit=100' `
    -Token $Token `
    -ExpectedStatus 200
  return @($response.Json.items)
}

function Test-ContainsApplication {
  param(
    [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Items,
    [Parameter(Mandatory)][string]$ApplicationId
  )

  return @($Items | Where-Object { $_.applicationId -eq $ApplicationId }).Count -gt 0
}

if (-not $Execute) {
  Write-RequestPlan
  return
}

if ([string]::IsNullOrWhiteSpace($PrimaryToken) -or [string]::IsNullOrWhiteSpace($SecondaryToken)) {
  throw 'Set JOB_TRACKER_TOKEN_A and JOB_TRACKER_TOKEN_B to existing, distinct users before using -Execute.'
}
if ($PrimaryToken -eq $SecondaryToken) {
  throw 'JOB_TRACKER_TOKEN_A and JOB_TRACKER_TOKEN_B must identify different users.'
}

if (-not $KeepArtifacts) {
  $aws = Get-Command aws -ErrorAction Stop
  & $aws.Source sts get-caller-identity `
    --profile $CleanupAwsProfile `
    --region $AwsRegion `
    --output json | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "AWS cleanup profile '$CleanupAwsProfile' is unavailable. No API mutations were attempted."
  }
}

$applicationId = $null
$attachmentBucket = $null
$attachmentKey = $null
$primaryDeleted = $false
$attachmentDeleted = $false
$failures = [System.Collections.Generic.List[string]]::new()
$runError = $null

try {
  Write-Host "Creating $runId"
  $create = Invoke-JobTrackerRequest `
    -Method POST `
    -Path '/api/applications' `
    -Token $PrimaryToken `
    -Body @{
      company = "$runId Company"
      title = 'Job Tracker Canary'
      appliedDate = $today
      status = 'Applied'
      notes = "$runId create"
      tags = @('canary', $runId)
      customFields = @{ canaryRun = $runId }
    } `
    -ExpectedStatus 201
  $applicationId = [string]$create.Json.applicationId
  if ([string]::IsNullOrWhiteSpace($applicationId)) {
    throw 'Create response did not include applicationId.'
  }
  $encodedApplicationId = [Uri]::EscapeDataString($applicationId)

  $primaryItems = @(Get-ApplicationItems -Token $PrimaryToken)
  if (-not (Test-ContainsApplication -Items $primaryItems -ApplicationId $applicationId)) {
    throw 'Primary list did not contain the newly created application.'
  }

  $update = Invoke-JobTrackerRequest `
    -Method PATCH `
    -Path "/api/applications/$encodedApplicationId" `
    -Token $PrimaryToken `
    -Body @{
      status = 'Interview'
      statusDate = $today
      notes = "$runId updated"
      tags = @('canary', 'updated', $runId)
    } `
    -ExpectedStatus 200
  if ($update.Json.applicationId -ne $applicationId -or $update.Json.status -ne 'Interview') {
    throw 'Update response did not preserve applicationId and status=Interview.'
  }

  $attachmentText = "$runId attachment"
  $attachmentBytes = [Text.Encoding]::UTF8.GetBytes($attachmentText)
  $attachmentFilename = "$runId.txt"
  $presign = Invoke-JobTrackerRequest `
    -Method POST `
    -Path '/api/attachments/presign' `
    -Token $PrimaryToken `
    -Body @{
      applicationId = $applicationId
      filename = $attachmentFilename
      contentType = 'text/plain'
      size = $attachmentBytes.Length
    } `
    -ExpectedStatus 200
  $attachmentBucket = [string]$presign.Json.bucket
  $attachmentKey = [string]$presign.Json.key
  if ([string]::IsNullOrWhiteSpace($presign.Json.uploadUrl) -or
      [string]::IsNullOrWhiteSpace($attachmentBucket) -or
      [string]::IsNullOrWhiteSpace($attachmentKey)) {
    throw 'Attachment presign response is missing uploadUrl, bucket, or key.'
  }

  $upload = Invoke-WebRequest `
    -Uri $presign.Json.uploadUrl `
    -Method PUT `
    -ContentType 'text/plain' `
    -Body $attachmentBytes `
    -UseBasicParsing
  if ([int]$upload.StatusCode -notin @(200, 204)) {
    throw "Attachment upload returned $([int]$upload.StatusCode); expected 200/204."
  }

  $attach = Invoke-JobTrackerRequest `
    -Method PATCH `
    -Path "/api/applications/$encodedApplicationId" `
    -Token $PrimaryToken `
    -Body @{
      attachments = @(
        @{
          key = $attachmentKey
          filename = $attachmentFilename
          contentType = 'text/plain'
          kind = 'canary'
          size = $attachmentBytes.Length
          uploadedAt = [DateTime]::UtcNow.ToString('o')
        }
      )
    } `
    -ExpectedStatus 200
  if (@($attach.Json.attachments).Count -ne 1 -or $attach.Json.attachments[0].key -ne $attachmentKey) {
    throw 'Application update did not retain the uploaded attachment metadata.'
  }

  $download = Invoke-JobTrackerRequest `
    -Method POST `
    -Path '/api/attachments/download' `
    -Token $PrimaryToken `
    -Body @{ key = $attachmentKey } `
    -ExpectedStatus 200
  $downloaded = Invoke-WebRequest `
    -Uri $download.Json.downloadUrl `
    -Method GET `
    -UseBasicParsing
  if ([int]$downloaded.StatusCode -ne 200 -or $downloaded.Content -ne $attachmentText) {
    throw 'Attachment download did not return the exact uploaded canary content.'
  }

  $secondaryItems = @(Get-ApplicationItems -Token $SecondaryToken)
  if (Test-ContainsApplication -Items $secondaryItems -ApplicationId $applicationId) {
    $failures.Add('Secondary user list exposed the primary user application.')
  }

  $crossDownload = Invoke-JobTrackerRequest `
    -Method POST `
    -Path '/api/attachments/download' `
    -Token $SecondaryToken `
    -Body @{ key = $attachmentKey }
  if ($crossDownload.Status -ne 403) {
    $failures.Add("Cross-user attachment download returned $($crossDownload.Status); expected 403.")
  }

  $crossPatch = Invoke-JobTrackerRequest `
    -Method PATCH `
    -Path "/api/applications/$encodedApplicationId" `
    -Token $SecondaryToken `
    -Body @{ notes = "$runId cross-user probe" }
  if ($crossPatch.Status -ne 404) {
    $failures.Add("Cross-user PATCH returned $($crossPatch.Status); expected 404. An older handler may have created a secondary-user shadow record.")
    Invoke-JobTrackerRequest `
      -Method DELETE `
      -Path "/api/applications/$encodedApplicationId" `
      -Token $SecondaryToken `
      -ExpectedStatus 200 | Out-Null
  }

  $crossDelete = Invoke-JobTrackerRequest `
    -Method DELETE `
    -Path "/api/applications/$encodedApplicationId" `
    -Token $SecondaryToken
  if ($crossDelete.Status -ne 404) {
    $failures.Add("Cross-user DELETE returned $($crossDelete.Status); expected 404.")
  }
  $primaryAfterCrossUser = @(Get-ApplicationItems -Token $PrimaryToken)
  if (-not (Test-ContainsApplication -Items $primaryAfterCrossUser -ApplicationId $applicationId)) {
    $failures.Add('Cross-user DELETE removed the primary user application.')
  }

  if (-not $KeepArtifacts) {
    Invoke-JobTrackerRequest `
      -Method DELETE `
      -Path "/api/applications/$encodedApplicationId" `
      -Token $PrimaryToken `
      -ExpectedStatus 200 | Out-Null
    $primaryDeleted = $true
    $afterDelete = @(Get-ApplicationItems -Token $PrimaryToken)
    if (Test-ContainsApplication -Items $afterDelete -ApplicationId $applicationId) {
      $failures.Add('Primary DELETE returned success but the application remained in the list.')
    }
  }
} catch {
  $runError = $_
} finally {
  if (-not $KeepArtifacts -and $applicationId -and -not $primaryDeleted) {
    try {
      $encodedApplicationId = [Uri]::EscapeDataString($applicationId)
      Invoke-JobTrackerRequest `
        -Method DELETE `
        -Path "/api/applications/$encodedApplicationId" `
        -Token $PrimaryToken `
        -ExpectedStatus 200 | Out-Null
      $primaryDeleted = $true
    } catch {
      $failures.Add("Application cleanup failed: $($_.Exception.Message)")
    }
  }

  if (-not $KeepArtifacts -and $attachmentBucket -and $attachmentKey) {
    try {
      $aws = Get-Command aws -ErrorAction Stop
      & $aws.Source s3api delete-object `
        --bucket $attachmentBucket `
        --key $attachmentKey `
        --region $AwsRegion `
        --profile $CleanupAwsProfile `
        --output json | Out-Null
      if ($LASTEXITCODE -ne 0) {
        throw "aws s3api delete-object exited with $LASTEXITCODE."
      }
      $attachmentDeleted = $true
    } catch {
      $failures.Add("Attachment cleanup failed: $($_.Exception.Message)")
    }
  }
}

if ($null -ne $runError) {
  throw $runError
}
if ($failures.Count) {
  throw "Job Tracker canary failed:`n - $($failures -join "`n - ")"
}

Write-Host "Job Tracker canary passed: $runId"
Write-Host "Application cleanup: $primaryDeleted"
Write-Host "Attachment cleanup: $attachmentDeleted"
