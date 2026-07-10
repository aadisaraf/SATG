<#
.SYNOPSIS
    Download + extract GTA5 "Playing for Data" dataset on Windows.
.DESCRIPTION
    Downloads all 10 parts of images (~57GB) and labels (~700MB) from
    TU Darmstadt, then extracts into:
      <output_dir>/images/   (24,966 PNGs)
      <output_dir>/labels/   (24,966 single-channel PNGs, labels 0-34)

    Resume-safe: counts existing images to determine which part to start from.
    Retry-safe: curl.exe --retry 5 + inner retry loop handles flaky connections.
    Sleep-safe: uses Windows API to prevent sleep while running.

    REQUIREMENTS:
      - Windows 10 or later (has curl.exe and tar.exe built-in)
      - PowerShell 5.0+ (should be default)

    Usage (Windows):
      # Open PowerShell as administrator (recommended for best results)
      # Or run as normal user — sleep prevention works without admin

      # Start the download:
      .\cloud\download_gta5.ps1

      # Or with a custom path:
      .\cloud\download_gta5.ps1 -OutputDir D:\data\GTA5

      # Monitor progress (in a separate PowerShell window):
      Get-Content .\gta5_download.log -Tail 10 -Wait

      # Quick count check:
      (Get-ChildItem .\data\GTA5\images\*.png).Count
      (Get-ChildItem .\data\GTA5\labels\*.png).Count

      # If it stalls, just re-run — it resumes where it left off.

    OUTPUT:
data/GTA5/images/  — 24,966 RGB PNGs (1914×1052)
data/GTA5/labels/  — 24,966 indexed PNGs (label IDs 0-34)
# Note: data/GTA5 is relative to the project root (SATG/data/GTA5)
#>

param(
    [string]$OutputDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "data\GTA5")
)

$BASE_URL = "https://download.visinf.tu-darmstadt.de/data/from_games/data"
$TOTAL_IMAGES = 24966
$IMAGES_PER_PART = [Math]::Floor($TOTAL_IMAGES / 10)  # ~2497

# --- Prevent sleep via Windows API (no admin required) ---
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class WakeLock {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const uint ES_AWAKE = 0x00000002;
    public const uint ES_DISPLAY_REQUIRED = 0x00000004;
}
"@
try {
    [WakeLock]::SetThreadExecutionState(0x80000003)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    Write-Host "Sleep prevention activated."
} catch {
    Write-Warning "Could not set wake lock. Download may pause if computer sleeps."
}

# Create output directories
New-Item -ItemType Directory -Force -Path $OutputDir, "$OutputDir\images", "$OutputDir\labels" | Out-Null

Write-Host "=== GTA5 Download Script (Windows) ==="
Write-Host "Output: $OutputDir"

# Count existing files (resume support)
$existingImages = @(Get-ChildItem "$OutputDir\images\*.png" -ErrorAction SilentlyContinue).Count
$existingLabels  = @(Get-ChildItem "$OutputDir\labels\*.png" -ErrorAction SilentlyContinue).Count
Write-Host "Existing: $existingImages images / $existingLabels labels"

# Calculate starting part
$startPart = [Math]::Floor($existingImages / $IMAGES_PER_PART) + 1
if ($startPart -gt 10) { $startPart = 10 }
if ($startPart -lt 1)  { $startPart = 1 }

Write-Host "Starting from part $(('{0:d2}' -f $startPart))"
Write-Host ""

for ($p = $startPart; $p -le 10; $p++) {
    $partStr = '{0:d2}' -f $p
    $imgZip = "${partStr}_images.zip"
    $labZip = "${partStr}_labels.zip"
    $imgZipPath = Join-Path $OutputDir $imgZip
    $labZipPath = Join-Path $OutputDir $labZip

    Write-Host "--- Part $partStr ---"

    # Download images zip with retries
    $downloadedImg = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "  Downloading $imgZip (~5.7GB) ..."
        # curl.exe is the real curl on Windows 10+, NOT the Invoke-WebRequest alias
        $result = & curl.exe -L -C - --retry 5 --retry-delay 10 -o "$imgZipPath" "$BASE_URL/$imgZip" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $downloadedImg = $true
            break
        }
        Write-Host "  Attempt $attempt failed, retrying in 30s ..."
        if (Test-Path $imgZipPath) { Remove-Item $imgZipPath -Force }
        Start-Sleep -Seconds 30
    }

    if (-not $downloadedImg) {
        Write-Host "  FATAL: $imgZip failed after 3 attempts."
        if (Test-Path $imgZipPath) { Remove-Item $imgZipPath -Force }
        [WakeLock]::SetThreadExecutionState(0x80000000)  # restore
        exit 1
    }

    # Verify zip integrity using .NET (reliable on all Windows versions)
    Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
    try {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($imgZipPath)
        $archive.Dispose()
    } catch {
        Write-Host "  FATAL: $imgZip is corrupt."
        Remove-Item $imgZipPath -Force
        [WakeLock]::SetThreadExecutionState(0x80000000)
        exit 1
    }

    # Download labels zip with retries
    $downloadedLab = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "  Downloading $labZip ..."
        $result = & curl.exe -L -C - --retry 5 --retry-delay 10 -o "$labZipPath" "$BASE_URL/$labZip" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $downloadedLab = $true
            break
        }
        Write-Host "  Attempt $attempt failed, retrying in 30s ..."
        if (Test-Path $labZipPath) { Remove-Item $labZipPath -Force }
        Start-Sleep -Seconds 30
    }

    if (-not $downloadedLab) {
        Write-Host "  FATAL: $labZip failed after 3 attempts."
        if (Test-Path $labZipPath) { Remove-Item $labZipPath -Force }
        [WakeLock]::SetThreadExecutionState(0x80000000)
        exit 1
    }

    # Verify labels zip
    try {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($labZipPath)
        $archive.Dispose()
    } catch {
        Write-Host "  FATAL: $labZip is corrupt."
        Remove-Item $labZipPath -Force
        [WakeLock]::SetThreadExecutionState(0x80000000)
        exit 1
    }

    # Extract to temp directory
    $tmpDir = Join-Path $OutputDir "tmp_$partStr"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

    Write-Host "  Extracting $imgZip ..."
    Expand-Archive -Path "$imgZipPath" -DestinationPath "$tmpDir" -Force

    Write-Host "  Extracting $labZip ..."
    Expand-Archive -Path "$labZipPath" -DestinationPath "$tmpDir" -Force

    # Move PNGs into flat directories
    $imgSource = Join-Path $tmpDir "images"
    $labSource = Join-Path $tmpDir "labels"
    if (Test-Path $imgSource) {
        Move-Item "$imgSource\*.png" "$OutputDir\images\" -Force
    }
    if (Test-Path $labSource) {
        Move-Item "$labSource\*.png" "$OutputDir\labels\" -Force
    }

    # Cleanup
    Remove-Item $tmpDir -Recurse -Force
    Remove-Item $imgZipPath -Force
    Remove-Item $labZipPath -Force

    $currentImages = @(Get-ChildItem "$OutputDir\images\*.png" -ErrorAction SilentlyContinue).Count
    Write-Host "  Done part $partStr  (total images: $currentImages / $TOTAL_IMAGES)"
    Write-Host ""
}

# Restore normal sleep behavior
[WakeLock]::SetThreadExecutionState(0x80000000)

# Final verification
Write-Host "=== Verifying ==="
$finalImages = @(Get-ChildItem "$OutputDir\images\*.png").Count
$finalLabels  = @(Get-ChildItem "$OutputDir\labels\*.png").Count
Write-Host "Images: $finalImages   (expected: $TOTAL_IMAGES)"
Write-Host "Labels:  $finalLabels   (expected: $TOTAL_IMAGES)"

if ($finalImages -eq $TOTAL_IMAGES -and $finalLabels -eq $TOTAL_IMAGES) {
    Write-Host ""
    Write-Host "=== Download complete! ==="
    Write-Host ""
    Write-Host "Next step:"
    Write-Host "  python -m precompute.preprocess_gta5_labels --label_root $OutputDir/labels"
} else {
    Write-Host ""
    Write-Host "  WARNING: File counts don't match. Expected $TOTAL_IMAGES, got $finalImages / $finalLabels."
    exit 1
}
