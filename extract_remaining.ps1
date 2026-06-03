[Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem') | Out-Null

$extractDir = 'c:\Users\shiva\Desktop\ASR\test_audio_batch_remote'
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

$existingFiles = @(
    "Shorts_call-prank-fake-police-off",
    "Shorts_Thalapathy-vijay-voice-pho",
    "Virat Kohli And Amir Khan Educati",
    "Walking Phone Call Prank dY",
    "normal_voicecall3",
    "normal_voicecall4",
    "office_colleagues_voicecall_recor",
    "uday.new",
    "Apdinu soneengale",
    "baba ko voice",
    "call galagali",
    "viral-trending",
    "Walking Phone Call Prank",
    "arshdeep",
    "normal_voicecall1",
    "office_collagues_call_recording",
    "teams-recording_new"
)

Write-Host "Extracting remaining from Audio_sample_lang.zip..."
$zip1 = [System.IO.Compression.ZipFile]::OpenRead('c:\Users\shiva\Desktop\ASR\Audio_sample_lang.zip')
$count1 = 0
foreach ($entry in $zip1.Entries) {
    if ($entry.FullName -match '\.mp3$' -and $entry.FullName -notmatch '__MACOSX') {
        $skip = $false
        foreach ($ex in $existingFiles) {
            if ($entry.Name -match $ex) { $skip = $true; break }
        }
        if (-not $skip) {
            $destPath = Join-Path $extractDir ('sample_' + $entry.Name)
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
            $count1++
        }
    }
}
$zip1.Dispose()

Write-Host "Extracting remaining from final_testing.zip..."
$zip2 = [System.IO.Compression.ZipFile]::OpenRead('c:\Users\shiva\Desktop\ASR\final_testing.zip')
$count2 = 0
foreach ($entry in $zip2.Entries) {
    if ($entry.FullName -match 'sravan_audio_files/.*\.mp3$' -and $entry.FullName -notmatch '__MACOSX') {
        $skip = $false
        foreach ($ex in $existingFiles) {
            if ($entry.Name -match $ex) { $skip = $true; break }
        }
        if (-not $skip) {
            $destPath = Join-Path $extractDir ('sravan_' + $entry.Name)
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
            $count2++
        }
    }
}
$zip2.Dispose()

Write-Host "Extraction complete! Found $count1 + $count2 files."
