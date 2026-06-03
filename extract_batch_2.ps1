[Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem')

$extractDir = 'c:\Users\shiva\Desktop\ASR\test_audio_batch_2'
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

$existingFiles = @(
    "Shorts_call-prank-fake-police-off",
    "Shorts_Thalapathy-vijay-voice-pho",
    "Virat Kohli And Amir Khan Educati",
    "Walking Phone Call Prank dY",
    "normal_voicecall3",
    "normal_voicecall4",
    "office_colleagues_voicecall_recor",
    "uday.new"
)

Write-Host "Extracting from Audio_sample_lang.zip..."
$zip1 = [System.IO.Compression.ZipFile]::OpenRead('c:\Users\shiva\Desktop\ASR\Audio_sample_lang.zip')
$count = 0
foreach ($entry in $zip1.Entries) {
    if ($entry.FullName -match '\.mp3$' -and $entry.FullName -notmatch '__MACOSX') {
        $skip = $false
        foreach ($ex in $existingFiles) {
            if ($entry.Name -match $ex) { $skip = $true; break }
        }
        if (-not $skip) {
            $destPath = Join-Path $extractDir ('sample_' + $entry.Name)
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
            $count++
            if ($count -ge 5) { break }
        }
    }
}
$zip1.Dispose()

Write-Host "Extracting from final_testing.zip..."
$zip2 = [System.IO.Compression.ZipFile]::OpenRead('c:\Users\shiva\Desktop\ASR\final_testing.zip')
$count = 0
foreach ($entry in $zip2.Entries) {
    if ($entry.FullName -match 'sravan_audio_files/.*\.mp3$' -and $entry.FullName -notmatch '__MACOSX') {
        $skip = $false
        foreach ($ex in $existingFiles) {
            if ($entry.Name -match $ex) { $skip = $true; break }
        }
        if (-not $skip) {
            $destPath = Join-Path $extractDir ('sravan_' + $entry.Name)
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
            $count++
            if ($count -ge 5) { break }
        }
    }
}
$zip2.Dispose()

Write-Host "Extraction complete!"
