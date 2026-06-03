[Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem')

$extractDir = 'c:\Users\shiva\Desktop\ASR\test_audio_batch'
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

Write-Host "Extracting from Audio_sample_lang.zip..."
$zip1 = [System.IO.Compression.ZipFile]::OpenRead('c:\Users\shiva\Desktop\ASR\Audio_sample_lang.zip')
$count = 0
foreach ($entry in $zip1.Entries) {
    if ($entry.FullName -match '\.mp3$' -and $entry.FullName -notmatch '__MACOSX') {
        $destPath = Join-Path $extractDir ('sample_' + $entry.Name)
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
        $count++
        if ($count -ge 4) { break }
    }
}
$zip1.Dispose()

Write-Host "Extracting from final_testing.zip..."
$zip2 = [System.IO.Compression.ZipFile]::OpenRead('c:\Users\shiva\Desktop\ASR\final_testing.zip')
$count = 0
foreach ($entry in $zip2.Entries) {
    if ($entry.FullName -match 'sravan_audio_files/.*\.mp3$' -and $entry.FullName -notmatch '__MACOSX') {
        $destPath = Join-Path $extractDir ('sravan_' + $entry.Name)
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
        $count++
        if ($count -ge 4) { break }
    }
}
$zip2.Dispose()

Write-Host "Extraction complete!"
