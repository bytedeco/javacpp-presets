param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,
    [Parameter(Mandatory = $true)]
    [string]$VisualStudioInstallationPath
)

$ErrorActionPreference = 'Stop'
$archiveTool = (Get-Command 7z.exe -ErrorAction Stop).Source
$extractionDirectory = Join-Path $env:RUNNER_TEMP 'cuda-13.4.2-arm64'
$toolkitDirectory = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit/CUDA/v13.4'

& $archiveTool x $ArchivePath "-o$extractionDirectory" -y -bb0 -bsp0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
New-Item -ItemType Directory -Force -Path $toolkitDirectory | Out-Null

$components = Get-ChildItem -LiteralPath $extractionDirectory -Directory | Where-Object {
    $_.Name -notlike '*_cross_*' -and
    ($_.Name -eq 'cccl' -or $_.Name -like 'cuda_*' -or $_.Name -like 'lib*' -or $_.Name -eq 'visual_studio_integration')
}
foreach ($component in $components) {
    foreach ($package in Get-ChildItem -LiteralPath $component.FullName -Directory) {
        foreach ($directory in Get-ChildItem -LiteralPath $package.FullName -Directory) {
            $destination = Join-Path $toolkitDirectory $directory.Name
            New-Item -ItemType Directory -Force -Path $destination | Out-Null
            Copy-Item -Path (Join-Path $directory.FullName '*') -Destination $destination -Recurse -Force
        }
    }
}
Copy-Item -LiteralPath "$extractionDirectory/CUDAToolkit/version.json" -Destination $toolkitDirectory -Force
Copy-Item -LiteralPath "$extractionDirectory/EULA.txt", "$extractionDirectory/license.txt" -Destination $toolkitDirectory -Force

$integrationDirectory = Join-Path $toolkitDirectory 'extras/visual_studio_integration/MSBuildExtensions'
$buildCustomizationDirectories = @(Get-ChildItem -LiteralPath "$VisualStudioInstallationPath/MSBuild/Microsoft/VC" -Directory -Recurse -Filter BuildCustomizations)
if ($buildCustomizationDirectories.Count -eq 0) { throw 'Visual Studio BuildCustomizations directory was not found.' }
foreach ($directory in $buildCustomizationDirectories) {
    Copy-Item -Path "$integrationDirectory/*" -Destination $directory.FullName -Force
    Write-Host "CUDA MSBuild integration: $($directory.FullName)"
}

foreach ($relativePath in @(
    'include/cuda.h',
    'include/crt/host_config.h',
    'bin/nvcc.exe',
    'bin/nvcc.profile',
    'bin/arm64/cudart64_13.dll',
    'lib/arm64/cudart.lib',
    'lib/arm64/cublas.lib',
    'lib/arm64/cusparse.lib',
    'lib/cmake/cccl/cccl-config.cmake',
    'nvvm/libdevice/libdevice.10.bc',
    'extras/CUPTI/include/cupti.h',
    'extras/visual_studio_integration/MSBuildExtensions/CUDA 13.4.props'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $toolkitDirectory $relativePath) -PathType Leaf)) {
        throw "CUDA installation is missing $relativePath"
    }
}

"CUDA_PATH=$toolkitDirectory" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"CUDA_PATH_V13_4=$toolkitDirectory" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"$toolkitDirectory/bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
"$toolkitDirectory/bin/arm64" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
$env:CUDA_PATH = $toolkitDirectory
$env:CUDA_PATH_V13_4 = $toolkitDirectory
$env:PATH = "$toolkitDirectory/bin;$toolkitDirectory/bin/arm64;$env:PATH"
& "$toolkitDirectory/bin/nvcc.exe" --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "CUDA Toolkit files installed in $toolkitDirectory"
