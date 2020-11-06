@echo on
set PROJ=%~1
cd %APPVEYOR_BUILD_FOLDER%

REM Create a writeable TMPDIR
mkdir %APPVEYOR_BUILD_FOLDER%\tmp
set TMPDIR=%APPVEYOR_BUILD_FOLDER%\tmp
mkdir %APPVEYOR_BUILD_FOLDER%\buildlogs

echo %NUMBER_OF_PROCESSORS%
set MAKEJ=%NUMBER_OF_PROCESSORS%

IF "%OS%"=="windows-x86_64" (
   set MSYSTEM=MINGW64
   echo Callings vcvarsall for amd64
   call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
   call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
)
IF "%OS%"=="windows-x86" (
   set MSYSTEM=MINGW32
   echo Callings vcvarsall for x86
   call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x86
   call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x86
)
echo on

if "%APPVEYOR_PULL_REQUEST_NUMBER%" == "" if "%APPVEYOR_REPO_BRANCH%" == "release" (
    set "MAVEN_RELEASE=-DperformRelease -DstagingRepositoryId=%STAGING_REPOSITORY%"
) else (
    set "MAVEN_RELEASE=-Dmaven.javadoc.skip=true"
)

rem C:\msys64\usr\bin\bash -lc "pacman -Syu --noconfirm"
rem C:\msys64\usr\bin\bash -lc "pacman -Su --noconfirm"
C:\msys64\usr\bin\bash -lc "pacman -S --needed --noconfirm base-devel git tar pkg-config unzip p7zip zip autoconf autoconf-archive automake make patch gnupg"
C:\msys64\usr\bin\bash -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-nasm mingw-w64-x86_64-toolchain mingw-w64-x86_64-gcc mingw-w64-i686-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-i686-gcc-fortran mingw-w64-x86_64-libwinpthread-git mingw-w64-i686-libwinpthread-git mingw-w64-x86_64-SDL mingw-w64-i686-SDL mingw-w64-x86_64-ragel"

SET "PATH=C:\Program Files\Python37;C:\Program Files (x86)\CMake\bin;C:\msys64\usr\bin\core_perl;C:\msys64\%MSYSTEM%\bin;C:\msys64\usr\bin;%PATH%"
C:\msys64\usr\bin\bash -c "ci/install-windows.sh %PROJ%"
if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit" (
    SET "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v11.1"
    SET "CUDA_PATH_V11_1=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v11.1"
    SET "PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v11.1\libnvvp;%PATH%"
    echo CUDA Version 11.1.182>"%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v11.1\version.txt"
)

echo Building for "%APPVEYOR_REPO_BRANCH%"
echo PR Number "%APPVEYOR_PULL_REQUEST_NUMBER%"

IF "%PARTIAL_CPPBUILD%"=="1" (
   C:\msys64\usr\bin\bash -c "bash cppbuild.sh install %PROJ% -platform=%OS% -extension=%EXT%"
   C:\msys64\usr\bin\bash -c "zip -r %PROJ%-cppbuild.zip %PROJ%/cppbuild"
   IF ERRORLEVEL 1 (
     echo Quitting with error  
     exit 1
   )
   TASKKILL /F /IM MSBuild.exe /T
   echo Exiting with success
   exit 0
)

IF "%APPVEYOR_PULL_REQUEST_NUMBER%"=="" (
   echo Deploy snaphot for %PROJ%
   call mvn deploy -B -U --settings .\ci\settings.xml -Dmaven.test.skip=true %MAVEN_RELEASE% -Djavacpp.platform=%OS% -Djavacpp.platform.extension=%EXT% -pl .,%PROJ%
   IF ERRORLEVEL 1 (
     echo Quitting with error  
     exit 1
   )
   FOR %%a in ("%PROJ:,=" "%") do (
    echo Deploy platform %%a 
    cd %%a
    if "%EXT%" == "" (set EXT2=) else (set EXT2=%EXT:~1%)
    call mvn deploy -B -U --settings ..\ci\settings.xml -f platform\%EXT2%\pom.xml -Dmaven.test.skip=true %MAVEN_RELEASE% -Djavacpp.platform=%OS% -Djavacpp.platform.extension=%EXT%
    IF ERRORLEVEL 1 (
      echo Quitting with error  
      exit 1
    )

    cd ..
   )
) ELSE (
   echo Install %PROJ%
   call mvn install -B -U --settings .\ci\settings.xml -Dmaven.test.skip=true %MAVEN_RELEASE% -Djavacpp.platform=%OS% -Djavacpp.platform.extension=%EXT% -pl .,%PROJ%
   IF ERRORLEVEL 1 (
      echo Quitting with error  
      exit 1
   )

)
TASKKILL /F /IM MSBuild.exe /T
echo Exiting with success
exit 0

