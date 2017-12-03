@echo on
set PROJ=%~1
cd %APPVEYOR_BUILD_FOLDER%

REM Create a writeable TMPDIR
mkdir %APPVEYOR_BUILD_FOLDER%\tmp
set TMPDIR=%APPVEYOR_BUILD_FOLDER%\tmp
mkdir %APPVEYOR_BUILD_FOLDER%\buildlogs

IF "%MSYS2_ARCH%"=="x86_64" (
   echo Callings vcvarsall for amd64
   call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
)
IF "%MSYS2_ARCH%"=="x86" (
   echo Callings vcvarsall for x86
   call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86
)

C:\%MSYS2_DIR%\usr\bin\bash -lc "pacman -Syu --noconfirm"
C:\%MSYS2_DIR%\usr\bin\bash -lc "pacman -Su --noconfirm"
C:\%MSYS2_DIR%\usr\bin\bash -lc "pacman -S --needed --noconfirm base-devel git tar nasm yasm pkg-config unzip autoconf automake libtool make patch"
C:\%MSYS2_DIR%\usr\bin\bash -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-toolchain mingw-w64-x86_64-libtool mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc mingw-w64-i686-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-i686-gcc-fortran mingw-w64-x86_64-libwinpthread-git mingw-w64-i686-libwinpthread-git"

C:\%MSYS2_DIR%\usr\bin\bash -lc "/c/projects/javacpp-presets/ci/install-windows.sh %PROJ%"
SET PATH=C:\%MSYS2_DIR%\usr\bin\core_perl;C:\%MSYS2_DIR%\%MSYSTEM%\bin;C:\%MSYS2_DIR%\usr\bin;%PATH%
echo %PATH%

echo Building for "%APPVEYOR_REPO_BRANCH%"
echo PR Number "%APPVEYOR_PULL_REQUEST_NUMBER%"
IF "%APPVEYOR_PULL_REQUEST_NUMBER%"=="" (
   echo Deploy snaphot for %PROJ%
   call mvn clean deploy -U -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=windows-%MSYS2_ARCH% -Djavacpp.copyResources --settings .\ci\settings.xml -pl .,%PROJ%
   IF ERRORLEVEL 1 (
     echo Quitting with error  
     exit /b 1
   )
   FOR %%a in ("%PROJ:,=" "%") do (
    echo Deploy platform %%a 
    cd %%a
    call mvn clean -U -f platform -Djavacpp.platform=windows-%MSYS2_ARCH% --settings ..\ci\settings.xml deploy
    IF ERRORLEVEL 1 (
      echo Quitting with error  
      exit /b 1
    )

    cd ..
   )
) ELSE (
   echo Install %PROJ%
   call mvn clean install -U -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=windows-%MSYS2_ARCH% -Djavacpp.copyResources -pl .,%PROJ%
   IF ERRORLEVEL 1 (
      echo Quitting with error  
      exit /b 1 
   )

)

