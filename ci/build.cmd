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

SET "PATH=C:\%MSYS2_DIR%\%MSYSTEM%\bin;C:\%MSYS2_DIR%\usr\bin;%PATH%"
bash -lc "pacman -S --needed --noconfirm pacman-mirrors"
bash -lc "pacman -Sy --noconfirm pacman"
bash -lc "pacman -Syu --noconfirm"
bash -lc "pacman -S --needed --noconfirm git"
bash -lc "pacman -S --needed --noconfirm base-devel tar nasm yasm pkg-config unzip autoconf automake libtool make patch"
bash -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-toolchain mingw-w64-x86_64-libtool mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc mingw-w64-i686-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-i686-gcc-fortran mingw-w64-x86_64-libwinpthread-git mingw-w64-i686-libwinpthread-git"

bash -lc "/c/projects/javacpp-presets/ci/install-windows.sh %PROJ%"


echo Starting main build now.. 
REM cd ..
REM cd javacpp 
REM echo Install javacpp
REM call mvn install -Djavacpp.copyResources -Dmaven.test.skip=true -Djavacpp.platform=windows-%MSYS2_ARCH% -Dmaven.javadoc.skip=true
REM cd ..
REM cd javacpp-presets
echo Building for "%APPVEYOR_REPO_BRANCH%"
echo PR Number "%APPVEYOR_PULL_REQUEST_NUMBER%"
echo XXXXXXX WARNING NOT CREATING CORRECT ERROR RETURN YET, USES STATUS OF LAST COMMAND..
IF "%APPVEYOR_PULL_REQUEST_NUMBER%"=="" (
   echo Deploy snaphot for %PROJ%
   call mvn deploy -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=windows-%MSYS2_ARCH% --settings .\ci\settings.xml -pl .,%PROJ%
   IF errorlevel 1 (
     exit /b %errorlevel%
   )
   FOR %%a in ("%PROJ:,=" "%") do (
    echo Deploy platform %%a 
    cd %%a
    call mvn -f platform -Djavacpp.platform=windows-%MSYS2_ARCH% --settings .\ci\settings.xml deploy
    IF errorlevel 1 (
      exit /b %errorlevel%
    )

    cd ..
   )
) ELSE (
   echo Install %PROJ%
   call mvn install -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=windows-%MSYS2_ARCH% -pl .,%PROJ%
   IF errorlevel 1 (
      exit /b %errorlevel%
   )

)

