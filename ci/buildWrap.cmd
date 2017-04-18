@echo off
set PROJ=%~1
set DROPAUTH=%~2
set CI_DEPLOY_USERNAME=%~3
set CI_DEPLOY_PASSWORD=%~4
cd %APPVEYOR_BUILD_FOLDER%

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
bash -lc "pacman -S --needed --noconfirm git"
bash -lc "pacman -Syu --noconfirm"
bash -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-toolchain base-devel tar nasm yasm pkg-config unzip autoconf automake libtool make patch mingw-w64-x86_64-libtool mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc mingw-w64-i686-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-i686-gcc-fortran mingw-w64-x86_64-libwinpthread-git mingw-w64-i686-libwinpthread-git"

bash -lc "/c/projects/javacpp-presets/ci/build.sh %PROJ% %DROP_AUTH_TOK% %CI_DEPLOY_USERNAME% %CI_DEPLOY_PASSWORD%"

