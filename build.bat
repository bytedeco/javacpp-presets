 @echo off

  cd %APPVEYOR_BUILD_FOLDER%

  echo Compiler: %COMPILER%
  echo Architecture: %MSYS2_ARCH%
  echo Platform: %PLATFORM%
  echo MSYS2 directory: %MSYS2_DIR%
  echo MSYS2 system: %MSYSTEM%
  echo Bits: %BIT%

  REM Create a writeable TMPDIR
  mkdir %APPVEYOR_BUILD_FOLDER%\tmp
  set TMPDIR=%APPVEYOR_BUILD_FOLDER%\tmp

  IF %COMPILER%==msys2 (
    @echo on
    SET "PATH=C:\%MSYS2_DIR%\%MSYSTEM%\bin;C:\%MSYS2_DIR%\usr\bin;%PATH%"
    bash -lc "pacman -S --needed --noconfirm pacman-mirrors"
    bash -lc "pacman -S --needed --noconfirm git"
    REM Update
    bash -lc "pacman -Syu --noconfirm"

    REM build tools
    bash -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-toolchain unzip autoconf automake libtool make patch mingw-w64-x86_64-libtool"

bash --version
g++ --version
java -version
mvn --version

dir
  )
