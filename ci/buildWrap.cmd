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

bash -lc "/c/projects/javacpp-presets/ci/build.sh %PROJ% %DROP_AUTH_TOK% %CI_DEPLOY_USERNAME% %CI_DEPLOY_PASSWORD%"

