 @echo off
  set projectName=%~1
  set DROPAUTH=%~2
  set CI_DEPLOY_USERNAME=%~3
  set CI_DEPLOY_PASSWORD=%~4
  cd %APPVEYOR_BUILD_FOLDER%

  echo Building "%projectName%"
  echo Compiler: %COMPILER%
  echo Architecture: %MSYS2_ARCH%
  echo Platform: %PLATFORM%
  echo MSYS2 directory: %MSYS2_DIR%
  echo MSYS2 system: %MSYSTEM%
  echo Bits: %BIT%

  REM Create a writeable TMPDIR
  mkdir %APPVEYOR_BUILD_FOLDER%\tmp
  set TMPDIR=%APPVEYOR_BUILD_FOLDER%\tmp
  mkdir %APPVEYOR_BUILD_FOLDER%\buildlogs

 IF %COMPILER%==msys2 (
    @echo on
    SET "PATH=C:\%MSYS2_DIR%\%MSYSTEM%\bin;C:\%MSYS2_DIR%\usr\bin;%PATH%"
    bash -lc "pacman -S --needed --noconfirm pacman-mirrors"
    bash -lc "pacman -S --needed --noconfirm git"
    REM Update
    bash -lc "pacman -Syu --noconfirm"

    REM build tools
    bash -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-toolchain base-devel tar nasm yasm pkg-config unzip autoconf automake libtool make patch mingw-w64-x86_64-libtool"

    bash --version
    g++ --version
    java -version
    mvn --version

    IF "%MSYS2_ARCH%"=="x86_64" (
       echo Callings vcvarsall for amd64
       call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
    )
    IF "%MSYS2_ARCH%"=="x86" (
       echo Callings vcvarsall for x86
       call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86
    )
 
    echo Perform download files out of main repo
    cd ..
    IF "%projectName%"=="flycapture" (
       echo Flycapture install
       IF "%MSYS2_ARCH%"=="x86_64" (
           curl -L -s -X POST --globoff  -o pgr.zip --header "Authorization: Bearer %DROPAUTH%" --header 'Dropbox-API-Arg: {"path": "/pgr.zip"}' https://content.dropboxapi.com/2/files/download
           unzip pgr.zip
           move "Point Grey Research" "c:\Program Files"
       )
       IF "%MSYS2_ARCH%"=="x86" (
           curl -L -s -X POST --globoff  -o pgr32.zip --header "Authorization: Bearer %DROPAUTH%" --header 'Dropbox-API-Arg: {"path": "/pgr32.zip"}' https://content.dropboxapi.com/2/files/download
           unzip pgr32.zip
           move "Point Grey Research" "c:\Program Files"
       )
       echo "Finished flycapture install"
    )

    IF "%projectName%"=="cuda" (
       curl -L -s -X POST --globoff  -o cudnn-8.0-windows10-x64-v6.0.zip --header "Authorization: Bearer %DROPAUTH%" --header 'Dropbox-API-Arg: {"path": "/cudnn-8.0-windows10-x64-v6.0.zip"}' https://content.dropboxapi.com/2/files/download
       @echo on
       curl.exe -L -o cuda_8.0.61_windows.exe "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_windows-exe"
       cuda_8.0.61_windows.exe -s 
       echo May need to wait while cuda installs..
       REM timeout /T 300
       unzip cudnn-8.0-windows10-x64-v6.0.zip
       move .\cuda\bin\cudnn64_6.dll "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin" 
       move .\cuda\include\cudnn.h "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include" 
       move .\cuda\lib\x64\cudnn.lib "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64" 
       echo Finished cuda install
    )

    IF "%projectName%"=="libdc1394" (
       @echo on
       curl.exe -L -o CMU.zip "https://www.dropbox.com/s/97boebrmdza18uu/CMU.zip?dl=0"
       dir
       unzip CMU.zip
       move CMU "c:\Program Files (x86)"
       echo Finished libdc1394 install
    )

    IF "%MSYS2_ARCH%"=="x86_64" (
       curl.exe -L -o hdf5.zip "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0-patch1/bin/windows/hdf5-1.10.0-patch1-win64-vs2013-shared.zip"
       unzip hdf5.zip 
       cd hdf5
       msiexec /i HDF5-1.10.0-win64.msi /quiet
    )

    IF "%MSYS2_ARCH%"=="x86" (
       echo 32bit copy for hdf5 
       curl.exe -L -o hdf5.zip "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0-patch1/bin/windows/hdf5-1.10.0-patch1-win32-vs2013-shared.zip"
       unzip hdf5.zip 
       cd hdf5
       msiexec /i HDF5-1.10.0-win32.msi /quiet
       xcopy "c:\Program Files (x86)\HDF_Group" "c:\Program Files\HDF_Group" /S /I /E
    )

    cd ..
    echo Finished hd5 install 
    

    echo Starting main build now.. 
    cd javacpp 
    mvn install -Dmaven.test.skip=true -Djavacpp.platform=windows-%MSYS2_ARCH% -Dmaven.javadoc.skip=true
    cd ..
    cd javacpp-presets
    mvn deploy -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=windows-%MSYS2_ARCH% --settings .\ci\settings.xml  -pl %projectName% 
  )
