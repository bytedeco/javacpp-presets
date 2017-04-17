#!/bin/bash
 
export projectName=$1
export DROPAUTH=$2
export CI_DEPLOY_USERNAME=$3
export CI_DEPLOY_PASSWORD=$4
cd $APPVEYOR_BUILD_FOLDER

echo Building $projectName
echo Compiler: $COMPILER
echo Architecture: $MSYS2_ARCH
echo MSYS2 directory: $MSYS2_DIR
echo MSYS2 system: $MSYSTEM
echo Bits: $BIT

#Create a writeable TMPDIR
mkdir $APPVEYOR_BUILD_FOLDER/tmp
export TMPDIR=$APPVEYOR_BUILD_FOLDER/tmp
mkdir $APPVEYOR_BUILD_FOLDER/buildlogs

if [ "$COMPILER" == "msys2" ]; then
    #export PATH="C:\$MSYS2_DIR\$MSYSTEM%\bin;C:\%MSYS2_DIR%\usr\bin;%PATH%"
    pacman -S --needed --noconfirm pacman-mirrors
    pacman -S --needed --noconfirm git
    pacman -Syu --noconfirm

    #build tools
    pacman -S --needed --noconfirm mingw-w64-x86_64-toolchain base-devel tar nasm yasm pkg-config unzip autoconf automake libtool make patch mingw-w64-x86_64-libtool

    bash --version
    g++ --version
    java -version
    mvn --version
fi

echo done

echo Perform download files out of main repo
cd ..
if [ "$projectName" == "flycapture" ]; then
       echo Flycapture install
       if [ "$MSYS2_ARCH"=="x86_64" ]; then
           curl -L -s -X POST --globoff  -o pgr.zip --header "Authorization: Bearer $DROPAUTH" --header 'Dropbox-API-Arg: {"path": "/pgr.zip"}' https://content.dropboxapi.com/2/files/download
           unzip pgr.zip
           mv "Point Grey Research" "/c/Program\ Files"
       elif [ "$MSYS2_ARCH"=="x86" ]; then
           curl -L -s -X POST --globoff  -o pgr32.zip --header "Authorization: Bearer $DROPAUTH" --header 'Dropbox-API-Arg: {"path": "/pgr32.zip"}' https://content.dropboxapi.com/2/files/download
           unzip pgr32.zip
           move "Point Grey Research" "/c/Program\ Files"
       fi
       echo "Finished flycapture install"
fi

if [ "$projectName" == "cuda" ]; then
       echo Installing cuda 
       curl -L -s -X POST --globoff  -o cudnn-8.0-windows10-x64-v6.0.zip --header "Authorization: Bearer $DROPAUTH" --header 'Dropbox-API-Arg: {"path": "/cudnn-8.0-windows10-x64-v6.0.zip"}' https://content.dropboxapi.com/2/files/download
       curl -L -o cuda_8.0.61_windows.exe "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_windows-exe"
       ./cuda_8.0.61_windows.exe -s 
       echo May need to wait while cuda installs..
       unzip cudnn-8.0-windows10-x64-v6.0.zip
       mv ./cuda/bin/cudnn64_6.dll "/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v8.0/bin" 
       mv ./cuda/include/cudnn.h "/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v8.0/include" 
       mv ./cuda/lib/x64/cudnn.lib "/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v8.0/lib/x64" 
       echo Finished cuda install
fi 

if [ "$projectName" == "libdc1394" ]; then 
       echo Installing libdc1394 
       curl -L -o CMU.zip "https://www.dropbox.com/s/97boebrmdza18uu/CMU.zip?dl=0"
       unzip CMU.zip
       mv CMU "/c/Program\ Files\ \(x86\)"
       echo Finished libdc1394 install
fi

if [[ "$projectName" =~ "hdf5" ]]; then
       echo Installing HDF5
       if [ "$MSYS2_ARCH" == "x86_64" ]; then 
          echo 64bit hdf5 
          curl -L -o hdf5.zip "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0-patch1/bin/windows/extra/hdf5-1.10.0-patch1-win64-vs2015-shared.zip"
          unzip hdf5.zip 
          cd hdf5
          msiexec /i HDF5-1.10.0-win64.msi /quiet
       fi
       elif [ "$MSYS2_ARCH" == "x86" ]; then
          echo 32bit copy for hdf5 
          curl -L -o hdf5.zip "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0-patch1/bin/windows/extra/hdf5-1.10.0-patch1-win32-vs2015-shared.zip"
          unzip hdf5.zip 
          cd hdf5
          msiexec /i HDF5-1.10.0-win32.msi /quiet
          mv /c/Program Files\ \(x86\)/HDF_Group /c/Program\ Files/HDF_Group
       cd ..
       echo Finished hd5 install 
fi


echo Starting main build now.. 
cd javacpp 
mvn install -Dmaven.test.skip=true -Djavacpp.platform=windows-$MSYS2_ARCH -Dmaven.javadoc.skip=true
cd ..
cd javacpp-presets
mvn deploy -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=windows-$MSYS2_ARCH --settings ./ci/settings.xml  -pl $projectName

