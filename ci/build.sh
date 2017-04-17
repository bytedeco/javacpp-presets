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
mkdir $APPVEYOR_BUILD_FOLDER\tmp
export TMPDIR=$APPVEYOR_BUILD_FOLDER\tmp
mkdir $APPVEYOR_BUILD_FOLDER\buildlogs

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
           curl -L -s -X POST --globoff  -o pgr.zip --header "Authorization: Bearer $DROPAUTH" --header 'Dropbox-API-Arg: {"path": "/pgr.z
ip"}' https://content.dropboxapi.com/2/files/download
           unzip pgr.zip
           move "Point Grey Research" "c:\Program Files"
       elif [ "$MSYS2_ARCH"=="x86" ]; then
           curl -L -s -X POST --globoff  -o pgr32.zip --header "Authorization: Bearer $DROPAUTH" --header 'Dropbox-API-Arg: {"path": "/pgr
32.zip"}' https://content.dropboxapi.com/2/files/download
           unzip pgr32.zip
           move "Point Grey Research" "c:\Program Files"
       fi
       echo "Finished flycapture install"
fi

echo Starting main build now.. 
cd javacpp 
mvn install -Dmaven.test.skip=true -Djavacpp.platform=windows-$MSYS2_ARCH -Dmaven.javadoc.skip=true
cd ..
cd javacpp-presets
mvn deploy -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=windows-$MSYS2_ARCH --settings .\ci\settings.xml  -pl $projectName

