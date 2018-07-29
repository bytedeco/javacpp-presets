#!/bin/bash

echo Reducing PATH size by removing duplicates and truncating to satisfy MKL, etc
PREVIFS="$IFS"
NEWPATH="${PATH%%:*}"
IFS=":"
for P in $PATH; do
    FOUND=0
    for P2 in $NEWPATH; do
        if [[ "$P" == "$P2" ]]; then
            FOUND=1
        fi
    done
    if [[ "$FOUND" == "0" ]] && [[ ${#NEWPATH} -lt 3000 ]]; then
        NEWPATH=$NEWPATH:$P
    fi
done
IFS="$PREVIFS"
echo ${#PATH}
echo ${#NEWPATH}
export PATH=$NEWPATH

set -vx

export PROJ=$1
cd $APPVEYOR_BUILD_FOLDER

echo Building $PROJ
echo Platform: $OS
echo MSYS2 system: $MSYSTEM
echo Extension: $EXT
echo Branch: $APPVEYOR_REPO_BRANCH

bash --version
g++ --version
java -version
mvn --version
/c/python27/python --version
pip --version
unzip --version
gpg --version

pip install requests

mkdir -p /c/Downloads

if [[ "$APPVEYOR_PULL_REQUEST_NUMBER" == "" ]] && [[ "$APPVEYOR_REPO_BRANCH" == "release" ]]; then
    /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py /c/Users/appveyor/settings.tar.gz
    tar xzf /c/Users/appveyor/settings.tar.gz -C /c/Users/appveyor/
fi

echo Perform download files out of main repo
cd ..
if [ "$PROJ" == "flycapture" ]; then
       echo Flycapture install
       if [ "$OS" == "windows-x86_64" ]; then
           if [[ $(find /c/Downloads/pgr.zip -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found flycap in cache and size seems ok"
           else
             echo "Downloading pgr.zip to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 0B2xpvMUzviShRFl3aWVWOVFPYlU /c/Downloads/pgr.zip 
           fi
           unzip /c/Downloads/pgr.zip
           mv Point\ Grey\ Research /c/Program\ Files
       elif [ "$OS" == "windows-x86" ]; then
           if [[ $(find /c/Downloads/pgr32.zip -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found flycap32 in cache and size seems ok"
           else
             echo "Downloading pgr32.zip to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 0B2xpvMUzviShQlpQSEFhZkUwc0U /c/Downloads/pgr32.zip 
           fi
           unzip /c/Downloads/pgr32.zip
           mv Point\ Grey\ Research /c/Program\ Files
       fi
       echo "Finished flycapture install"
fi

if [ "$PROJ" == "mkl" ]; then
       echo Installing mkl 
       curl -L  -o mkl.exe "http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/13037/w_mkl_2018.3.210.exe"
       ./mkl.exe --s --x --f .
       ./setup.exe install --output=mkllog.txt -eula=accept
       sleep 60
       cat mkllog.txt
       echo Finished mkl 
fi

if [ "$PROJ" == "cuda" ] || [ "$EXT" == "-gpu" ]; then
       echo Installing cuda 
       curl -L -o cuda_9.2.88_windows.exe "https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda_9.2.88_windows"
       curl -L -o cuda_9.2.88.1_windows.exe "https://developer.nvidia.com/compute/cuda/9.2/Prod/patches/1/cuda_9.2.88.1_windows"
       curl -L -o cudnn-9.2-windows7-x64-v7.1.zip "http://developer.download.nvidia.com/compute/redist/cudnn/v7.1.4/cudnn-9.2-windows7-x64-v7.1.zip"
       ./cuda_9.2.88_windows.exe -s
       sleep 60
       ./cuda_9.2.88.1_windows.exe -s
       sleep 10
       unzip ./cudnn-9.2-windows7-x64-v7.1.zip
       mv ./cuda/bin/*.dll /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.2/bin
       mv ./cuda/include/*.h /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.2/include
       mv ./cuda/lib/x64/*.lib /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.2/lib/x64
       echo Finished cuda install
fi 

echo Finished setting up env in setup.sh

