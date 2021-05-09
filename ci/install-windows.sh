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
export APPVEYOR_BUILD_FOLDER=`pwd`

echo Building $PROJ
echo Platform: $OS
echo MSYS2 system: $MSYSTEM
echo Extension: $EXT
echo Branch: $APPVEYOR_REPO_BRANCH

bash --version
git --version
g++ --version
java -version
mvn --version
/c/python27/python --version
pip --version
unzip --version
gpg --version

/c/python27/python -m pip install requests

mkdir -p /c/Downloads

if [[ "$APPVEYOR_PULL_REQUEST_NUMBER" == "" ]] && [[ "$APPVEYOR_REPO_BRANCH" == "release" ]]; then
    /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py /c/Users/appveyor/settings.tar.gz
    tar xzf /c/Users/appveyor/settings.tar.gz -C /c/Users/appveyor/
    tar xzf /c/Users/appveyor/settings.tar.gz -C /home/appveyor/
fi

echo Perform download files out of main repo
cd ..
if [[ "$PROJ" =~ flycapture ]]; then
       echo Flycapture install
       if [ "$OS" == "windows-x86_64" ]; then
           if [[ $(find /c/Downloads/FlyCapture_2.13.3.31_x64.msi -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found flycap in cache and size seems ok"
           else
             echo "Downloading flycap to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 14QM7W5RHhvZanF1UBobgEIvwdy6VwTht /c/Downloads/FlyCapture_2.13.3.31_x64.msi
           fi
           # we can get this msi file by starting the installation from the exe file
           cmd.exe //c 'msiexec /quiet /i C:\Downloads\FlyCapture_2.13.3.31_x64.msi ADDLOCAL=ALL'
       elif [ "$OS" == "windows-x86" ]; then
           if [[ $(find /c/Downloads/FlyCapture_2.13.3.31_x86.msi -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found flycap32 in cache and size seems ok"
           else
             echo "Downloading flycap32 to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1ctSSAMF5IkxTKWiiLtID-ltmm27pHFdr /c/Downloads/FlyCapture_2.13.3.31_x86.msi
           fi
           # we can get this msi file by starting the installation from the exe file
           cmd.exe //c 'msiexec /quiet /i C:\Downloads\FlyCapture_2.13.3.31_x86.msi ADDLOCAL=ALL'
       fi
       echo "Finished flycapture install"
fi

if [[ "$PROJ" =~ spinnaker ]]; then
       echo Spinnaker install
       if [ "$OS" == "windows-x86_64" ]; then
           if [[ $(find /c/Downloads/Spinnaker_2.4.0.143_*_v140_x64.msi -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found spinnaker in cache and size seems ok"
           else
             echo "Downloading spinnaker to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1NiMxxDflip1F5rJuf73xhfoI9vWUeUkQ /c/Downloads/Spinnaker_2.4.0.143_Binaries_v140_x64.msi
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1FKatuVsLIQLbDb9R9W2ax_tav8_7--HI /c/Downloads/Spinnaker_2.4.0.143_SourceCode_v140_x64.msi
           fi
           # we can get these msi files by starting the installation from the exe file
           cmd.exe //c 'msiexec /quiet /i C:\Downloads\Spinnaker_2.4.0.143_Binaries_v140_x64.msi ADDLOCAL=ALL'
           cmd.exe //c 'msiexec /quiet /i C:\Downloads\Spinnaker_2.4.0.143_SourceCode_v140_x64.msi ADDLOCAL=ALL'
       elif [ "$OS" == "windows-x86" ]; then
           if [[ $(find /c/Downloads/Spinnaker_2.4.0.143_*_v140_x86.msi -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found spinnaker32 in cache and size seems ok"
           else
             echo "Downloading spinnaker32 to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 15i6tV-vahgm_uNWT96c_KX7tFdZxmev- /c/Downloads/Spinnaker_2.4.0.143_Binaries_v140_x86.msi
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1ikOyG-gVettvjoJGQWkHxOcBWuTrGyfz /c/Downloads/Spinnaker_2.4.0.143_SourceCode_v140_x86.msi
           fi
           # we can get these msi files by starting the installation from the exe file
           cmd.exe //c 'msiexec /quiet /i C:\Downloads\Spinnaker_2.4.0.143_Binaries_v140_x86.msi ADDLOCAL=ALL'
           cmd.exe //c 'msiexec /quiet /i C:\Downloads\Spinnaker_2.4.0.143_SourceCode_v140_x86.msi ADDLOCAL=ALL'
       fi
       echo "Finished spinnaker install"
fi

if [ "$PROJ" == "mkl" ]; then
       echo Installing mkl 
       curl -L  -o mkl.exe "https://registrationcenter-download.intel.com/akdlm/irc_nas/tec/17173/w_mkl_2020.4.311.exe"
       ./mkl.exe --s --x --f .
       ./install.exe install --output=mkllog.txt -eula=accept
       sleep 60
       cat mkllog.txt
       echo Finished mkl 
fi

if [ "$PROJ" == "cuda" ] || [ "$PROJ" == "tensorrt" ] || [ "$EXT" == "-gpu" ]; then
       echo Installing cuda 
       curl -L -o cuda_11.1.1_456.81_win10.exe "https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_456.81_win10.exe"
       curl -L -o cudnn-11.1-windows-x64-v8.0.4.30.zip "https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.4/cudnn-11.1-windows-x64-v8.0.4.30.zip"
       ./cuda_11.1.1_456.81_win10.exe -s
       sleep 60
       unzip ./cudnn-11.1-windows-x64-v8.0.4.30.zip
       mv ./cuda/bin/*.dll /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.1/bin
       mv ./cuda/include/*.h /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.1/include
       mv ./cuda/lib/x64/*.lib /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.1/lib/x64
       echo Finished cuda install
fi 

if [ "$PROJ" == "tensorrt" ] || [ "$EXT" == "-gpu" ]; then
       echo Installing tensorrt 
       /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1wp0W7FyEFpEI92uIDWWRvgEsvqMhqXEe /c/Downloads/tensorrt.zip
       unzip -o /c/Downloads/tensorrt.zip -d /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/
       ln -sf /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/TensorRT* /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/TensorRT
       echo Finished tensorrt install
fi

if [ "$PROJ" == "tensorflow" ]; then
       curl -L http://downloads.sourceforge.net/project/swig/swigwin/swigwin-3.0.12/swigwin-3.0.12.zip -o swigwin-3.0.12.zip
       unzip -o swigwin-3.0.12.zip -d /c/

       echo "adding bazel for tensorflow"
       curl -L https://github.com/bazelbuild/bazel/releases/download/0.25.3/bazel-0.25.3-windows-x86_64.exe -o /c/msys64/usr/bin/bazel.exe; export CURL_STATUS=$?
       if [ "$CURL_STATUS" != "0" ]; then
         echo "Download failed here, so can't proceed with the build.. Failing.."
         exit 1
       fi
fi

# copy Python 3.x back to default installation directory
cp -a "/c/Python36-x64" "/C/Program Files/Python36"
cp -a "/c/Python37-x64" "/C/Program Files/Python37"
cp -a "/c/Python38-x64" "/C/Program Files/Python38"

# install an older less buggy version of GCC
#/c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1CmH1Eq0LoAQQ5mWaqYiU9kYOfMGjLXlz /c/Downloads/mingw-w64-old.tar
#tar xvf /c/Downloads/mingw-w64-old.tar
#pacman -U --noconfirm *.pkg.tar.xz

# get rid of some stuff we don't use to avoid running out of disk space and that may actually interfere with our builds
rm -Rf /c/go*
rm -Rf /c/Qt*
#rm -Rf /c/Ruby*
rm -Rf /c/cygwin*
#rm -Rf /c/Miniconda*
#rm -Rf /c/Libraries/boost*
rm -Rf /c/Libraries/llvm*
rm -Rf /c/Program\ Files/LLVM*
rm -Rf /c/ProgramData/Microsoft/AndroidNDK*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ DirectX\ SDK*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/Community/VC/Tools/MSVC/14.12*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/Community/VC/Redist/MSVC/14.12*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/Community/VC/Auxiliary/Build/14.12/
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Tools/MSVC/14.16*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Redist/MSVC/14.16*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Auxiliary/Build/14.16/
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Tools/MSVC/14.2[01234567]*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Redist/MSVC/14.2[01234567]*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Auxiliary/Build/14.2[01234567]/
ls -l /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Tools/MSVC/
ls -l /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Redist/MSVC/
ls -l /c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2019/Community/VC/Auxiliary/Build/
rm -Rf /usr/bin/link.exe
pacman -Rc --noconfirm python python2 mingw-w64-i686-python3 mingw-w64-x86_64-python3
df -h

# try to download partial builds, which doesn't work from AppVeyor's hosted VMs always returning "Connection state changed (MAX_CONCURRENT_STREAMS == 100)!" for some reason
#DOWNLOAD_FILE="$PROJ-cppbuild.zip"
#DOWNLOAD_ADDRESS="https://ci.appveyor.com/api/projects/bytedeco/javacpp-presets/artifacts/$DOWNLOAD_FILE"
#if curl -fsSL -G -v -o "$DOWNLOAD_FILE" "$DOWNLOAD_ADDRESS" --data-urlencode "all=true" --data-urlencode "job=Environment: PROJ=$PROJ, OS=$OS, EXT=$EXT, PARTIAL_CPPBUILD=1"; then
#    unzip -o $DOWNLOAD_FILE -d $APPVEYOR_BUILD_FOLDER
#fi

du -csh $HOME/* $HOME/.cache/* $HOME/.ccache/* /c/Users/appveyor/* /c/Users/appveyor/.m2/* /c/Users/downloads/*

echo Finished setting up env in setup.sh

