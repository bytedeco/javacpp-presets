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
           cmd /c 'msiexec /quiet /i C:\Downloads\FlyCapture_2.13.3.31_x64.msi ADDLOCAL=ALL'
       elif [ "$OS" == "windows-x86" ]; then
           if [[ $(find /c/Downloads/FlyCapture_2.13.3.31_x86.msi -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found flycap32 in cache and size seems ok"
           else
             echo "Downloading flycap32 to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1ctSSAMF5IkxTKWiiLtID-ltmm27pHFdr /c/Downloads/FlyCapture_2.13.3.31_x86.msi
           fi
           # we can get this msi file by starting the installation from the exe file
           cmd /c 'msiexec /quiet /i C:\Downloads\FlyCapture_2.13.3.31_x86.msi ADDLOCAL=ALL'
       fi
       echo "Finished flycapture install"
fi

if [[ "$PROJ" =~ spinnaker ]]; then
       echo Spinnaker install
       if [ "$OS" == "windows-x86_64" ]; then
           if [[ $(find /c/Downloads/Spinnaker_*_v140_x64.msi -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found spinnaker in cache and size seems ok"
           else
             echo "Downloading spinnaker to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1sjFe7KyvjxPEmEFp9xP3wIs3QCOHAN1m /c/Downloads/Spinnaker_Binaries_v140_x64.msi
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1gLfpCE3XkcXbWaoFqT6kQqQm_EO55vYS /c/Downloads/Spinnaker_SourceCode_v140_x64.msi
           fi
           # we can get these msi files by starting the installation from the exe file
           cmd /c 'msiexec /quiet /i C:\Downloads\Spinnaker_Binaries_v140_x64.msi ADDLOCAL=ALL INSTALLFOLDER="C:\Program Files\Point Grey Research\Spinnaker"'
           cmd /c 'msiexec /quiet /i C:\Downloads\Spinnaker_SourceCode_v140_x64.msi ADDLOCAL=ALL INSTALLFOLDER="C:\Program Files\Point Grey Research\Spinnaker"'
       elif [ "$OS" == "windows-x86" ]; then
           if [[ $(find /c/Downloads/Spinnaker_*_v140_x86.msi -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found spinnaker32 in cache and size seems ok"
           else
             echo "Downloading spinnaker32 to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1YJcLAaf8Bf2XzC7Puv4ZwXdibdmVcwNg /c/Downloads/Spinnaker_Binaries_v140_x86.msi
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 1BG51avW4q605c2KGGJ3ehjhUlVFBObqh /c/Downloads/Spinnaker_SourceCode_v140_x86.msi
           fi
           # we can get these msi files by starting the installation from the exe file
           cmd /c 'msiexec /quiet /i C:\Downloads\Spinnaker_Binaries_v140_x86.msi ADDLOCAL=ALL INSTALLFOLDER="C:\Program Files (x86)\Point Grey Research\Spinnaker"'
           cmd /c 'msiexec /quiet /i C:\Downloads\Spinnaker_SourceCode_v140_x86.msi ADDLOCAL=ALL INSTALLFOLDER="C:\Program Files (x86)\Point Grey Research\Spinnaker"'
       fi
       echo "Finished spinnaker install"
fi

if [ "$PROJ" == "mkl" ]; then
       echo Installing mkl 
       curl -L  -o mkl.exe "http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15098/w_mkl_2019.2.190.exe"
       ./mkl.exe --s --x --f .
       ./install.exe install --output=mkllog.txt -eula=accept
       sleep 60
       cat mkllog.txt
       echo Finished mkl 
fi

if [ "$PROJ" == "cuda" ] || [ "$EXT" == "-gpu" ]; then
       echo Installing cuda 
       curl -L -o cuda_10.0.130_411.31_windows.exe "https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_windows"
       curl -L -o cudnn-10.0-windows7-x64-v7.4.1.5.zip "https://developer.download.nvidia.com/compute/redist/cudnn/v7.4.1/cudnn-10.0-windows7-x64-v7.4.1.5.zip"
       ./cuda_10.0.130_411.31_windows.exe -s
       sleep 60
       unzip ./cudnn-10.0-windows7-x64-v7.4.1.5.zip
       mv ./cuda/bin/*.dll /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v10.0/bin
       mv ./cuda/include/*.h /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v10.0/include
       mv ./cuda/lib/x64/*.lib /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v10.0/lib/x64
       echo Finished cuda install
fi 

if [ "$PROJ" == "tensorflow" ]; then
       /c/Python27/python -m pip install numpy
       curl -L http://downloads.sourceforge.net/project/swig/swigwin/swigwin-3.0.12/swigwin-3.0.12.zip -o swigwin-3.0.12.zip
       unzip -o swigwin-3.0.12.zip -d /c/

       echo "adding bazel for tensorflow"
       curl -L https://github.com/bazelbuild/bazel/releases/download/0.19.2/bazel-0.19.2-windows-x86_64.exe -o /c/msys64/usr/bin/bazel.exe; export CURL_STATUS=$?
       if [ "$CURL_STATUS" != "0" ]; then
         echo "Download failed here, so can't proceed with the build.. Failing.."
         exit 1
       fi
fi

# copy Python 3.6 back to default installation directory
cp -a "/c/Python36-x64" "/C/Program Files/Python36"

# install an older less buggy version of GCC
curl -L -o mingw-w64-i686-gcc-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-gcc-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-i686-gcc-ada-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-gcc-ada-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-i686-gcc-objc-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-gcc-objc-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-i686-gcc-libs-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-gcc-libs-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-i686-gcc-fortran-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-gcc-fortran-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-i686-gcc-libgfortran-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-gcc-libgfortran-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-i686-binutils-2.31.1-1-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-binutils-2.31.1-1-any.pkg.tar.xz
curl -L -o mingw-w64-i686-crt-git-6.0.0.5136.897300fe-1-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-crt-git-6.0.0.5136.897300fe-1-any.pkg.tar.xz
curl -L -o mingw-w64-i686-tools-git-6.0.0.5111.3bc5ab74-1-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-tools-git-6.0.0.5111.3bc5ab74-1-any.pkg.tar.xz
curl -L -o mingw-w64-i686-headers-git-6.0.0.5136.897300fe-1-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-headers-git-6.0.0.5136.897300fe-1-any.pkg.tar.xz
curl -L -o mingw-w64-i686-libmangle-git-6.0.0.5079.3b7a42fd-1-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-libmangle-git-6.0.0.5079.3b7a42fd-1-any.pkg.tar.xz
curl -L -o mingw-w64-i686-winstorecompat-git-5.0.0.4760.d3089b5-1-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-winstorecompat-git-5.0.0.4760.d3089b5-1-any.pkg.tar.xz
curl -L -o mingw-w64-i686-winpthreads-git-6.0.0.5134.2416de71-1-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-winpthreads-git-6.0.0.5134.2416de71-1-any.pkg.tar.xz
curl -L -o mingw-w64-i686-libwinpthread-git-6.0.0.5134.2416de71-1-any.pkg.tar.xz http://repo.msys2.org/mingw/i686/mingw-w64-i686-libwinpthread-git-6.0.0.5134.2416de71-1-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-gcc-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-gcc-ada-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-ada-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-gcc-objc-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-objc-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-gcc-libs-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-libs-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-gcc-fortran-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-fortran-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-gcc-libgfortran-7.3.0-2-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-gcc-libgfortran-7.3.0-2-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-binutils-2.31.1-1-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-binutils-2.31.1-1-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-crt-git-6.0.0.5136.897300fe-1-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-crt-git-6.0.0.5136.897300fe-1-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-tools-git-6.0.0.5111.3bc5ab74-1-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-tools-git-6.0.0.5111.3bc5ab74-1-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-headers-git-6.0.0.5136.897300fe-1-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-headers-git-6.0.0.5136.897300fe-1-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-libmangle-git-6.0.0.5079.3b7a42fd-1-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-libmangle-git-6.0.0.5079.3b7a42fd-1-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-winstorecompat-git-5.0.0.4760.d3089b5-1-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-winstorecompat-git-5.0.0.4760.d3089b5-1-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-winpthreads-git-6.0.0.5134.2416de71-1-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-winpthreads-git-6.0.0.5134.2416de71-1-any.pkg.tar.xz
curl -L -o mingw-w64-x86_64-libwinpthread-git-6.0.0.5134.2416de71-1-any.pkg.tar.xz http://repo.msys2.org/mingw/x86_64/mingw-w64-x86_64-libwinpthread-git-6.0.0.5134.2416de71-1-any.pkg.tar.xz
pacman -U --noconfirm *.pkg.tar.xz

# get rid of some stuff we don't use to avoid running out of disk space and that may actually interfere with our builds
rm -Rf /c/go*
rm -Rf /c/Qt*
#rm -Rf /c/Ruby*
rm -Rf /c/cygwin*
#rm -Rf /c/Miniconda*
#rm -Rf /c/Libraries/boost*
rm -Rf /c/Libraries/llvm*
rm -Rf /c/Program\ Files/LLVM*
rm -Rf /c/Program\ Files\ \(x86\)/Microsoft\ DirectX\ SDK*
rm -Rf /c/ProgramData/Microsoft/AndroidNDK*
df -h

# try to download partial builds, which doesn't work from AppVeyor's hosted VMs always returning "Connection state changed (MAX_CONCURRENT_STREAMS == 100)!" for some reason
#DOWNLOAD_FILE="$PROJ-cppbuild.zip"
#DOWNLOAD_ADDRESS="https://ci.appveyor.com/api/projects/bytedeco/javacpp-presets/artifacts/$DOWNLOAD_FILE"
#if curl -fsSL -G -v -o "$DOWNLOAD_FILE" "$DOWNLOAD_ADDRESS" --data-urlencode "all=true" --data-urlencode "job=Environment: PROJ=$PROJ, OS=$OS, EXT=$EXT, PARTIAL_CPPBUILD=1"; then
#    unzip -o $DOWNLOAD_FILE -d $APPVEYOR_BUILD_FOLDER
#fi

du -csh $HOME/* $HOME/.cache/* $HOME/.ccache/* /c/Users/appveyor/* /c/Users/appveyor/.m2/* /c/Users/downloads/*

echo Finished setting up env in setup.sh

