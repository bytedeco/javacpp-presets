#!/bin/bash
 
export projectName=$1
cd $APPVEYOR_BUILD_FOLDER

echo Building $projectName
echo Compiler: $COMPILER
echo Architecture: $MSYS2_ARCH
echo MSYS2 directory: $MSYS2_DIR
echo MSYS2 system: $MSYSTEM
echo Bits: $BIT
echo Branch: $APPVEYOR_REPO_BRANCH

bash --version
g++ --version
java -version
mvn --version
/c/python27/python --version
pip --version
unzip --version

pip install requests

mkdir -p /c/Downloads

echo Perform download files out of main repo
cd ..
if [ "$projectName" == "flycapture" ]; then
       echo Flycapture install
       if [ "$MSYS2_ARCH" == "x86_64" ]; then
           if [[ $(find /c/Downloads/pgr.zip -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found flycap in cache and size seems ok"
           else
             echo "Downloading pgr.zip to cache as not found"
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 0B2xpvMUzviShRFl3aWVWOVFPYlU /c/Downloads/pgr.zip 
           fi
           unzip /c/Downloads/pgr.zip
           mv Point\ Grey\ Research /c/Program\ Files
       elif [ "$MSYS2_ARCH" == "x86" ]; then
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

if [ "$projectName" == "mkl" ]; then
       echo Installing mkl 
       curl -L  -o mkl.exe "http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/12079/w_mkl_2018.0.124.exe"
       ./mkl.exe --s --x --f .
       ./setup.exe install --output=mkllog.txt -eula=accept
       sleep 60
       echo Finished mkl 
fi

if [ "$projectName" == "cuda" ] || [ "$projectName" == "opencv" ]; then
       echo Installing cuda 
       if [[ $(find /c/Downloads/cudnn-9.0-windows10-x64-v7.zip -type f -size +1000000c 2>/dev/null) ]]; then
         echo "Found cudnn in cache and size seems OK"
       else
         echo "Downloading cudnn as not found in cache"
         /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 0B8dyy7cU8B67cDRUVFFtZzFwREE /c/Downloads/cudnn-9.0-windows10-x64-v7.zip
       fi
       curl -L -o cuda_9.0.176_windows-exe "https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_windows-exe"
       ./cuda_9.0.176_windows-exe -s 
       echo May need to wait while cuda installs..
       unzip /c/Downloads/cudnn-9.0-windows10-x64-v7.zip
       mv ./cuda/bin/*.dll /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.0/bin
       mv ./cuda/include/*.h /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.0/include
       mv ./cuda/lib/x64/*.lib /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.0/lib/x64
       echo Finished cuda install
fi 

if [ "$projectName" == "libdc1394" ]; then 
       echo Installing libdc1394 
       /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 0B2xpvMUzviShVnNJM3JCclpuTE0 CMU.zip
       unzip CMU.zip
       mv CMU /c/Program\ Files\ \(x86\)
       echo Finished libdc1394 install
fi

if [[ "$projectName" =~ "hdf5" ]]; then
       echo Installing HDF5
       if [ "$MSYS2_ARCH" == "x86_64" ]; then 
          echo 64bit hdf5 
          if [[ $(find /c/Downloads/hdf5-64.zip -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found hdf5-64 in cache and size seems OK"
          else
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 0B2xpvMUzviShTEtPYjU5SDhIZWM /c/Downloads/hdf5-64.zip 
          fi
          unzip /c/Downloads/hdf5-64.zip 
          cd hdf 
          msiexec //i HDF5-1.10.1-win64.msi //quiet
       elif [ "$MSYS2_ARCH" == "x86" ]; then
          echo 32bit copy for hdf5 
          if [[ $(find /c/Downloads/hdf5-32.zip -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found hdf5-32 in cache and size seems OK"
          else
             /c/python27/python $APPVEYOR_BUILD_FOLDER/ci/gDownload.py 0B2xpvMUzviShU1BzVTd1YzZGczg /c/Downloads/hdf5-32.zip 
          fi
          unzip /c/Downloads/hdf5-32.zip 
          cd hdf 
          msiexec //i HDF5-1.10.1-win32.msi //quiet
       fi
       cd ..
       echo Finished hd5 install 
fi

echo Finished setting up env in setup.sh

