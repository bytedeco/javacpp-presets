#!/bin/bash
 
export projectName=$1
cd $APPVEYOR_BUILD_FOLDER

echo Building $projectName
echo Compiler: $COMPILER
echo Architecture: $MSYS2_ARCH
echo MSYS2 directory: $MSYS2_DIR
echo MSYS2 system: $MSYSTEM
echo Bits: $BIT

bash --version
g++ --version
java -version
mvn --version

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
             curl -L -o /c/Downloads/pgr.zip "https://www.dropbox.com/s/vywbsds5difobpq/pgr.zip?dl=0"
           fi
           unzip /c/Downloads/pgr.zip
           mv Point\ Grey\ Research /c/Program\ Files
       elif [ "$MSYS2_ARCH" == "x86" ]; then
           if [[ $(find /c/Downloads/pgr32.zip -type f -size +1000000c 2>/dev/null) ]]; then
             echo "Found flycap32 in cache and size seems ok"
           else
             echo "Downloading pgr32.zip to cache as not found"
             curl -L -o /c/Downloads/pgr32.zip "https://www.dropbox.com/s/ofwly7sqdh7667v/pgr32.zip?dl=0"
           fi
           unzip /c/Downloads/pgr32.zip
           mv Point\ Grey\ Research /c/Program\ Files
       fi
       echo "Finished flycapture install"
fi

if [ "$projectName" == "cuda" ]; then
       echo Installing cuda 
       if [[ $(find /c/Downloads/cudnn-8.0-windows10-x64-v6.0.zip -type f -size +1000000c 2>/dev/null) ]]; then
         echo "Found cudnn in cache and size seems OK"
       else
         echo "Downloading cudnn as not found in cache"
         curl -L -o /c/Downloads/cudnn-8.0-windows10-x64-v6.0.zip "https://www.dropbox.com/s/wp0x29p2pz60icn/cudnn-8.0-windows10-x64-v6.0.zip?dl=0"
       fi
       curl -L -o cuda_8.0.61_windows.exe "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_windows-exe"
       ./cuda_8.0.61_windows.exe -s 
       echo May need to wait while cuda installs..
       unzip /c/Downloads/cudnn-8.0-windows10-x64-v6.0.zip
       mv ./cuda/bin/cudnn64_6.dll /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v8.0/bin
       mv ./cuda/include/cudnn.h /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v8.0/include
       mv ./cuda/lib/x64/cudnn.lib /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v8.0/lib/x64
       echo Finished cuda install
fi 

if [ "$projectName" == "libdc1394" ]; then 
       echo Installing libdc1394 
       curl -L -o CMU.zip "https://www.dropbox.com/s/97boebrmdza18uu/CMU.zip?dl=0"
       unzip CMU.zip
       mv CMU /c/Program\ Files\ \(x86\)
       echo Finished libdc1394 install
fi

if [[ "$projectName" =~ "hdf5" ]]; then
       echo Installing HDF5
       if [ "$MSYS2_ARCH" == "x86_64" ]; then 
          echo 64bit hdf5 
          curl -L -o hdf5.zip "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0-patch1/bin/windows/extra/hdf5-1.10.0-patch1-win64-vs2015-shared.zip"
          unzip hdf5.zip 
          cd hdf5
          msiexec //i HDF5-1.10.0-win64.msi //quiet
       elif [ "$MSYS2_ARCH" == "x86" ]; then
          echo 32bit copy for hdf5 
          curl -L -o hdf5.zip "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0-patch1/bin/windows/extra/hdf5-1.10.0-patch1-win32-vs2015-shared.zip"
          unzip hdf5.zip 
          cd hdf5
          msiexec //i HDF5-1.10.0-win32.msi //quiet
       fi
       cd ..
       echo Finished hd5 install 
fi

echo Finished setting up env in setup.sh

