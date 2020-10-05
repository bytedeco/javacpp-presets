#!/bin/bash 
set -vx

# Prevent Travis CI from terminating builds after 10 minutes with no output
while true; do uptime; sleep 60; done &

# Abort before the maximum build time to be able to save the cache
# (needs to be less than 2 hours for this to work on Mac as well)
(sleep 6600; sudo killall -s SIGINT java; sudo killall bazel; sudo killall make) &

# Allocate a swapfile on Linux as it's not enabled by default
sudo fallocate -l 4GB /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
df -h

# Fix issue with Sectigo CA root certificate on Ubuntu Xenial
sudo sed -i '/AddTrust_External_Root/d' /etc/ca-certificates.conf
sudo update-ca-certificates -f

mkdir ./buildlogs
mkdir $TRAVIS_BUILD_DIR/downloads
sudo chown -R travis:travis $HOME
du -csh $HOME/* $HOME/.m2/* $HOME/.cache/* $HOME/.ccache/* $HOME/downloads/*
curl https://bootstrap.pypa.io/get-pip.py | sudo python
sudo pip install requests
touch $HOME/vars.list

export MAKEJ=2
echo "export MAKEJ=2" | tee --append $HOME/vars.list

# Try to use ccache to speed up the build
curl -L https://github.com/ccache/ccache/releases/download/v3.7.10/ccache-3.7.10.tar.gz -o $HOME/ccache-3.7.10.tar.gz
tar xvf $HOME/ccache-3.7.10.tar.gz -C $HOME
export CCACHE_DIR=$HOME/.ccache
export PATH=/usr/lib64/ccache/:/usr/lib/ccache/:$PATH
echo "export CCACHE_DIR=$HOME/.ccache" | tee --append $HOME/vars.list
echo "export PATH=/usr/lib64/ccache/:/usr/lib/ccache/:\$PATH" | tee --append $HOME/vars.list
echo -e "log_file = $HOME/ccache.log\nmax_size = 5.0G\nhash_dir = false\nsloppiness = file_macro,include_file_ctime,include_file_mtime,pch_defines,time_macros" > $CCACHE_DIR/ccache.conf

if [[ "$TRAVIS_PULL_REQUEST" == "false" ]] && [[ "$TRAVIS_BRANCH" == "release" ]]; then
    python $TRAVIS_BUILD_DIR/ci/gDownload.py $HOME/settings.tar.gz
    tar xzf $HOME/settings.tar.gz -C $HOME
    MAVEN_RELEASE="-Dgpg.homedir=$HOME/.gnupg/ -DperformRelease -DstagingRepositoryId=$STAGING_REPOSITORY"
else
    MAVEN_RELEASE="-Dmaven.javadoc.skip=true"
fi

if [ "$TRAVIS_OS_NAME" == "osx" ]; then export JAVA_HOME=$(/usr/libexec/java_home); fi

if [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]] || [[ "$OS" =~ android ]]; then
  CENTOS_VERSION=6
  SCL_ENABLE="devtoolset-7 rh-python36 python27"
  if [[ "hyperscan tensorflow onnx ngraph onnxruntime qt skia " =~ "$PROJ " ]] || [[ "$OS" =~ android ]]; then
    CENTOS_VERSION=7
    SCL_ENABLE="devtoolset-7"
  fi
  echo "Starting docker for x86_64 and x86 linux"
  docker run -d -ti -e CI_DEPLOY_USERNAME -e CI_DEPLOY_PASSWORD -e GPG_PASSPHRASE -e STAGING_REPOSITORY -v $HOME:$HOME -v $TRAVIS_BUILD_DIR/../:$HOME/build centos:$CENTOS_VERSION /bin/bash
  DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
  echo "Container id is $DOCKER_CONTAINER_ID please wait while updates applied"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -q -y install centos-release-scl-rh epel-release"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -y repolist"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -q -y install rh-java-common-ant $SCL_ENABLE boost-devel ccache clang gcc-c++ gcc-gfortran java-1.8.0-openjdk-devel ant python python36-devel python36-pip swig file which wget unzip tar bzip2 gzip xz patch make autoconf-archive libtool bison flex perl nasm alsa-lib-devel freeglut-devel gtk2-devel libusb-devel libusb1-devel curl-devel expat-devel gettext-devel openssl-devel zlib-devel SDL-devel libva-devel libxkbcommon-devel libxkbcommon-x11-devel xcb-util* fontconfig-devel libffi-devel ragel ocl-icd-devel"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -y update"
  if [ "$OS" == "linux-x86" ]; then
    docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "rpm -qa | sed s/.x86_64$/.i686/ | xargs yum -q -y install"
    docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "find /var/cache/yum/ -name *.rpm | xargs rpm -i --force"
    if [[ "$SCL_ENABLE" =~ devtoolset-7 ]]; then
      docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "wget --no-directories --no-parent -r https://www.repo.cloudlinux.com/cloudlinux/$CENTOS_VERSION/sclo/devtoolset-7/i386/ -P $HOME"
      docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "rpm -i --force --nodeps $HOME/*.rpm"
    fi
  fi
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "rm -f /usr/lib/libgfortran.so.3* /usr/lib64/libgfortran.so.3*" # not required for GCC 7+
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "source scl_source enable $SCL_ENABLE || true; gcc --version"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "gpg --version"

  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf $HOME/.m2 /root/.m2"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf $HOME/.cache /root/.cache"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf $HOME/.ccache /root/.ccache"

  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "curl -L https://github.com/Kitware/CMake/releases/download/v3.16.6/cmake-3.16.6-Linux-x86_64.tar.gz -o $HOME/cmake-3.16.6-Linux-x86_64.tar.gz"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "curl -L https://archive.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz -o $HOME/apache-maven-3.3.9-bin.tar.gz"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "curl -L https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.18.3.tar.gz -o $HOME/git-2.18.3.tar.gz"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xzf $HOME/cmake-3.16.6-Linux-x86_64.tar.gz -C /opt/"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xzf $HOME/apache-maven-3.3.9-bin.tar.gz -C /opt/"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xzf $HOME/git-2.18.3.tar.gz -C $HOME"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd $HOME/git-2.18.3; make prefix=/usr/local/; make prefix=/usr/local/ install"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /usr/bin/python3.6 /usr/bin/python3"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /opt/cmake-3.16.6-Linux-x86_64/bin/* /usr/bin/"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /opt/apache-maven-3.3.9/bin/mvn /usr/bin/mvn"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cmake -version"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "git --version"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mvn -version"

  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp -a /opt/rh/httpd24/root/usr/lib64/* /usr/lib64/"

  if [[ "$PROJ" =~ flycapture ]]; then
    if [ "$OS" == "linux-x86_64" ]; then
        if [[ $(find $HOME/downloads/flycapture2-2.13.3.31-amd64-pkg_xenial.tgz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found flycap64 in cache and size seems ok" 
        else
          echo "Downloading flycap64 as not found in cache or too small" 
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 1YtVjdnbQLZHX_ocQ6xAmiq6pjftuPOPd $HOME/downloads/flycapture2-2.13.3.31-amd64-pkg_xenial.tgz
        fi
        tar xzvf $HOME/downloads/flycapture2-2.13.3.31-amd64-pkg_xenial.tgz -C $TRAVIS_BUILD_DIR/../
        ls $TRAVIS_BUILD_DIR/../flycapture2-2.13.3.31-amd64/*.deb | while read fName; do ar vx $fName; tar -xvf data.tar.xz; done;
        cp -a usr $TRAVIS_BUILD_DIR/../
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp -a $HOME/build/usr/* /usr/"
    elif [ "$OS" == "linux-x86" ]; then
        if [[ $(find $HOME/downloads/flycapture2-2.13.3.31-i386-pkg_xenial.tgz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found flycap32 in cache and size seems ok" 
        else
          echo "Downloading flycap32 as not found in cache or too small" 
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 1BOpSik1Fndagzjf4ykwzermt2qlTzsWI $HOME/downloads/flycapture2-2.13.3.31-i386-pkg_xenial.tgz
        fi
        tar xzvf $HOME/downloads/flycapture2-2.13.3.31-i386-pkg_xenial.tgz -C $TRAVIS_BUILD_DIR/../
        ls $TRAVIS_BUILD_DIR/../flycapture2-2.13.3.31-i386/*.deb | while read fName; do ar vx $fName; tar -xvf data.tar.xz; done;
        cp -a usr $TRAVIS_BUILD_DIR/../
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp -a $HOME/build/usr/* /usr/"
    fi 
  fi 
  if [[ "$PROJ" =~ spinnaker ]]; then
    if [ "$OS" == "linux-x86_64" ]; then
        if [[ $(find $HOME/downloads/spinnaker-1.27.0.48-Ubuntu16.04-amd64-pkg.tar.gz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found spinnaker in cache and size seems ok"
        else
          echo "Downloading spinnaker as not found in cache or too small"
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 1DiI8mG1fyewEGWxW7keg-dqXk0rLwmAK $HOME/downloads/spinnaker-1.27.0.48-Ubuntu16.04-amd64-pkg.tar.gz
        fi
        tar xzvf $HOME/downloads/spinnaker-1.27.0.48-Ubuntu16.04-amd64-pkg.tar.gz -C $TRAVIS_BUILD_DIR/../
        ls $TRAVIS_BUILD_DIR/../spinnaker-1.27.0.48-amd64/*.deb | while read fName; do ar vx $fName; tar -xvf data.tar.xz; done;
        ln -s libSpinnaker_C.so.1.27.0.48 usr/lib/libSpinnaker_C.so.1
        ln -s libSpinnaker.so.1.27.0.48 usr/lib/libSpinnaker.so.1
        cp -a usr $TRAVIS_BUILD_DIR/../
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp -a $HOME/build/usr/* /usr/"
    fi
  fi
  if [[ "$PROJ" == "mkl" ]] && [[ "$OS" =~ linux ]]; then
         #don't put in download dir as will be cached and we can use direct url instead
         curl -L http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/16903/l_mkl_2020.3.279.tgz -o $HOME/mkl.tgz
         tar xzvf $HOME/mkl.tgz -C $TRAVIS_BUILD_DIR/../
         sed -i -e 's/decline/accept/g' $TRAVIS_BUILD_DIR/../l_mkl_2020.3.279/silent.cfg
         docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "$HOME/build/l_mkl_2020.3.279/install.sh -s $HOME/build/l_mkl_2020.3.279/silent.cfg"
  fi
  if [ "$PROJ" == "tensorflow" ]; then
        echo "adding bazel for tensorflow"
        curl -L https://github.com/bazelbuild/bazel/releases/download/0.25.3/bazel-0.25.3-installer-linux-x86_64.sh -o $HOME/downloads/bazel.sh; export CURL_STATUS=$?
        if [ "$CURL_STATUS" != "0" ]; then
          echo "Download failed here, so can't proceed with the build.. Failing.."
          exit 1  
        fi
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "bash $HOME/downloads/bazel.sh"
        export TEST_TMPDIR=$HOME/.cache/bazel
        echo "export TEST_TMPDIR=$HOME/.cache/bazel" | tee --append $HOME/vars.list
  fi
  if [[ "$PROJ" =~ cuda ]] || [[ "$PROJ" == "tensorrt" ]] || [[ "$EXT" =~ gpu ]]; then
        echo "installing cuda, cudnn, and nccl.."
        curl -L https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-rhel7-11-0-local-11.0.3_450.51.06-1.x86_64.rpm -o $HOME/cuda-repo-rhel7-11-0-local-11.0.3_450.51.06-1.x86_64.rpm
        curl -L https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.3/cudnn-11.0-linux-x64-v8.0.3.33.tgz -o $HOME/cudnn-11.0-linux-x64-v8.0.3.33.tgz
        curl -L https://developer.download.nvidia.com/compute/redist/nccl/v2.7/nccl_2.7.8-1+cuda11.0_x86_64.txz -o $HOME/nccl_x86_64.txz

        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "rpm -i $HOME/cuda-repo-rhel7-11-0-local-11.0.3_450.51.06-1.x86_64.rpm"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd /var/cuda-repo-rhel7-11-0-local/; rpm -i --nodeps cuda*.rpm libc*.rpm libn*.rpm"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/cuda/lib64/libnvidia-ml.so"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar hxvf $HOME/cudnn-11.0-linux-x64-v8.0.3.33.tgz -C /usr/local/"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar hxvf $HOME/nccl_x86_64.txz --strip-components=1 -C /usr/local/cuda/"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mv /usr/local/cuda/lib/* /usr/local/cuda/lib64/"
        # work around issues with CUDA 10.2/11.0
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mv /usr/include/cublas* /usr/include/nvblas* /usr/local/cuda/include/"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mv /usr/lib64/libcublas* /usr/lib64/libnvblas* /usr/local/cuda/lib64/"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "for f in /usr/local/cuda/lib64/*.so.10; do ln -s \$f \$f.2; done"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "for f in /usr/local/cuda/lib64/*.so.10; do ln -s \$f \${f:0:-1}1; done"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so; cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so.1"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/lib64/libnvidia-ml.so; cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libnvidia-ml.so.1"

        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd $HOME/ccache-3.7.10/; ./configure; make; make install"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "echo 'CCACHE_CC=/usr/local/cuda/bin/nvcc /usr/local/bin/ccache compiler \"\$@\"' > /usr/local/cuda/bin/nvcccache"
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "chmod 755 /usr/local/cuda/bin/nvcccache"
  fi
  if [[ "$PROJ" == "tensorrt" ]] || [[ "$EXT" =~ gpu ]]; then
        python $TRAVIS_BUILD_DIR/ci/gDownload.py 1xHWJHMNJ6g29YnMNeFKwJZF1KdRl_N3e $HOME/downloads/tensorrt.tar.gz
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar hxvf $HOME/downloads/tensorrt.tar.gz -C /usr/local/; ln -sf /usr/local/TensorRT* /usr/local/tensorrt"
  fi
fi

if [ "$OS" == "linux-armhf" ]; then
    echo "Setting up tools for linux-armhf build"
    sudo dpkg --add-architecture i386
    sudo apt-get update
    sudo apt-get -y install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1
    sudo apt-get -y install ccache clang git file wget unzip tar bzip2 gzip patch autoconf-archive autogen automake libtool bison flex perl nasm libasound2-dev freeglut3-dev libgtk2.0-dev libusb-dev libffi-dev zlib1g-dev
    curl -L https://github.com/raspberrypi/tools/archive/master.tar.gz -o $HOME/tools-master.tar.gz
    curl -L https://github.com/raspberrypi/userland/archive/master.tar.gz -o $HOME/userland-master.tar.gz
    mkdir -p $HOME/tools $HOME/userland
    tar xzf $HOME/tools-master.tar.gz --strip-components=1 -C $HOME/tools
    tar xzf $HOME/userland-master.tar.gz --strip-components=1 -C $HOME/userland
    ln -s $HOME/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian/bin/arm-linux-gnueabihf-gfortran $HOME/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/
    ln -s $HOME/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian/arm-linux-gnueabihf/lib/libgfortran* $HOME/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/lib/
    export PATH=$PATH:$HOME/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/
    export BUILD_COMPILER=-Djavacpp.platform.compiler=arm-linux-gnueabihf-g++
    export BUILD_OPTIONS=-Djava.library.path=$HOME/tools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/lib/
    pushd $HOME/userland
    # https://github.com/raspberrypi/userland/issues/603
    sed -i -e 's/"${INSTALL_TARGETS} vcilcs"/${INSTALL_TARGETS} vcilcs/g' interface/vmcs_host/CMakeLists.txt
    bash buildme
    popd

    if [[ "$PROJ" =~ flycapture ]]; then
        if [[ $(find $HOME/downloads/flycapture.2.13.3.31_armhf.tar.gz -type f -size +1000000c 2>/dev/null) ]]; then
            echo "Found flycap-armhf in cache and size seems ok" 
        else
            echo "Downloading flycap-armhf as not found in cache or too small" 
            python $TRAVIS_BUILD_DIR/ci/gDownload.py 16NuUBs2MXQpVYqzDCEr9KdMng-6rHuDI $HOME/downloads/flycapture.2.13.3.31_armhf.tar.gz
        fi
        cp $HOME/downloads/flycapture.2.13.3.31_armhf.tar.gz $TRAVIS_BUILD_DIR/downloads/
    fi
fi

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
   # get rid of some stuff we don't use to avoid running out of disk space
   sudo rm -Rf /Library/Developer/CoreSimulator/*

   echo "performing brew update and install of dependencies, please wait.."
   brew update
   brew upgrade cmake
   brew install ccache curl swig autoconf-archive libomp libtool libusb xz sdl gpg1 bison flex perl nasm ragel

   brew tap AdoptOpenJDK/openjdk
   brew cask install adoptopenjdk8
   export JAVA_HOME=$(/usr/libexec/java_home -v1.8)

   # Try to use ccache to speed up the build and work around issue with Sectigo CA root certificate
   export PATH=/usr/local/opt/ccache/libexec/:/usr/local/opt/curl/bin/:/usr/local/opt/gpg1/libexec/gpgbin/:/usr/local/opt/bison/bin/:/usr/local/opt/flex/bin/:$PATH

   if [[ "$PROJ" =~ arpack-ng ]] || [[ "$PROJ" =~ cminpack ]] || [[ "$PROJ" =~ mkl-dnn ]] || [[ "$PROJ" =~ openblas ]] || [[ "$PROJ" =~ scipy ]]; then
       brew install gcc@7
       brew link --overwrite gcc@7

       # Remove "fixed" header files that are actually broken
       sudo rm -Rf $(find /usr/local/Cellar/gcc@7/ -iname include-fixed)

       # Fix up some binaries to support rpath
       sudo install_name_tool -add_rpath /usr/local/lib/gcc/7/ -add_rpath @loader_path/. -id @rpath/libgomp.1.dylib /usr/local/lib/gcc/7/libgomp.1.dylib
       sudo install_name_tool -add_rpath /usr/local/lib/gcc/7/ -add_rpath @loader_path/. -id @rpath/libstdc++.6.dylib /usr/local/lib/gcc/7/libstdc++.6.dylib
       sudo install_name_tool -add_rpath /usr/local/lib/gcc/7/ -add_rpath @loader_path/. -id @rpath/libgfortran.4.dylib /usr/local/lib/gcc/7/libgfortran.4.dylib
       sudo install_name_tool -add_rpath /usr/local/lib/gcc/7/ -add_rpath @loader_path/. -id @rpath/libquadmath.0.dylib /usr/local/lib/gcc/7/libquadmath.0.dylib
       sudo install_name_tool -add_rpath /usr/local/lib/gcc/7/ -add_rpath @loader_path/. -id @rpath/libgcc_s.1.dylib /usr/local/lib/gcc/7/libgcc_s.1.dylib
       sudo install_name_tool -change /usr/local/Cellar/gcc@7/7.5.0_2/lib/gcc/7/libquadmath.0.dylib @rpath/libquadmath.0.dylib /usr/local/lib/gcc/7/libgfortran.4.dylib
       sudo install_name_tool -change /usr/local/lib/gcc/7/libgcc_s.1.dylib @rpath/libgcc_s.1.dylib /usr/local/lib/gcc/7/libgomp.1.dylib
       sudo install_name_tool -change /usr/local/lib/gcc/7/libgcc_s.1.dylib @rpath/libgcc_s.1.dylib /usr/local/lib/gcc/7/libstdc++.6.dylib
       sudo install_name_tool -change /usr/local/lib/gcc/7/libgcc_s.1.dylib @rpath/libgcc_s.1.dylib /usr/local/lib/gcc/7/libgfortran.4.dylib
       sudo install_name_tool -change /usr/local/lib/gcc/7/libgcc_s.1.dylib @rpath/libgcc_s.1.dylib /usr/local/lib/gcc/7/libquadmath.0.dylib
   fi

   sudo install_name_tool -add_rpath @loader_path/. -id @rpath/libomp.dylib /usr/local/opt/libomp/lib/libomp.dylib
   sudo install_name_tool -add_rpath @loader_path/. -id @rpath/libSDL-1.2.0.dylib /usr/local/lib/libSDL-1.2.0.dylib
   sudo install_name_tool -add_rpath @loader_path/. -id @rpath/libusb-1.0.0.dylib /usr/local/lib/libusb-1.0.0.dylib

   mvn -version
   /usr/local/bin/gcc-? --version
   gpg --version

fi

if [[ "$OS" =~ android ]]; then
   echo "Install android requirements.."
   DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')

   curl -L https://dl.google.com/android/repository/android-ndk-r18b-linux-x86_64.zip -o $HOME/ndk.zip; export CURL_STATUS=$?
   if [ "$CURL_STATUS" != "0" ]; then
    echo "Download failed here, so can't proceed with the build.. Failing.."
    exit 1
   fi

   unzip -qq $HOME/ndk.zip -d $HOME/
   ln -sf $HOME/android-ndk-r18b $HOME/android-ndk
   echo "Android NDK setup done"
   echo "export ANDROID_NDK=$HOME/android-ndk/" | tee --append $HOME/vars.list
   cat $HOME/vars.list
   echo "export BUILD_COMPILER=-Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++" | tee --append $HOME/vars.list
   echo "export BUILD_OPTIONS=-Djava.library.path=" | tee --append $HOME/vars.list
   echo "export BUILD_ROOT=-Djavacpp.platform.root=$HOME/android-ndk/" | tee --append $HOME/vars.list
fi


echo "Download dependencies" 
if [ "$TRAVIS_OS_NAME" == "osx" ]; then

      if [[ "arrow mxnet tensorflow onnx ngraph onnxruntime " =~ "$PROJ " ]]; then
        curl -L https://www.python.org/ftp/python/3.6.6/python-3.6.6-macosx10.9.pkg -o $HOME/python.pkg
        echo "Install python pkg"
        sudo installer -store -pkg $HOME/python.pkg -target /
      fi

      if [ "$PROJ" == "mkl" ]; then
        #don't put in download dir as will be cached and we can use direct url instead
        curl -L http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/16902/m_mkl_2020.3.279.dmg -o $HOME/mkl.dmg
        echo "Mount mkl dmg"
        hdiutil mount $HOME/mkl.dmg
        sleep 10
        cp /Volumes/m_mkl_2020.3.279/m_mkl_2020.3.279.app/Contents/MacOS/silent.cfg $HOME/silent.cfg
        sed -i -e 's/decline/accept/g' $HOME/silent.cfg
        sudo /Volumes/m_mkl_2020.3.279/m_mkl_2020.3.279.app/Contents/MacOS/install.sh -s $HOME/silent.cfg; export BREW_STATUS=$?
        echo "mkl status $BREW_STATUS"
        if [ $BREW_STATUS -ne 0 ]; then
          echo "mkl Failed"
          exit $BREW_STATUS
        fi
      fi

      if [[ "$PROJ" =~ cuda ]] || [[ "$EXT" =~ gpu ]]; then
        echo "installing cuda.."
        #don't put in download dir as will be cached and we can use direct url instead
        curl -L http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_mac.dmg -o $HOME/cuda_10.2.89_mac.dmg
        curl -L https://developer.download.nvidia.com/compute/redist/cudnn/v7.6.5/cudnn-10.1-osx-x64-v7.6.5.32.tgz -o $HOME/cudnn-10.1-osx-x64-v7.6.5.32.tgz

        echo "Mount dmg"
        hdiutil mount $HOME/cuda_10.2.89_mac.dmg
        sleep 5
        ls -ltr /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS 
        sudo /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS/CUDAMacOSXInstaller --accept-eula --no-window; export BREW_STATUS=$? 
        echo "Brew status $BREW_STATUS"
        if [ $BREW_STATUS -ne 0 ]; then
          echo "Brew Failed"
          exit $BREW_STATUS
        fi

        tar xvf $HOME/cudnn-10.1-osx-x64-v7.6.5.32.tgz
        sudo cp ./cuda/include/*.h /usr/local/cuda/include/
        sudo cp ./cuda/lib/*.dylib /usr/local/cuda/lib/
        sudo cp ./cuda/lib/*.a /usr/local/cuda/lib/
        sudo cp /usr/local/cuda/lib/* /usr/local/lib/
        sudo cp /usr/local/cuda/lib/libcuda.dylib /usr/local/cuda/lib/stubs/

        # work around issues with CUDA 10.2
        for f in /usr/local/cuda/lib/*.10.dylib; do sudo ln -s $f ${f/%.10.dylib/.10.2.dylib}; done

        cd $HOME/ccache-3.7.10/; ./configure; make; sudo make install; cd $TRAVIS_BUILD_DIR
        echo 'CCACHE_CC=/usr/local/cuda/bin/nvcc /usr/local/bin/ccache compiler "$@"' | sudo tee /usr/local/cuda/bin/nvcccache
        sudo chmod 755 /usr/local/cuda/bin/nvcccache
      fi

      if [ "$PROJ" == "tensorflow" ]; then
        echo "adding bazel for tensorflow"
        curl -L https://github.com/bazelbuild/bazel/releases/download/0.25.3/bazel-0.25.3-installer-darwin-x86_64.sh -o $HOME/bazel.sh; export CURL_STATUS=$?
        if [ "$CURL_STATUS" != "0" ]; then
          echo "Download failed here, so can't proceed with the build.. Failing.."
          exit 1
        fi
        sudo bash $HOME/bazel.sh
        export TEST_TMPDIR=$HOME/.cache/bazel
     fi
fi  



echo "Running install for $PROJ"
if  [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]] || [[ "$OS" =~ android ]]; then
   DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
   echo "container id is $DOCKER_CONTAINER_ID"
    if [ "$1" == "nodeploy" ]; then
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "source scl_source enable $SCL_ENABLE || true; . $HOME/vars.list; cd $HOME/build/javacpp-presets; bash cppbuild.sh install $PROJ -platform=$OS -extension=$EXT"; export BUILD_STATUS=0
    elif [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
       echo "Not a pull request so attempting to deploy using docker"
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "source scl_source enable $SCL_ENABLE || true; . $HOME/vars.list; cd $HOME/build/javacpp-presets; mvn clean deploy -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings ./ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE \$BUILD_COMPILER \$BUILD_OPTIONS \$BUILD_ROOT -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT -pl .,$PROJ"; export BUILD_STATUS=$?
       if [ $BUILD_STATUS -eq 0 ]; then
         echo "Deploying platform"
         for i in ${PROJ//,/ }
         do
          docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "source scl_source enable $SCL_ENABLE || true; . $HOME/vars.list; cd $HOME/build/javacpp-presets/$i; mvn clean deploy -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings ../ci/settings.xml -f platform/${EXT:1}/pom.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT "; export BUILD_STATUS=$?
          if [ $BUILD_STATUS -ne 0 ]; then
           echo "Build Failed"
           exit $BUILD_STATUS
          fi
         done
       fi
        
     else
       echo "Pull request so install using docker"
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "source scl_source enable $SCL_ENABLE || true; . $HOME/vars.list; cd $HOME/build/javacpp-presets; mvn clean install -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings ./ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE \$BUILD_COMPILER \$BUILD_OPTIONS \$BUILD_ROOT -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT -pl .,$PROJ"; export BUILD_STATUS=$?
    fi

   echo "Build status $BUILD_STATUS"
   if [ $BUILD_STATUS -ne 0 ]; then  
     echo "Build Failed"
     exit $BUILD_STATUS
   fi

else	
   echo "Building $PROJ, with additional build flags $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT"
   if [ "$1" == "nodeploy" ]; then
      bash cppbuild.sh install $PROJ -platform=$OS -extension=$EXT; export BUILD_STATUS=0
   elif [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
      echo "Not a pull request so attempting to deploy"
      mvn clean deploy -B -U -Dmaven.repo.local=$HOME/.m2/repository  --settings $TRAVIS_BUILD_DIR/ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT -pl .,$PROJ; export BUILD_STATUS=$?
      if [ $BUILD_STATUS -eq 0 ]; then
        echo "Deploying platform step"
        for i in ${PROJ//,/ }
        do
	  cd $i
          mvn clean deploy -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings $TRAVIS_BUILD_DIR/ci/settings.xml -f platform/${EXT:1}/pom.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT; export BUILD_STATUS=$?
          cd ..
        done
      fi
    else
      echo "Pull request so install only"
      mvn clean install -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings $TRAVIS_BUILD_DIR/ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT -pl .,$PROJ; export BUILD_STATUS=$?
   fi

   echo "Build status $BUILD_STATUS"
   if [ $BUILD_STATUS -ne 0 ]; then
     echo "Build Failed"
     #echo "Dump of config.log output files found follows:"
     #find . -name config.log | xargs cat
     exit $BUILD_STATUS
   fi
fi


#finally, shutdown any container used for docker builds
if [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]]; then
   DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
   docker stop $DOCKER_CONTAINER_ID
   docker rm -v $DOCKER_CONTAINER_ID
fi

sudo chown -R travis:travis $HOME
du -csh $HOME/* $HOME/.m2/* $HOME/.cache/* $HOME/.ccache/* $HOME/downloads/*
exit 0

