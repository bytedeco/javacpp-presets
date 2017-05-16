#!/bin/bash 
set -vx
export

mkdir ./buildlogs
ls -ltr $HOME/downloads
ls -ltr $HOME/.m2
pip install requests
git clone https://github.com/bytedeco/javacpp.git
cd javacpp
mvn install -l javacppBuild.log -Dmaven.test.skip=true -Dmaven.javadoc.skip=true
cd ..
export PYTHON_BIN_PATH=$(which python) # For tensorflow
if [ "$TRAVIS_OS_NAME" == "osx" ]; then export JAVA_HOME=$(/usr/libexec/java_home); fi
  if [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]]; then
    if [ "$OS" == "linux-x86_64" ]; then
      echo "starting docker"
      docker run -d -ti -e CI_DEPLOY_USERNAME -e CI_DEPLOY_PASSWORD -e "container=docker" -v $HOME/.m2:/root/.m2 -v $HOME/downloads:/root/downloads -v $TRAVIS_BUILD_DIR/../:/root/build -v /sys/fs/cgroup:/sys/fs/cgroup nvidia/cuda:8.0-cudnn6-devel-centos7 /usr/sbin/init > /dev/null
      DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
      echo "container id is $DOCKER_CONTAINER_ID please wait while updates applied"
      docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -y install epel-release" > /dev/null
      docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -y install clang gcc-c++ gcc-gfortran java-devel maven python numpy swig git file which wget unzip tar bzip2 gzip xz patch make cmake3 libtool perl nasm yasm alsa-lib-devel freeglut-devel glfw-devel gtk2-devel libusb-devel libusb1-devel zlib-devel openblas-devel" > /dev/null
      if [ "$PROJ" == "flycapture" ]; then
        if [[ $(find $HOME/downloads/flycap.tar.gz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found flycap in cache and size seems ok" 
        else
          echo "Downloading flycap as not found in cache or too small" 
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShS1I1MzN0SmQ1MHc $HOME/downloads/flycap.tar.gz 
        fi
          tar xzvf $HOME/downloads/flycap.tar.gz -C $TRAVIS_BUILD_DIR/../
          docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp -R /root/build/include/* /usr/include; cp -R /root/build/lib/* /usr/lib" 
      fi 
      if [ "$PROJ" == "mkl" ]; then
         #don't put in download dir as will be cached and we can use direct url instead
         curl -L http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/11306/l_mkl_2017.2.174.tgz -o $HOME/mkl.tgz
         tar xzvf $HOME/mkl.tgz -C $TRAVIS_BUILD_DIR/../
         sed -i -e 's/decline/accept/g' $TRAVIS_BUILD_DIR/../l_mkl_2017.2.174/silent.cfg
         docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "/root/build/l_mkl_2017.2.174/install.sh -s /root/build/l_mkl_2017.2.174/silent.cfg"
      fi
      if [ "$PROJ" == "tensorflow" ]; then
        echo "adding bazel for tensorflow"
        curl -L https://github.com/bazelbuild/bazel/releases/download/0.4.4/bazel-0.4.4-installer-linux-x86_64.sh -o $HOME/downloads/bazel.sh; export CURL_STATUS=$?
        if [ "$CURL_STATUS" != "0" ]; then
          echo "Download failed here, so can't proceed with the build.. Failing.."
          return 1
        fi
         docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "bash /root/downloads/bazel.sh"
      fi 
  fi
fi

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
   echo "performing brew update and install of dependencies, please wait.."
   brew update > /dev/null
   brew install gcc5 swig bazel libtool libusb nasm yasm xz 
   ln -s /usr/local/opt/gcc\@5 /usr/local/opt/gcc5
 fi

if [[ "$OS" =~ android ]]; then
   echo "Install android requirements.."
   #sudo apt-get install yasm nasm
   pip install numpy
   curl -L https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip -o $HOME/ndk.zip; export CURL_STATUS=$?
   if [ "$CURL_STATUS" != "0" ]; then
    echo "Download failed here, so can't proceed with the build.. Failing.."
    return 1
   fi

   unzip -qq $HOME/ndk.zip -d $HOME/
   ln -s $HOME/android-ndk-r14b $HOME/android-ndk
   echo "Android NDK setup done"
   if [ "$OS" == "android-arm" ]; then
      echo "Setting build for android-arm"
      export ANDROID_NDK=$HOME/android-ndk/
      export PATH=$PATH:$HOME/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin
      export ANDROID_FLAGS="-Djavacpp.platform.root=$HOME/android-ndk/ -Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-g++"
   fi
   if [ "$OS" == "android-x86" ]; then
      echo "Setting build for android-x86"
      export ANDROID_NDK=$HOME/android-ndk/
      export ANDROID_FLAGS="-Djavacpp.platform.root=$HOME/android-ndk/ -Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/x86-4.9/prebuilt/linux-x86_64/bin/i686-linux-android-g++"
   fi
   if [ "$PROJ" == "tensorflow" ]; then
      echo "adding bazel for tensorflow"
      curl -L  https://github.com/bazelbuild/bazel/releases/download/0.4.4/bazel-0.4.4-installer-linux-x86_64.sh -o bazel.sh; export CURL_STATUS=$?
      if [ "$CURL_STATUS" != "0" ]; then
        echo "Download failed here, so can't proceed with the build.. Failing.."
        return 1
      fi
      sudo bash bazel.sh
   fi
fi

