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

    echo "download dependencies" 
    if [ "$TRAVIS_OS_NAME" == "osx" ]; then
      if [[ "$PROJ" =~ mxnet ]]; then 
        export PKG_CONFIG_PATH=$TRAVIS_BUILD_DIR/opencv/cppbuild/macosx-x86_64/lib/pkgconfig
      fi 
      if [[ "$PROJ" =~ cuda ]] || [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ caffe ]]; then 
        echo "installing cuda.."
        while true; do echo .; sleep 60; done &
        export CHILDPID=$!
        echo "Child PID $CHILDPID"
        python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShUzFIUHhkQnBQWWM cuda.dmg
        echo "Mount dmg"
        hdiutil mount cuda.dmg
        sleep 5
        ls -ltr /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS 
        sudo /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS/CUDAMacOSXInstaller --accept-eula --no-window; export BREW_STATUS=$? 
        echo "Brew status $BREW_STATUS"
        if [ $BREW_STATUS -ne 0 ]; then
          echo "Brew Failed"
          return $BREW_STATUS
        fi
        kill $CHILDPID

        if [[ $(find $HOME/downloads/cudnn-8.0-osx-x64-v6.0.tgz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found cudnn in cache and size seems ok" 
        else
          echo "Downloading cudnn as not found in cache" 
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShc2JlWXNjNTlIVnc $HOME/downloads/cudnn-8.0-osx-x64-v6.0.tgz
        fi
        tar xvf $HOME/downloads/cudnn-8.0-osx-x64-v6.0.tgz
        sudo cp ./cuda/include/cudnn.h /usr/local/cuda/include/cudnn.h
        sudo cp ./cuda/lib/libcudnn.6.dylib /usr/local/cuda/lib/libcudnn.6.dylib
        sudo cp ./cuda/lib/libcudnn.dylib /usr/local/cuda/lib/libcudnn.dylib
        sudo cp ./cuda/lib/libcudnn_static.a /usr/local/cuda/lib/libcudnn_static.a
      fi  
    fi  
    echo "starting script"
    echo "running for $PROJ"
    if  [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]]; then
      DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
      echo "container id is $DOCKER_CONTAINER_ID"
      if [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ openblas ]]; then
        echo "redirecting log output, tailing log every 5 mins to prevent timeout.."
        while true; do echo .; docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tail -10 /root/build/javacpp-presets/buildlogs/$PROJ.log"; sleep 300; done &
        if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
          echo "Not a pull request so attempting to deploy"
          docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd /root/build/javacpp-presets;mvn deploy -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS -l /root/build/javacpp-presets/buildlogs/$PROJ.log -pl $PROJ"; export BUILD_STATUS=$?
        else
          echo "Pull request so install only"
          docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd /root/build/javacpp-presets;mvn install -Djavacpp.copyResources -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS -l /root/build/javacpp-presets/buildlogs/$PROJ.log -pl $PROJ"; export BUILD_STATUS=$?
        fi
      else
        if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
          echo "Not a pull request so attempting to deploy"
          docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd /root/build/javacpp-presets;mvn deploy -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS -pl $PROJ"; export BUILD_STATUS=$?
        else
          echo "Pull request so install only"
          docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd /root/build/javacpp-presets;mvn install -Djavacpp.copyResources -Dmaven.test.skip=true -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS -pl $PROJ"; export BUILD_STATUS=$?
        fi
      fi
      echo "Build status $BUILD_STATUS"
      if [ $BUILD_STATUS -ne 0 ]; then  
        echo "Build Failed"
        return $BUILD_STATUS
      fi
    else	
     if [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ openblas ]]; then
       echo "redirecting log output, tailing log every 5 mins to prevent timeout.."
       while true; do echo .; tail -10 $TRAVIS_BUILD_DIR/buildlogs/$PROJ.log; sleep 300; done &
       if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
         echo "Not a pull request so attempting to deploy"
         #sed -i -e 's/sonatype-nexus-snapshots/intNexus/g' pom.xml
         #sed -i -e 's|https://oss.sonatype.org/content/repositories/snapshots/|http://bytedeconexus.ddns.net:15081/nexus/content/repositories/bytedecoInt/|g' pom.xml
         mvn deploy -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS $ANDROID_FLAGS -l $TRAVIS_BUILD_DIR/buildlogs/$PROJ.log -pl $PROJ; export BUILD_STATUS=$?
       else
         echo "Pull request so install only"
         mvn install -Dmaven.javadoc.skip=true -Djavacpp.copyResources -Djavacpp.platform=$OS $ANDROID_FLAGS -l $TRAVIS_BUILD_DIR/buildlogs/$PROJ.log -pl $PROJ; export BUILD_STATUS=$?
       fi
     else
       echo "Building $PROJ"
       echo $ANDROID_FLAGS
       if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
         echo "Not a pull request so attempting to deploy"
         #sed -i -e 's/sonatype-nexus-snapshots/intNexus/g' pom.xml
         #sed -i -e 's|https://oss.sonatype.org/content/repositories/snapshots/|http://bytedeconexus.ddns.net:15081/nexus/content/repositories/bytedecoInt/|g' pom.xml
         mvn deploy --settings ./ci/settings.xml -Djavacpp.copyResources -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS $ANDROID_FLAGS -pl $PROJ; export BUILD_STATUS=$?
       else
         echo "Pull request so install only"
         mvn install -Dmaven.javadoc.skip=true -Djavacpp.copyResources -Djavacpp.platform=$OS $ANDROID_FLAGS -pl $PROJ; export BUILD_STATUS=$?
       fi
     fi
      echo "Build status $BUILD_STATUS"
      if [ $BUILD_STATUS -ne 0 ]; then
        echo "Build Failed"
        return $BUILD_STATUS
      fi
    fi
    if [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]]; then
      DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
      docker stop $DOCKER_CONTAINER_ID
      docker rm -v $DOCKER_CONTAINER_ID
    fi

