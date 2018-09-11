#!/bin/bash 
set -vx

while true; do echo .; sleep 60; done &

sudo fallocate -l 4GB /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

mkdir ./buildlogs
mkdir $TRAVIS_BUILD_DIR/downloads
ls -ltr $HOME/downloads
ls -ltr $HOME/.m2
curl https://bootstrap.pypa.io/get-pip.py | sudo python
sudo pip install requests
export PYTHON_BIN_PATH=$(which python) # For tensorflow
touch $HOME/vars.list

export MAKEJ=2
echo "export MAKEJ=2" | tee --append $HOME/vars.list

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
  SCL_ENABLE="devtoolset-6 python27"
  if [[ "librealsense cpython mxnet tensorflow onnx skia " =~ "$PROJ " ]] || [[ "$OS" =~ android ]]; then
    CENTOS_VERSION=7
    SCL_ENABLE="python36-devel python36-setuptools"
  fi
  echo "Starting docker for x86_64 and x86 linux"
  docker run -d -ti -e CI_DEPLOY_USERNAME -e CI_DEPLOY_PASSWORD -e GPG_PASSPHRASE -e STAGING_REPOSITORY -e "container=docker" -v $HOME:$HOME -v $TRAVIS_BUILD_DIR/../:$HOME/build -v /sys/fs/cgroup:/sys/fs/cgroup nvidia/cuda:9.2-cudnn7-devel-centos$CENTOS_VERSION /bin/bash
  DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
  echo "Container id is $DOCKER_CONTAINER_ID please wait while updates applied"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -q -y --disablerepo=cuda install centos-release-scl-rh epel-release"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -q -y --disablerepo=cuda install rh-java-common-ant $SCL_ENABLE clang gcc-c++ gcc-gfortran java-1.8.0-openjdk-devel ant maven python numpy swig git file which wget unzip tar bzip2 gzip xz patch make cmake3 autoconf-archive libtool perl nasm yasm alsa-lib-devel freeglut-devel gtk2-devel libusb-devel libusb1-devel zlib-devel SDL-devel libva-devel"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -y --disablerepo=cuda update"
  if [ "$OS" == "linux-x86" ]; then
    docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "curl -L https://copr.fedorainfracloud.org/coprs/mlampe/devtoolset-6.1/repo/epel-$CENTOS_VERSION/mlampe-devtoolset-6.1-epel-$CENTOS_VERSION.repo -o /etc/yum.repos.d/mlampe-devtoolset-6.1-epel-$CENTOS_VERSION.repo"
    docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "sed -i 's/\$basearch/i386/g' /etc/yum.repos.d/mlampe-devtoolset-6.1-epel-$CENTOS_VERSION.repo"
    docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "rpm -qa | sed s/.x86_64$/.i686/ | xargs yum -q -y --disablerepo=cuda install"
    docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "find /var/cache/yum/ -name *.rpm | xargs rpm -i --force"
  fi
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so; cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib64/libcuda.so.1"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "source scl_source enable $SCL_ENABLE || true; gcc --version"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "gpg --version"

  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "curl -L https://archive.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz -o $HOME/apache-maven-3.3.9-bin.tar.gz"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xzf $HOME/apache-maven-3.3.9-bin.tar.gz -C /opt/"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /usr/bin/python3.6 /usr/bin/python3"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /opt/apache-maven-3.3.9/bin/mvn /usr/bin/mvn"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mvn -version"

  if [ "$PROJ" == "flycapture" ]; then
    if [ "$OS" == "linux-x86_64" ]; then
        if [[ $(find $HOME/downloads/flycap.tar.gz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found flycap64 in cache and size seems ok" 
        else
          echo "Downloading flycap64 as not found in cache or too small" 
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShS1I1MzN0SmQ1MHc $HOME/downloads/flycap.tar.gz 
        fi
        tar xzvf $HOME/downloads/flycap.tar.gz -C $TRAVIS_BUILD_DIR/../
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp -R $HOME/build/include/* /usr/include; cp -R $HOME/build/lib/* /usr/lib" 
    elif [ "$OS" == "linux-x86" ]; then
        if [[ $(find $HOME/downloads/flycaplinux32.tar.gz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found flycap32 in cache and size seems ok" 
        else
          echo "Downloading flycap32 as not found in cache or too small" 
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShaDhTN1FOUTE3UkE $HOME/downloads/flycaplinux32.tar.gz 
        fi
        tar xzvf $HOME/downloads/flycaplinux32.tar.gz -C $TRAVIS_BUILD_DIR/../
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cp -R $HOME/build/include/* /usr/include; cp -R $HOME/build/lib/* /usr/lib" 
    fi 
  fi 
  if [ "$PROJ" == "spinnaker" ]; then
    if [ "$OS" == "linux-x86_64" ]; then
        if [[ $(find $HOME/downloads/spinnaker_local_v.1.15.0.63.tar.gz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found spinnaker in cache and size seems ok"
        else
          echo "Downloading flycap64 as not found in cache or too small"
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 1IYtvqzpNHJgZK-TPztW_WDYuDEyo56D_ $HOME/downloads/spinnaker_local_v.1.15.0.63.tar.gz
        fi
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xvf $HOME/downloads/spinnaker_local_v.1.15.0.63.tar.gz -C /"
    fi
  fi
  if [[ "$PROJ" == "mkl" ]] && [[ "$OS" =~ linux ]]; then
         #don't put in download dir as will be cached and we can use direct url instead
         curl -L http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/13005/l_mkl_2018.3.222.tgz -o $HOME/mkl.tgz
         tar xzvf $HOME/mkl.tgz -C $TRAVIS_BUILD_DIR/../
         sed -i -e 's/decline/accept/g' $TRAVIS_BUILD_DIR/../l_mkl_2018.3.222/silent.cfg
         docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "$HOME/build/l_mkl_2018.3.222/install.sh -s $HOME/build/l_mkl_2018.3.222/silent.cfg"
  fi
  if [ "$PROJ" == "tensorflow" ]; then
        echo "adding bazel for tensorflow"
        curl -L https://github.com/bazelbuild/bazel/releases/download/0.15.2/bazel-0.15.2-installer-linux-x86_64.sh -o $HOME/downloads/bazel.sh; export CURL_STATUS=$?
        if [ "$CURL_STATUS" != "0" ]; then
          echo "Download failed here, so can't proceed with the build.. Failing.."
          exit 1  
        fi
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "bash $HOME/downloads/bazel.sh"
  fi
  if [ "$PROJ" == "tensorrt" ]; then
        python $TRAVIS_BUILD_DIR/ci/gDownload.py 1tznGJWMlq-Q9jJuHDGpYGAe2IbJfpYHs $HOME/downloads/tensorrt.tar.gz
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xvf $HOME/downloads/tensorrt.tar.gz -C /usr/local/; ln -s /usr/local/TensorRT* /usr/local/tensorrt"
  fi
fi

if [ "$OS" == "linux-armhf" ]; then
	echo "Setting up tools for linux-armhf build"
	sudo dpkg --add-architecture i386
	sudo apt-get update
	sudo apt-get -y install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1
	sudo apt-get -y install clang git file wget unzip tar bzip2 gzip patch autoconf-archive autogen automake libtool perl nasm yasm libasound2-dev freeglut3-dev libgtk2.0-dev libusb-dev zlib1g
	git -C $HOME clone https://github.com/raspberrypi/tools
	git -C $HOME clone https://github.com/raspberrypi/userland
	export PATH=$PATH:$HOME/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian/bin
	export BUILD_COMPILER=-Djavacpp.platform.compiler=arm-linux-gnueabihf-g++
	export BUILD_OPTIONS=-Djava.library.path=
	pushd $HOME/userland
	bash buildme
	popd
	if [ "$PROJ" == "flycapture" ]; then
          if [[ $(find $HOME/downloads/flycapture.2.11.3.121_armhf.tar.gz -type f -size +1000000c 2>/dev/null) ]]; then
            echo "Found flycap-armhf in cache and size seems ok" 
          else
            echo "Downloading flycap-armhf as not found in cache or too small" 
            python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShMjVXZFlveXpyWE0 $HOME/downloads/flycapture.2.11.3.121_armhf.tar.gz
          fi
	  cp $HOME/downloads/flycapture.2.11.3.121_armhf.tar.gz $TRAVIS_BUILD_DIR/downloads/ 
        fi

fi

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
   echo "performing brew update and install of dependencies, please wait.."
   brew update
   brew upgrade maven
   brew install gcc@7 swig autoconf-archive libtool libusb nasm yasm xz sdl gpg1
   brew link --overwrite gcc
   export PATH=/usr/local/opt/gpg1/libexec/gpgbin/:$PATH

   mvn -version
   /usr/local/bin/gcc-? --version
   gpg --version

   sudo install_name_tool -add_rpath @loader_path/. -id @rpath/libSDL-1.2.0.dylib /usr/local/lib/libSDL-1.2.0.dylib
   sudo install_name_tool -add_rpath @loader_path/. -id @rpath/libusb-1.0.0.dylib /usr/local/lib/libusb-1.0.0.dylib
fi

if [[ "$OS" =~ android ]]; then
   echo "Install android requirements.."
   DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
   #docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "pip install numpy" 

   curl -L https://dl.google.com/android/repository/android-ndk-r15c-linux-x86_64.zip -o $HOME/ndk.zip; export CURL_STATUS=$?
   if [ "$CURL_STATUS" != "0" ]; then
    echo "Download failed here, so can't proceed with the build.. Failing.."
    exit 1
   fi

   unzip -qq $HOME/ndk.zip -d $HOME/
   ln -s $HOME/android-ndk-r15c $HOME/android-ndk
   if [ "$PROJ" == "tensorflow" ]; then
     echo "modifying ndk version 14 to 12 as per tensorflow cppbuild.sh suggestion"
     sed -i 's/15/12/g' $HOME/android-ndk/source.properties
   fi
   echo "Android NDK setup done"
   echo "export ANDROID_NDK=$HOME/android-ndk/" | tee --append $HOME/vars.list
   cat $HOME/vars.list
   echo "export BUILD_OPTIONS=-Djava.library.path=" | tee --append $HOME/vars.list
   echo "export BUILD_ROOT=-Djavacpp.platform.root=$HOME/android-ndk/" | tee --append $HOME/vars.list
   if [ "$OS" == "android-arm" ]; then
      echo "export PATH=\$PATH:$HOME/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin" | tee --append $HOME/vars.list
      echo "export BUILD_COMPILER=-Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-g++" | tee --append $HOME/vars.list
   fi
   if [ "$OS" == "android-arm64" ]; then
      echo "export PATH=\$PATH:$HOME/android-ndk/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin" | tee --append $HOME/vars.list
      echo "export BUILD_COMPILER=-Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-g++" | tee --append $HOME/vars.list
   fi
   if [ "$OS" == "android-x86" ]; then
      echo "Setting build for android-x86"
      echo "export BUILD_COMPILER=-Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/x86-4.9/prebuilt/linux-x86_64/bin/i686-linux-android-g++" | tee --append $HOME/vars.list
   fi
   if [ "$OS" == "android-x86_64" ]; then
      echo "Setting build for android-x86_64"
      echo "export BUILD_COMPILER=-Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/x86_64-4.9/prebuilt/linux-x86_64/bin/x86_64-linux-android-g++" | tee --append $HOME/vars.list
   fi
   if [ "$PROJ" == "tensorflow" ]; then
      echo "adding bazel for tensorflow"
      curl -L  https://github.com/bazelbuild/bazel/releases/download/0.15.2/bazel-0.15.2-installer-linux-x86_64.sh -o $HOME/bazel.sh; export CURL_STATUS=$?
      if [ "$CURL_STATUS" != "0" ]; then
        echo "Download failed here, so can't proceed with the build.. Failing.."
        exit 1
      fi
      docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "sudo bash $HOME/bazel.sh"
   fi
fi


echo "Download dependencies" 
if [ "$TRAVIS_OS_NAME" == "osx" ]; then

      if [ "$PROJ" == "cpython" ]; then
        curl -L https://www.python.org/ftp/python/3.6.6/python-3.6.6-macosx10.9.pkg -o $HOME/python.pkg
        echo "Install python pkg"
        sudo installer -store -pkg $HOME/python.pkg -target /
      fi

      if [ "$PROJ" == "mkl" ]; then
        #don't put in download dir as will be cached and we can use direct url instead
        curl -L http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/13012/m_mkl_2018.3.185.dmg -o $HOME/mkl.dmg
        echo "Mount mkl dmg"
        hdiutil mount $HOME/mkl.dmg
        sleep 10
        cp /Volumes/m_mkl_2018.3.185/m_mkl_2018.3.185.app/Contents/MacOS/silent.cfg $HOME/silent.cfg
        sed -i -e 's/decline/accept/g' $HOME/silent.cfg
        sudo /Volumes/m_mkl_2018.3.185/m_mkl_2018.3.185.app/Contents/MacOS/install.sh -s $HOME/silent.cfg; export BREW_STATUS=$?
        echo "mkl status $BREW_STATUS"
        if [ $BREW_STATUS -ne 0 ]; then
          echo "mkl Failed"
          exit $BREW_STATUS
        fi
      fi

      if [[ "$PROJ" =~ cuda ]] || [[ "$EXT" =~ gpu ]]; then
        echo "installing cuda.."
        #don't put in download dir as will be cached and we can use direct url instead
        curl -L https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda_9.2.64_mac -o $HOME/cuda_9.2.64_mac.dmg
        curl -L https://developer.nvidia.com/compute/cuda/9.2/Prod/patches/1/cuda_9.2.64.1_mac -o $HOME/cuda_9.2.64.1_mac.dmg
        curl -L https://developer.download.nvidia.com/compute/redist/cudnn/v7.2.1/cudnn-9.2-osx-x64-v7.2.1.38.tgz -o $HOME/cudnn-9.2-osx-x64-v7.2.1.38.tgz

        echo "Mount dmg"
        hdiutil mount $HOME/cuda_9.2.64_mac.dmg
        sleep 5
        ls -ltr /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS 
        sudo /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS/CUDAMacOSXInstaller --accept-eula --no-window; export BREW_STATUS=$? 
        echo "Brew status $BREW_STATUS"
        if [ $BREW_STATUS -ne 0 ]; then
          echo "Brew Failed"
          exit $BREW_STATUS
        fi
        hdiutil mount $HOME/cuda_9.2.64.1_mac.dmg
        sleep 5
        ls -ltr /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS 
        sudo /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS/CUDAMacOSXInstaller --accept-eula --no-window; export BREW_STATUS=$? 
        echo "Brew status $BREW_STATUS"
        if [ $BREW_STATUS -ne 0 ]; then
          echo "Brew Failed"
          exit $BREW_STATUS
        fi

        tar xvf $HOME/cudnn-9.2-osx-x64-v7.2.1.38.tgz
        sudo cp ./cuda/include/*.h /usr/local/cuda/include/
        sudo cp ./cuda/lib/*.dylib /usr/local/cuda/lib/
        sudo cp ./cuda/lib/*.a /usr/local/cuda/lib/
        sudo cp /usr/local/cuda/lib/* /usr/local/lib/
      fi

      if [ "$PROJ" == "tensorflow" ]; then
        echo "adding bazel for tensorflow"
        curl -L https://github.com/bazelbuild/bazel/releases/download/0.15.2/bazel-0.15.2-installer-darwin-x86_64.sh -o $HOME/bazel.sh; export CURL_STATUS=$?
        if [ "$CURL_STATUS" != "0" ]; then
          echo "Download failed here, so can't proceed with the build.. Failing.."
          exit 1
        fi
        sudo bash $HOME/bazel.sh
     fi
fi  



echo "Running install for $PROJ"
if  [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]] || [[ "$OS" =~ android ]]; then
   DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
   echo "container id is $DOCKER_CONTAINER_ID"
    if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
       echo "Not a pull request so attempting to deploy using docker"
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets; source scl_source enable $SCL_ENABLE || true; mvn clean deploy -B -U --settings ./ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE \$BUILD_COMPILER \$BUILD_OPTIONS \$BUILD_ROOT -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT -pl .,$PROJ"; export BUILD_STATUS=$?
       if [ $BUILD_STATUS -eq 0 ]; then
         echo "Deploying platform"
         for i in ${PROJ//,/ }
         do
          docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd $HOME/build/javacpp-presets/$i; source scl_source enable $SCL_ENABLE || true; mvn clean deploy -B -U --settings ../ci/settings.xml -f platform/pom.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT "; export BUILD_STATUS=$?
          if [ $BUILD_STATUS -ne 0 ]; then
           echo "Build Failed"
           exit $BUILD_STATUS
          fi
         done
       fi
        
     else
       echo "Pull request so install using docker"
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets; source scl_source enable $SCL_ENABLE || true; mvn clean install -B -U --settings ./ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE \$BUILD_COMPILER \$BUILD_OPTIONS \$BUILD_ROOT -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT -pl .,$PROJ"; export BUILD_STATUS=$?
    fi

   echo "Build status $BUILD_STATUS"
   if [ $BUILD_STATUS -ne 0 ]; then  
     echo "Build Failed"
     exit $BUILD_STATUS
   fi

else	
   echo "Building $PROJ, with additional build flags $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT"
   if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
      echo "Not a pull request so attempting to deploy"
      mvn clean deploy -B -U --settings $TRAVIS_BUILD_DIR/ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT -pl .,$PROJ; export BUILD_STATUS=$?
      if [ $BUILD_STATUS -eq 0 ]; then
        echo "Deploying platform step"
        for i in ${PROJ//,/ }
        do
	  cd $i
          mvn clean deploy -B -U --settings $TRAVIS_BUILD_DIR/ci/settings.xml -f platform -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT; export BUILD_STATUS=$?
          cd ..
        done
      fi
    else
      echo "Pull request so install only"
      mvn clean install -B -U --settings $TRAVIS_BUILD_DIR/ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS -Djavacpp.platform.extension=$EXT $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT -pl .,$PROJ; export BUILD_STATUS=$?
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

