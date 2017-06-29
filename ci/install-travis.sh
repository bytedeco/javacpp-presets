#!/bin/bash 
set -vx
#export

mkdir ./buildlogs
mkdir $TRAVIS_BUILD_DIR/downloads
ls -ltr $HOME/downloads
ls -ltr $HOME/.m2
pip install requests
export PYTHON_BIN_PATH=$(which python) # For tensorflow
touch $HOME/vars.list

if [ "$TRAVIS_OS_NAME" == "osx" ]; then export JAVA_HOME=$(/usr/libexec/java_home); fi

if [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]] || [[ "$OS" =~ android ]]; then
  echo "Starting docker for x86_64 and x86 linux"
  docker run -d -ti -e CI_DEPLOY_USERNAME -e CI_DEPLOY_PASSWORD -e "container=docker" -v $HOME:$HOME -v $TRAVIS_BUILD_DIR/../:$HOME/build -v /sys/fs/cgroup:/sys/fs/cgroup nvidia/cuda:8.0-cudnn6-devel-centos7 /usr/sbin/init > /dev/null
  DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
  echo "Container id is $DOCKER_CONTAINER_ID please wait while updates applied"
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -y install epel-release" > /dev/null
  docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "yum -y install clang gcc-c++ gcc-gfortran java-devel maven python numpy swig git file which wget unzip tar bzip2 gzip xz patch make cmake3 libtool perl nasm yasm alsa-lib-devel freeglut-devel glfw-devel gtk2-devel libusb-devel libusb1-devel zlib-devel openblas-devel" > /dev/null
  if [ "$OS" == "linux-x86" ]; then
    docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "rpm -qa | sed s/.x86_64$/.i686/ | xargs yum -y install > /dev/null"
  fi
  
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
  if [ "$PROJ" == "mkl" ]; then
         #don't put in download dir as will be cached and we can use direct url instead
         curl -L http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/11306/l_mkl_2017.2.174.tgz -o $HOME/mkl.tgz
         tar xzvf $HOME/mkl.tgz -C $TRAVIS_BUILD_DIR/../
         sed -i -e 's/decline/accept/g' $TRAVIS_BUILD_DIR/../l_mkl_2017.2.174/silent.cfg
         docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "$HOME/build/l_mkl_2017.2.174/install.sh -s $HOME/build/l_mkl_2017.2.174/silent.cfg"
  fi
  if [ "$PROJ" == "tensorflow" ]; then
        echo "adding bazel for tensorflow"
        curl -L https://github.com/bazelbuild/bazel/releases/download/0.5.1/bazel-0.5.1-installer-linux-x86_64.sh -o $HOME/downloads/bazel.sh; export CURL_STATUS=$?
        if [ "$CURL_STATUS" != "0" ]; then
          echo "Download failed here, so can't proceed with the build.. Failing.."
          exit 1  
        fi
         docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "bash $HOME/downloads/bazel.sh"
  fi 
fi


if [ "$OS" == "linux-armhf" ]; then
	echo "Setting up tools for linux-armhf build"
	sudo dpkg --add-architecture i386
	sudo apt-get update
	sudo apt-get -y install libc6:i386 libncurses5:i386 libstdc++6:i386 lib32z1
	sudo apt-get -y install clang git file  wget unzip tar bzip2 gzip  patch automake libtool perl nasm yasm libasound2-dev freeglut3-dev libglfw3-dev libgtk2.0-dev libusb-dev zlib1g
	git -C $HOME clone https://github.com/raspberrypi/tools
	export PATH=$PATH:$HOME/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian/bin
	export BUILD_COMPILER=-Djavacpp.platform.compiler=arm-linux-gnueabihf-g++
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

if [ "$OS" == "linux-ppc64le" ]; then
      echo "Setting up tools for linux-ppc64le  build"
      sudo apt-get install gfortran-powerpc64le-linux-gnu gcc-powerpc64le-linux-gnu g++-powerpc64le-linux-gnu gfortran-powerpc64le-linux-gnu linux-libc-dev-ppc64el-cross binutils-multiarch
      sudo dpkg --add-architecture ppc64el
      sudo add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty main restricted universe multiverse"
      sudo add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-updates main restricted universe multiverse"
      sudo add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-backports main restricted universe multiverse"
      sudo add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-security main restricted universe multiverse"
      sudo apt-get update
      sudo apt-get install libcairo2-dev:ppc64el libglib2.0-dev:ppc64el libatk1.0-dev:ppc64el libglfw-dev:ppc64el libfreetype6-dev:ppc64el libasound2-dev:ppc64el freeglut3-dev:ppc64el libgtk2.0-dev:ppc64el libusb-dev:ppc64el zlib1g:ppc64el gir1.2-atk-1.0:ppc64el gir1.2-gtk-2.0:ppc64el libpango1.0-dev:ppc64el gir1.2-pango-1.0:ppc64el libgdk-pixbuf2.0-dev:ppc64el gir1.2-gdkpixbuf-2.0:ppc64el gir1.2-freedesktop:ppc64el gir1.2-glib-2.0:ppc64el  libgirepository-1.0-1:ppc64el
      sudo ln -s  /usr/lib/powerpc64le-linux-gnu/glib-2.0/include/glibconfig.h /usr/include/glib-2.0/glibconfig.h
      sudo ln -s  /usr/lib/powerpc64le-linux-gnu/gtk-2.0/include/gdkconfig.h /usr/include/gtk-2.0/gdk/gdkconfig.h
      export BUILD_COMPILER="-Djavacpp.platform.compiler=powerpc64le-linux-gnu-g++"
      
      #if [[ "$PROJ" =~ cuda ]] || [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ caffe ]]; then
      if [[ "$PROJ" =~ cuda ]]; then
	echo "installing cuda.."
	curl -L -o cuda.deb https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2v2_8.0.61-1_ppc64el-deb 
        ar x cuda.deb
        tar xvf data.tar.gz
        sudo dpkg --force-all -i ./var/cuda-repo-8-0-local-ga2v2/*.deb
        if [[ $(find $HOME/downloads/cudnn-8.0-linux-ppc64le-v6.0.tgz -type f -size +1000000c 2>/dev/null) ]]; then
          echo "Found cudnn in cache and size seems ok" 
        else
          echo "Downloading cudnn as not found in cache" 
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShdFJveFVxWlF3UnM $HOME/downloads/cudnn-8.0-linux-ppc64le-v6.0.tgz
        fi
        tar xvf $HOME/downloads/cudnn-8.0-linux-ppc64le-v6.0.tgz
        sudo mv ./cuda/targets/ppc64le-linux/include/* /usr/local/cuda/targets/ppc64le-linux/include
        sudo mv ./cuda/targets/ppc64le-linux/lib/* /usr/local/cuda/targets/ppc64le-linux/lib
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
   DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
   #docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "pip install numpy" 

   curl -L https://dl.google.com/android/repository/android-ndk-r15b-linux-x86_64.zip -o $HOME/ndk.zip; export CURL_STATUS=$?
   if [ "$CURL_STATUS" != "0" ]; then
    echo "Download failed here, so can't proceed with the build.. Failing.."
    exit 1
   fi

   unzip -qq $HOME/ndk.zip -d $HOME/
   ln -s $HOME/android-ndk-r15b $HOME/android-ndk
   if [ "$PROJ" == "tensorflow" ]; then
     echo "modifying ndk version 14 to 12 as per tensorflow cppbuild.sh suggestion"
     sed -i 's/15/12/g' $HOME/android-ndk/source.properties
   fi
   echo "Android NDK setup done"
   echo "export ANDROID_NDK=$HOME/android-ndk/" | tee --append $HOME/vars.list
   cat $HOME/vars.list
   echo "export BUILD_ROOT=-Djavacpp.platform.root=$HOME/android-ndk/" | tee --append $HOME/vars.list
   if [ "$OS" == "android-arm" ]; then
      echo "export PATH=\$PATH:$HOME/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin" | tee --append $HOME/vars.list
      echo "export BUILD_COMPILER=-Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-g++" | tee --append $HOME/vars.list
   fi
   if [ "$OS" == "android-x86" ]; then
      echo "Setting build for android-x86"
      echo "export BUILD_COMPILER=-Djavacpp.platform.compiler=$HOME/android-ndk/toolchains/x86-4.9/prebuilt/linux-x86_64/bin/i686-linux-android-g++" | tee --append $HOME/vars.list
   fi
   if [ "$PROJ" == "tensorflow" ]; then
      echo "adding bazel for tensorflow"
      curl -L  https://github.com/bazelbuild/bazel/releases/download/0.5.1/bazel-0.5.1-installer-linux-x86_64.sh -o $HOME/bazel.sh; export CURL_STATUS=$?
      if [ "$CURL_STATUS" != "0" ]; then
        echo "Download failed here, so can't proceed with the build.. Failing.."
        exit 1
      fi
      docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "sudo bash $HOME/bazel.sh"
   fi
fi


echo "Download dependencies" 
if [ "$TRAVIS_OS_NAME" == "osx" ]; then
      if [[ "$PROJ" =~ mxnet ]]; then 
        export PKG_CONFIG_PATH=$TRAVIS_BUILD_DIR/opencv/cppbuild/macosx-x86_64/lib/pkgconfig
      fi 
      if [ "$PROJ" == "mkl" ]; then
        #don't put in download dir as will be cached and we can use direct url instead
        curl -L http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/11563/m_mkl_2017.3.181.dmg -o $HOME/mkl.dmg
        echo "Mount mkl dmg"
        hdiutil mount $HOME/mkl.dmg
        sleep 10
        cp /Volumes/m_mkl_2017.3.181/m_mkl_2017.3.181.app/Contents/MacOS/silent.cfg $HOME/silent.cfg
        sed -i -e 's/decline/accept/g' $HOME/silent.cfg
        sudo /Volumes/m_mkl_2017.3.181/m_mkl_2017.3.181.app/Contents/MacOS/install.sh -s $HOME/silent.cfg; export BREW_STATUS=$?
        echo "mkl status $BREW_STATUS"
        if [ $BREW_STATUS -ne 0 ]; then
          echo "mkl Failed"
          exit $BREW_STATUS
        fi
      fi

      if [[ "$PROJ" =~ cuda ]] || [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ caffe ]]; then 
        echo "installing cuda.."
        while true; do echo .; sleep 60; done &
        export CHILDPID=$!
        echo "Child PID $CHILDPID"
        #if [[ $(find $HOME/downloads/cuda.dmg -type f -size +1000000c 2>/dev/null) ]]; then
        if [[ $(hdiutil verify $HOME/downloads/cuda.dmg) ]]; then
          echo "Found cuda in cache and verified image ok" 
        else
          echo "Downloading cuda as problem in cache" 
          python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShUzFIUHhkQnBQWWM $HOME/downloads/cuda.dmg
        fi

        echo "Mount dmg"
        hdiutil mount $HOME/downloads/cuda.dmg
        sleep 5
        ls -ltr /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS 
        sudo /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS/CUDAMacOSXInstaller --accept-eula --no-window; export BREW_STATUS=$? 
        echo "Brew status $BREW_STATUS"
        if [ $BREW_STATUS -ne 0 ]; then
          echo "Brew Failed"
          exit $BREW_STATUS
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



echo "Running install for $PROJ"
if  [[ "$OS" == "linux-x86" ]] || [[ "$OS" == "linux-x86_64" ]] || [[ "$OS" =~ android ]]; then
   DOCKER_CONTAINER_ID=$(docker ps | grep centos | awk '{print $1}')
   echo "container id is $DOCKER_CONTAINER_ID"
   if [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ openblas ]]; then
     echo "redirecting log output, tailing log every 5 mins to prevent timeout.."
     while true; do echo .; docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tail -10 $HOME/build/javacpp-presets/buildlogs/$PROJ.log"; sleep 300; done &
     if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
       echo "Not a pull request so attempting to deploy using docker"
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn deploy -U -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.test.skip=true -Dmaven.javadoc.skip=true \$BUILD_COMPILER \$BUILD_ROOT -Djavacpp.platform=$OS -l $HOME/build/javacpp-presets/buildlogs/$PROJ.log -pl .,$PROJ"; export BUILD_STATUS=$?
     else
       echo "Pull request so install using docker"
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn install -U --settings ./ci/settings.xml -Djavacpp.copyResources -Dmaven.test.skip=true \$BUILD_COMPILER \$BUILD_ROOT -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS -l $HOME/build/javacpp-presets/buildlogs/$PROJ.log -pl .,$PROJ"; export BUILD_STATUS=$?
     fi
   else
     if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
       echo "Not a pull request so attempting to deploy using docker"
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn deploy -U -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.test.skip=true -Dmaven.javadoc.skip=true \$BUILD_COMPILER \$BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
       if [ $BUILD_STATUS -eq 0 ]; then
         echo "Deploying platform"
         for i in ${PROJ//,/ }
         do
          docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd $HOME/build/javacpp-presets/$i; mvn -U -f platform/pom.xml -Djavacpp.platform=$OS --settings ../ci/settings.xml deploy"; export BUILD_STATUS=$?
          if [ $BUILD_STATUS -ne 0 ]; then
           echo "Build Failed"
           exit $BUILD_STATUS
          fi
         done
       fi
        
     else
       echo "Pull request so install using docker"
       docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn install -U --settings ./ci/settings.xml -Djavacpp.copyResources -Dmaven.test.skip=true -Dmaven.javadoc.skip=true \$BUILD_COMPILER \$BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
     fi
   fi
   echo "Build status $BUILD_STATUS"
   if [ $BUILD_STATUS -ne 0 ]; then  
     echo "Build Failed"
     exit $BUILD_STATUS
   fi

else	
  if [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ openblas ]]; then
    echo "redirecting log output, tailing log every 5 mins to prevent timeout.."
    while true; do echo .; tail -10 $TRAVIS_BUILD_DIR/buildlogs/$PROJ.log; sleep 300; done &
    if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
      echo "Not a pull request so attempting to deploy"
      mvn deploy -U -Djavacpp.copyResources --settings $TRAVIS_BUILD_DIR/ci/settings.xml -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS $BUILD_COMPILER $BUILD_ROOT -l $TRAVIS_BUILD_DIR/buildlogs/$PROJ.log -pl .,$PROJ; export BUILD_STATUS=$?
    else
      echo "Pull request so install only"
      mvn install -U -Dmaven.javadoc.skip=true -Djavacpp.copyResources --settings $TRAVIS_BUILD_DIR/ci/settings.xml -Djavacpp.platform=$OS $BUILD_COMPILER $BUILD_ROOT -l $TRAVIS_BUILD_DIR/buildlogs/$PROJ.log -pl .,$PROJ; export BUILD_STATUS=$?
    fi
  else
    echo "Building $PROJ, with additional build flags $BUILD_COMPILER $BUILD_ROOT"
    if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
      echo "Not a pull request so attempting to deploy"
      mvn deploy -U --settings $TRAVIS_BUILD_DIR/ci/settings.xml -Djavacpp.copyResources -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS $BUILD_COMPILER $BUILD_ROOT -pl .,$PROJ; export BUILD_STATUS=$?
      if [ $BUILD_STATUS -eq 0 ]; then
        echo "Deploying platform step"
        for i in ${PROJ//,/ }
        do
	  cd $i
          mvn -U -f platform -Djavacpp.platform=$OS --settings $TRAVIS_BUILD_DIR/ci/settings.xml deploy; export BUILD_STATUS=$?
          cd ..
        done
      fi
    else
      echo "Pull request so install only"
      mvn install -U -Dmaven.javadoc.skip=true -Djavacpp.copyResources --settings $TRAVIS_BUILD_DIR/ci/settings.xml -Djavacpp.platform=$OS $BUILD_COMPILER $BUILD_ROOT -pl .,$PROJ; export BUILD_STATUS=$?
    fi
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

