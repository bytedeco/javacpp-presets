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

export BUILD_COMPILER=-Djavacpp.platform.compiler=powerpc64le-linux-gnu-g++

echo "Starting docker for ppc cross compile"
docker run -d -ti -e CI_DEPLOY_USERNAME -e CI_DEPLOY_PASSWORD -e "container=docker" -v $HOME:$HOME -v $TRAVIS_BUILD_DIR/../:$HOME/build -v /sys/fs/cgroup:/sys/fs/cgroup ubuntu:xenial /sbin/init 
DOCKER_CONTAINER_ID=$(docker ps | grep xenial | awk '{print $1}')
echo "Container id is $DOCKER_CONTAINER_ID please wait while updates applied"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get update"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install apt-utils software-properties-common pkg-config openjdk-8-jdk maven python python-dev python-numpy gfortran-powerpc64le-linux-gnu gcc-powerpc64le-linux-gnu g++-powerpc64le-linux-gnu gfortran-powerpc64le-linux-gnu linux-libc-dev-ppc64el-cross binutils-multiarch swig git file wget unzip tar bzip2 patch automake make libtool perl nasm yasm curl cmake"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "dpkg --add-architecture ppc64el"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports xenial main restricted universe multiverse"'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports xenial-updates main restricted universe multiverse"'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports xenial-backports main restricted universe multiverse"'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports xenial-security main restricted universe multiverse"'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get update"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install libasound2-dev:ppc64el libusb-dev:ppc64el libusb-1.0-0-dev:ppc64el zlib1g-dev:ppc64el libxcb1-dev:ppc64el"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install libcairo2-dev:ppc64el libpango1.0-dev:ppc64el libglib2.0-dev:ppc64el libatk1.0-dev:ppc64el libperl-dev:ppc64el libqt4-dev:ppc64el libgtk2.0-dev:ppc64el gir1.2-atk-1.0:ppc64el gir1.2-gtk-2.0:ppc64el gir1.2-pango-1.0:ppc64el libgdk-pixbuf2.0-dev:ppc64el gir1.2-gdkpixbuf-2.0:ppc64el gir1.2-freedesktop:ppc64el gir1.2-glib-2.0:ppc64el libgirepository-1.0-1:ppc64el"

if [[ "$PROJ" =~ cuda ]]; then
   echo "Setting up for cuda build"
   #cp /usr/include/cudnn.h /usr/local/cuda/include/  
   python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShdFlrT3FtNXFSRW8 $HOME/downloads/cudappc64.tar
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xvf $HOME/downloads/cudappc64.tar -C /usr/local/"  
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /usr/local/cuda-9.0 /usr/local/cuda"  
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so"  
   python $TRAVIS_BUILD_DIR/ci/gDownload.py 0B2xpvMUzviShdFJveFVxWlF3UnM $HOME/downloads/cudnnppc64.tar
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xvf $HOME/downloads/cudnnppc64.tar -C /usr/local/"  
fi

echo "Running install for $PROJ"
echo "container id is $DOCKER_CONTAINER_ID"
if [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ openblas ]]; then
   echo "redirecting log output, tailing log every 5 mins to prevent timeout.."
   while true; do echo .; docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tail -10 $HOME/build/javacpp-presets/buildlogs/$PROJ.log"; sleep 300; done &
   if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
     echo "Not a pull request so attempting to deploy using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn deploy -U -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.test.skip=true -Dmaven.javadoc.skip=true $BUILD_COMPILER $BUILD_ROOT -Djavacpp.platform=$OS -l $HOME/build/javacpp-presets/buildlogs/$PROJ.log -pl .,$PROJ"; export BUILD_STATUS=$?
   else
     echo "Pull request so install using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn install -U --settings ./ci/settings.xml -Djavacpp.copyResources -Dmaven.test.skip=true $BUILD_COMPILER $BUILD_ROOT -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS -l $HOME/build/javacpp-presets/buildlogs/$PROJ.log -pl .,$PROJ"; export BUILD_STATUS=$?
   fi
 else
   if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
     echo "Not a pull request so attempting to deploy using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn deploy -U -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.test.skip=true -Dmaven.javadoc.skip=true $BUILD_COMPILER $BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
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
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn install -U --settings ./ci/settings.xml -Djavacpp.copyResources -Dmaven.test.skip=true -Dmaven.javadoc.skip=true $BUILD_COMPILER $BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
   fi
 fi
 echo "Build status $BUILD_STATUS"
 if [ $BUILD_STATUS -ne 0 ]; then  
   echo "Build Failed"
   echo "Dump of config.log output files found follows:"
   find . -name config.log | xargs cat
   exit $BUILD_STATUS
 fi


docker stop $DOCKER_CONTAINER_ID
docker rm -v $DOCKER_CONTAINER_ID

