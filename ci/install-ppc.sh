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
docker run -d -ti -e CI_DEPLOY_USERNAME -e CI_DEPLOY_PASSWORD -e "container=docker" -v $HOME:$HOME -v $TRAVIS_BUILD_DIR/../:$HOME/build -v /sys/fs/cgroup:/sys/fs/cgroup ubuntu:trusty /sbin/init 
DOCKER_CONTAINER_ID=$(docker ps | grep trusty | awk '{print $1}')
echo "Container id is $DOCKER_CONTAINER_ID please wait while updates applied"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "dpkg --add-architecture ppc64el"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-security main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "sed -i 's/deb http/deb [arch=i386,amd64] http/g' /etc/apt/sources.list"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get update > /dev/null"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install python python2.7 python-minimal python2.7-minimal libgtk2.0-dev:ppc64el libasound2-dev:ppc64el libusb-dev:ppc64el libusb-1.0-0-dev:ppc64el zlib1g-dev:ppc64el libxcb1-dev:ppc64el > /dev/null"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install pkg-config gcc-powerpc64le-linux-gnu g++-powerpc64le-linux-gnu gfortran-powerpc64le-linux-gnu linux-libc-dev-ppc64el-cross binutils-multiarch default-jdk maven python python-dev python-numpy swig git file wget unzip tar bzip2 patch autogen automake make libtool perl nasm yasm curl cmake3 > /dev/null"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y remove libxdmcp-dev libx11-dev libxcb1-dev libxt-dev > /dev/null"

if [[ "$PROJ" =~ cuda ]]; then
   echo "Setting up for cuda build"
   cd $HOME/downloads 
   curl -L https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_ppc64el-deb -o $HOME/downloads/cuda-ppc64.deb
   curl -L http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.5/cudnn-9.0-linux-ppc64le-v7.tgz -o $HOME/downloads/cudnn.tgz
   ar vx $HOME/downloads/cuda-ppc64.deb
   tar xvf data.tar.xz
   mkdir $HOME/downloads/cudaFS
   cd var; find . -name *.deb | while read line; do ar vx $line; tar --totals -xf data.tar.xz -C $HOME/downloads/cudaFS; done
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd /; cp -R $HOME/downloads/cudaFS/* ." 
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /usr/local/cuda-9.0 /usr/local/cuda"  
   #docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so"  
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar --totals -xzf $HOME/downloads/cudnn.tgz -C /usr/local/"
   #as we're downloading each time, clear downloads dir to prevent time wasted/errors with cache updates
   rm -rf $HOME/downloads
fi

echo "Running install for $PROJ"
echo "container id is $DOCKER_CONTAINER_ID"
if [[ "$PROJ" =~ tensorflow ]] || [[ "$PROJ" =~ openblas ]]; then
   echo "redirecting log output, tailing log every 5 mins to prevent timeout.."
   while true; do echo .; docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tail -10 $HOME/build/javacpp-presets/buildlogs/$PROJ.log"; sleep 300; done &
   if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
     echo "Not a pull request so attempting to deploy using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn clean deploy -B -U -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.test.skip=true -Dmaven.javadoc.skip=true $BUILD_COMPILER $BUILD_ROOT -Djavacpp.platform=$OS -l $HOME/build/javacpp-presets/buildlogs/$PROJ.log -pl .,$PROJ"; export BUILD_STATUS=$?
   else
     echo "Pull request so install using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn clean install -B -U --settings ./ci/settings.xml -Djavacpp.copyResources -Dmaven.test.skip=true $BUILD_COMPILER $BUILD_ROOT -Dmaven.javadoc.skip=true -Djavacpp.platform=$OS -l $HOME/build/javacpp-presets/buildlogs/$PROJ.log -pl .,$PROJ"; export BUILD_STATUS=$?
   fi
 else
   if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
     echo "Not a pull request so attempting to deploy using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn clean deploy -B -U -Djavacpp.copyResources --settings ./ci/settings.xml -Dmaven.test.skip=true -Dmaven.javadoc.skip=true $BUILD_COMPILER $BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
     if [ $BUILD_STATUS -eq 0 ]; then
       echo "Deploying platform"
       for i in ${PROJ//,/ }
       do
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd $HOME/build/javacpp-presets/$i; mvn -B -U -f platform/pom.xml -Djavacpp.platform=$OS --settings ../ci/settings.xml deploy"; export BUILD_STATUS=$?
        if [ $BUILD_STATUS -ne 0 ]; then
         echo "Build Failed"
         exit $BUILD_STATUS
        fi
       done
     fi
        
   else
     echo "Pull request so install using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn clean install -B -U --settings ./ci/settings.xml -Djavacpp.copyResources -Dmaven.test.skip=true -Dmaven.javadoc.skip=true $BUILD_COMPILER $BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
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

