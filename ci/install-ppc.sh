#!/bin/bash 
set -vx

while true; do echo .; sleep 60; done &

mkdir ./buildlogs
mkdir $TRAVIS_BUILD_DIR/downloads
ls -ltr $HOME/downloads
ls -ltr $HOME/.m2
pip install requests
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

export BUILD_COMPILER=-Djavacpp.platform.compiler=powerpc64le-linux-gnu-g++
export BUILD_OPTIONS=-Djava.library.path=

echo "Starting docker for ppc cross compile"
docker run -d -ti -e CI_DEPLOY_USERNAME -e CI_DEPLOY_PASSWORD -e GPG_PASSPHRASE -e STAGING_REPOSITORY -e "container=docker" -v $HOME:$HOME -v $TRAVIS_BUILD_DIR/../:$HOME/build -v /sys/fs/cgroup:/sys/fs/cgroup ubuntu:trusty /sbin/init
DOCKER_CONTAINER_ID=$(docker ps | grep trusty | awk '{print $1}')
echo "Container id is $DOCKER_CONTAINER_ID please wait while updates applied"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "dpkg --add-architecture ppc64el"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-security main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "sed -i 's/deb http/deb [arch=i386,amd64] http/g' /etc/apt/sources.list"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get update"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install python python2.7 python-minimal python2.7-minimal libgtk2.0-dev:ppc64el libasound2-dev:ppc64el libusb-dev:ppc64el libusb-1.0-0-dev:ppc64el zlib1g-dev:ppc64el libxcb1-dev:ppc64el"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install pkg-config gcc-powerpc64le-linux-gnu g++-powerpc64le-linux-gnu gfortran-powerpc64le-linux-gnu linux-libc-dev-ppc64el-cross binutils-multiarch default-jdk ant maven python python-dev python-numpy swig git file wget unzip tar bzip2 patch autoconf-archive autogen automake make libtool perl nasm yasm curl cmake3"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y remove libxdmcp-dev libx11-dev libxcb1-dev libxt-dev"

if [[ "$PROJ" =~ cuda ]]; then
   echo "Setting up for cuda build"
   cd $HOME/
   curl -L https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_ppc64el -o $HOME/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_ppc64el.deb
   curl -L https://developer.nvidia.com/compute/cuda/9.2/Prod/patches/1/cuda-repo-ubuntu1604-9-2-local-cublas-update-1_1.0-1_ppc64el -o $HOME/cuda-repo-ubuntu1604-9-2-local-cublas-update-1_1.0-1_ppc64el.deb
   curl -L http://developer.download.nvidia.com/compute/redist/cudnn/v7.1.4/cudnn-9.2-linux-ppc64le-v7.1.tgz -o $HOME/cudnn-9.2-linux-ppc64le-v7.1.tgz
   ar vx $HOME/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_ppc64el.deb
   tar xvf data.tar.xz
   mkdir $HOME/cudaFS
   cd var; find . -name *.deb | while read line; do ar vx $line; tar --totals -xf data.tar.xz -C $HOME/cudaFS; done
   cd ..
   rm -Rf var
   ar vx $HOME/cuda-repo-ubuntu1604-9-2-local-cublas-update-1_1.0-1_ppc64el.deb
   tar xvf data.tar.xz
   cd var; find . -name *.deb | while read line; do ar vx $line; tar --totals -xf data.tar.xz -C $HOME/cudaFS; done
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd /; cp -R $HOME/cudaFS/* ."
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /usr/local/cuda-9.2 /usr/local/cuda"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -s /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/cuda/lib64/libnvidia-ml.so"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar --totals -xf $HOME/cudnn-9.2-linux-ppc64le-v7.1.tgz -C /usr/local/"
fi

docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "powerpc64le-linux-gnu-gcc --version"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "gpg --version"

echo "Running install for $PROJ"
echo "container id is $DOCKER_CONTAINER_ID"
 if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
     echo "Not a pull request so attempting to deploy using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets; mvn clean deploy -B -U --settings ./ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
     if [ $BUILD_STATUS -eq 0 ]; then
       echo "Deploying platform"
       for i in ${PROJ//,/ }
       do
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd $HOME/build/javacpp-presets/$i; mvn clean deploy -B -U --settings ../ci/settings.xml -f platform/pom.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS"; export BUILD_STATUS=$?
        if [ $BUILD_STATUS -ne 0 ]; then
         echo "Build Failed"
         exit $BUILD_STATUS
        fi
       done
     fi
        
   else
     echo "Pull request so install using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn clean install -B -U --settings ./ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
 fi

 echo "Build status $BUILD_STATUS"
 if [ $BUILD_STATUS -ne 0 ]; then  
   echo "Build Failed"
   #echo "Dump of config.log output files found follows:"
   #find . -name config.log | xargs cat
   exit $BUILD_STATUS
 fi


docker stop $DOCKER_CONTAINER_ID
docker rm -v $DOCKER_CONTAINER_ID

