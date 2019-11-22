#!/bin/bash 
set -vx

# Prevent Travis CI from terminating builds after 10 minutes with no output
while true; do uptime; sleep 60; done &

# Abort before the maximum build time to be able to save the cache
# (needs to be less than 2 hours for this to work on Mac as well)
(sleep 7000; sudo killall -s SIGINT java; sudo killall bazel) &

# Allocate a swapfile on Linux as it's not enabled by default
sudo fallocate -l 4GB /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
df -h

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

export BUILD_COMPILER=-Djavacpp.platform.compiler=powerpc64le-linux-gnu-g++
export BUILD_OPTIONS=-Djava.library.path=/usr/powerpc64le-linux-gnu/lib/

echo "Starting docker for ppc cross compile"
docker run -d -ti -e CI_DEPLOY_USERNAME -e CI_DEPLOY_PASSWORD -e GPG_PASSPHRASE -e STAGING_REPOSITORY -v $HOME:$HOME -v $TRAVIS_BUILD_DIR/../:$HOME/build ubuntu:xenial /bin/bash
DOCKER_CONTAINER_ID=$(docker ps | grep xenial | awk '{print $1}')
echo "Container id is $DOCKER_CONTAINER_ID please wait while updates applied"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "dpkg --add-architecture ppc64el"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports xenial main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports xenial-updates main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports xenial-backports main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports xenial-security main restricted universe multiverse" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec 'echo "deb [arch=amd64] http://ppa.launchpad.net/openjdk-r/ppa/ubuntu xenial main" >> /etc/apt/sources.list'
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "sed -i 's/deb http/deb [arch=i386,amd64] http/g' /etc/apt/sources.list"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EB9B1D8886F44E2A"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get update"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install python python2.7 python-minimal python2.7-minimal libgtk2.0-dev:ppc64el libasound2-dev:ppc64el libusb-dev:ppc64el libusb-1.0-0-dev:ppc64el libffi-dev:ppc64el zlib1g-dev:ppc64el libxcb1-dev:ppc64el"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "apt-get -y install pkg-config ccache gcc-powerpc64le-linux-gnu g++-powerpc64le-linux-gnu gfortran-powerpc64le-linux-gnu linux-libc-dev-ppc64el-cross binutils-multiarch openjdk-8-jdk-headless ant python python-dev swig git file wget unzip tar bzip2 patch autoconf-archive autogen automake make libtool perl nasm yasm curl cmake libffi-dev zlib1g-dev"

if [[ "$PROJ" =~ cuda ]]; then
   echo "Setting up for cuda build"
   cd $HOME/
   curl -L http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_ppc64el.deb -o $HOME/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_ppc64el.deb
   curl -L https://developer.download.nvidia.com/compute/redist/cudnn/v7.6.4/cudnn-10.1-linux-ppc64le-v7.6.4.38.tgz -o $HOME/cudnn-10.1-linux-ppc64le-v7.6.4.38.tgz
   python $TRAVIS_BUILD_DIR/ci/gDownload.py 1gZKnRGxBw77uClSSKINfWsylSu1si125 $HOME/nccl_ppc64le.txz
   ar vx $HOME/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_ppc64el.deb
   tar xvf data.tar.xz
   mkdir $HOME/cudaFS
   cd var; find . -name *.deb | while read line; do ar vx $line; tar --totals -xf data.tar.xz -C $HOME/cudaFS; done
   cd ..
   rm -Rf var
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd /; cp -R $HOME/cudaFS/* ."
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /usr/local/cuda-10.1 /usr/local/cuda"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/cuda/lib64/libnvidia-ml.so"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar hxvf $HOME/cudnn-10.1-linux-ppc64le-v7.6.4.38.tgz -C /usr/local/"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar hxvf $HOME/nccl_ppc64le.txz --strip-components=1 -C /usr/local/cuda/"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mv /usr/local/cuda/lib/* /usr/local/cuda/lib64/"
   # work around issues with CUDA 10.1
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mv /usr/include/cublas* /usr/include/nvblas* /usr/local/cuda/include/"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mv /usr/lib/powerpc64le-linux-gnu/libcublas* /usr/lib/powerpc64le-linux-gnu/libnvblas* /usr/local/cuda/lib64/"
   docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "for f in /usr/local/cuda/lib64/*.so.10; do ln -s \$f \$f.1; done"
fi

docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "powerpc64le-linux-gnu-gcc --version"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "gpg --version"

docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf $HOME/.m2 /root/.m2"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf $HOME/.cache /root/.cache"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf $HOME/.ccache /root/.ccache"

docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "curl -L https://archive.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz -o $HOME/apache-maven-3.3.9-bin.tar.gz"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "tar xzf $HOME/apache-maven-3.3.9-bin.tar.gz -C /opt/"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "ln -sf /opt/apache-maven-3.3.9/bin/mvn /usr/bin/mvn"
docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "mvn -version"

echo "Running install for $PROJ"
echo "container id is $DOCKER_CONTAINER_ID"
 if [ "$TRAVIS_PULL_REQUEST" = "false" ]; then 
     echo "Not a pull request so attempting to deploy using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets; mvn clean deploy -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings ./ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
     if [ $BUILD_STATUS -eq 0 ]; then
       echo "Deploying platform"
       for i in ${PROJ//,/ }
       do
        docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec "cd $HOME/build/javacpp-presets/$i; mvn clean deploy -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings ../ci/settings.xml -f platform/pom.xml -Dmaven.test.skip=true $MAVEN_RELEASE -Djavacpp.platform=$OS"; export BUILD_STATUS=$?
        if [ $BUILD_STATUS -ne 0 ]; then
         echo "Build Failed"
         exit $BUILD_STATUS
        fi
       done
     fi
        
   else
     echo "Pull request so install using docker"
     docker exec -ti $DOCKER_CONTAINER_ID /bin/bash -xec ". $HOME/vars.list; cd $HOME/build/javacpp-presets;mvn clean install -B -U -Dmaven.repo.local=$HOME/.m2/repository --settings ./ci/settings.xml -Dmaven.test.skip=true $MAVEN_RELEASE $BUILD_COMPILER $BUILD_OPTIONS $BUILD_ROOT -Djavacpp.platform=$OS -pl .,$PROJ"; export BUILD_STATUS=$?
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

sudo chown -R travis:travis $HOME
du -csh $HOME/* $HOME/.m2/* $HOME/.cache/* $HOME/.ccache/* $HOME/downloads/*
exit 0

