#!/bin/bash 
set -vx
export

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
         sed -i -e 's/sonatype-nexus-snapshots/intNexus/g' pom.xml
         sed -i -e 's|https://oss.sonatype.org/content/repositories/snapshots/|http://bytedeconexus.ddns.net:15081/nexus/content/repositories/bytedecoInt/|g' pom.xml
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
         sed -i -e 's/sonatype-nexus-snapshots/intNexus/g' pom.xml
         sed -i -e 's|https://oss.sonatype.org/content/repositories/snapshots/|http://bytedeconexus.ddns.net:15081/nexus/content/repositories/bytedecoInt/|g' pom.xml
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

