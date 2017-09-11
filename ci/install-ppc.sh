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


if [ "$OS" == "linux-ppc64le" ]; then
      echo "Setting up tools for linux-ppc64le  build"
      #sudo apt-get dist-upgrade 
      sudo apt-get install gcc gfortran-powerpc64le-linux-gnu gcc-powerpc64le-linux-gnu g++-powerpc64le-linux-gnu gfortran-powerpc64le-linux-gnu linux-libc-dev-ppc64el-cross binutils-multiarch
      sudo dpkg --add-architecture ppc64el
      sudo add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty main restricted universe multiverse"
      sudo add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-updates main restricted universe multiverse"
      sudo add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-backports main restricted universe multiverse"
      sudo add-apt-repository "deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports trusty-security main restricted universe multiverse"
      sudo apt-get update
      sudo apt-get clean 
      sudo apt-get autoclean 
      sudo apt-get dist-upgrade 
      sudo apt-get -f install 
      #sudo apt-get install libcairo2-dev:ppc64el libglib2.0-dev:ppc64el libatk1.0-dev:ppc64el libglfw-dev:ppc64el libfreetype6-dev:ppc64el libasound2-dev:ppc64el freeglut3-dev:ppc64el libgtk2.0-dev:ppc64el libusb-dev:ppc64el zlib1g:ppc64el gir1.2-atk-1.0:ppc64el gir1.2-gtk-2.0:ppc64el libpango1.0-dev:ppc64el gir1.2-pango-1.0:ppc64el libgdk-pixbuf2.0-dev:ppc64el gir1.2-gdkpixbuf-2.0:ppc64el gir1.2-freedesktop:ppc64el gir1.2-glib-2.0:ppc64el  libgirepository-1.0-1:ppc64el
      sudo apt-get install libcairo2-dev:ppc64el libglib2.0-dev:ppc64el libatk1.0-dev:ppc64el libfreetype6-dev:ppc64el libasound2-dev:ppc64el libgtk2.0-dev:ppc64el libusb-dev:ppc64el zlib1g:ppc64el gir1.2-atk-1.0:ppc64el gir1.2-gtk-2.0:ppc64el gir1.2-pango-1.0:ppc64el libgdk-pixbuf2.0-dev:ppc64el gir1.2-gdkpixbuf-2.0:ppc64el gir1.2-freedesktop:ppc64el gir1.2-glib-2.0:ppc64el  libgirepository-1.0-1:ppc64el
      sudo apt-get -f install 
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


echo "Running install for $PROJ"
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


